import torch
import random
import numpy as np
import os
from PIL import Image

from omegaconf import OmegaConf
from hydra.utils import instantiate

from torch.nn.utils.rnn import pad_sequence
from training.utils.structure import BackboneOut

def get_frame_indices(
    num_frames, vlen, sample='rand', fix_start=None, 
    input_fps=1, max_num_frames=-1, start=None, end=None
):
    # Calculate frame indices for the start and end times
    if start is not None or end is not None:
        start_frame = int(start * input_fps) if start is not None else 0
        end_frame = int(end * input_fps) if end is not None else vlen
        # Clamp the range to ensure it is within video length
        start_frame = max(0, start_frame)
        end_frame = min(vlen, end_frame)
        # Adjust video length for sampling
        vlen = end_frame - start_frame
    else:
        start_frame = 0
        end_frame = vlen

    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        # Split the video into `acc_samples` intervals, and sample from each interval
        intervals = np.linspace(start=start_frame, stop=end_frame, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
        
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e + start_frame for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices






def mask_downsampling(mask, size):
    mask = mask.unsqueeze(1)
    mask_outputs = torch.nn.functional.interpolate(
        mask,
        size=size,
        align_corners=False,
        mode="bilinear",
        antialias=True,  # use antialias for downsampling
    )
    mask_outputs = (mask_outputs >= 0.5).float().squeeze()
    return mask_outputs





def load_model_from_config(config_file, device="cuda", resolution=None, checkpoint_path=None):
    """
    Loads the model defined in the config file, and loads the weights from checkpoint.
    
    Args:
        config_file (str): Path to the YAML configuration file.
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Returns:
        model: The instantiated model loaded with checkpoint weights.
    """
    # Load the configuration.
    cfg = OmegaConf.load(config_file)
    if checkpoint_path != None:
        cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = checkpoint_path
    if resolution != None:
        cfg.scratch.resolution = resolution
    
    
    # The model configuration is under the trainer.model key (as in your YAML).
    model_cfg = cfg.trainer.model
    model = instantiate(model_cfg)
    
    # --- Load checkpoint ---
    # The checkpoint configuration is defined under trainer.checkpoint.
    # It uses a model_weight_initializer with an embedded state dict loader.
    # Here we extract the checkpoint path.
    ckpt_config = cfg.trainer.checkpoint
    checkpoint_path = ckpt_config.model_weight_initializer.state_dict.checkpoint_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint. (Map to CPU; then move the model to the target device.)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # The checkpoint dictionary should have a 'model' key containing the state dict.
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()
    return model





import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np




def group_and_pad_videos(
    video_list,
    object_num_list,
    backbone_out_list,
    max_deviation: int = 2,
    max_object_num: int = 512
):
    """
    Groups videos under two constraints:
      1. Within each group, max(frame_count) - min(frame_count) <= max_deviation
      2. Sum of object counts in each group <= max_object_num

    For each group:
      - Pads all videos (shape: (T, H, W, C)) to the same length (T_max) with zero‐frames,
        then stacks into a tensor (B, T_max, H, W, C) via pad_sequence.
      - Pads all backbone output sequences (instances of BackboneOut) by duplicating their
        last element until length == T_max, producing a BackboneOut sequence of length T_max.

    **Important**: We assume there exists a class `BackboneOut` such that:
      - `bb_seq` (each element of backbone_out_list) is a non‐empty instance of `BackboneOut`.
      - `BackboneOut(elt)` constructs a sequence initialized with `elt`.
      - Each `BackboneOut` supports `len(seq)`, indexing `seq[i]`, slicing `seq[i:j]`, and `seq.append(...)`.

    Args:
      video_list (List[Tensor]): each tensor has shape (T, H, W, C).
      object_num_list (List[int]): object count per video.
      backbone_out_list (List[BackboneOut]): each is a non‐empty sequence of length T.
      max_deviation (int): allowed frame‐count deviation within a group.
      max_object_num (int): allowed sum of objects within a group.

    Returns:
      batched_videos    (List[Tensor]): each tensor has shape (B, T_max, H, W, C).
      padded_backbones  (List[List[BackboneOut]]): for each group, a list of padded BackboneOut
                                                 sequences, each of length T_max.
      group_indices     (List[List[int]]): original indices of videos per group.
    """
    assert len(video_list) == len(object_num_list) == len(backbone_out_list), \
        "All input lists must have the same length."

    # 1. Pair up (original_index, video, object_count, backbone_sequence)
    indexed = []
    for i in range(len(video_list)):
        indexed.append((i, video_list[i], object_num_list[i], backbone_out_list[i]))
    # 2. Sort by temporal length T of the video
    indexed.sort(key=lambda x: x[1].shape[0])

    groups_vid = []      # List of lists of video tensors
    groups_bb = []       # List of lists of BackboneOut sequences
    group_indices = []   # List of lists of original indices

    current_vid_group = []
    current_bb_group = []
    current_idx_group = []
    group_min_T = group_max_T = None
    group_obj_sum = 0

    # 3. Build groups under both constraints
    for idx, vid, obj_count, bb_seq in indexed:
        T = vid.shape[0]
        if not current_vid_group:
            # Start a fresh group
            current_vid_group = [vid]
            current_bb_group = [bb_seq]
            current_idx_group = [idx]
            group_min_T = group_max_T = T
            group_obj_sum = obj_count
        else:
            new_min = group_min_T if group_min_T < T else T
            new_max = group_max_T if group_max_T > T else T
            if (new_max - new_min <= max_deviation
                and group_obj_sum + obj_count <= max_object_num):
                # Add to current group
                current_vid_group.append(vid)
                current_bb_group.append(bb_seq)
                current_idx_group.append(idx)
                group_min_T, group_max_T = new_min, new_max
                group_obj_sum += obj_count
            else:
                # Finalize the old group
                groups_vid.append(current_vid_group)
                groups_bb.append(current_bb_group)
                group_indices.append(current_idx_group)
                # Start a new group
                current_vid_group = [vid]
                current_bb_group = [bb_seq]
                current_idx_group = [idx]
                group_min_T = group_max_T = T
                group_obj_sum = obj_count

    # 4. Append the last group if present
    if current_vid_group:
        groups_vid.append(current_vid_group)
        groups_bb.append(current_bb_group)
        group_indices.append(current_idx_group)

    # 5. For each group: pad videos via pad_sequence, pad backbones by BackboneOut constructor
    batched_videos = []
    padded_backbones = []

    for vid_group, bb_group in zip(groups_vid, groups_bb):
        # Determine T_max from the video group
        T_max = 0
        for v in vid_group:
            if v.shape[0] > T_max:
                T_max = v.shape[0]

        # 5a. Pad videos with pad_sequence (zero-padding along time dim)
        batched_v = pad_sequence(vid_group, batch_first=True)  # shape: (B, T_max, H, W, C)
        batched_videos.append(batched_v)

        # 5b. Pad each BackboneOut sequence by duplicating its last element
        group_padded_bb = []
        for bb_seq in bb_group:
            T_seq = len(bb_seq)  # number of elements in this sequence
            assert T_seq > 0, "BackboneOut sequences must be non-empty."

            # Initialize with the first element
            padded_seq = BackboneOut(bb_seq[0])
            # Append original elements 1 to T_seq-1
            for t in range(1, T_seq):
                padded_seq.append(bb_seq[t])

            # If T_seq < T_max, duplicate last element
            if T_seq < T_max:
                last_elem = bb_seq[T_seq - 1]
                for _ in range(T_max - T_seq):
                    padded_seq.append(last_elem)

            # Now padded_seq has length T_max
            group_padded_bb.append(padded_seq)

        padded_backbones.append(group_padded_bb)
    return batched_videos, padded_backbones, group_indices




def save_video_per_frame(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(folder_path, f"{i:02d}.png"))
        
        
        

class GPU_time_wrapper:
    def __init__(self):
        self.collections = []
    
    def update(self, func, args, process_name):
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t1.record()
        return_val = func(**args)
        t2.record()
        torch.cuda.synchronize()
        self.collections.append(t1.elapsed_time(t2)/1000)
        # print(process_name, "processing time:", t1.elapsed_time(t2)/1000)
        return return_val
    
    def calculate(self):
        return np.mean(np.array(self.collections))
    
    