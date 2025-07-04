import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import random
import cv2
# Import the BatchedVideoDatapoint class from your training utils.
# (Make sure that your PYTHONPATH includes the root directory of your project.)
from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np
import decord
import torch.nn.functional as F

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



def run_segmentation_batches(all_images, model, threshold=0.1, max_mask_num=128, min_mask_area = 16, chunk_size=8):

    chunks = list(torch.split(all_images, chunk_size))
    # extract output logits from model inference
    logits = []
    for images in chunks:
        with torch.no_grad():
            o = model(pixel_values=images)
            log = o.logits.float()
            logits.append(log)
    logits = torch.cat(logits)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=list(images.shape[2:]),
        mode="bilinear",
        align_corners=False,
    )
    # eccencially an edge detector
    probs = torch.sigmoid(upsampled_logits)[:, 0].cpu().detach().numpy()
    binarilized = (probs < threshold).astype(np.uint8) 
    
    # convert edge detection to segmentation mask
    mask_batch, mask_lens = [], []
    for bi in binarilized: 
        _, labels = cv2.connectedComponents(bi)
        masks = [labels == i for i in range(1, labels.max() + 1)]
        
        # kernel = np.ones((3,3), np.uint8)  # 3x3 structuring element
        # for i, m in enumerate(masks):
        #     masks[i] = cv2.dilate(m.astype(np.uint8), kernel, iterations=2).astype(bool)
        
        masks.sort(key=lambda x: x.sum(), reverse=True)
        # filter out masks that are too small
        try: min_index = (torch.tensor([x.sum() for x in masks]) <= min_mask_area).nonzero()[0].item()
        except: min_index = len(masks)
        masks = masks[:min_index]
        # filter out masks that exceed maximum number
        masks = masks[:max_mask_num]
        mask_batch.append(masks)
        mask_lens.append(len(masks))
        
    return mask_batch, mask_lens




def load_model_from_config(config_file, device="cuda"):
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



def sample_and_save_short_video(video_path: np.array,sample: bool = True, frame_number: int = 8):
        """_uniformly sample desired frames from original video, and save to disk (help dataloading during training)
        
        Args:
            save_video (np.array): shape (T,W,H,C)
            video_path (str)
            sample (bool, optional)
        """
        
        
        video_reader = decord.VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        
        if sample:
            indices = get_frame_indices(num_frames=frame_number, sample='middle', vlen=vlen)
            save_video = video_reader.get_batch(indices).asnumpy()
        else:
            indices = [_ for _ in range(vlen)]
            save_video = video_reader.get_batch(indices).asnumpy()
            
        return save_video
    
    
def preprocess_video(video, image_size, image_mean = [0.485, 0.456, 0.406], image_std = [0.229, 0.224, 0.225]):
    # Resize and normalize for video
    # T, W, H, _ = video.shape
    video = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    video = F.interpolate(video, size=(image_size, image_size), mode='bilinear', align_corners=False)
    mean = torch.tensor(image_mean).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
    std = torch.tensor(image_std).view(1, 3, 1, 1)    # Shape (1, C, 1, 1)
    video = video.float() / 255.0  # Scale to [0, 1]
    video = (video - mean) / std  # Normalize
    return video
    

def main():
    import sys
    
    config_file = 'sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model from configuration and checkpoint.
    segmentor = AutoModelForSemanticSegmentation.from_pretrained("chendelong/DirectSAM-1800px-0424").to('cuda').eval()
        
    model = load_model_from_config(config_file, device)
    
    num_frames = 8
    video1 = sample_and_save_short_video('/weka/chenhaoz/home/videotok/breakdance-flare.mp4', frame_number=num_frames)
    video2 = sample_and_save_short_video('/weka/chenhaoz/home/videotok/dogs-jump.mp4', frame_number=num_frames)
    input_video1 = preprocess_video(video1, image_size=512).cuda()
    input_video2 = preprocess_video(video2, image_size=512).cuda()
    
    
    
    starting_frames = torch.stack([input_video1[0], input_video2[0]])
    mask_batch, mask_lens = run_segmentation_batches(starting_frames, segmentor, max_mask_num=128)
    
    # construct video batch
    img_batch = torch.stack([input_video1, input_video2], dim=1)
    # construct masks
    masks = torch.tensor(np.concatenate([np.stack(mask_frame) for mask_frame in mask_batch])).unsqueeze(0).cuda().bool()
    # construct obj_to_frame_idx
    frame_idx = torch.arange(num_frames).unsqueeze(1).repeat(1,sum(mask_lens))
    video_idx = torch.tensor([0]*mask_lens[0]+[1]*mask_lens[1]).unsqueeze(0).repeat(num_frames,1)
    obj_to_frame_idx = torch.stack([frame_idx, video_idx], dim=-1).cuda()
    input = BatchedVideoDatapoint(img_batch=img_batch, obj_to_frame_idx=obj_to_frame_idx, masks=masks, metadata=None, dict_key="a")

    # Get the model output.
    with torch.no_grad():
        output = model(input)
    
    #  list of length frame number. each element is a dict, in which 'pred_masks_high_res' is a tensor of torch.Size([num_object, 1, H, W])
    output_masks = torch.stack([(l['pred_masks_high_res']>0).squeeze() for l in output]) # [frame_numer, n_object, h, w]
    
    # for i in range(output_masks.shape[1]):
    #     object_mask = torch.cat([output_masks[:,i][j] for j in range(output_masks.shape[0])]).cpu().numpy()
    #     Image.fromarray(object_mask).save(f"mask_{i:02d}.png")
        

if __name__ == "__main__":
    main()