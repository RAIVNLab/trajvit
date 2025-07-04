import torch
import random
import cv2
from PIL import Image
import numpy as np
import decord
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import os
import glob
import imageio
from einops import rearrange, repeat
from scenedetect import detect, ContentDetector, AdaptiveDetector, HistogramDetector, HashDetector, ThresholdDetector
from collections import Counter
from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from training.utils.structure import BackboneOut
from training.utils.tool import load_model_from_config,  get_frame_indices, mask_downsampling, group_and_pad_videos, save_video_per_frame
from tqdm import tqdm



    

def map_segments_vectorized(masks1, masks2, start_frame, threshold=0.5):
    """
    Efficiently map segments from segmentation1 to segmentation2 using vectorized IoU computation.
    
    Args:
        segmentation1: List of Tensors, each of shape (1, H, W) representing a segment in segment_map1.
        segmentation2: List of Tensors, each of shape (1, H, W) representing a segment in segment_map2.
        threshold: float, minimum IoU to consider two segments as corresponding.
    
    Returns:
        index_map: Tensor of shape (N2,), where each element is the index of the corresponding segment in segmentation1,
                   or the index of itself if no segment matches.
        frame_map: Tensor of shape (N2,), boolean map indicating if the segment in segmentation2 has a correspondence in segmentation1.
    """
    # Stack the lists of masks into a single tensor for each segmentation map
    # masks1 = torch.cat(segmentation1, dim=0).cuda()  # Shape (N1, H, W)
    # if type(segmentation2) == list:  masks2 = torch.cat(segmentation2, dim=0).cuda()  # Shape (N2, H, W)
    # else: masks2 = segmentation2

    # Compute intersection areas
    intersections = torch.einsum('ihw,jhw->ij', masks1.float(), masks2.float())
    
    # Compute areas of each segment
    areas1 = masks1.sum(dim=(1, 2))  # Shape (N1,)
    areas2 = masks2.sum(dim=(1, 2))  # Shape (N2,)
    
    # Compute IoU
    unions = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersections
    iou_matrix = intersections / unions
    
    # Initialize the index map with default values (index of itself)
    index_map = torch.arange(len(masks2)).cuda()
    # Initialize the frame map as all False
    frame_map = torch.zeros(len(masks2)).cuda() + start_frame + 1
    
    # Find the best matches based on IoU
    max_ious, max_indices = iou_matrix.max(dim=0)
    valid_mask = max_ious >= threshold
    index_map[valid_mask] = max_indices[valid_mask]
    frame_map[valid_mask] = start_frame
    
    return index_map, frame_map


def build_correspondence(segment_list, key_frame_idxs):
    coo_token_list, coo_frame_list  = [], []
    
    # TODO: can be optimized, but not a bottleneck
    for chunk in range(len(segment_list)):
        start_frame_idx = key_frame_idxs[chunk]
        segment_chunks = segment_list[chunk]
        
        if start_frame_idx == 0:
            coo_token_list.append(torch.arange(len(segment_list[0][0])).cuda())
            coo_frame_list.append(torch.zeros(len(segment_list[0][0])).cuda())
        
        length = len(segment_chunks) -1 if chunk!=len(segment_list)-1 else len(segment_chunks)
        for i in range(1, length):
            coo_token_list.append(torch.arange(len(segment_chunks[i])).cuda())
            coo_frame_list.append(torch.zeros(len(segment_chunks[i])).cuda()+start_frame_idx+i-1)
            
        if chunk!=len(segment_list)-1:
            index_map, frame_map = map_segments_vectorized(segment_chunks[len(segment_chunks)-1], segment_list[chunk+1][0], start_frame=start_frame_idx+len(segment_chunks)-2)
            coo_token_list.append(index_map)
            coo_frame_list.append(frame_map)
    return coo_token_list, coo_frame_list


def build_global_graph(coo_token_list, coo_frame_list):
    T = len(coo_token_list)
    next_start_tok_id = 0
    tok_id_list = []
    
    for t in range(T):
        have_mapping_mask  = (coo_frame_list[t]!=t)
        last_frame_index = coo_token_list[t][have_mapping_mask]
    
        frame_tok_id = torch.zeros(len(coo_token_list[t]), dtype=torch.int64).cuda()
        mask_idx_new = torch.where(~have_mapping_mask)[0]
        frame_tok_id[mask_idx_new] = torch.arange(len(mask_idx_new)).cuda() + next_start_tok_id
        next_start_tok_id += len(mask_idx_new)
        
        if t >= 1: 
            frame_tok_id[have_mapping_mask] = tok_id_list[t-1][last_frame_index]
    
        tok_id_list.append(frame_tok_id)
    
    video_graph = torch.zeros(next_start_tok_id, T, dtype=torch.int64).cuda()
    for t, frame_tok_id in enumerate(tok_id_list):
        video_graph[:,t][frame_tok_id] = torch.arange(len(frame_tok_id)).cuda() + 1  # index 0 is for invalid mask
    
    return video_graph



def binary_segmentation_to_object_id(segment_list, resize=True, size=(256,256)):
 # put segmentation masks together in object id format
    return_masks = []
    avg_object_num = []
    for k, segs in enumerate(segment_list):
        length = len(segs) if k == len(segment_list)-1 else len(segs) - 1
        for idx in range(length):
            masks = segs[idx].cuda().float()
            if resize: masks = mask_downsampling(masks, size=size)
            object_num = len(masks)
            masks = masks * torch.arange(1,len(masks)+1)[:,None,None].cuda()
            gather_mask = torch.sum(masks, dim=0).int()
            return_masks.append(gather_mask)
            avg_object_num.append(object_num)
            
    avg_object_num = torch.mean(torch.tensor(avg_object_num).float()).item()
    return_masks = torch.stack(return_masks, dim=0)
    return return_masks, avg_object_num


class TrajGenPipeline:
    def __init__(self, 
        sam2_config,
        image_size = 512,
        frame_num = 16,
        sam2_checkpoint_path="../checkpoints/sam2.1_hiera_small.pt"
    ):
        self.image_size = image_size
        self.frame_num = frame_num
        self.max_clip_length = frame_num // 4
        self.max_divide_factor = (self.image_size//512)**2 * max((self.frame_num // 16),1)
        
        self.segmentor = AutoModelForSemanticSegmentation.from_pretrained("chendelong/DirectSAM-1800px-0424").to('cuda').eval()
        self.tracker = load_model_from_config(sam2_config, checkpoint_path=sam2_checkpoint_path, resolution=image_size)
        

    def sample_video(self, video_path, save_video_path=None, sample: bool = True,):
        """_uniformly sample desired frames from original video, and save to disk (help dataloading during training)
        
        Args:
            save_video (np.array): shape (T,W,H,C)
            video_path (str)
            sample (bool, optional)
        """
        
        if os.path.isdir(video_path):
            image_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
            save_video = np.stack([np.array(Image.open(ip)) for ip in image_paths])
            if sample:
                indices = get_frame_indices(num_frames=self.frame_num, sample='middle', vlen=len(image_paths))
                save_video = save_video[indices]
        else:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            vlen = len(video_reader)
            
            if sample:
                indices = get_frame_indices(num_frames=self.frame_num, sample='middle', vlen=vlen)
                save_video = video_reader.get_batch(indices).asnumpy()
            else:
                indices = [_ for _ in range(vlen)]
                save_video = video_reader.get_batch(indices).asnumpy()
            
        if save_video_path: imageio.mimsave(save_video_path, save_video, fps=5, codec="libx264")
        return save_video
    
    def preprocess_video(self, video, image_mean = [0.485, 0.456, 0.406], image_std = [0.229, 0.224, 0.225]):
            # Resize and normalize for mp4
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
            video = F.interpolate(video, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            mean = torch.tensor(image_mean).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
            std = torch.tensor(image_std).view(1, 3, 1, 1)    # Shape (1, C, 1, 1)
            video = video.float() / 255.0  # Scale to [0, 1]
            video = (video - mean) / std  # Normalize
            return video


    def unnormalize_video(self, video, image_mean = [0.485, 0.456, 0.406], image_std = [0.229, 0.224, 0.225]):
        mean = torch.tensor(image_mean).view(1, 3, 1, 1).cuda()  # Shape (1, C, 1, 1)
        std = torch.tensor(image_std).view(1, 3, 1, 1).cuda()    # Shape (1, C, 1, 1)
        video = video * std + mean
        video = video* 255.0  # Scale to [0, 1]
        return video


    def detect_key_frame(self, path):
        detector = AdaptiveDetector(
            adaptive_threshold=1,
            min_scene_len=1,
            window_width=1,
            min_content_val=10,
        )
        scenes = detect(path,  detector, show_progress=False)
        outlist1 = [a.frame_num for a,b in scenes]
        
        detector = ContentDetector(
            min_scene_len=1, threshold=27
        )
        scenes = detect(path,  detector, show_progress=False)
        outlist2 = [a.frame_num for a,b in scenes]
        
        detector = HistogramDetector(min_scene_len=1, threshold=0.15)
        scenes = detect(path,  detector, show_progress=False)
        outlist3 = [a.frame_num for a,b in scenes]

        all_elements = outlist1 + outlist2 + outlist3
        element_counts = Counter(all_elements)

        # Find elements that appear in at least 3 lists
        common_elements = sorted([key for key, count in element_counts.items() if count >= 2])
        if len(common_elements) == 0: common_elements = [0]
        
        return common_elements



    def cut_long_scene(self, key_frames, max_duration):
        adjusted_frames = [key_frames[0]]
        for i in range(1, len(key_frames)):
            start = key_frames[i - 1]
            end = key_frames[i]
            
            while end - start > max_duration + 1:
                start += max_duration
                adjusted_frames.append(start)
            adjusted_frames.append(end)
        
        return adjusted_frames

    def run_segmentation(self,all_images, threshold=0.2, max_mask_num=128, min_mask_area = 16,):
        chunk_size = 32 // self.max_divide_factor
        chunks = list(torch.split(all_images, chunk_size))
        # extract output logits from model inference
        logits = []
        for images in chunks:
            with torch.no_grad():
                o = self.segmentor(pixel_values=images)
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
            # masks = [labels == i for i in range(1, labels.max() + 1)]
            oh_np = np.eye(labels.max()+1, dtype=bool)[labels] # oh_np.shape == (H, W, L+1)
            masks = oh_np[..., 1:].transpose(2, 0, 1)             # shape: (L, H, W)
            
            # sort based on area
            L, H, W = masks.shape
            areas = masks.reshape(L, -1).sum(axis=1)
            order = np.argsort(-areas)    # largest first
            masks = masks[order]
            
            # filter out small masks
            sorted_areas = areas[order]
            keep = sorted_areas > min_mask_area
            masks = masks[keep]
            
            # keep mask number below max_mask_num
            masks = masks[:max_mask_num]

            kernel = np.ones((5,5), np.uint8)  # 3x3 structuring element
            for i, m in enumerate(masks):
                masks[i] = cv2.dilate(m.astype(np.uint8), kernel, iterations=2).astype(bool)
            #     masks[i] = torch.tensor(masks[i])
            # masks.sort(key=lambda x: x.sum(), reverse=True)
            masks = torch.from_numpy(masks).cuda()
            
            mask_batch.append(masks)
            mask_lens.append(len(masks))
            
        return mask_batch, mask_lens



    def _prepare_backbone_feature_for_tracking(self, img_batch):
        batch_size = 32 // (self.image_size//512)**2
        img_batch_chunks = list(torch.split(img_batch, batch_size))
        
        backbone_out_chunks = []
        with torch.no_grad():
            for images in img_batch_chunks:
                backbone_out_chunks.append(self.tracker.forward_image(images))
        
        backbone_out = {'vision_features': None, 'vision_pos_enc': None, 'backbone_fpn': None}
        backbone_out['vision_features'] = torch.cat([b['vision_features'] for b in backbone_out_chunks])
        backbone_out['vision_pos_enc'] = [torch.cat([b['vision_pos_enc'][i] for b in backbone_out_chunks]) for i in range(len(backbone_out_chunks[0]['vision_pos_enc']))]
        backbone_out['backbone_fpn'] = [torch.cat([b['backbone_fpn'][i] for b in backbone_out_chunks]) for i in range(len(backbone_out_chunks[0]['backbone_fpn']))]
        return BackboneOut(backbone_out)
    
        
    def track_step(self, video_group, backbone_out_group, object_num_per_clip, masks_first_frame):
        n_group, t_group, _, _, _ = video_group.shape
        n_object = sum(object_num_per_clip)
        
        img_batch = rearrange(video_group, 'b t h w c -> t b h w c')
        backbone_input = backbone_out_group[0].join(backbone_out_group[1:])
        
        _frame_idx = torch.arange(t_group).unsqueeze(1).repeat(1,n_object)
        _batch_idx = torch.tensor(sum([[i]*obj_num for i,obj_num in enumerate(object_num_per_clip)], start=[])).unsqueeze(0).repeat(t_group,1)
        obj_to_frame_idx = torch.stack([_frame_idx, _batch_idx], dim=-1).cuda()
        input = BatchedVideoDatapoint(img_batch=img_batch, obj_to_frame_idx=obj_to_frame_idx, masks=masks_first_frame, metadata=None, dict_key="a")
        with torch.no_grad():
            output = self.tracker(input, backbone_input.store)
        output_masks = torch.stack([(l['pred_masks_high_res']>0).squeeze() for l in output]) # [frame_number, n_object, h, w]
        return output_masks
        

    def traj_generation(self, video: np.array, video_path=None, use_key_frame_selection=False, save_to_disk=False, version_ext='', verbose=False):
        """generate panoptic trajectory for a video

        Args:
            video (np.array): shape (T,W,H,C)
            video_path (str)
            use_key_frame_selection (bool, optional)
        """
        
        T, W, H, _ = video.shape
        
        if type(video) == np.ndarray: 
            video = self.preprocess_video(video)
        video = video.cuda()
        
        
        # TODO: key frame detection
        # print("run key frame selection")
        if use_key_frame_selection: 
            assert video_path != None
            key_frame_idxs = raw_key_idxs = self.detect_key_frame(video_path) + [T]
            # key_frame_idxs = self.cut_long_scene(raw_key_idxs, max_duration=self.max_clip_length)
        else: 
            key_frame_idxs = [i for i in range(0, T, self.max_clip_length)] + [T]
        print("key frames", key_frame_idxs)
            
        video_chunks = [video[key_frame_idxs[i]:key_frame_idxs[i+1]+1] for i in range(len(key_frame_idxs)-1)]
        
        print("run segmentation")
        video_start_frames = torch.stack([v[0] for v in video_chunks]).cuda()
        segmentation_masks, masks_len = self.run_segmentation(video_start_frames)
        
        
        print("run tracking")
        # prepare tracking backbone features
        backbone_out = self._prepare_backbone_feature_for_tracking(video)
        backbone_out_chunk = [backbone_out[key_frame_idxs[i]:key_frame_idxs[i+1]+1] for i in range(len(key_frame_idxs)-1)]
        
        max_object_num = 256 // self.max_divide_factor
        padded_video_groups, padded_backbone_out_groups, group_indices = group_and_pad_videos(video_chunks, masks_len, backbone_out_chunk, max_deviation=2, max_object_num=max_object_num)
        gather_output_masks = [None] * len(video_chunks)
        for video_group, backbone_out_group, video_idxs in zip(padded_video_groups, padded_backbone_out_groups, group_indices):
            object_num_per_clip = [masks_len[i] for i in video_idxs]
            masks_first_frame = torch.cat([segmentation_masks[i] for i in video_idxs])
            
            if sum(object_num_per_clip) > max_object_num:  # even for single clip, the object number exceed
                assert video_group.shape[0] == 1
                num_chunks = sum(object_num_per_clip) // max_object_num + 1
                masks_first_frame_chunks = torch.chunk(masks_first_frame, chunks=num_chunks)
                
                output_masks = []
                for sub_mask_first_frame in masks_first_frame_chunks:
                    sub_object_num = [len(sub_mask_first_frame)]
                    sub_output_masks = self.track_step(video_group, backbone_out_group, sub_object_num, sub_mask_first_frame.unsqueeze(0))
                    output_masks.append(sub_output_masks)
                output_masks = torch.cat(output_masks, dim=1)

            else: output_masks = self.track_step(video_group, backbone_out_group, object_num_per_clip, masks_first_frame.unsqueeze(0))

            
            # gather masks
            cum_obj_num = 0
            for vididx, obj_num in zip(video_idxs, object_num_per_clip):
                vid_output_mask = output_masks[:, cum_obj_num:cum_obj_num+obj_num]
                actual_frame_num = video_chunks[vididx].shape[0]
                vid_output_mask = vid_output_mask[:actual_frame_num]
                cum_obj_num += obj_num
                gather_output_masks[vididx] = vid_output_mask
                
        print("build video graph")
        coo_token_list, coo_frame_list = build_correspondence(gather_output_masks, key_frame_idxs)
        video_graph = build_global_graph(coo_token_list, coo_frame_list)
        return_masks, avg_object_num = binary_segmentation_to_object_id(gather_output_masks, resize=True, size=(W,H))
        

        # save created trajectory
        if save_to_disk:
            save_return_masks = return_masks.cpu().numpy().astype(np.uint8)
            save_video_graph = video_graph.cpu().numpy()
            
            extension = os.path.splitext(os.path.basename(video_path))[1]
            video_path = video_path.replace(f"_short"+extension, extension)
            np.savez_compressed(video_path.replace(extension, f"_mask{version_ext}.npz"), save_return_masks)
            np.savez_compressed(video_path.replace(extension, f"_graph{version_ext}.npz"), tensor=save_video_graph)
            if verbose: print("complete", video_path.replace(extension, f"_mask{version_ext}.npz"))

        return return_masks, video_graph
        
        
    

    def visualize(self, video, video_graph, return_masks, visualize_dir, video_name, seed_value=1, ):
        """visualize panoptic trajectory for a video

        Args:
            video (np.array): shape (T,W,H,C)
            video_graph (np.array): shape (M,T), M is number of trajectories in a video
            return_masks (np.array): shape (T,W,H)
        """
        os.makedirs(visualize_dir, exist_ok=True)
        T = video.shape[0]
        resize_video = []
        resize_masks = []
        for t in range(T):
            resize_video.append(cv2.resize(video[t], (600,300)))
            resize_masks.append(cv2.resize(return_masks[t].cpu().numpy(), (600,300), interpolation=cv2.INTER_NEAREST))
        video = np.stack(resize_video)
        return_masks = np.stack(resize_masks)
                    
        
        print("visualize the trajectory (may take long time)")
        canvas_list = []
        raw_canvas_list = []
        random.seed(seed_value)         # Python’s built-in random
        np.random.seed(seed_value)      # NumPy

        colorlist = np.random.randint(0, 255, (2000, 3), dtype=np.uint8)
        
        for t in range(T):
            frame_graph, frame_segments = video_graph[:,t].cpu().numpy(), return_masks[t]

            raw_canvas = video[t].copy()
            
            color_canvas = np.zeros_like(raw_canvas)
            for m in range(len(frame_graph)):
                if frame_graph[m] == 0: continue
                mask = (frame_segments == frame_graph[m])
                color_canvas[mask.astype(np.bool_)] = colorlist[m].tolist()
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(color_canvas, contours, -1, (50, 50, 50), 1) 

            canvas = (0.3 * raw_canvas + 0.7 * color_canvas).astype(np.uint8)
            canvas_list.append(canvas)
            raw_canvas_list.append(raw_canvas)
        video_array = np.stack(canvas_list).astype(np.uint8)
        raw_video_array = np.stack(raw_canvas_list).astype(np.uint8)
        
        all_video_array = np.concatenate([raw_video_array, video_array], axis=2)
        imageio.mimsave(os.path.join(visualize_dir, video_name.split(".")[0] + "_all.gif"), all_video_array, duration=400//(T//16), loop=0)
        
        
        
    def pixel_labels(
        self,
        mask: torch.LongTensor,         # (B,T,H,W)  object-IDs per pixel
        video_graph: torch.LongTensor,  # (B,M,T)    object-IDs per traj & time
        ignore_index: int = -1,
    ):
        """
        Returns
        labels  : (B,N)  each entry ∈ {0..M-1} or ignore_index
        valid   : (B,M)  valid[b,m]=False if traj m is *padding* for batch-b
        """
        B,T,H,W = mask.shape
        B_,M,T_ = video_graph.shape
        assert (B,T) == (B_,T_), "shape mismatch"

        mask_f = mask.view(B, T, -1)                         # (B,T,HW)
        # (B,M,T,HW)  matches[b,m,t,n] == (traj m’s ID at t) == (pixel n’s ID at t)
        matches = video_graph.unsqueeze(-1).eq(
                mask_f.unsqueeze(1))                       # bool

        # Hungarian will need to know which GT trajs are real:
        valid_traj = (video_graph != ignore_index).any(-1)   # (B,M) bool

        # Argmax over M gives the (first) trajectory-index for every pixel
        # non-matching pixels (background) are marked ignore_index
        has_match = matches.any(1)                           # (B,T,HW)
        label_map = torch.where(
            has_match,
            matches.float().argmax(1),                       # (B,T,HW) long
            torch.full_like(mask_f, ignore_index),           # background
        )
        # TODO：check the correctness of this transformation
        return label_map.view(B, -1)[0], valid_traj[0]             # (B,N), (B,M)
    
    
    
    
    
    def visualize_labels(self, video, video_graph, return_masks, visualize_dir, video_name, seed_value=1, individual=False, ):
        """visualize panoptic trajectory for a video

        Args:
            video (np.array): shape (T,W,H,C)
            video_graph (np.array): shape (M,T), M is number of trajectories in a video
            return_masks (np.array): shape (T,W,H)
        """
        T, W, H, _ = video.shape
        label_map, valid_traj = self.pixel_labels(return_masks.unsqueeze(0), video_graph.unsqueeze(0), ignore_index=0)
        label_map = label_map.reshape(T,56, 56).cpu().numpy()
        
        
        resize_video = []
        resize_label = []
        for t in range(T):
            resize_video.append(cv2.resize(video[t], (600,300)))
            resize_label.append(cv2.resize(label_map[t], (600,300), interpolation=cv2.INTER_NEAREST))
        video = np.stack(resize_video)
        resize_label = np.stack(resize_label)
        
        
        print("visualize the trajectory (may take long time)")
        canvas_list = []
        raw_canvas_list = []
        random.seed(seed_value)         # Python’s built-in random
        np.random.seed(seed_value)      # NumPy

        colorlist = np.random.randint(0, 255, (2000, 3), dtype=np.uint8)
        
        for t in range(T):
            frame_graph, frame_segments = video_graph[:,t].cpu().numpy(), resize_label[t]

            raw_canvas = video[t].copy()
            
            color_canvas = colorlist[resize_label[t].astype(np.int64)]
            canvas = (0.3 * raw_canvas + 0.7 * color_canvas).astype(np.uint8)
            # Image.fromarray(canvas).save(os.path.join(visualize_dir, f"{video_name}_{t:02d}.png"))
            canvas_list.append(canvas)
            raw_canvas_list.append(raw_canvas)
        video_array = np.stack(canvas_list).astype(np.uint8)
        
        raw_video_array = np.stack(raw_canvas_list).astype(np.uint8)
        
        all_video_array = np.concatenate([raw_video_array, video_array], axis=2)
        imageio.mimsave(os.path.join(visualize_dir, video_name.split(".")[0] + "_all.gif"), all_video_array, duration=400//(T//16), loop=0)
        
        
if __name__ == "__main__":
    video_path = '/weka/chenhaoz/home/videotok/0000075_00002.mp4'
    traj_gen = TrajGenPipeline(
        sam2_config='sam2/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune.yaml',
        frame_num=16
    )
    video = traj_gen.sample_video(video_path, save_video_path="example.mp4")
    mask, graph = traj_gen.traj_generation(video=video, video_path="example.mp4", use_key_frame_selection=True)
    # video = traj_gen.sample_video(video_path,)
    # mask, graph = traj_gen.traj_generation(video=video,)
    traj_gen.visualize(video, video_graph=graph, return_masks=mask, 
        visualize_dir=".", video_name="example", )