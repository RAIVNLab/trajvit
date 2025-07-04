from torch.utils.data import Dataset
from dataset.utils import load_image_from_path
import random
import logging
import os
import numpy as np
import torch
import cv2

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""
    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video"]
        self.anno_list = None  # list(dict), each dict contains {"image": str, # image or video path}
        self.transform = None
        self.video_reader = None
        self.num_tries = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


    def resize_masks(self, masks, size):
        T = masks.shape[0]
        resized_masks = np.empty((T, size[0], size[1]), dtype=masks.dtype)
        for t in range(T): resized_masks[t] = cv2.resize(masks[t], (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return resized_masks


    def get_mask_and_graph(self, ann, image_size):
        data_path = ann['image']
        extension = os.path.splitext(os.path.basename(data_path))[1]
        data_path = data_path.replace(f"_short"+extension, extension)

        mask_path = ann.get('mask', data_path.replace(extension, f'_mask{self.version_ext}.npz'))
        graph_path = ann.get('graph', data_path.replace(extension, f'_graph{self.version_ext}.npz'))

        masks =  np.load(mask_path, allow_pickle=True)['arr_0']
        masks = self.resize_masks(masks, image_size)
        masks = torch.from_numpy(masks)
        
        graphs = np.load(graph_path, allow_pickle=True)['tensor']
        graphs = torch.from_numpy(graphs)
    
        return masks, graphs
    
    
    def load_and_transform_media_data(self, index, disable_augmentation=False, special_transform=None, sample_frame=None, load_graph_and_mask=False):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, disable_augmentation=disable_augmentation, load_graph_and_mask=load_graph_and_mask)
        else:
            return self.load_and_transform_media_data_video(index, 
                                    disable_augmentation=disable_augmentation, 
                                    special_transform=special_transform, 
                                    sample_frame=sample_frame,
                                    load_graph_and_mask=load_graph_and_mask,
                                    )


    def load_and_transform_media_data_image(self, index, disable_augmentation=False, load_graph_and_mask=False):
        ann = self.anno_list[index]
        data_path = ann["image"]
        image = load_image_from_path(data_path)
        if not disable_augmentation: image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index, disable_augmentation=False, special_transform=None, sample_frame=None, load_graph_and_mask=False):
        for i in range(self.num_tries):
            ann = self.anno_list[index]
            data_path = ann["image"]
            
            if special_transform is not None: data_path = special_transform(data_path)
            # TODO 
            try:
                max_num_frames = self.max_num_frames \
                    if hasattr(self, "max_num_frames") else -1
                num_frames = self.num_frames if sample_frame is None else sample_frame
                frames, frame_indices, _ = self.video_reader(
                    data_path, self.num_frames, self.sample_type,
                    max_num_frames=max_num_frames
                )
                if len(frame_indices)!=self.num_frames: raise Exception
                
                
                if not disable_augmentation: frames = self.transform(frames)
            
                if load_graph_and_mask:
                    masks, graphs = self.get_mask_and_graph(ann, frames.shape[-2:])
                    return frames, index, len(frame_indices), masks, graphs
                else:
                    return frames, index, len(frame_indices)
            
                
            except Exception as e:
                index = random.randint(0, len(self) - 1)
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement")
                continue
            
            
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )
            
