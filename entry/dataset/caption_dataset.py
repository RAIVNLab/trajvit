from dataset.utils import pre_text
from os.path import basename
import torch
import numpy as np
import cv2
import zlib
import glob
import os

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno, PanopticPosAugmentation
from dataset.video_utils import VIDEO_READER_FUNCS, decode_video_from_bytes, get_frame_indices
from dataset.template import kinetics_templates
import logging

logger = logging.getLogger(__name__)


class ImgTxtRetTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False):
        super(ImgTxtRetTrainDataset, self).__init__()
        
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_vision_gt = has_multi_vision_gt
        self.match_ids = {}

        n = 0
        for ann in self.anno_list:
            key = ann["caption"] if has_multi_vision_gt else basename(ann["image"])
            if type(key) == list: key = key[0]
            if key not in self.match_ids:
                self.match_ids[key] = n
                n += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index)
        caption = pre_text(ann["caption"])
        key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
        return image, caption, self.match_ids[key]


class VidTxtRetTrainDataset(ImgTxtRetTrainDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=3,
            is_paragraph_retrieval=False, has_multi_vision_gt=False
    ):
        super(VidTxtRetTrainDataset, self).__init__(ann_file, transform, has_multi_vision_gt)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval

        if is_paragraph_retrieval:
            self.anno_list = preprocess_para_retrieval_data(self.anno_list)




import random
class VidGraphTrainDataset(VidTxtRetTrainDataset):
    media_type = 'video'
    
    def __init__(self, image_res=224, mask_down_factor=2, version_ext="", eval=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        if version_ext == '_v0': version_ext = ''
        self.version_ext = version_ext
        self.image_mask_augmentation = PanopticPosAugmentation(size=image_res)
        self.eval = eval
    

    def sample_mask_and_graph(self, masks, graphs, num_frames):
        indices = get_frame_indices(num_frames=num_frames, vlen=masks.shape[0])
        masks = masks[indices]
        graphs = graphs[:, indices]
        return masks, graphs
        
        
    def __getitem__(self, index):
        video, index, frame_number, masks, graphs = self.load_and_transform_media_data(
            index, 
            disable_augmentation=True if not self.eval else False,
            load_graph_and_mask=True
        )
        ann = self.anno_list[index]
        
        ori_caption = ann["caption"]
        if type(ori_caption) == list: ori_caption = random.choice(ori_caption)
        caption = pre_text(ori_caption)
        key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
        
        if not self.eval:  video, masks = self.image_mask_augmentation(video, masks)
        
        masks = self.resize_masks(masks.numpy(), size=(self.image_res//self.mask_down_factor,self.image_res//self.mask_down_factor))
        masks = torch.from_numpy(masks)
        if masks.shape[0] != frame_number:
            print("sub sample mask and graph,", masks.shape[0], "->", frame_number)
            masks, graphs = self.sample_mask_and_graph(masks, graphs, frame_number)
        
        masks[masks>graphs.max()] = 0
        graphs[graphs>masks.max()] = 0  # some seg idx is invalid due to vanished segmentation size
        valid_rows = torch.nonzero(torch.sum(graphs, dim=1) > 0)[:,0]
        graphs = graphs[valid_rows]
        
        num_video_token = graphs.shape[0]
        
        return video, caption, self.match_ids[key], masks, graphs, num_video_token
        
    
    
    
    
    
class ImgGraphTrainDataset(ImgTxtRetTrainDataset):
    media_type = 'image'
    
    def __init__(self, image_res=224, mask_down_factor=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        self.num_frames = 1
        self.image_mask_augmentation = PanopticPosAugmentation(size=image_res)
    
    def get_mask_and_graph(self, data_path, image_size):
        extension = os.path.splitext(os.path.basename(data_path))[1]
        masks = np.load(data_path.replace(extension, '_mask.npz'))['arr_0'][None, ...]
        masks = self.resize_masks(masks, image_size)
        masks = torch.from_numpy(masks)
        
        return masks
        
    
    def __getitem__(self, index):
        video, index = self.load_and_transform_media_data(index, disable_augmentation=True)
        ann = self.anno_list[index]
        
        caption = pre_text(ann.get('short_caption', ann["caption"]))
        key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
        masks = self.get_mask_and_graph(ann["image"], video.shape[-2:])
        video, masks = self.image_mask_augmentation(video, masks)
        
        masks = self.resize_masks(masks.numpy(), size=(self.image_res//self.mask_down_factor,self.image_res//self.mask_down_factor))
        masks = torch.from_numpy(masks)
        graphs = torch.arange(1,masks.max()+1).unsqueeze(1)  # (N,T) where T=1
        num_video_token = graphs.shape[0]
        return video, caption, self.match_ids[key], masks, graphs, num_video_token
    
    
    


class ImgTxtRetEvalDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False):
        super(ImgTxtRetEvalDataset, self).__init__()
        self.raw_anno_list = load_anno(ann_file)
        self.transform = transform
        self.has_multi_vision_gt = has_multi_vision_gt  # each caption has multiple image as ground_truth

        self.text = None
        self.image = None
        self.txt2img = None
        self.img2txt = None
        self.build_data()

    def build_data(self):
        self.anno_list = []
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        if self.has_multi_vision_gt:
            self.build_data_multi_img_gt()
        else:
            self.build_data_multi_txt_gt()
            
    def build_data_multi_img_gt(self):
        """each text may have multiple ground_truth image, e.g., ssv2"""
        img_id = 0
        txt_id = 0
        caption_id_dict = dict()
        for ann in self.raw_anno_list:
            if ann["caption"] not in caption_id_dict.keys():
                templated_texts = [template.format(pre_text(ann['caption'])) for template in kinetics_templates]
                self.text.append(templated_texts)
                self.txt2img[txt_id] = []
                caption_id_dict[ann['caption']] = txt_id
                txt_id += 1
                
            cur_txt_id = caption_id_dict[ann['caption']]
            image = ann['image']
            self.image.append(image)
            self.anno_list.append(ann)
            self.txt2img[cur_txt_id].append(img_id)
            self.img2txt[img_id] = cur_txt_id
            img_id += 1
        logger.info(f"building multi img gt -- length of text classes: {len(self.txt2img)}")

    def build_data_multi_txt_gt(self):
        """each image may have multiple ground_truth text, e.g., COCO and Flickr30K"""
        txt_id = 0
        img_id = 0
        img_id_dict = dict()
        for ann in self.raw_anno_list:
            if ann['image'] not in img_id_dict.keys():
                self.image.append(ann['image'])
                self.anno_list.append(ann)
                self.img2txt[img_id] = []
                img_id_dict[ann['image']] = img_id
                img_id += 1 
            
            cur_img_id = img_id_dict[ann['image']]
            caption = ann['caption']
            self.text.append(caption)
            self.img2txt[cur_img_id].append(txt_id)
            self.txt2img[txt_id] = cur_img_id
            txt_id += 1
        
        logger.info(f"building multi txt gt -- length of image classes: {len(self.img2txt)}")

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        image, index = self.load_and_transform_media_data(index)
        return image, index


class VidTxtRetEvalDataset(ImgTxtRetEvalDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=3,
            is_paragraph_retrieval=False, has_multi_vision_gt=False
    ):
        super(VidTxtRetEvalDataset, self).__init__(ann_file, transform, has_multi_vision_gt)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval

        if is_paragraph_retrieval:
            self.anno_list = preprocess_para_retrieval_data(self.raw_anno_list)
        self.build_data()


        
class VidGraphEvalDataset(VidTxtRetEvalDataset):
    media_type = "video"
    
    def __init__(self, image_res=224, mask_down_factor=2, version_ext="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        if version_ext == '_v0': version_ext = ''
        self.version_ext = version_ext
    
    def sample_mask_and_graph(self, masks, graphs, cur_frame_number):
        indices = get_frame_indices(num_frames=cur_frame_number, vlen=masks.shape[0])
        masks = masks[indices]
        graphs = graphs[:, indices]
        return masks, graphs
    
    
    def __getitem__(self, index):
        # if self.num_frames!= 64: st = common_transform
        # else: st = special_transform
        
        video, index, cur_frame_number, masks, graphs = self.load_and_transform_media_data(index, load_graph_and_mask=True)
        masks = self.resize_masks(masks.numpy(), size=(self.image_res//self.mask_down_factor,self.image_res//self.mask_down_factor))
        masks = torch.from_numpy(masks)
        
        caption = "placeholder"
        
        if masks.shape[0] != cur_frame_number:
            print("sub sample mask and graph,", masks.shape[0], "->", cur_frame_number)
            masks, graphs = self.sample_mask_and_graph(masks, graphs, cur_frame_number)
        
        masks[masks>graphs.max()] = 0
        graphs[graphs>masks.max()] = 0  # some seg idx is invalid due to vanished segmentation size
        valid_rows = torch.nonzero(torch.sum(graphs, dim=1) > 0)[:,0]
        graphs = graphs[valid_rows]
        num_video_token = graphs.shape[0]
        
        return video, caption, index, masks, graphs, num_video_token





class ImgGraphEvalDataset(ImgTxtRetEvalDataset):
    media_type = "image"
    
    def __init__(self, image_res=224, mask_down_factor=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        self.num_frames = 1
        
    
    
    def get_mask_and_graph(self, data_path):
        extension = os.path.splitext(os.path.basename(data_path))[1]
        masks =  np.load(data_path.replace(extension, '_mask.npz'))['arr_0'][None, ...]
        masks = self.resize_masks(masks, size=(self.image_res//self.mask_down_factor,self.image_res//self.mask_down_factor))
        masks = torch.from_numpy(masks) 
        
        graphs = torch.arange(1,masks.max()+1).unsqueeze(1)  # (N,T) where T=1
        num_video_token = graphs.shape[0]
        return masks, graphs, num_video_token
    
    def __getitem__(self, index):
        video, index = self.load_and_transform_media_data(index)
        ann = self.anno_list[index]
        caption = "placeholder"
        
        
        masks, graphs, num_video_token = self.get_mask_and_graph(ann["image"])
        
        return video, caption, index, masks, graphs, num_video_token


def preprocess_para_retrieval_data(anno_list):
    processed_anno_list = []
    for d in anno_list:
        d["caption"] = " ".join(d.pop("caption"))
        processed_anno_list.append(d)
    return processed_anno_list


class VidTxtRetMCEvalDataset(ImageVideoBaseDataset):
    """For MSRVTT-MC test task"""
    media_type = "video"

    def __init__(self, ann_file, transform, num_frames=4,
                 video_reader_type="decord", sample_type="rand", num_tries=3):
        super(VidTxtRetMCEvalDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # video args
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index)
        caption = [pre_text(e) for e in ann["caption"]]  # len=5
        answer = ann["answer"]
        return image, caption, answer, ann



