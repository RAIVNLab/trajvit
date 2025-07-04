from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import torch
import cv2
import os
import sys
import decord
sys.path.append("traj_gen")
sys.path.append("entry")

from traj_gen.training.pipeline import TrajGenPipeline
from entry.share_models.traj_transformer import VideoTokenViT
from entry.share_models.vit3d import ViT3D

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

import yaml

    
def init_traj_vit(config_path="entry/configs/pretrain.yaml"):
    """ init our model """
    with open(config_path) as f: config = yaml.safe_load(f)   
    model = VideoTokenViT(config['traj_model'], pos_config=config['traj_pos'], perceiver_config=config['perceiver'], norm_layer=None)
    return model


def create_transform(image_size):
    """ create image transform """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.))

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC),
        type_transform,
        normalize,
    ])




def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='model',
):
    if pretrained is None: 
        logger.info("pretrained model path is None, return")
        return encoder
    
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    
    pretrained_dict = checkpoint[checkpoint_key]

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    m_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('vision_encoder.vision_encoder.'): k = k[len('vision_encoder.'):]
        elif k.startswith('text_encoder'): continue
        else: k = k.replace('vision_encoder.', '')
        m_pretrained_dict[k] = v
    pretrained_dict = m_pretrained_dict    
    

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder





def preprocess_mask_and_graph(masks, graphs, image_size=224, mask_res_down_factor=4):
    """ resize mask before doing inference.  mask: tensor of shape (T,H,W)"""
    def resize_masks(masks, size):
        T = masks.shape[0]
        resized_masks = np.empty((T, size[0], size[1]), dtype=masks.dtype)
        for t in range(T): resized_masks[t] = cv2.resize(masks[t], (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return resized_masks
    
    masks = resize_masks(masks, size=(image_size//mask_res_down_factor, image_size//mask_res_down_factor))
    masks = torch.from_numpy(masks)
    graphs = torch.from_numpy(graphs)
    
    masks[masks>graphs.max()] = 0
    graphs[graphs>masks.max()] = 0  # some seg idx is invalid due to vanished segmentation size
    graphs = graphs[~(graphs == 0).all(dim=1)] # filter out rows that are all 0
    return masks, graphs
    

from tqdm import tqdm
import argparse
if __name__ == "__main__":
    # hyper-parameters 

    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, default="./example/example.mp4", help='input video path.')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames.')
    parser.add_argument('--traj_gen_resolution', type=int, default=512, help='Resolution for trajectory generation.')
    parser.add_argument('--inference_resolution', type=int, default=224, help='Resolution for inference.')
    parser.add_argument('--mask_res_down_factor', type=int, default=4, help='Downscale factor for mask resolution.')
    parser.add_argument('--vittok_ckpt_path', type=str, default=None, help='Path to ViTtok checkpoint.')
    parser.add_argument('--visualize_tracks', default=False, action='store_true', help='visualize generated trajectories')

    args = parser.parse_args()

    print(args)
    ######################## First step: pre-generate all the trajectories and save them to disk #################################
    trajGenModel = TrajGenPipeline(
        sam2_config='./traj_gen/sam2/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune.yaml',
        image_size=args.traj_gen_resolution,
        frame_num=args.num_frames,
        sam2_checkpoint_path='checkpoints/sam2.1_hiera_small.pt'
    )
    
    os.makedirs("results/tmp", exist_ok=True)
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_video_path = os.path.join('results/tmp', os.path.basename(video_path))
    sample_video = trajGenModel.sample_video(video_path=video_path, save_video_path=save_video_path)


    return_masks, video_graph = trajGenModel.traj_generation(
        video=sample_video,
        video_path=save_video_path,
        use_key_frame_selection=True,
        save_to_disk=True,
        verbose=True,
    )
    
    if args.visualize_tracks:
        trajGenModel.visualize(sample_video, video_graph, return_masks, visualize_dir="results/traj/", video_name=video_name)


    ######################### Start inference right now. ######################
    # init models
    vittok = init_traj_vit()
    
    # load pretrained model
    vittok = load_pretrained(vittok, pretrained=args.vittok_ckpt_path).eval().cuda()
    
    # init image transformation (Test time)
    transform = create_transform(image_size=args.inference_resolution)

    # load inputs
    video_reader = decord.VideoReader(save_video_path, num_threads=1)
    video = video_reader.get_batch(range(len(video_reader))).asnumpy()   
    extension = os.path.splitext(os.path.basename(save_video_path))[1]
    masks =  np.load(save_video_path.replace(extension, f'_mask.npz'))['arr_0']
    graphs = np.load(save_video_path.replace(extension, f'_graph.npz'))['tensor']
    
    # preprocess
    print("preprocess data for inference")
    video = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    input_video = transform(video)
    input_mask, input_graph = preprocess_mask_and_graph(masks, graphs, image_size=args.inference_resolution, mask_res_down_factor=args.mask_res_down_factor)
    
    # make batch dimension & move to gpu
    input_video = input_video.unsqueeze(0).cuda()
    input_mask = input_mask.unsqueeze(0).cuda()
    input_graph = input_graph.unsqueeze(0).cuda()

    # model inference
    print("start inference")
    with torch.no_grad():
        vittok_output = vittok(input_video, segmask=input_mask, video_graph=input_graph)

    print("output shape", vittok_output.shape)