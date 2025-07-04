import torch
import copy
from models.utils import interpolate_pos_embed, interpolate_pos_relative_bias_beit, load_temp_embed_with_mismatch
from models.tokenization_bert import BertTokenizer

from utils.scheduler import create_scheduler
from utils.optimizer import create_optimizer

import logging
import os
logger = logging.getLogger(__name__)


def load_model_ckpt(model_without_ddp, pretrained_path):
    logger.info(f"Loading checkpoint from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    # load temporal_embeddings, clip or expand when necessary
    state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
        temp_embed_old=state_dict["temporal_embeddings"],
        temp_embed_new=model_without_ddp.temporal_embeddings.data
    )

    msg = model_without_ddp.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    logger.info(f"Loaded checkpoint from {pretrained_path}")  


import re
def get_largest_ckpt(folder_path):
    if not os.path.exists(folder_path): return None
    # List all files in the folder
    files = os.listdir(folder_path)
    # Use a regex to match filenames of the format "ckpt_X.pth"
    ckpt_files = [f for f in files if re.match(r'ckpt_\d+\.pth', f)]
    if len(ckpt_files) == 0: return None
    # Extract the numbers from the filenames
    ckpt_numbers = [(int(re.search(r'\d+', f).group()), f) for f in ckpt_files]
    # Find the file with the largest number
    largest_ckpt = max(ckpt_numbers, key=lambda x: x[0])[1]
    return os.path.join(folder_path, largest_ckpt)


def get_sorted_ckpts(folder_path):
    if not os.path.exists(folder_path): return None
    # List all files in the folder
    files = os.listdir(folder_path)
    # Use a regex to match filenames of the format "ckpt_X.pth"
    ckpt_files = [f for f in files if re.match(r'ckpt_\d+\.pth', f)]
    if len(ckpt_files) == 0: return None
    # Extract the numbers from the filenames
    ckpt_number_pairs = sorted([(int(re.search(r'\d+', f).group()), f) for f in ckpt_files], key=lambda x: x[0])
    return [os.path.join(folder_path, k[1]) for k in ckpt_number_pairs]



def setup_model(config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
    model = model_cls(config=config, tokenizer=tokenizer)

    num_param = sum(p.numel() for p in model.parameters()) / 10**6
    logger.info(f"total parameters: {num_param:.3f} M")
    
        
    model = model.to(torch.device(config.device))
    model_without_ddp = model
    
    
    
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters  # `False` for image-only task
        )

    if not config.evaluate:
        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    else:
        optimizer, scheduler, scaler = None, None, None

    start_epoch = 0
    global_step = 0
    
    # TODO: make pretrained_path necessary
    
    
    
    if config.pretrained_path: ckpt_path = config.pretrained_path
    else: ckpt_path = None
    
    if config.resume and not ckpt_path: ckpt_path = get_largest_ckpt(config.output_dir)
    if ckpt_path and os.path.exists(ckpt_path) and not os.path.isdir(ckpt_path):
        
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        global_step = checkpoint["global_step"]

        if config.evaluate:
            pass
        elif config.partial_resume:
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        elif config.resume:
            logger.info("full resume mode")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]


        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")
        
        
        
    image_ckpt_path = config.image_pretrained_path
    if image_ckpt_path and os.path.exists(image_ckpt_path) and not os.path.isdir(image_ckpt_path):
        logger.info(f"Loading checkpoint from {image_ckpt_path}")
        checkpoint = torch.load(image_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        msg = model_without_ddp.load_image_model(state_dict, load_only_vision=config.load_vision_only)
        logger.info(msg)
        logger.info(f"Loaded image pretrained checkpoint from {image_ckpt_path}")
        
    return model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step





