import pandas as pd
import time
import datetime
import wandb
from os.path import join
import logging
from tqdm import tqdm
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import Singularity, VideoViT, VideoTokCLIP

from utils.logger import log_dict_to_wandb, setup_wandb
from utils.config_utils import setup_main
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed, remove_files_if_exist
from utils.distributed import get_rank, get_world_size, is_main_process
from dataset import create_dataset, create_sampler, create_loader, MetaLoader
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model

logger = logging.getLogger(__name__)

logger.info("finish importing package")



def test_example_ret(model, test_name2loaders, tokenizer, device, config):
    model.eval()
    res = {}
    media_type = 'video'
    logger.info(f"ret datasets: {list(test_name2loaders.keys())}")
    for ret_name, ret_loader in test_name2loaders.items():
        logger.info(f"starting {ret_name}") 
        img2txt, txt2img = [], []
        for i, _batch in enumerate(ret_loader):
            assert len(_batch) == 1
            batch = _batch[0]
            
            image, caption, segment, graph = batch
            image = image.to(device, non_blocking=True)
            text_input = tokenizer(
                caption, padding="max_length", truncation=True,
                max_length=config.max_txt_l[media_type], return_tensors="pt"
            ).to(device)  # change from "longest" to "max_length"
            if config.vit_type == 'vittok':
                segment = segment.to(device, non_blocking=True)
                graph = graph.to(device, non_blocking=True)
                
            vision_input = (image, segment, graph)
            imgacc, txtacc = retrieval_accuracy(model, vision_input, text_input)
            img2txt.append(imgacc)
            txt2img.append(txtacc)
            
            if i % 10 == 0 and is_main_process():
                logger.info(f"{ret_name}: progress {i}/{len(ret_loader)}; img2txt accuracy {imgacc}; txt2img accuracy {txtacc}")
                
        img2txt = torch.mean(torch.tensor(img2txt)).item()
        txt2img = torch.mean(torch.tensor(txt2img)).item()
        res.update({ret_name + "_img2txt": img2txt, ret_name + "_txt2img": txt2img})
        
    
    logger.info(f"{res}")

    return {"temporal_retrieval/": res}
        

@torch.no_grad()
def retrieval_accuracy(model, vision_input, positive_texts, negative_texts):
    """
    Calculate retrieval accuracy for a batch of images with one positive and two negative texts each.
    """
    # Encode images and texts
    image_embeddings = model.encode_image(vision_input)[1][:,0,:]  # [batch_size, embed_dim]
    text_embeddings = model.encode_text(positive_texts)[1]  # [batch_size, embed_dim]

    # Compute cosine similarity, scaled by temperature
    logits_per_image = (image_embeddings.unsqueeze(1) @ text_embeddings.transpose(-1, -2)).squeeze(1) / model.temp  # [batch_size, 3]
    logits_per_text = (text_embeddings.unsqueeze(1) @ image_embeddings.transpose(-1, -2)).squeeze(1) / model.temp  # [batch_size, 3]

    labels = torch.arange(logits_per_image.size(0), dtype=torch.long, device=logits_per_image.device)  # positive text is at index 0
    
    image_correct = (logits_per_image.argmax(dim=1) == labels).sum().item()
    image_accuracy = image_correct / logits_per_image.size(0)
    
    text_correct = (logits_per_text.argmax(dim=1) == labels).sum().item()
    text_accuracy = text_correct / logits_per_text.size(0)
    return image_accuracy, text_accuracy



def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")

    # test datasets, a mapping from dataset name to data loader
    test_datasets, test_collate_fn, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets, [None] * len(test_datasets),
        batch_size=[config.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[config.num_workers_test] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[test_collate_fn] * len(test_datasets)
    )
    test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    
    
    return test_name2loaders,  None



def main(config):
    print("is main process?", is_main_process())
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config, eval=True)

    config.video_input.num_frames = 16
    config.fp16 = False
    config.evaluate = True

    # logger.info(f"config: \n{config}")
    logger.info(f"test_file: {config.test_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    test_name2loaders, ret_name2loaders = setup_dataloaders(config, mode="pt")
    

    if config.vit_type == 'vittok': model_cls = VideoTokCLIP
    elif config.vit_type == 'vit3d': model_cls = VideoViT
    else: model_cls = Singularity
    
    model, model_without_ddp, _, _, _, \
        tokenizer, start_epoch, global_step = setup_model(
            config,
            model_cls=model_cls,
            has_decoder=False,
            pretrain=True,
            find_unused_parameters=True,
        )

    logger.info("Start zero-shot evaluation")
    logger.info(f"Global step {global_step}")
    with torch.cuda.amp.autocast(enabled=config.fp16):
        eval_res = {}
        for test_name, test_loader in test_name2loaders.items():
            res = evaluation_wrapper(
                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name)
            eval_res.update(res)
        
        if is_main_process():
            if config.wandb.enable:
                for p, v in eval_res.items():
                    log_dict_to_wandb(v, step=global_step, prefix=p)

            eval_res = pd.DataFrame(eval_res)
            logger.info(f"\n{eval_res.transpose()}")
            
    dist.barrier()
            
                    
    

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
