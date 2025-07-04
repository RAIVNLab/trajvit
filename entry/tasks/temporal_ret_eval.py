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

from models.model_pretrain import Singularity, VideoViT, VideoTokCLIP, VivitViT, TokenLearnerViT

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
        for i, _batch in tqdm(enumerate(ret_loader), total=len(ret_loader), desc=ret_name):
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
            
            if i % 100 == 0 and is_main_process():
                logger.info(f"{ret_name}: progress {i}/{len(ret_loader)}; img2txt accuracy {imgacc}; txt2img accuracy {txtacc}")
                
        img2txt = torch.mean(torch.tensor(img2txt)).item()
        txt2img = torch.mean(torch.tensor(txt2img)).item()
        res.update({ret_name + "_img2txt": img2txt, ret_name + "_txt2img": txt2img})
        
    
    logger.info(f"{res}")

    return {"temporal_retrieval/": res}
        

@torch.no_grad()
def retrieval_accuracy(model, vision_input, texts):
    """
    Calculate retrieval accuracy for a batch of images with one positive and two negative texts each.
    """
    # Encode images and texts
    image_embeddings = model.encode_image(vision_input)[1]  # [batch_size, 1, embed_dim]
    text_embeddings = model.encode_text(texts)[1]  # [batch_size, embed_dim]

    # Compute cosine similarity, scaled by temperature
    logits_per_image, logits_per_text = model.get_sim(image_embeddings, text_embeddings)

    labels = torch.arange(logits_per_image.size(0), dtype=torch.long, device=logits_per_image.device)  
    
    image_correct = (logits_per_image.argmax(dim=1) == labels).sum().item()
    image_accuracy = image_correct / logits_per_image.size(0)
    
    text_correct = (logits_per_text.argmax(dim=1) == labels).sum().item()
    text_accuracy = text_correct / logits_per_text.size(0)
    return image_accuracy, text_accuracy



def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")

    # test datasets, a mapping from dataset name to data loader
    # test_datasets, test_collate_fn, test_dataset_names = create_dataset(f"{mode}_eval", config)
    # test_loaders = create_loader(
    #     test_datasets, [None] * len(test_datasets),
    #     batch_size=[config.batch_size_test[d.media_type] for d in test_datasets],
    #     num_workers=[config.num_workers_test] * len(test_datasets),
    #     is_trains=[False] * len(test_datasets),
    #     collate_fns=[test_collate_fn] * len(test_datasets)
    # )
    # test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    
    ret_datasets, ret_collate_fn, ret_dataset_names = create_dataset(f"{mode}_example_ret", config)
    ret_loaders = create_loader(
        ret_datasets, [None] * len(ret_datasets),
        batch_size=[1 for d in ret_datasets],
        num_workers=[config.num_workers_test] * len(ret_datasets),
        is_trains=[False] * len(ret_datasets),
        collate_fns=[ret_collate_fn] * len(ret_datasets)
    )
    ret_name2loaders = {k: v for k, v in zip(ret_dataset_names, ret_loaders)}
    
    return None, ret_name2loaders



def main(config):
    print("is main process?", is_main_process())
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config, eval=True, manual_version='temporal_eval:1.0')

    if config.mask_ratio > 0:
        config.scheduler.epochs = config.scheduler.epochs * 2
        config.scheduler.finetune_epochs = config.scheduler.epochs // 6
        
    config.video_input.num_frames = 16

    # logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    test_name2loaders, ret_name2loaders = setup_dataloaders(config, mode="pt")
    
    num_steps_per_epoch = 10000  # dummy value
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    

    if config.vit_type == 'vittok': model_cls = VideoTokCLIP
    elif config.vit_type == 'vit3d': model_cls = VideoViT
    elif config.vit_type == 'vivit': model_cls = VivitViT
    elif config.vit_type == 'token_learner': model_cls = TokenLearnerViT
    else: model_cls = Singularity
    
    model, model_without_ddp, _, _, _, \
        tokenizer, start_epoch, global_step = setup_model(
            config,
            model_cls=model_cls,
            has_decoder=False,
            pretrain=True,
            find_unused_parameters=True,
        )
    assert global_step == 0

    logger.info("Start zero-shot evaluation")
    start_time = time.time()
        

    for epoch in range(1, config.scheduler.epochs - config.scheduler.finetune_epochs):
        save_freq = config.scheduler.epochs // 10
        if (epoch % save_freq == 0 or epoch == config.scheduler.epochs-1) :
            ckpt_path = join(config.output_dir, f"ckpt_{epoch:02d}.pth")           
            if ckpt_path and os.path.exists(ckpt_path) and not os.path.isdir(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                state_dict = checkpoint["model"]
                global_step = checkpoint["global_step"]
                msg = model_without_ddp.load_state_dict(state_dict, strict=False)
                logger.info(msg)
                logger.info(f"Loaded checkpoint from {ckpt_path}")
            else:
                break
            
            
        

            with torch.cuda.amp.autocast(enabled=config.fp16):
                eval_res = {}
                res = test_example_ret(model_without_ddp, ret_name2loaders, tokenizer, device, config)
                eval_res.update(res)
                # for test_name, test_loader in test_name2loaders.items():
                #     res = evaluation_wrapper(
                #         model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name)
                #     eval_res.update(res)
                
                if is_main_process():
                    if config.wandb.enable:
                        for p, v in eval_res.items():
                            log_dict_to_wandb(v, step=global_step, prefix=p)

                    eval_res = pd.DataFrame(eval_res)
                    logger.info(f"Epoch {epoch}")
                    logger.info(f"\n{eval_res.transpose()}")
                    
            dist.barrier()
            
    if config.scheduler.finetune_epochs > 0:
        for epoch in range(config.scheduler.epochs - config.scheduler.finetune_epochs, config.scheduler.epochs):
            save_freq = max(config.scheduler.epochs // 10, 1)
            if (epoch % save_freq == 0 or epoch == config.scheduler.epochs-1) :
                if epoch != 0:
                    ckpt_path = join(config.output_dir, f"ckpt_{epoch:02d}.pth")
                    
                    loaded = False
                    while not loaded:                  
                        if ckpt_path and os.path.exists(ckpt_path) and not os.path.isdir(ckpt_path):
                            checkpoint = torch.load(ckpt_path, map_location="cpu")
                            state_dict = checkpoint["model"]
                            global_step = checkpoint["global_step"]
                            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
                            logger.info(msg)
                            logger.info(f"Loaded checkpoint from {ckpt_path}")
                            loaded=True
                        else:
                            logger.info(f"haven't found checkpoint {ckpt_path}")
                            time.sleep(60*20)  # 20 minutes of sleeping
                else:
                    continue
                        
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    eval_res = {}
                    res = test_example_ret(model_without_ddp, ret_name2loaders, tokenizer, device, config)
                    eval_res.update(res)
                    # for test_name, test_loader in test_name2loaders.items():
                    #     res = evaluation_wrapper(
                    #         model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name)
                    #     eval_res.update(res)
                    
                    if is_main_process():
                        if config.wandb.enable:
                            for p, v in eval_res.items():
                                log_dict_to_wandb(v, step=global_step, prefix=p)

                        eval_res = pd.DataFrame(eval_res)
                        logger.info(f"Epoch {epoch}")
                        logger.info(f"\n{eval_res.transpose()}")
                        
                dist.barrier()
                    
    

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
