import pandas as pd
import time
import datetime
import wandb
from os.path import join
import logging
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import sys
# sys.path.append('/weka/chenhaoz/home/videotok/entry')
from models.model_pretrain import Singularity, VideoViT, VideoTokCLIP

from utils.logger import log_dict_to_wandb, setup_wandb
from utils.config_utils import setup_main
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed, remove_files_if_exist
from utils.distributed import get_rank, get_world_size, is_main_process, init_distributed_mode
from dataset import create_dataset, create_sampler, create_loader, MetaLoader
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model


from dataset.random_token_collator import RandomMaskCollator
from dataset.random_tube_collator import TubeMaskCollator

logger = logging.getLogger(__name__)

logger.info("finish importing package")


def gather_across_gpus(value):
    # Gather values from all processes
    gathered_values = [torch.zeros_like(value) for _ in range(get_world_size())]
    dist.all_gather(gathered_values, value)
    return gathered_values

def train(model, train_loaders, optimizer, tokenizer, epoch, global_step, device, scheduler, scaler, config, prefix='train/'):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=30, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=30, fmt="{value:.4f}"))
    # loss_names = ["loss_mlm", "loss_ita", "loss_itm"]
    loss_names = ["loss_ita", "accuracy_ita"]
    media_types = [loader.dataset.media_type for loader in train_loaders]
    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(f"{m}-{name}", SmoothedValue(window=30, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    
    for i, batch in enumerate(iterator):  
        # if 'debug' in config.output_dir:
        #     # print("forwarding", i)
        #     if i > 100: break
        
        media_type, real_batch = batch
        
        if config.mask_ratio > 0:
            (image, text, idx, segment, graph, num_tokens), masks_enc = real_batch
            masks_enc = masks_enc.to(device, non_blocking=True)
        else:
            image, text, idx, segment, graph, num_tokens = real_batch
            masks_enc = None

        image = image.to(device, non_blocking=True)
        if config.vit_type == 'vittok':
            segment = segment.to(device, non_blocking=True)
            graph = graph.to(device, non_blocking=True)
            num_tokens = num_tokens.to(device, non_blocking=True)
        
        text_input = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.max_txt_l[media_type], return_tensors="pt"
        ).to(device)  # change from "longest" to "max_length"

        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model((image, segment, graph, num_tokens), text_input, idx=None, training_mask=masks_enc)
            loss = loss_dict["loss_ita"]

        # change finishes
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.optimizer.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable \
                and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix=prefix)

        global_step += 1

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        
    return global_step


@torch.no_grad()
def eval_train(model, eval_loaders, tokenizer, global_step, device, config):
    model.eval()
    avg_eval_accuracy, avg_eval_loss = [], []
    media_types = [loader.dataset.media_type for loader in eval_loaders]
    eval_loader = MetaLoader(name2loader=dict(list(zip(media_types, eval_loaders))))
    for i, batch in enumerate(eval_loader):
        media_type,  (image, text, idx, segment, graph, num_tokens) = batch
        
        # if 'debug' in config.output_dir:
        #     if i > 100: break
            # print("forwarding", i)

        image = image.to(device, non_blocking=True)
        if config.vit_type == 'vittok':
            segment = segment.to(device, non_blocking=True)
            graph = graph.to(device, non_blocking=True)
            num_tokens = num_tokens.to(device, non_blocking=True)
        text_input = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.max_txt_l[media_type], return_tensors="pt"
        ).to(device)  # change from "longest" to "max_length"
    
        with torch.no_grad(): loss_dict = model((image, segment, graph, num_tokens), text_input, idx=None)
        avg_eval_accuracy.append(loss_dict["accuracy_ita"])
        avg_eval_loss.append(loss_dict["loss_ita"])
    
    avg_eval_loss = torch.mean(torch.tensor(avg_eval_loss)).item()
    avg_eval_accuracy = torch.mean(torch.tensor(avg_eval_accuracy)).item()


    if is_main_process():
        logger.info(f"Eval Accuracy: {avg_eval_accuracy}, total batch: {i}")
        if config.wandb.enable: 
            log_dict_to_wandb({"video-text matching accuracy": avg_eval_accuracy}, step=global_step, prefix="eval/")
            log_dict_to_wandb({"eval set loss": avg_eval_loss}, step=global_step, prefix="eval/")
    

        


def setup_dataloaders(config, mode="pt", finetune_stage=False):
    
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    train_datasets, collate_fn = create_dataset(f"{mode}_train", config)
    media_types = [d.media_type for d in train_datasets]

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler(
            train_datasets, [True] * len(media_types), num_tasks, global_rank)
    else:
        samplers = [None] * len(media_types)
    
    if finetune_stage: config.mask_ratio = 0
    if config.mask_ratio > 0:
        if config.vit_type == 'vittok': 
            mask_collate_cls = RandomMaskCollator
        elif config.vit_type == 'vit3d': 
            mask_collate_cls = TubeMaskCollator
        else: raise NotImplementedError

    collect_collators = []
    for m in media_types:
        if config.mask_ratio > 0 and m == 'video':
            collect_collators.append(mask_collate_cls(collate_fn, ratio=config.mask_ratio))
        else:
            collect_collators.append(collate_fn)
    
    
    train_loaders = create_loader(
        train_datasets, samplers,
        batch_size=[config.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=collect_collators,
    )  # [0]
    
    if finetune_stage: return train_loaders

    eval_datasets, eval_collate_fn = create_dataset(f"{mode}_train_eval", config)
    eval_media_types = [d.media_type for d in eval_datasets]
    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        eval_samplers = create_sampler(
            eval_datasets, [True] * len(eval_media_types), num_tasks, global_rank)
    else:
        eval_samplers = [None] * len(eval_media_types)
    eval_loaders = create_loader(
        eval_datasets, eval_samplers,
        batch_size=[config.batch_size[k] for k in eval_media_types],
        num_workers=[config.num_workers] * len(eval_media_types),
        is_trains=[True] * len(eval_media_types),
        collate_fns=[eval_collate_fn] * len(eval_media_types),
    )  # [0]

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
    
    
    return train_loaders, eval_loaders, test_name2loaders, media_types




import os
def main(config):
    print("is main process?", is_main_process())
    print("global rank", os.environ["RANK"], "local rank", os.environ["LOCAL_RANK"])
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config, eval=True)
        
    config.optimizer.lr = config.optimizer.lr / 10
    
    if config.vit_type == 'vit3d': config.pretrained_path = '/weka/chenhaoz/home/videotok/results/ckpts/pt_panda_1m/panda_1m_2025_vit3d/ckpt_29.pth'
    elif config.vit_type == 'vittok': config.pretrained_path = '/weka/chenhaoz/home/videotok/results/ckpts/pt_panda_4m/panda_4m_cny_vittok/ckpt_29.pth'
    # logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, eval_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="pt", finetune_stage=False)
    
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    # print("----------------------------")
    # print("step", config.scheduler.num_training_steps, config.scheduler.num_warmup_steps)
    # print("----------------------------")
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    if config.vit_type == 'vittok': model_cls = VideoTokCLIP
    elif config.vit_type == 'vit3d': model_cls = VideoViT
    else: model_cls = Singularity
    
    model, model_without_ddp, optimizer, scheduler, scaler, \
        tokenizer, start_epoch, global_step = setup_model(
            config,
            model_cls=model_cls,
            has_decoder=False,
            pretrain=True,
            find_unused_parameters=True,
        )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    logger.info("Start training")
    start_time = time.time()
    
    
    epoch = 0
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
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res.transpose()}")

    for epoch in range(1, config.scheduler.epochs - config.scheduler.finetune_epochs):

        global_step = train(
            model, train_loaders, optimizer, tokenizer, epoch, global_step,
            device, scheduler, scaler, config
        )
        
        dist.barrier()
        
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
                logger.info(f"Epoch {epoch}")
                logger.info(f"\n{eval_res.transpose()}")
                

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()

if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    