import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import glob
import os

from dataset.caption_dataset import (
    ImgTxtRetTrainDataset, ImgTxtRetEvalDataset,
    VidTxtRetTrainDataset, VidTxtRetEvalDataset,
    ImgGraphTrainDataset,ImgGraphEvalDataset,
    VidTxtRetMCEvalDataset, VidGraphTrainDataset, 
    VidGraphEvalDataset, 
)
from dataset.qa_dataset import ImageQADataset, VideoQADataset
from dataset.dataloader import MetaLoader
from dataset.utils import graph_custom_collate_fn, example_retrieval_custom_collate_fn, GaussianBlur


def get_media_type(dataset_config):
    if len(dataset_config) == 3 and dataset_config[2] == "video":
        return "video"
    else:
        return "image"


def create_dataset(dataset_type, config, non_transform=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.))



    train_transform = test_transform = transforms.Compose([
        transforms.Resize(
            (config.image_res, config.image_res),
            interpolation=InterpolationMode.BICUBIC),
        type_transform,
        normalize,
    ])
    
    if non_transform: 
        train_transform = test_transform = transforms.Compose([
            transforms.Resize(
                (config.image_res, config.image_res),
                interpolation=InterpolationMode.BICUBIC),
        ])

    video_only_dataset_kwargs_train = dict(
        video_reader_type=config.video_input.reader,
        sample_type=config.video_input.sample_type,
        num_frames=config.video_input.num_frames,
        num_tries=5,  # false tolerance
        version_ext=config.video_input.version_ext,
    )
    video_only_dataset_kwargs_eval = dict(
        video_reader_type=config.video_input.reader,
        sample_type=config.video_input.sample_type_test,
        num_frames=config.video_input.num_frames_test,
        num_tries=3,  # we want to have predictions for all videos
        version_ext=config.video_input.version_ext,
    )


    if dataset_type in ["ret_train", "ret_eval"]:  # for didemo and activitynet captions
        is_paragraph_retrieval = config.get("is_paragraph_retrieval", False)
        video_only_dataset_kwargs_eval["is_paragraph_retrieval"] = is_paragraph_retrieval
        video_only_dataset_kwargs_train["is_paragraph_retrieval"] = is_paragraph_retrieval

    
    if dataset_type in ["pt_train", "ret_train", "pt_train_eval"]:
        # convert to list of lists
        if dataset_type in ["pt_train", "ret_train"]:
            video_only_dataset_kwargs_eval['eval'] = False
            train_files = [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
            
        else:
            video_only_dataset_kwargs_eval['eval'] = True
            train_files = [config.eval_file] if isinstance(config.eval_file[0], str) else config.eval_file
            
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))
        
        if dataset_type == "ret_train":
            assert len(train_media_types) == 1, \
                f"retrieval downstream should only have one media type, got {train_media_types}"

        train_datasets = []
        for m in train_media_types: 
            dataset_cls = ImgGraphTrainDataset if m == "image" else VidGraphTrainDataset
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]
            dataset_kwargs = dict(
                ann_file=_train_files, transform=train_transform,
                has_multi_vision_gt=False,  # true for ssv2 ret
                image_res = config.image_res,
                mask_down_factor=config.mask_down_factor
            )
            
            if m == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_train)

            train_datasets.append(dataset_cls(**dataset_kwargs))  # replace
            
        return train_datasets, graph_custom_collate_fn


    elif dataset_type in ["pt_eval", "ret_eval"]:

        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        
        if config.video_input.num_frames_test == 1: test_file = config.test_file_image
        elif config.test_corpus != 'all': test_file = {config.test_corpus: config.test_file[config.test_corpus]}
        else: test_file = config.test_file
        
        for name, (data_cfg, has_multi_vision_gt) in test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_cls = ImgGraphEvalDataset if media_type == "image" else VidGraphEvalDataset
            test_dataset_names.append(name)
            dataset_kwargs = dict(
                ann_file=[data_cfg], transform=test_transform,
                image_res = config.image_res,
                has_multi_vision_gt=has_multi_vision_gt,  # true for ssv2 ret
                mask_down_factor=config.mask_down_factor
            )
            if media_type == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_eval)
            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        # test_datasets[0].__getitem__(0)
        return test_datasets, graph_custom_collate_fn, test_dataset_names

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in \
            zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True,
            timeout=300
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
