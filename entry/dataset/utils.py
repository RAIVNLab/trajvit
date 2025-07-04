from utils.distributed import is_main_process, get_rank, get_world_size
import logging
import torch.distributed as dist
import torch
import os
import json
import re
import numpy as np
from os.path import join
from tqdm import trange
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import PILToTensor
import imageio
import torch.nn.functional as F
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def graph_custom_collate_fn(batch):
    
    video = torch.stack([b[0] for b in batch])
    caption = [b[1] for b in batch]
    match_id = torch.tensor([b[2] for b in batch])
    masks = torch.stack([b[3] for b in batch])
    graphs_list = [b[4] for b in batch]
    num_token = torch.tensor([b[5] for b in batch])
    
    
    # Find the maximum M (first dimension) in the batch
    max_M = max([tensor.shape[0] for tensor in graphs_list]) 
    T, batch_size, dtype = graphs_list[0].shape[1], len(graphs_list), graphs_list[0].dtype
    padded_graphs = torch.zeros(batch_size, max_M, T, dtype=dtype)
    # Copy tensors into the padded_batch
    for i, tensor in enumerate(graphs_list):
        padded_graphs[i, :num_token[i], :] = tensor  # Copy the tensor without for-loop padding
        
    return video, caption, match_id, masks, padded_graphs, num_token






def example_retrieval_custom_collate_fn(batch):
    
    gathered_batch = []
    for bitem in batch:
        video = torch.stack(bitem[0])
        caption = bitem[1]
        masks = torch.stack(bitem[2])
        graphs_list = bitem[3]
        
        # Find the maximum M (first dimension) in the batch
        max_M = max([tensor.shape[0] for tensor in graphs_list]) 
        T, batch_size, dtype = graphs_list[0].shape[1], len(graphs_list), graphs_list[0].dtype
        padded_graphs = torch.zeros(batch_size, max_M, T, dtype=dtype)
        # Copy tensors into the padded_batch
        for i, tensor in enumerate(graphs_list):
            padded_graphs[i, :graphs_list[i].shape[0], :] = tensor  # Copy the tensor without for-loop padding
            
        gathered_batch.append([video, caption, masks, padded_graphs])
        
    return gathered_batch



def load_segmentation_and_transform(input_dir):
    # Get all image files in the directory, sorted by frame number
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    video_arr = []
    # Load each image and append it to the array
    for t, image_file in enumerate(image_files):
        video_arr.append(imageio.imread(os.path.join(input_dir, image_file)))
    
    video_arr = np.stack(video_arr)
    
    return torch.from_numpy(video_arr)


def load_image_from_path(image_path):
    image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image


def load_anno(ann_file_list):
    """[summary]

    Args:
        ann_file_list (List[List[str, str]] or List[str, str]):
            the latter will be automatically converted to the former.
            Each sublist contains [anno_path, image_root], (or [anno_path, video_root, 'video'])
            which specifies the data type, video or image

    Returns:
        List(dict): each dict is {
            image: str or List[str],  # image_path,
            caption: str or List[str]  # caption text string
        }
    """
    if isinstance(ann_file_list[0], str):
        ann_file_list = [ann_file_list]

    ann = []
    for d in ann_file_list:
        data_root = d[1]
        fp = d[0]
        is_video = len(d) == 3 and d[2] == "video"
        cur_ann = json.load(open(fp, "r"))
        iterator = trange(len(cur_ann), desc=f"Loading {fp}") \
            if is_main_process() else range(len(cur_ann))
        for idx in iterator:
            key = "video" if is_video else "image"
            # unified to have the same key for data path
            cur_ann[idx]["image"] = os.path.join(data_root, cur_ann[idx][key])
        ann += cur_ann
    return ann


def pre_text(text, max_l=None):
    text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
    text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    text = re.sub(r"\s{2,}", ' ', text)
    text = text.rstrip('\n').strip(' ')

    if max_l:  # truncate
        words = text.split(' ')
        if len(words) > max_l:
            text = ' '.join(words[:max_l])
    return text


logger = logging.getLogger(__name__)


def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(
            result_dir, '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, '%s_rank%d.pth' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    result = None
    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)

    return result


def sync_save_result(result, result_dir, filename, is_json=True, is_list=True):
    """gather results from multiple GPUs"""
    if is_json:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.pth' % (filename, get_rank()))
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:
            json.dump(result, open(final_result_file, 'w'))
        else:
            torch.save(result, final_result_file)

        logger.info('result file saved to %s' % final_result_file)
    dist.barrier()
    return final_result_file, result


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths





class GaussianBlur:
    """
    Gaussian blur augmentation for PyTorch tensors.
    Inspired by SimCLR: https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma=[0.1, 2.0], kernel_size=5):
        self.sigma = sigma
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for GaussianBlur.")

    def __call__(self, x):
        """
        Apply Gaussian blur to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W).
        Returns:
            torch.Tensor: Blurred tensor.
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must have shape (C, H, W).")

        # Choose a random sigma value
        sigma = random.uniform(self.sigma[0], self.sigma[1])

        # Create the Gaussian kernel
        kernel = self._create_gaussian_kernel(sigma, self.kernel_size, x.device)

        # Apply the kernel to each channel separately
        x = x.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        x = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=x.size(1))  # Channel-wise convolution
        return x.squeeze(0)  # Remove batch dimension

    def _create_gaussian_kernel(self, sigma, kernel_size, device):
        """
        Create a 2D Gaussian kernel for a given sigma and kernel size.
        Args:
            sigma (float): Standard deviation of the Gaussian.
            kernel_size (int): Size of the kernel (must be odd).
            device (torch.device): Device for the kernel.
        Returns:
            torch.Tensor: Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
        """
        # Create a 1D Gaussian kernel
        x = torch.arange(kernel_size, device=device) - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss /= gauss.sum()  # Normalize

        # Outer product to form a 2D Gaussian kernel
        kernel = torch.outer(gauss, gauss)
        kernel /= kernel.sum()  # Normalize
        kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Shape (1, 1, H, W)
        kernel = kernel.repeat(3, 1, 1, 1)  # Shape (C, 1, H, W) for RGB images

        return kernel
    
    
    
    
    
    
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


import torchvision.transforms.v2 as tv2   # torchvision ≥ 0.15

class PanopticPosAugmentation:
    def __init__(self, size, eval=False):
        # self.image_transform = transforms.Compose([
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=(5,5))], p=0.5),
        # ])
        
        self.image_transform = tv2.Compose([
            tv2.RandomApply(                               # p = 0.8
                [tv2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),

            tv2.RandomGrayscale(p=0.2),                    # p = 0.2

            tv2.RandomApply(                               # p = 0.5
                [tv2.GaussianBlur(kernel_size=(5, 5),
                                sigma=(0.1, 2.0))], p=0.5),
        ])

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.normalize = transforms.Normalize(mean, std)
        # self.type_transform = transforms.Lambda(lambda x: x.float().div(255.))
        self.size = size  # Size for resizing/cropping
        # self.i = 0
        

    def __call__(self, image, mask):
        # Random resized crop
        image = image.to(torch.float32) / 255.       # tensor-ise + scale
        mask  = mask.to(torch.uint8)                 # keep mask cheap

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.)
        )
        # image = self.type_transform(image)
        image_crop = F.resized_crop(image, i, j, h, w, (self.size, self.size))
        mask = F.resized_crop(mask, i, j, h, w, (self.size, self.size), interpolation=Image.NEAREST)

        if torch.isnan(image_crop).any():
            print("Image crop contains NaN!")
            
        # Random horizontal flip
        if random.random() > 0.5:
            image_crop = F.hflip(image_crop)
            mask = F.hflip(mask)
        # Apply image-only transformations
        image_transform = self.image_transform(image_crop)
        
        # imageio.mimsave(f"sample_{i}.mp4", np.array(image_transform.permute(0,2,3,1)*255).astype(np.uint8), fps=5, codec="libx264")
        
        # Image.fromarray(np.array(image[0].permute(1,2,0)*255).astype(np.uint8)).save(f"example_{i}.png")
        # self.i += 1
        # Normalize the image
        image_normalize = self.normalize(image_transform)
        
        if torch.isnan(image_normalize).any():
            print("Image normalize contains NaN!")

        return image_normalize, mask
    