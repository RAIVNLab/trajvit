
import torch

def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    if type(masks) is not list: masks = [masks]
    all_x = []
    for m in masks:
        if len(x.shape) == 3:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        else:
            mask_keep = m
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)





def compute_centers(masks, normalize=True):
    
    N, w, h = masks.shape
    x_coords = torch.arange(w, device=masks.device).view(1, -1, 1).expand(N, w, h)
    y_coords = torch.arange(h, device=masks.device).view(1, 1, -1).expand(N, w, h)
    
    if normalize: 
        x_coords = x_coords / w
        y_coords = y_coords / h
    
    
    x_sum = torch.sum(masks * x_coords, dim=(1, 2))
    y_sum = torch.sum(masks * y_coords, dim=(1, 2))
    area = torch.sum(masks, dim=(1, 2))
    
    
    area_nonzero = area.clamp(min=1)  # Prevent division by zero
    
    x_center = x_sum / area_nonzero
    y_center = y_sum / area_nonzero
    
    centers = torch.stack((x_center, y_center), dim=1)
    
    centers[area == 0] = -1
    
    return centers


def compute_bounding_boxes(masks, normalize=True):
    """
    Computes the bounding boxes for a set of binary masks.

    Args:
        masks: A tensor of shape (N, w, h) where N is the number of masks.
        normalize: Whether to normalize the bounding box coordinates by the width and height.

    Returns:
        A tensor of shape (N, 4) containing bounding boxes in (x_min, y_min, x_max, y_max) format.
        If normalize is True, the coordinates will be in the range [0, 1].
    """
    N, w, h = masks.shape
    
    # Get x and y coordinates
    x_coords = torch.arange(w, device=masks.device).view(1, -1, 1).expand(N, w, h)
    y_coords = torch.arange(h, device=masks.device).view(1, 1, -1).expand(N, w, h)
    
    # Mask out coordinates where there are no objects
    masks_flattened = masks.reshape(N, -1)
    x_coords_flattened = x_coords.reshape(N, -1)
    y_coords_flattened = y_coords.reshape(N, -1)

    # Compute min and max for each mask
    x_min = torch.where(masks_flattened, x_coords_flattened, w).min(dim=1)[0]
    x_max = torch.where(masks_flattened, x_coords_flattened, -1).max(dim=1)[0]
    y_min = torch.where(masks_flattened, y_coords_flattened, h).min(dim=1)[0]
    y_max = torch.where(masks_flattened, y_coords_flattened, -1).max(dim=1)[0]
    
    # Normalize if required
    if normalize:
        x_min = x_min / w
        x_max = x_max / w
        y_min = y_min / h
        y_max = y_max / h

    # Stack into bounding box format (x_min, y_min, x_max, y_max)
    bounding_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    
    # Handle empty masks (no object): set bounding box to -1
    bounding_boxes[(x_min == w) & (y_min == h)] = -1
    
    return bounding_boxes



def decompose_masks(masks):
    """ 
    inputs: masks (B,T,W,H) with value index; 
    outputs: return_masks (B,T,N,W,H) with binary value; where N is maximum number of objects
    """
    masks = masks.long() 
    max_N = masks.max() + 1
    
    mask = masks.unsqueeze(2)
    one_hot = (mask == torch.arange(0, max_N, device=masks.device, dtype=masks.dtype).view(1, 1, max_N, 1, 1))

    one_hot[:,:,0] = 0
    return one_hot
    
from einops import rearrange, repeat
def decompose_masks_no_t(masks):
    """ 
    inputs: masks (B,T,W,H) with value index; 
    outputs: return_masks (B*T,N,W,H) with binary value; where N is maximum number of objects
    """
    masks = rearrange(masks, 'B T W H -> (B T) W H')
    masks = masks.long() 
    max_N = masks.max() + 1
    
    mask = masks.unsqueeze(1)
    one_hot = (mask == torch.arange(0, max_N, device=masks.device, dtype=masks.dtype).view(1, max_N, 1, 1))

    one_hot[:,0] = 0
    return one_hot 
    
# Function to print GPU memory usage
def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
