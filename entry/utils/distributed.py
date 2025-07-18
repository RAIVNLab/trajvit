import os
import torch
import torch.distributed as dist
import logging


logger = logging.getLogger(__name__)


def setup_for_distributed(is_master):
    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)

    if not is_master:
        logging.disable()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def init_distributed_mode(args):
    # job started by torch.distributed.launch
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
   
    args.distributed = True
    args.dist_backend = 'nccl'

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
        world_size=args.world_size,  # Set by the launcher
        rank=args.rank,  # Unique rank of the current process
        
    )
    print(f"Initialized process group with rank {args.rank}/{args.world_size}; local rank {args.gpu}")
    

    dist.barrier()
    setup_for_distributed(args.rank == 0)


# Copyright (c) Facebook, Inc. and its affiliates.
# copied from https://github.com/facebookresearch/vissl/blob/master/vissl/utils/distributed_gradients.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# copied from megavlt
def gather_tensor_along_batch_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    tensor_list = GatherLayer.apply(tensor)
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


@torch.no_grad()
def gather_tensor_along_batch(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list
