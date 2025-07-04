""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim
import logging
logger = logging.getLogger(__name__)
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay, no_decay_list=(), filter_bias_and_bn=True):
    named_param_tuples = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen weights
        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
            named_param_tuples.append([name, param, 0])
        elif name in no_decay_list:
            named_param_tuples.append([name, param, 0])
        else:
            named_param_tuples.append([name, param, weight_decay])
    return named_param_tuples

def add_different_lr(named_param_tuples, diff_lr_names, diff_lr, default_lr):
    named_param_tuples_with_lr = []
    for name, param, wd in named_param_tuples:
        if not param.requires_grad:
            continue  # Skip non-trainable parameters
        use_diff_lr = any(diff_name in name for diff_name in diff_lr_names)
        named_param_tuples_with_lr.append([name, param, wd, diff_lr if use_diff_lr else default_lr])
    return named_param_tuples_with_lr

def create_optimizer_params_group(named_param_tuples_with_lr):
    group = {}
    for name, param, wd, lr in named_param_tuples_with_lr:
        if not param.requires_grad:
            continue  # Skip non-trainable parameters
        if wd not in group:
            group[wd] = {}
        if lr not in group[wd]:
            group[wd][lr] = []
        group[wd][lr].append(param)

    optimizer_params_group = []
    for wd, lr_groups in group.items():
        for lr, params in lr_groups.items():
            optimizer_params_group.append(dict(
                params=params,
                weight_decay=wd,
                lr=lr
            ))
            logger.info(f"optimizer -- lr={lr} wd={wd} len(params)={len(params)}")
    return optimizer_params_group

def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    # Check for modules that require different learning rates
    if hasattr(args, "different_lr") and args.different_lr.enable:
        diff_lr_module_names = args.different_lr.module_names
        diff_lr = args.different_lr.lr
    else:
        diff_lr_module_names = []
        diff_lr = None

    no_decay = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else {}

    # Filter only parameters that require gradients before proceeding
    named_param_tuples = add_weight_decay(
        model, weight_decay, no_decay, filter_bias_and_bn
    )
    named_param_tuples = add_different_lr(
        named_param_tuples, diff_lr_module_names, diff_lr, args.lr
    )
    parameters = create_optimizer_params_group(named_param_tuples)

    # Additional optimizer configuration
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps'):
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas'):
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args'):
        opt_args.update(args.opt_args)

    # Optimizer selection
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_type = opt_lower.split('_')[-1]
    if opt_type == 'sgd' or opt_type == 'nesterov':
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_type == 'momentum':
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_type == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError("Invalid optimizer type")
    return optimizer