from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR

from .tristage_scheduler import TriStageLRScheduler
from .cosine_warmup_scheduler import CosineWarmupLRScheduler


def fetch_scheduler(type_, optimizer, train_iters, warmup_steps=0, min_lr_ratio=0.0):
    """
    Fetch learning rate scheduler.
    
    Args:
        type_: Scheduler type ('constant', 'cosine', 'cosine_warmup', 'tristage_flower')
        optimizer: PyTorch optimizer
        train_iters: Total training iterations
        warmup_steps: Number of warmup steps (only used for 'cosine_warmup')
        min_lr_ratio: Minimum learning rate as a ratio of base lr (default: 0.0)
    """
    if type_ == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=train_iters)
    elif type_ == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=min_lr_ratio * optimizer.param_groups[0]['lr'])
    elif type_ == "cosine_warmup":
        scheduler = CosineWarmupLRScheduler(
            optimizer, 
            warmup_steps=warmup_steps, 
            total_steps=train_iters,
            min_lr_ratio=min_lr_ratio  # Use provided min_lr_ratio
        )
    elif type_ == "tristage_flower":
        scheduler = TriStageLRScheduler(optimizer)
    else:
        raise NotImplementedError(f"Scheduler type '{type_}' not implemented")

    return scheduler
