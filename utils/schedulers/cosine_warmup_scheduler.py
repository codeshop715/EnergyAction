"""Cosine learning rate scheduler with linear warmup."""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLRScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.
    
    During warmup phase (0 to warmup_steps):
        lr = base_lr * (current_step / warmup_steps)
    
    After warmup (warmup_steps to total_steps):
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        where progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of base lr (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]

