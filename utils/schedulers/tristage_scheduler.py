import math

from torch.optim.lr_scheduler import _LRScheduler


class TriStageLRScheduler(_LRScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations
        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations
        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::
      init_lr = cfg.init_lr_scale * cfg.lr
      lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
      lr = lrs[update_num]

    During hold::
      lr = cfg.lr

    During decay::
      decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
      lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)
    
    Updated it to be CosineAnneaLLR Scheduler

    After that::
      lr = cfg.lr * cfg.final_lr_scale

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer,
        init_lr=2e-5,
        init_lr_scale=0.1,
        final_lr_scale=0.5,
        total_steps=50000,
        phase_ratio="(0.05, 0.1, 0.85)",
        lr=2e-5
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr

        self.phase_ratio = eval(phase_ratio)

        self.warmup_steps = int(total_steps * self.phase_ratio[0])
        self.hold_steps = int(total_steps * self.phase_ratio[1])
        self.decay_steps = int(total_steps * self.phase_ratio[2])

        self.peak_lr = lr
        self.init_lr = init_lr_scale * lr
        self.final_lr = final_lr_scale * lr

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        self.update_step = 0
        self.lr = self.init_lr

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            # self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
            self.lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1 + math.cos(steps_in_stage / self.decay_steps * math.pi))
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]
