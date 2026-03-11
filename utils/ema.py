import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMA:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1,
            power=2/3 are good values for models you plan to train for a million
            or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at
            1M steps), gamma=1, power=3/4 for models you plan to train for less
            (reaches decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup.
                Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

    def copy_weights(self, new_model, ema_model):
        ema_model.load_state_dict(new_model.state_dict())

    def get_decay(self, optimization_step):
        """Compute the decay factor."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        return max(self.min_value, min(value, self.max_value))

    @torch.inference_mode()
    def step(self, new_model, ema_model, use_ema, optimization_step):
        if not use_ema:
            self.copy_weights(new_model, ema_model)
            return

        decay = self.get_decay(optimization_step)

        for module, ema_module in zip(new_model.modules(), ema_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - decay)
