import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as BaseScheduler


class DDPMScheduler(BaseScheduler):
    """A wrapper class for DDPM which handles sampling noise for training."""

    def sample_noise_step(self, num_noise, device):
        timesteps = torch.randint(
            0,
            self.config.num_train_timesteps,
            (num_noise,), device=device
        ).long()

        return timesteps

    def prepare_target(self, noise, gt):
        if self.config.prediction_type == "epsilon":
            return noise
        elif self.config.prediction_type == "sample":
            return gt
        else:
            raise NotImplementedError

    def step(
        self,
        model_output: torch.Tensor,
        timestep_ind: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ):
        return super().step(
            model_output=model_output,
            timestep=self.timesteps[timestep_ind].to(model_output.device),
            sample=sample,
            generator=generator,
            return_dict=return_dict
        )
