from .ddim import DDIMScheduler
from .ddpm import DDPMScheduler
from .rectified_flow import RFScheduler


def fetch_schedulers(denoise_model, denoise_timesteps):
    if denoise_model == "ddpm":
        position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    elif denoise_model == "ddim":
        position_noise_scheduler = DDIMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        rotation_noise_scheduler = DDIMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    elif denoise_model in ("rectified_flow", "unit", "pi0", "flow_uniform"):
        noise_sampler_config = {"mean": 0, "std": 1.5}
        if denoise_model == "unit":
            noise_sampler_config = {"mean": 0, "std": 1.0}
        samplers = {
            "rectified_flow": "logit_normal",
            "unit": "logit_normal",
            "pi0": "pi0",
            "flow_uniform": "uniform"
        }
        position_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
        rotation_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
    return position_noise_scheduler, rotation_noise_scheduler
