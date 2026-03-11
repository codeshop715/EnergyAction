from copy import deepcopy
import os
import random
import sys

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from modeling.policy.bimanual_composer import BimanualComposer
from modeling.encoder.text import fetch_tokenizers
from ..common_utils import count_parameters
from ..depth2cloud import fetch_depth2cloud
from ..data_preprocessors import fetch_data_preprocessor
from ..ema import EMA
from ..schedulers import fetch_scheduler
from .utils import compute_metrics


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        self.preprocessor = fetch_data_preprocessor(self.args.dataset)(
            self.args.keypose_only,
            self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud(self.args.dataset)
        )

        if not self.args.eval_only:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=f"{self.args.train_data_dir}/train.zarr",
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size,
            max_demos_per_task=getattr(self.args, 'max_demos_per_task', -1)
        )
        val_dataset = self.dataset_cls(
            root=f"{self.args.eval_data_dir}/val.zarr",
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size,
            max_demos_per_task=getattr(self.args, 'max_demos_per_task', -1)
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=None,
            drop_last=True,
            generator=g,
            prefetch_factor=4,
            persistent_workers=True
        )
        # No sampler for val!
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size_val // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            prefetch_factor=4,
            persistent_workers=True
        )
        return train_loader, val_loader, None

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        model_kwargs = {
            'backbone': self.args.backbone,
            'finetune_backbone': self.args.finetune_backbone,
            'finetune_text_encoder': self.args.finetune_text_encoder,
            'num_vis_instr_attn_layers': self.args.num_vis_instr_attn_layers,
            'fps_subsampling_factor': self.args.fps_subsampling_factor,
            'embedding_dim': self.args.embedding_dim,
            'num_attn_heads': self.args.num_attn_heads,
            'nhist': self.args.num_history,
            'nhand': 2 if self.args.bimanual else 1,
            'num_shared_attn_layers': self.args.num_shared_attn_layers,
            'relative': self.args.relative_action,
            'rotation_format': self.args.rotation_format,
            'denoise_timesteps': self.args.denoise_timesteps,
            'denoise_model': self.args.denoise_model,
            'lv2_batch_size': self.args.lv2_batch_size
        }
        
        # Add EBM-specific parameters if available
        if hasattr(self.args, 'use_ebm_composition') and self.args.use_ebm_composition:
            model_kwargs.update({
                'freeze_flow_weights': getattr(self.args, 'freeze_flow_weights', False),
                'use_performance_guarantee': getattr(self.args, 'use_performance_guarantee', True),
                'enable_energy_in_eval': getattr(self.args, 'enable_energy_in_eval', True),  # Enable energy in eval by default
                'enable_adaptive_denoising': getattr(self.args, 'enable_adaptive_denoising', True),
                'min_denoise_steps': getattr(self.args, 'min_denoise_steps', 5),
                'max_denoise_steps': getattr(self.args, 'max_denoise_steps', 10),
                'energy_threshold_low': getattr(self.args, 'energy_threshold_low', 1.0),
                'energy_threshold_high': getattr(self.args, 'energy_threshold_high', 10.0),
                'enable_early_stopping': getattr(self.args, 'enable_early_stopping', False),
                'early_stop_check_interval': getattr(self.args, 'early_stop_check_interval', 2),
                'enable_coordination_constraints': getattr(self.args, 'enable_coordination_constraints', True),
                'coord_constraint_weight': getattr(self.args, 'coord_constraint_weight', 1.0),
                'coord_weight_hidden_dim': getattr(self.args, 'coord_weight_hidden_dim', 64),
                'min_ee_distance': getattr(self.args, 'min_ee_distance', 0.15),
                'min_joint_distance': getattr(self.args, 'min_joint_distance', 0.12)
            })
            print(f"EBM parameters added: "
                  f"enable_energy_in_eval={model_kwargs['enable_energy_in_eval']}")
            print(f"Adaptive denoising parameters: enable_adaptive_denoising={model_kwargs['enable_adaptive_denoising']}, "
                  f"min_denoise_steps={model_kwargs['min_denoise_steps']}, max_denoise_steps={model_kwargs['max_denoise_steps']}, "
                  f"energy_threshold_low={model_kwargs['energy_threshold_low']}, energy_threshold_high={model_kwargs['energy_threshold_high']}")
            print(f"Coordination constraints: enable={model_kwargs['enable_coordination_constraints']}, "
                  f"weight={model_kwargs['coord_constraint_weight']}, "
                  f"min_ee_distance={model_kwargs['min_ee_distance']}, min_joint_distance={model_kwargs['min_joint_distance']}")
        
        _model = self.model_cls(**model_kwargs)

        # For BimanualComposer, we need to load pretrained weights first to create sub-models
        if isinstance(_model, BimanualComposer):
            if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
                normalizer = self.get_workspace_normalizer()
                _model.set_workspace_normalizer(normalizer)
            # Load pretrained weights to create sub-models BEFORE counting parameters
            _model.load_pretrained_weights(self.args.left_pretrained_path, self.args.right_pretrained_path)

        # Print basic modules' parameters (after sub-models are created for BimanualComposer)
        count_parameters(_model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=f"{self.args.train_data_dir}/train.zarr",
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size,
            max_demos_per_task=getattr(self.args, 'max_demos_per_task', -1)
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # # Track unique parameters by id to avoid duplicates from weight sharing
        # seen_params = set()
        
        # Now split parameters based on name
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip if we've already added this parameter (handles weight sharing)
            # param_id = id(param)
            # if param_id in seen_params:
            #     continue
            # seen_params.add(param_id)
            
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        
        # # Print parameter counts for debugging
        # total_params = sum(len(group["params"]) for group in optimizer_grouped_parameters)
        # print(f"Optimizer parameter groups: {[len(group['params']) for group in optimizer_grouped_parameters]}")
        # print(f"Total unique parameters in optimizer: {total_params}")
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
        return optimizer

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader, train_sampler = self.get_loaders()

        # Get model
        model = self.get_model()
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
            if not isinstance(model, BimanualComposer):
                normalizer = self.get_workspace_normalizer()
                model.workspace_normalizer.copy_(normalizer)

        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters,
            warmup_steps=getattr(self.args, 'warmup_steps', 0),
            min_lr_ratio=getattr(self.args, 'min_lr_ratio', 0.0)
        )
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        # make sure to compile before DDP!
        if self.args.use_compile and not isinstance(model, BimanualComposer):
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)

        # Initialize EMA copy
        ema_model = deepcopy(model)
        # Use fixed EMA decay if specified, otherwise use default warmup schedule
        ema_decay = getattr(self.args, 'ema_decay', 0.999)
        if self.args.use_ema:
            # Use fixed decay for stable training
            self.ema = EMA(
                update_after_step=0,
                inv_gamma=1.0,
                power=1.0,  # power=1 for constant decay
                min_value=ema_decay,
                max_value=ema_decay
            )
            print(f"EMA enabled with fixed decay: {ema_decay}")
        else:
            # Default EMA (will just copy weights)
            self.ema = EMA()
            print("EMA disabled (weights will be copied without averaging)")

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            start_iter, best_loss = self.load_checkpoint(model, ema_model, optimizer)
        
        if model.workspace_normalizer is None:
            model.workspace_normalizer = self.get_workspace_normalizer()
        
        print(model.workspace_normalizer)

        # Eval only
        if self.args.eval_only:
            print("Test evaluation.......")
            model.eval()
            self.evaluate_nsteps(
                ema_model if self.args.use_ema else model,
                val_loader, step_id=-1,
                val_iters=-1
            )
            return ema_model if self.args.use_ema else model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1

        # Training loop
        model.train()
        iter_loader = iter(train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it
                # and increment the epoch
                epoch += 1
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            loss_value = self.train_one_step(model, optimizer, scaler, lr_scheduler, sample)
            self.ema.step(model, ema_model, self.args.use_ema, step_id)
            
            # Log training loss to tensorboard and print periodically
            if not self.args.eval_only:
                self.writer.add_scalar('train/loss', loss_value, step_id)
                
            # Print loss every 100 steps or when step_id is small
            if (step_id + 1) % 100 == 0 or step_id < 10:
                current_lr = lr_scheduler.get_last_lr()[0]
                
                loss_info = f"Step {step_id + 1}: Energy = {loss_value:.6f}, LR = {current_lr:.2e}"
                
                if not self.args.eval_only:
                    self.writer.add_scalar('train/learning_rate', current_lr, step_id)
                
                print(loss_info)
                sys.stdout.flush()  # Force flush the output

            if (step_id + 1) % self.args.val_freq == 0:
                print("Train evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id,
                    val_iters=1250
                )
                # save model
                best_loss = self.save_checkpoint(
                    model, ema_model, optimizer, step_id,
                    new_loss, best_loss
                )
                model.train()

        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        pass  # implement in children

    def _model_forward(self, model, batch, training=True):
        if self.args.pre_tokenize and isinstance(batch["instr"], list):
            batch["instr"] = self.tokenizer(batch["instr"]).cuda(non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(model, BimanualComposer):
                if training:
                    out = model.compute_loss(batch)
                else:
                    out = model.compute_trajectory(**batch)
            else:
                out = model(
                    batch["action"], batch["action_mask"],
                    batch["rgb"], batch.get("rgb2d"), batch["pcd"],
                    batch["instr"], batch["proprioception"],
                    run_inference=not training
                )
        return out  # loss if training, else action

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample):
        """Run a single training step."""
        optimizer.zero_grad()

        # Prepare batch and move to device
        batch = self.prepare_batch(sample, augment=True)

        # Forward pass
        loss = self._model_forward(model, batch, training=True)

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients (use configurable max_norm) - DISABLED
        scaler.unscale_(optimizer)
        # grad_clip_norm = getattr(self.args, 'grad_clip_norm', 1.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # Update
        scaler.step(optimizer)
        scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()
        
        return loss.item()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            batch = self.prepare_batch(sample, augment=False)
            pred_action = self._model_forward(model, batch, training=False)
            
            if isinstance(model, BimanualComposer):
                gt_action = torch.stack(
                    [batch["left_action"], batch["right_action"]], dim=2
                ).cuda(non_blocking=True)
            else:
                gt_action = batch["action"].cuda(non_blocking=True)

            if self.args.relative_action:
                # This part might need adjustment for bimanual case if used
                proprio_for_rel = batch.get("proprioception", batch.get("left_proprioception"))
                pred_action = relative_to_absolute(
                    pred_action[:, :, 0],
                    proprio_for_rel.cuda(non_blocking=True)[:, :, 0]
                )
                gt_action = relative_to_absolute(
                    gt_action[:, :, 0],
                    proprio_for_rel.cuda(non_blocking=True)[:, :, 0]
                )

            losses, losses_B = compute_metrics(pred_action, gt_action)

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

        # Log all statistics
        values = {k: v.mean().item() for k, v in values.items()}
        if step_id > -1:
            for key, val in values.items():
                self.writer.add_scalar(key, val, step_id)

        # Also log to terminal
        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value:.03f}")

        return -values[f'{split}-losses/mean/traj_pos_acc_001']

    def load_checkpoint(self, model, ema_model, optimizer):
        """Load from checkpoint."""
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            # For non-composer models, compute workspace bounds here
            if not isinstance(model, BimanualComposer):
                print('The main process will compute workspace bounds')
                normalizer = self.get_workspace_normalizer()
                model.workspace_normalizer.copy_(normalizer)
            return 0, None

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            ema_model.load_state_dict(model_dict["ema_weight"], strict=True)
        # Useful for resuming training
        if 'optimizer' in model_dict and not self.args.eval_only:
            optimizer.load_state_dict(model_dict["optimizer"])
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        model_state = model.state_dict()
        ema_state = ema_model.state_dict() if self.args.use_ema else None

        # Best checkpoint
        if best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        torch.save({
            "weight": model_state,
            "ema_weight": ema_state,
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % self.args.interm_ckpt_freq == 0:
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    """Custom collate_fn, measured to be faster than default."""
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        if key not in batch[0].keys():
            continue
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


def actions_collate_fn(batch):
    return {"action": torch.cat([item["action"] for item in batch])}


def relative_to_absolute(action, proprio):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([pos, orn, action[..., 6:]], -1)
