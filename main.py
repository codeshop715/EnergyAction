"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import sys

import torch

# Import setproctitle for process name management
try:
    import setproctitle
    HAS_SETPROCTITLE = True
except ImportError:
    HAS_SETPROCTITLE = False
    print("Warning: setproctitle not available. Process name will not be set.")

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none
from utils.trainers import fetch_train_tester


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract"),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        ('max_demos_per_task', int, -1),  # limit demos per task (-1 = use all)
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Training and testing arguments
        ('checkpoint', str_none, None),
        ('left_pretrained_path', str_none, None),
        ('right_pretrained_path', str_none, None),
        ('val_freq', int, 4000),
        ('interm_ckpt_freq', int, 1000000),
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('warmup_steps', int, 0),  # Number of warmup steps for lr scheduler
        ('min_lr_ratio', float, 0.0),  # Minimum LR as ratio of base LR (e.g., 0.05 for 5% of base LR)
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('ema_decay', float, 0.999),  # EMA decay rate
        # ('grad_clip_norm', float, 1.0),  # Gradient clipping max norm - DISABLED
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.05),
        ('weight_sharing', str2bool, False),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 4),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        # EBM arguments
        ('use_ebm_composition', str2bool, False),

        ('enable_energy_in_eval', str2bool, True),
        ('freeze_flow_weights', str2bool, True),
        ('enable_adaptive_denoising', str2bool, True),
        ('min_denoise_steps', int, 5),
        ('max_denoise_steps', int, 10),
        ('energy_threshold_low', float, 1.0),
        ('energy_threshold_high', float, 10.0),
        ('enable_early_stopping', str2bool, False),
        ('early_stop_check_interval', int, 2),
        ('enable_coordination_constraints', str2bool, True),
        ('coord_constraint_weight', float, 1.0),
        ('coord_weight_hidden_dim', int, 64),
        ('min_ee_distance', float, 0.15),
        ('min_joint_distance', float, 0.12)
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


if __name__ == '__main__':
    # Set process name from environment variable if available
    process_name = os.environ.get('PROCESS_NAME')
    if process_name and HAS_SETPROCTITLE:
        setproctitle.setproctitle(process_name)
        print(f"Process name set to: {process_name}")
    elif process_name:
        print(f"Process name requested ({process_name}) but setproctitle not available")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    
    # Automatically switch to EBM version if requested
    if args.use_ebm_composition and args.model_type == 'bimanual_composer':
        print("EBM composition enabled - using EBMBimanualComposer")
        model_class = fetch_model_class('ebm_bimanual_composer')
    else:
        model_class = fetch_model_class(args.model_type)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_tester.main()
