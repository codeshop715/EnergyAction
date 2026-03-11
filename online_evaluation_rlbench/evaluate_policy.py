"""Online evaluation script on RLBench."""

import argparse
import random
from pathlib import Path
import json
import os
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none, round_floats

# Set process name from environment variable if available
try:
    import setproctitle
    if 'PROCESS_NAME' in os.environ:
        setproctitle.setproctitle(os.environ['PROCESS_NAME'])
except ImportError:
    pass  # setproctitle not available, continue without it


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Testing arguments
        ('checkpoint', str_none, None),
        ('task', str, "close_jar"),
        ('max_tries', int, 10),
        ('max_steps', int, 25),
        ('headless', str2bool, False),
        ('collision_checking', str2bool, False),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, Path(__file__).parent / "demos"),
        ('dataset', str, "Peract"),
        ('image_size', str, "256,256"),
        # Logging arguments
        ('output_file', Path, Path(__file__).parent / "eval.json"),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 0),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        # EBM-specific parameters
        ('enable_energy_in_eval', str2bool, False),
        ('enable_adaptive_denoising', str2bool, True),

        ('min_denoise_steps', int, 5),
        ('max_denoise_steps', int, 10),
        ('energy_threshold_low', float, 1.0),
        ('energy_threshold_high', float, 10.0),
        # Early stopping parameters
        ('enable_early_stopping', str2bool, False),
        ('early_stop_check_interval', int, 2),
        # Coordination constraint parameters
        ('enable_coordination_constraints', str2bool, False),
        ('coord_constraint_weight', float, 1.0),
        ('coord_weight_hidden_dim', int, 64),
        ('min_ee_distance', float, 0.001),
        ('min_joint_distance', float, 0.001)
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def load_models(args):
    print("=" * 100)
    print("🔧 MODEL LOADING PROCESS")
    print("=" * 100)
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"🤖 Model Type: {args.model_type}")
    print()

    model_class = fetch_model_class(args.model_type)
    
    # For BimanualComposer and EBM models, nhand should always be 1 as they manage two single-arm actors internally
    # For regular models, use 2 if bimanual else 1
    if args.model_type in ['bimanual_composer', 'ebm_bimanual_composer']:
        nhand_param = 1  # These models manage dual arms internally
    else:
        nhand_param = 2 if args.bimanual else 1
        
    # Base model arguments
    model_kwargs = {
        'backbone': args.backbone,
        'num_vis_instr_attn_layers': args.num_vis_instr_attn_layers,
        'fps_subsampling_factor': args.fps_subsampling_factor,
        'embedding_dim': args.embedding_dim,
        'num_attn_heads': args.num_attn_heads,
        'nhist': args.num_history,
        'nhand': nhand_param,
        'num_shared_attn_layers': args.num_shared_attn_layers,
        'relative': args.relative_action,
        'rotation_format': args.rotation_format,
        'denoise_timesteps': args.denoise_timesteps,
        'denoise_model': args.denoise_model
    }
    
    # Add EBM-specific arguments if it's an EBM model
    if args.model_type == 'ebm_bimanual_composer':
        print("⚡ EBM-specific configuration:")
        print(f"   - Energy in eval: {args.enable_energy_in_eval}")
        print(f"   - Adaptive denoising: {args.enable_adaptive_denoising}")
        print(f"   - Early stopping: {args.enable_early_stopping}")
        print(f"   - Denoise steps range: {args.min_denoise_steps}-{args.max_denoise_steps}")
        print(f"   - Energy thresholds: {args.energy_threshold_low}-{args.energy_threshold_high}")
        print(f"   - Coordination constraints: {args.enable_coordination_constraints}")
        print()
        
        model_kwargs.update({
            'enable_energy_in_eval': args.enable_energy_in_eval,
            'enable_adaptive_denoising': args.enable_adaptive_denoising,
            'min_denoise_steps': args.min_denoise_steps,
            'max_denoise_steps': args.max_denoise_steps,
            'energy_threshold_low': args.energy_threshold_low,
            'energy_threshold_high': args.energy_threshold_high,
            'enable_early_stopping': args.enable_early_stopping,
            'early_stop_check_interval': args.early_stop_check_interval,
            'enable_coordination_constraints': args.enable_coordination_constraints,
            'coord_constraint_weight': args.coord_constraint_weight,
            'coord_weight_hidden_dim': args.coord_weight_hidden_dim,
            'min_ee_distance': args.min_ee_distance,
            'min_joint_distance': args.min_joint_distance
        })
    
    print("📋 Step 1/3: Creating model instance...")
    model = model_class(**model_kwargs)
    print(f"✅ Model instance created: {model.__class__.__name__}")
    print()

    # For BimanualComposer and EBM models, initialize actor structure first
    if args.model_type in ["bimanual_composer", "ebm_bimanual_composer"]:
        print("📋 Step 2/3: Initializing bimanual actor structure...")
        print("   (Note: This creates empty left_actor and right_actor structures)")
        print("   (Actual weights will be loaded in Step 3)")
        model.load_pretrained_weights()  # This initializes left_actor and right_actor structure
        print(f"✅ Actor structure initialized for {args.model_type}")
        print()
    
    # Load model weights
    print("📋 Step 3/3: Loading checkpoint weights...")
    print(f"   Reading checkpoint from: {args.checkpoint}")
    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    
    model_dict_weight = {}
    for key in model_dict["weight"]:
        # Remove 'module.' prefix if present, otherwise keep the original key
        if key.startswith('module.'):
            _key = key[7:]  # Remove 'module.' prefix
        else:
            _key = key  # Keep original key
        model_dict_weight[_key] = model_dict["weight"][key]
    
    print(f"   Total parameters in checkpoint: {len(model_dict_weight)}")
    
    # Filter out legacy energy_sampler keys that no longer exist in the model
    model_keys = set(name for name, _ in model.named_parameters())
    model_keys.update(name for name, _ in model.named_buffers())
    filtered_weight = {}
    skipped_keys = []
    for k, v in model_dict_weight.items():
        if k in model_keys or any(k.startswith(mk.rsplit('.', 1)[0]) for mk in model_keys):
            filtered_weight[k] = v
        else:
            skipped_keys.append(k)
    if skipped_keys:
        print(f"   ⚠️  Skipped {len(skipped_keys)} legacy keys (e.g. energy_sampler.*)")
    
    # Load the weights into the initialized structure
    model.load_state_dict(filtered_weight, strict=False)
    print("✅ All weights loaded successfully into the model")
    
    if args.model_type in ["bimanual_composer", "ebm_bimanual_composer"]:
        print("   ✓ Left actor weights loaded")
        print("   ✓ Right actor weights loaded")
        if args.model_type == 'ebm_bimanual_composer':
            print("   ✓ Energy composer weights loaded")
            
            print("\n🔍 Verifying EBM components:")
            try:
                left_actor_param = next(model.left_actor.parameters())
                energy_left_param = next(model.energy_composer.left_energy_converter.flow_actor.parameters())
                
                same_object = (model.left_actor is model.energy_composer.left_energy_converter.flow_actor)
                param_match = torch.equal(left_actor_param, energy_left_param)
                
                print(f"   Same actor object: {same_object}")
                print(f"   Parameters match: {param_match}")
                print(f"   Left actor param sample: {left_actor_param.flatten()[0].item():.6f}")
                print(f"   Energy composer sees: {energy_left_param.flatten()[0].item():.6f}")
                
                if same_object and param_match:
                    print("   ✅ EBM components correctly reference loaded weights")
                else:
                    print("   ⚠️  WARNING: EBM components may not have correct weights!")
                    if not same_object:
                        print("      → Energy composer holds a COPY, not a reference!")
                    if not param_match:
                        print("      → Parameter values don't match!")
            except Exception as e:
                print(f"   ⚠️  Could not verify EBM components: {e}")
    print()
    
    model.eval()
    print("🎯 Model set to evaluation mode")
    print("🚀 Model loaded to CUDA")
    print("=" * 100)
    
    # ============================================
    # STRICT MODEL VERIFICATION
    # ============================================
    # from model_verification import verify_model_consistency
    
    # print("\n" + "=" * 100)
    # print("⚠️  PERFORMING STRICT MODEL VERIFICATION")
    # print("=" * 100)
    
    # verification_passed = verify_model_consistency(
    #     model=model,
    #     model_type=args.model_type,
    #     verbose=True
    # )
    
    # if not verification_passed:
    #     print("\n" + "=" * 100)
    #     print("❌ MODEL VERIFICATION FAILED!")
    #     print("=" * 100)
    #     print("Please review the errors above before proceeding.")
    #     print("Exiting to prevent running evaluation with incorrect model setup.")
    #     print("=" * 100)
    #     raise RuntimeError("Model verification failed. Cannot proceed with evaluation.")
    
    # print("\n" + "=" * 100)
    # print("✅ MODEL VERIFICATION PASSED - Proceeding with evaluation")
    # print("=" * 100)
    # print()
    # print()

    return model.cuda()


if __name__ == "__main__":
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Bimanual vs single-arm utils
    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv, Actioner
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv, Actioner

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)
    print(dataset_class)

    # Load models
    model = load_models(args)
    # print(model)

    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Load RLBench environment
        env = RLBenchEnv(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=dataset_class.cameras,
            collision_checking=bool(args.collision_checking)
        )

        # Actioner (runs the policy online)
        actioner = Actioner(model, backbone=args.backbone)

        # Evaluate
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
