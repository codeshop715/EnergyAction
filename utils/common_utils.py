import argparse


def str_none(value):
    if value.lower() in ['none', 'null', 'nil'] or len(value) == 0:
        return None
    else:
        return value


def str2bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def count_parameters(model):
    """
    Count trainable parameters in a model, properly handling shared parameters.
    Uses parameter IDs to avoid double-counting shared weights.
    """
    # Use a set to track unique parameter IDs (handles shared parameters correctly)
    unique_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_id = id(param)
            if param_id not in unique_params:
                unique_params[param_id] = (name, param.numel())
    
    total_trainable = sum(numel for _, numel in unique_params.values())
    print(f"Trainable model parameters: {total_trainable:,}")
    
    # Print number of trainable parameters for main modules (also with deduplication)
    for module_name, submodule in model.named_modules():
        if '.' not in module_name and module_name:  # Top-level modules only, skip empty name
            unique_submodule_params = {}
            for name, param in submodule.named_parameters():
                if param.requires_grad:
                    param_id = id(param)
                    if param_id not in unique_submodule_params:
                        unique_submodule_params[param_id] = param.numel()
            
            submodule_params = sum(unique_submodule_params.values())
            if submodule_params > 0:
                print(f"  {module_name} - trainable params: {submodule_params:,}")
    
    # Additional info: detect shared parameters (summary only)
    param_id_to_modules = {}
    for module_name, submodule in model.named_modules():
        if '.' not in module_name and module_name:  # Top-level modules only, skip empty name
            for param_name, param in submodule.named_parameters():
                if param.requires_grad:
                    param_id = id(param)
                    if param_id not in param_id_to_modules:
                        param_id_to_modules[param_id] = []
                    param_id_to_modules[param_id].append(module_name)
    
    # Find shared parameters
    shared_params = {pid: modules for pid, modules in param_id_to_modules.items() if len(modules) > 1}
    if shared_params:
        total_shared_params = sum(unique_params[pid][1] for pid in shared_params if pid in unique_params)
        num_shared_groups = len(shared_params)
        print(f"\nParameter sharing detected:")
        print(f"  Total shared parameters: {total_shared_params:,} ({num_shared_groups} parameter tensors shared across modules)")
    else:
        print("\nNo parameter sharing detected (all weights are independent)")
