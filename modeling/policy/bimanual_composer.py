import torch
import torch.nn as nn
import os

from .denoise_actor_3d import DenoiseActor
from typing import Dict, Union, Optional


class BimanualComposer(nn.Module):
    """
    A composer model that manages two separate single-arm DenoiseActor models
    for bimanual robot control.

    This model acts as a wrapper. It does not have its own forward pass for
    action generation but orchestrates the two sub-models.
    Supports weight sharing where encoder and shared attention layers are shared
    between left and right arms, but action prediction heads are separate.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the BimanualComposer.
        It creates two instances of the DenoiseActor model, one for each arm.
        All arguments are passed down to the DenoiseActor constructors.
        """
        super().__init__()
        # Register the workspace normalizer as a buffer.
        # This makes it part of the model's state_dict but not a trainable parameter.
        # We initialize it with a placeholder tensor.
        self.register_buffer('workspace_normalizer', torch.zeros(2, 3), persistent=True)
        self.model_args = args
        self.model_kwargs = kwargs
        self.weight_sharing = kwargs.get('weight_sharing', False)
        self.left_actor = None
        self.right_actor = None

    def set_workspace_normalizer(self, normalizer: torch.Tensor):
        """
        Sets the workspace normalizer for the composer and its sub-models.
        """
        # Assign the new normalizer to the buffer
        self.workspace_normalizer.copy_(normalizer)
        
        # Propagate to sub-models if they exist
        if self.left_actor:
            self.left_actor.workspace_normalizer = self.workspace_normalizer
        if self.right_actor:
            self.right_actor.workspace_normalizer = self.workspace_normalizer

    def load_pretrained_weights(self, left_path=None, right_path=None):
        # Initialize actors with default config if no pretrained models
        if not left_path and not right_path:
            print("No pretrained models specified. Creating new actors from scratch.")
            # Use the provided model kwargs with single-arm configuration
            left_config = self.model_kwargs.copy()
            left_config['nhand'] = 1  # Force single-arm for each actor
            right_config = self.model_kwargs.copy() 
            right_config['nhand'] = 1  # Force single-arm for each actor
            
            self.left_actor = DenoiseActor(*self.model_args, **left_config)
            # print(f"DEBUG: Left actor nhand = {getattr(self.left_actor, 'nhand', 'UNKNOWN')}")
            if self.weight_sharing:
                print("Weight sharing enabled: creating right actor with shared components.")
                self.right_actor = self._create_weight_shared_actor(self.left_actor, right_config)
            else:
                self.right_actor = DenoiseActor(*self.model_args, **right_config)
            # print(f"DEBUG: Right actor nhand = {getattr(self.right_actor, 'nhand', 'UNKNOWN')}")
            
            print("New left and right actors created successfully.")
            return
            
        if left_path:
            print(f"Loading left arm from {left_path}")
            left_checkpoint = torch.load(left_path, map_location='cpu')

            # Infer model config from checkpoint
            if 'args' in left_checkpoint:
                args = left_checkpoint['args']
                config = {
                    'embedding_dim': getattr(args, 'embedding_dim', 120),
                    'num_attn_heads': getattr(args, 'num_attn_heads', 8),
                    'nhist': getattr(args, 'num_history', 3),
                    'nhand': 1,  # Force single-arm for composer
                    'num_shared_attn_layers': getattr(args, 'num_shared_attn_layers', 4),
                    'denoise_timesteps': getattr(args, 'denoise_timesteps', 10),
                    'denoise_model': getattr(args, 'denoise_model', 'rectified_flow'),
                    'rotation_format': getattr(args, 'rotation_format', 'quat_xyzw'),
                    'num_vis_instr_attn_layers': getattr(args, 'num_vis_instr_attn_layers', 2),
                    'relative': getattr(args, 'relative_action', False),
                    'backbone': getattr(args, 'backbone', 'clip'),
                    'fps_subsampling_factor': getattr(args, 'fps_subsampling_factor', 5)
                }
            else:
                # Fallback to init args if no config in checkpoint
                config = self.model_kwargs.copy()
                config['nhand'] = 1  # Force single-arm for composer
            
            self.left_actor = DenoiseActor(*self.model_args, **config)
            
            # Handling both DataParallel and single model checkpoints
            if 'model' in left_checkpoint:
                left_state_dict = left_checkpoint['model']
            elif 'weight' in left_checkpoint:
                left_state_dict = left_checkpoint['weight']
            else:
                left_state_dict = left_checkpoint

            # Adjust keys if they are prefixed with 'module.'
            new_left_state_dict = {k.replace('module.', ''): v for k, v in left_state_dict.items()}
            
            missing_keys, unexpected_keys = self.left_actor.load_state_dict(new_left_state_dict, strict=False)
            print(f"Left arm missing keys: {missing_keys}")
            print(f"Left arm unexpected keys: {unexpected_keys}")
            print("Left arm checkpoint loaded successfully.")
            
        if right_path:
            print(f"Loading right arm from {right_path}")
            right_checkpoint = torch.load(right_path, map_location='cpu')

            # Infer model config from checkpoint
            if 'args' in right_checkpoint:
                args = right_checkpoint['args']
                config = {
                    'embedding_dim': getattr(args, 'embedding_dim', 120),
                    'num_attn_heads': getattr(args, 'num_attn_heads', 8),
                    'nhist': getattr(args, 'num_history', 3),
                    'nhand': 1,  # Force single-arm for composer
                    'num_shared_attn_layers': getattr(args, 'num_shared_attn_layers', 4),
                    'denoise_timesteps': getattr(args, 'denoise_timesteps', 10),
                    'denoise_model': getattr(args, 'denoise_model', 'rectified_flow'),
                    'rotation_format': getattr(args, 'rotation_format', 'quat_xyzw'),
                    'num_vis_instr_attn_layers': getattr(args, 'num_vis_instr_attn_layers', 2),
                    'relative': getattr(args, 'relative_action', False),
                    'backbone': getattr(args, 'backbone', 'clip'),
                    'fps_subsampling_factor': getattr(args, 'fps_subsampling_factor', 5)
                }
            else:
                # Fallback to init args if no config in checkpoint
                config = self.model_kwargs.copy()
                config['nhand'] = 1  # Force single-arm for composer

            self.right_actor = DenoiseActor(*self.model_args, **config)

            if 'model' in right_checkpoint:
                right_state_dict = right_checkpoint['model']
            elif 'weight' in right_checkpoint:
                right_state_dict = right_checkpoint['weight']
            else:
                right_state_dict = right_checkpoint
            
            new_right_state_dict = {k.replace('module.', ''): v for k, v in right_state_dict.items()}

            missing_keys, unexpected_keys = self.right_actor.load_state_dict(new_right_state_dict, strict=False)
            print(f"Right arm missing keys: {missing_keys}")
            print(f"Right arm unexpected keys: {unexpected_keys}")
            print("Right arm checkpoint loaded successfully.")
        
        # If only one pretrained model is provided, create the other actor from scratch
        if left_path and not right_path:
            print("Creating right actor from scratch (no pretrained model provided)")
            right_config = self.model_kwargs.copy()
            right_config['nhand'] = 1
            if self.weight_sharing:
                print("Weight sharing enabled: creating right actor with shared components from left actor.")
                self.right_actor = self._create_weight_shared_actor(self.left_actor, right_config)
            else:
                self.right_actor = DenoiseActor(*self.model_args, **right_config)
            # print(f"DEBUG: Right actor nhand = {getattr(self.right_actor, 'nhand', 'UNKNOWN')}")
            print("New right actor created successfully.")
            
        if right_path and not left_path:
            print("Creating left actor from scratch (no pretrained model provided)")
            left_config = self.model_kwargs.copy()
            left_config['nhand'] = 1
            if self.weight_sharing:
                print("Weight sharing enabled: creating left actor with shared components from right actor.")
                self.left_actor = self._create_weight_shared_actor(self.right_actor, left_config)
            else:
                self.left_actor = DenoiseActor(*self.model_args, **left_config)
            # print(f"DEBUG: Left actor nhand = {getattr(self.left_actor, 'nhand', 'UNKNOWN')}")
            print("New left actor created successfully.")

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Computes the total loss for a batch of bimanual data.

        The method calculates the loss for each arm independently and then
        sums them to get the final loss.

        Args:
            batch: A dictionary containing the training data. It is expected
                   to have separate keys for left and right arm data, e.g.,
                   'left_action', 'right_action', 'left_proprioception', etc.
                   Shared data like vision inputs ('rgb', 'pcd') and language
                   instructions ('instr') are used for both arms.

        Returns:
            A tensor representing the total combined loss for both arms.
        """
        if not self.left_actor or not self.right_actor:
            raise RuntimeError("Models are not loaded. Call load_pretrained_weights first.")

        # --- Left Arm Loss ---
        # Create a batch for the left arm
        left_batch = {
            "gt_trajectory": batch["left_action"],
            "proprio": batch["left_proprioception"],
            "rgb3d": batch.get("rgb"),
            "pcd": batch.get("pcd"),
            "rgb2d": batch.get("rgb2d"),
            "instruction": batch.get("instr"),
            # Trajectory mask is created internally by the model
            "trajectory_mask": None,
        }
        # Pop trajectory_mask as it's not an argument for DenoiseActor's compute_loss
        left_batch.pop('trajectory_mask', None)

        loss_left = self.left_actor.compute_loss(**left_batch)

        # --- Right Arm Loss ---
        # Create a batch for the right arm
        right_batch = {
            "gt_trajectory": batch["right_action"],
            "proprio": batch["right_proprioception"],
            "rgb3d": batch.get("rgb"),
            "pcd": batch.get("pcd"),
            "rgb2d": batch.get("rgb2d"),
            "instruction": batch.get("instr"),
            "trajectory_mask": None,
        }
        # Pop trajectory_mask as it's not an argument for DenoiseActor's compute_loss
        right_batch.pop('trajectory_mask', None)

        loss_right = self.right_actor.compute_loss(**right_batch)

        # Total loss is the sum of individual arm losses
        total_loss = loss_left + loss_right
        
        # Print all losses in one line
        # print(f"[BIMANUAL] Left: {loss_left.item():.6f}, Right: {loss_right.item():.6f}, Total: {total_loss.item():.6f}")
        return total_loss

    def compute_trajectory(self, **kwargs):
        """
        Generates a bimanual trajectory.

        It calls the compute_trajectory method of each single-arm model
        and combines the resulting actions into a single bimanual action.

        Args:
            **kwargs: A dictionary of observation data. It is expected to
                      contain shared vision/language data and separate
                      proprioception for each arm ('left_proprioception',
                      'right_proprioception').
                      Can also include 'early_stop_callback' for adaptive denoising.

        Returns:
            A tensor representing the combined bimanual action trajectory,
            with a shape of (B, T, 2, 8).
        """
        if not self.left_actor or not self.right_actor:
            raise RuntimeError("Models are not loaded. Call load_pretrained_weights first.")
        
        # Extract early stop callback if provided
        early_stop_callback = kwargs.pop('early_stop_callback', None)
            
        # --- Left Arm Action ---
        # DenoiseActor.compute_trajectory only accepts specific parameters
        # Handle trajectory_mask: use it if provided, otherwise use action_mask
        trajectory_mask = kwargs.get('trajectory_mask')
        if trajectory_mask is None:
            trajectory_mask = kwargs.get('action_mask')
            if trajectory_mask is None:
                # Generate default mask based on action shape
                left_action = kwargs.get('left_action')
                if left_action is not None:
                    # Validate left_action shape to prevent common errors
                    if len(left_action.shape) < 2:
                        raise ValueError(f"left_action must have at least 2 dimensions (B, T, ...), got shape {left_action.shape}")
                    elif len(left_action.shape) == 2:
                        # Assume (B*T, action_dim) and reshape to (B, T, action_dim)
                        print(f"WARNING: left_action has 2D shape {left_action.shape}, assuming batch_size=1 and reshaping")
                        batch_size = 1
                        seq_len = left_action.shape[0]
                        action_dim = left_action.shape[1]
                        left_action = left_action.view(batch_size, seq_len, action_dim)
                        print(f"Reshaped left_action to {left_action.shape}")
                    elif len(left_action.shape) == 3:
                        # Expected shape: (B, T, action_dim)
                        pass
                    elif len(left_action.shape) > 3:
                        raise ValueError(f"left_action has too many dimensions {left_action.shape}. Expected (B, T, action_dim)")
                    
                    # Add nhand dimension: from (B, T, action_dim) to (B, T, 1)
                    trajectory_mask = torch.ones(left_action.shape[:-1] + (1,), dtype=torch.bool, device=left_action.device)
                    # print(f"DEBUG: Created trajectory_mask with shape {trajectory_mask.shape} from left_action shape {left_action.shape}")
                else:
                    raise ValueError("Either trajectory_mask, action_mask, or left_action must be provided")
        
        # Create single-arm trajectory masks for each actor
        # If trajectory_mask has nhand=2, we need to reduce it to nhand=1 for each actor
        if trajectory_mask.shape[-1] == 2:
            # Take only the first arm's mask for both actors (assuming symmetric mask)
            single_arm_mask = trajectory_mask[..., :1]  # (B, T, 1)
        else:
            # Already single arm mask
            single_arm_mask = trajectory_mask
        
        left_params = {
            'trajectory_mask': single_arm_mask,
            'rgb3d': kwargs.get('rgb3d') if kwargs.get('rgb3d') is not None else kwargs.get('rgb'),
            'rgb2d': kwargs.get('rgb2d'),
            'pcd': kwargs.get('pcd'),
            'instruction': kwargs.get('instruction') if kwargs.get('instruction') is not None else kwargs.get('instr'),
            'proprio': kwargs.get('left_proprioception')
        }
        
        # Get left callback if provided
        left_callback = None
        if early_stop_callback is not None:
            if isinstance(early_stop_callback, dict):
                left_callback = early_stop_callback.get('left')
            else:
                left_callback = early_stop_callback
        
        action_left = self.left_actor.compute_trajectory(**left_params, early_stop_callback=left_callback)

        # --- Right Arm Action ---
        # DenoiseActor.compute_trajectory only accepts specific parameters
        right_params = {
            'trajectory_mask': single_arm_mask,  # Use the single-arm mask
            'rgb3d': kwargs.get('rgb3d') if kwargs.get('rgb3d') is not None else kwargs.get('rgb'),
            'rgb2d': kwargs.get('rgb2d'),
            'pcd': kwargs.get('pcd'),
            'instruction': kwargs.get('instruction') if kwargs.get('instruction') is not None else kwargs.get('instr'),
            'proprio': kwargs.get('right_proprioception')
        }
        
        # Get right callback if provided
        right_callback = None
        if early_stop_callback is not None:
            if isinstance(early_stop_callback, dict):
                right_callback = early_stop_callback.get('right')
            else:
                right_callback = early_stop_callback

        action_right = self.right_actor.compute_trajectory(**right_params, early_stop_callback=right_callback)

        # --- Combine Actions ---
        # Stack the actions along a new dimension to create the bimanual action
        # The shape of each action is (B, T, 1, 8), stacking results in (B, T, 2, 8)
        bimanual_action = torch.cat([action_left, action_right], dim=2)
        
        return bimanual_action

    def forward(self, **kwargs):
        """
        The forward method defaults to trajectory generation.
        """
        return self.compute_trajectory(**kwargs)
    
    def _create_weight_shared_actor(self, source_actor, target_config):
        """
        Creates a new actor that shares weights with the source actor.
        Only the encoder will be shared between arms, while the trajectory encoder
        and all prediction head components will be separate for each arm.
        """
        target_actor = DenoiseActor(*self.model_args, **target_config)
        
        # Share only the encoder (visual-language encoder)
        target_actor.encoder = source_actor.encoder
        
        # Keep separate: trajectory encoder and entire prediction head
        # (trajectory encoder and prediction head remain as initialized in target_actor)
        
        print("Weight sharing setup complete: only encoder is shared, trajectory encoder and prediction heads are separate.")
        return target_actor

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that automatically creates actors when loading weights.
        """
        # Check if state_dict contains left_actor or right_actor weights
        has_left_actor = any(key.startswith('left_actor.') for key in state_dict.keys())
        has_right_actor = any(key.startswith('right_actor.') for key in state_dict.keys())
        
        # Initialize actors if they don't exist and weights are present
        if (has_left_actor or has_right_actor) and (self.left_actor is None or self.right_actor is None):
            print("Detected actor weights in state_dict. Initializing actors...")
            
            # Use the provided model kwargs with single-arm configuration
            left_config = self.model_kwargs.copy()
            left_config['nhand'] = 1  # Force single-arm for each actor
            right_config = self.model_kwargs.copy() 
            right_config['nhand'] = 1  # Force single-arm for each actor
            
            self.left_actor = DenoiseActor(*self.model_args, **left_config)
            # print(f"DEBUG: Created left actor nhand = {getattr(self.left_actor, 'nhand', 'UNKNOWN')}")
            if self.weight_sharing:
                print("Weight sharing enabled: creating right actor with shared components.")
                self.right_actor = self._create_weight_shared_actor(self.left_actor, right_config)
            else:
                self.right_actor = DenoiseActor(*self.model_args, **right_config)
            # print(f"DEBUG: Created right actor nhand = {getattr(self.right_actor, 'nhand', 'UNKNOWN')}")
            
            print("Actors created for state_dict loading.")
        
        # Call the parent load_state_dict
        return super().load_state_dict(state_dict, strict=strict)
