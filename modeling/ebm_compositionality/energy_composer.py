"""
Energy Composer

Implements the core energy-based composition logic for bimanual manipulation.
Combines individual arm energies into joint energy: E_joint = E_left + E_right
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple, Optional
from .flow_to_energy import FlowToEnergyConverter


class EnergyComposer(nn.Module):
    """
    Composes individual arm energies into joint bimanual energy.
    
    The joint energy is computed as:
    E_joint(left_action, right_action | observation) = E_left(left_action | observation) + E_right(right_action | observation)
    
    This enables the model to reason about bimanual coordination through energy minimization.
    """
    
    def __init__(self, left_flow_actor, right_flow_actor, 
                 freeze_flow_weights: bool = True,
                 energy_scale: float = 1.0,
                 # Coordination constraint parameters
                 enable_coordination_constraints: bool = True,
                 coord_constraint_weight: float = 1.0,
                 coord_weight_hidden_dim: int = 64,
                 min_ee_distance: float = 0.15,
                 min_joint_distance: float = 0.12):
        """
        Initialize the energy composer.
        
        Args:
            left_flow_actor: Pre-trained DenoiseActor for left arm
            right_flow_actor: Pre-trained DenoiseActor for right arm  
            freeze_flow_weights: Whether to freeze flow model weights
            energy_scale: Scaling factor for energy values
            enable_coordination_constraints: Whether to enable bimanual coordination constraints
            coord_constraint_weight: Weight for coordination constraint energy
            coord_weight_hidden_dim: Hidden dimension for constraint weight predictor
            min_ee_distance: Minimum safe distance between end-effectors (meters)
            min_joint_distance: Minimum safe distance between joints (meters)
        """
        super().__init__()
        
        # Create energy converters for each arm
        # Convert boolean freeze_flow_weights to training_mode string
        training_mode = "frozen" if freeze_flow_weights else "full"
        
        self.left_energy_converter = FlowToEnergyConverter(
            left_flow_actor, training_mode=training_mode
        )
        self.right_energy_converter = FlowToEnergyConverter(
            right_flow_actor, training_mode=training_mode
        )
        
        # Learnable energy scaling parameter in log space
        # Using log space ensures energy_scale is always positive and provides better gradients
        # Initialize to log(10.0) to increase energy signal strength
        # self.log_energy_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Fixed bias (not learnable) to prevent negative loss values
        self.energy_bias = 0.0
        
        # Store references to the original actors
        self.left_actor = left_flow_actor
        self.right_actor = right_flow_actor
        
        # Initialize coordination constraints
        self.enable_coordination_constraints = enable_coordination_constraints
        self.coord_constraint_weight = coord_constraint_weight
        
        if enable_coordination_constraints:
            from .bimanual_coordination_constraints import BimanualCoordinationConstraints
            
            self.coord_constraints = BimanualCoordinationConstraints(
                action_dim=8,
                min_ee_distance=min_ee_distance,
                min_joint_distance=min_joint_distance,
                weight_hidden_dim=coord_weight_hidden_dim
            )
            print(f"✅ Coordination constraints enabled (weight={coord_constraint_weight}, "
                  f"min_ee_dist={min_ee_distance}m, min_joint_dist={min_joint_distance}m)")
        else:
            self.coord_constraints = None
            print("❌ Coordination constraints disabled")
    
    def compute_individual_energies(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute individual arm energies.
        
        Args:
            batch: Bimanual batch data containing:
                - left_action: Left arm actions (B, T, 1, action_dim)
                - right_action: Right arm actions (B, T, 1, action_dim)
                - left_proprioception: Left arm proprioception
                - right_proprioception: Right arm proprioception
                - Shared observations (rgb, pcd, instr, etc.)
        
        Returns:
            Tuple of (left_energy, right_energy), each of shape (B,)
        """
        # Prepare left arm batch
        # Add nhand dimension: (B, T, 8) -> (B, T, 1, 8)
        left_batch = {
            "gt_trajectory": batch["left_action"].unsqueeze(2),
            "proprio": batch["left_proprioception"].unsqueeze(2), 
            "rgb3d": batch.get("rgb"),
            "pcd": batch.get("pcd"),
            "rgb2d": batch.get("rgb2d"),
            "instruction": batch.get("instr"),
        }
        
        # Prepare right arm batch
        # Add nhand dimension: (B, T, 8) -> (B, T, 1, 8)  
        right_batch = {
            "gt_trajectory": batch["right_action"].unsqueeze(2),
            "proprio": batch["right_proprioception"].unsqueeze(2),
            "rgb3d": batch.get("rgb"),
            "pcd": batch.get("pcd"), 
            "rgb2d": batch.get("rgb2d"),
            "instruction": batch.get("instr"),
        }
        
        # Compute individual energies
        left_energy = self.left_energy_converter.compute_energy(left_batch)
        right_energy = self.right_energy_converter.compute_energy(right_batch)
        
        return left_energy, right_energy
    
    def compute_joint_energy(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute joint bimanual energy.
        
        The joint energy is computed as the sum of individual arm energies
        plus optional coordination constraints.
        
        Args:
            batch: Bimanual batch data
            
        Returns:
            Joint energy tensor (scalar for batch)
        """
        left_energy, right_energy = self.compute_individual_energies(batch)
        
        # Simple additive composition - sum of Flow Matching losses
        joint_energy = left_energy + right_energy
        
        # Add coordination constraint energy
        if self.coord_constraints is not None:
            # Extract actions - they are already (B, T, 8) without nhand dimension
            left_action = batch["left_action"]   # (B, T, 8)
            right_action = batch["right_action"]  # (B, T, 8)
            
            # Compute coordination constraints
            constraints = self.coord_constraints(left_action, right_action)
            
            # Total constraint energy (average over time dimension)
            coord_energy = constraints['total'].mean(dim=1)  # (B,)
            
            # Add weighted constraint energy to joint energy
            joint_energy = joint_energy + self.coord_constraint_weight * coord_energy
        
        # Apply learnable scaling (bias is fixed at 0.0 to prevent negative loss)
        # Use exp mapping from log space to ensure energy_scale is always positive
        # Clamp to reasonable range [0.1, 100.0] for numerical stability

        # energy_scale = torch.exp(self.log_energy_scale).clamp(min=0.1, max=100.0)
        # joint_energy = energy_scale * joint_energy

        # joint_energy = 0.5 * joint_energy
        # print(joint_energy)
        
        return joint_energy
    
    def compute_energy(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute energy for a batch of data.
        
        Args:
            batch_data: Dictionary containing:
                - gt_trajectory: Ground truth trajectory [B, T, 2, action_dim]
                - Other observation data
                
        Returns:
            Energy values [B]
        """
        # Extract trajectories for each arm
        gt_trajectory = batch_data['gt_trajectory']  # [B, T, 2, action_dim]
        left_actions = gt_trajectory[:, :, 0, :]   # [B, T, action_dim]
        right_actions = gt_trajectory[:, :, 1, :]  # [B, T, action_dim]
        
        # Prepare batch in the format expected by compute_joint_energy
        # Map field names to match what compute_individual_energies expects
        # Handle rgb3d with fallback to rgb
        rgb_data = batch_data.get('rgb3d')
        if rgb_data is None:
            rgb_data = batch_data.get('rgb')
        
        # Handle instruction with fallback to instr
        instruction_data = batch_data.get('instruction')
        if instruction_data is None:
            instruction_data = batch_data.get('instr')
        
        left_proprio = batch_data.get('left_proprioception')
        right_proprio = batch_data.get('right_proprioception')
        
        if left_proprio is None or right_proprio is None:
            proprio = batch_data.get('proprio')
            if proprio is not None:
                # proprio shape: (B, nhist, 2, 8)
                left_proprio = proprio[:, :, 0, :]   # (B, nhist, 8)
                right_proprio = proprio[:, :, 1, :]  # (B, nhist, 8)
        
        formatted_batch = {
            'left_action': left_actions,
            'right_action': right_actions,
            'left_proprioception': left_proprio,
            'right_proprioception': right_proprio,
            'rgb': rgb_data,
            'rgb3d': rgb_data,
            'rgb2d': batch_data.get('rgb2d'),
            'pcd': batch_data.get('pcd'),
            'instr': instruction_data,
        }
        
        # Compute joint energy
        energy = self.compute_joint_energy(formatted_batch)
        
        return energy
    
    def sample_individual_actions(self, observation: Dict[str, Any], 
                                 early_stop_callback=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from individual flow models (baseline sampling).
        
        Args:
            observation: Observation data with arm-specific proprioception
            early_stop_callback: Optional callback for early stopping during denoising
            
        Returns:
            Tuple of (left_action, right_action)
        """
        # Prepare parameters for each arm (similar to original BimanualComposer)
        # Handle trajectory_mask: use it if provided, otherwise use action_mask
        trajectory_mask = observation.get('trajectory_mask')
        if trajectory_mask is None:
            trajectory_mask = observation.get('action_mask')
            if trajectory_mask is None:
                # Generate default mask based on action shape or visual input
                left_action = observation.get('left_action')
                if left_action is not None:
                    # Add nhand dimension: from (B, T) to (B, T, 1)
                    trajectory_mask = torch.ones(left_action.shape[:-1] + (1,), dtype=torch.bool, device=left_action.device)
                else:
                    # Fallback: use visual input to determine batch size and device
                    rgb_input = observation.get('rgb3d')
                    if rgb_input is None:
                        rgb_input = observation.get('rgb')
                    if rgb_input is not None:
                        batch_size = rgb_input.shape[0]
                        device = rgb_input.device
                        # Default trajectory mask: (B, T, 1) for single timestep, single hand
                        # For single-arm flow models: (B, chunk_size, 1)
                        chunk_size = 1  # Default chunk size from args
                        trajectory_mask = torch.ones((batch_size, chunk_size, 1), dtype=torch.bool, device=device)
                    else:
                        raise ValueError("Either trajectory_mask, action_mask, left_action, or visual input must be provided")
        
        # For single-arm models, we need trajectory_mask with shape (B, T, 1) not (B, T, 2)
        # If trajectory_mask has nhand dimension > 1, we need to select or reshape it
        if trajectory_mask.shape[-1] > 1:
            # trajectory_mask is (B, T, 2) for bimanual tasks
            # For single-arm models, create a mask with nhand=1 by taking the max/any across hands
            # This ensures we process all timesteps that have any active hand
            single_arm_mask = trajectory_mask.any(dim=-1, keepdim=True)  # (B, T, 1)
        else:
            # trajectory_mask is already (B, T, 1)
            single_arm_mask = trajectory_mask
        
        # Get rgb3d with fallback to rgb
        rgb3d_data = observation.get('rgb3d')
        if rgb3d_data is None:
            rgb3d_data = observation.get('rgb')
        
        # Get instruction with fallback to instr
        instruction_data = observation.get('instruction')
        if instruction_data is None:
            instruction_data = observation.get('instr')
        
        left_params = {
            'trajectory_mask': single_arm_mask,
            'rgb3d': rgb3d_data,
            'rgb2d': observation.get('rgb2d'),
            'pcd': observation.get('pcd'),
            'instruction': instruction_data,
            'proprio': observation.get('left_proprioception')
        }
        
        right_params = {
            'trajectory_mask': single_arm_mask,
            'rgb3d': rgb3d_data,
            'rgb2d': observation.get('rgb2d'),
            'pcd': observation.get('pcd'),
            'instruction': instruction_data,
            'proprio': observation.get('right_proprioception')
        }
        
        # Sample from individual flow models with arm-specific callbacks
        left_callback = early_stop_callback.get('left') if isinstance(early_stop_callback, dict) else early_stop_callback
        right_callback = early_stop_callback.get('right') if isinstance(early_stop_callback, dict) else early_stop_callback
        
        left_action = self.left_energy_converter.sample_from_flow(left_params, early_stop_callback=left_callback)
        right_action = self.right_energy_converter.sample_from_flow(right_params, early_stop_callback=right_callback)
        
        return left_action, right_action
    
    def enable_training(self):
        """Enable training mode for energy components."""
        self.train()
        self.left_energy_converter.enable_training()
        self.right_energy_converter.enable_training()
    
    def enable_inference(self):
        """Enable inference mode (freeze flow models)."""
        self.eval()
        self.left_energy_converter.enable_inference() 
        self.right_energy_converter.enable_inference()
    
    def get_flow_actors(self) -> Tuple:
        """Get the underlying flow actors."""
        return self.left_actor, self.right_actor
