"""
EBM Bimanual Composer

Main interface that replaces BimanualComposer with energy-based implementation
while maintaining complete interface compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os

# Import the original components
from ..policy.bimanual_composer import BimanualComposer
from ..policy.denoise_actor_3d import DenoiseActor

# Import EBM components
from .energy_composer import EnergyComposer
# Note: EnergyGuidedSampler removed - not used in current implementation


class EBMBimanualComposer(BimanualComposer):
    """
    Energy-Based Model version of BimanualComposer.
    
    Maintains complete interface compatibility with the original BimanualComposer
    while internally using energy-based optimization for bimanual coordination.
    
    Key features:
    - compute_trajectory() uses energy-guided sampling with adaptive denoising
    - Internal composition through joint energy minimization
    - Optional early stopping based on energy thresholds
    """
    
    def __init__(self, *args, 
                 freeze_flow_weights: bool = True,
                 use_performance_guarantee: bool = True,
                 enable_energy_in_eval: bool = False,
                 enable_adaptive_denoising: bool = True,
                 min_denoise_steps: int = 5,
                 max_denoise_steps: int = 10,
                 energy_threshold_low: float = 1.0,
                 energy_threshold_high: float = 10.0,
                 enable_early_stopping: bool = True,
                 early_stop_check_interval: int = 2,
                 enable_coordination_constraints: bool = True,
                 coord_constraint_weight: float = 1.0,
                 coord_weight_hidden_dim: int = 64,
                 min_ee_distance: float = 0.15,
                 min_joint_distance: float = 0.12,
                 **kwargs):
        """
        Initialize EBM Bimanual Composer.
        
        Args:
            *args, **kwargs: Same arguments as original BimanualComposer
            freeze_flow_weights: Whether to freeze pre-trained flow weights
            use_performance_guarantee: Whether to guarantee performance >= flow baseline
        """
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # EBM-specific parameters
        self.freeze_flow_weights = freeze_flow_weights
        self.use_performance_guarantee = use_performance_guarantee
        self.enable_energy_in_eval = enable_energy_in_eval
        
        self.enable_adaptive_denoising = enable_adaptive_denoising
        self.min_denoise_steps = min_denoise_steps
        self.max_denoise_steps = max_denoise_steps
        self.energy_threshold_low = energy_threshold_low
        self.energy_threshold_high = energy_threshold_high
        self.enable_early_stopping = enable_early_stopping
        self.early_stop_check_interval = early_stop_check_interval
        
        self.enable_coordination_constraints = enable_coordination_constraints
        self.coord_constraint_weight = coord_constraint_weight
        self.coord_weight_hidden_dim = coord_weight_hidden_dim
        self.min_ee_distance = min_ee_distance
        self.min_joint_distance = min_joint_distance
        
        self.original_denoise_steps = None
        
        # EBM components (initialized after loading pretrained weights)
        self.energy_composer = None
        # Note: energy_sampler removed - not used in current implementation
        
        # Training mode flag
        self._ebm_training_mode = False
    
    def load_pretrained_weights(self, left_path=None, right_path=None):
        """
        Load pretrained weights and initialize EBM components.
        Maintains the same interface as original BimanualComposer.
        """
        # Call parent implementation to load flow models
        super().load_pretrained_weights(left_path, right_path)
        
        # Initialize EBM components after flow models are loaded
        self._initialize_ebm_components()
        
        print("EBM components initialized successfully.")
    
    def _initialize_ebm_components(self):
        """Initialize energy composer."""
        if self.left_actor is None or self.right_actor is None:
            raise RuntimeError("Flow actors must be loaded before initializing EBM components")
        
        # Create energy composer with coordination constraints
        self.energy_composer = EnergyComposer(
            left_flow_actor=self.left_actor,
            right_flow_actor=self.right_actor,
            freeze_flow_weights=self.freeze_flow_weights,
            enable_coordination_constraints=self.enable_coordination_constraints,
            coord_constraint_weight=self.coord_constraint_weight,
            coord_weight_hidden_dim=self.coord_weight_hidden_dim,
            min_ee_distance=self.min_ee_distance,
            min_joint_distance=self.min_joint_distance
        )
        
        # Note: EnergyGuidedSampler initialization removed - not used in current implementation
        
        self.original_denoise_steps = {
            'left': getattr(self.left_actor, 'n_steps', 10),
            'right': getattr(self.right_actor, 'n_steps', 10)
        }
        
        print(f"Original denoise steps: left={self.original_denoise_steps['left']}, right={self.original_denoise_steps['right']}")
        print(f"Adaptive denoising enabled: {self.enable_adaptive_denoising}")
        if self.enable_adaptive_denoising:
            print(f"Denoise steps range: {self.min_denoise_steps} - {self.max_denoise_steps}")
            print(f"Energy thresholds: low={self.energy_threshold_low}, high={self.energy_threshold_high}")
            print(f"Early stopping enabled: {self.enable_early_stopping}")
            if self.enable_early_stopping:
                print(f"Early stop check interval: every {self.early_stop_check_interval} steps")
        
        # Set appropriate modes
        if self._ebm_training_mode:
            self.energy_composer.enable_training()
        else:
            self.energy_composer.enable_inference()
    
    def compute_trajectory(self, **kwargs):
        """
        Generate trajectory using energy-guided sampling.
        
        Maintains the same interface as original BimanualComposer.compute_trajectory().
        
        Args:
            **kwargs: Same format as original - observation data
            
        Returns:
            Bimanual action tensor of shape (B, T, 2, 8) (same as original)
        """
        if not self.energy_composer:
            raise RuntimeError("EBM components not initialized. Call load_pretrained_weights first.")
        
        if not self.training and not self.enable_energy_in_eval:
            return super().compute_trajectory(**kwargs)
        
        if not self.training and self.enable_energy_in_eval and self.enable_adaptive_denoising:
            if self.enable_early_stopping:
                return self._inference_with_early_stopping(kwargs)
            else:
                adaptive_steps = self.compute_adaptive_denoise_steps(kwargs)
                
                original_left_steps = getattr(self.left_actor, 'n_steps', 10)
                original_right_steps = getattr(self.right_actor, 'n_steps', 10)
                
                self.set_denoise_steps(adaptive_steps['left'], adaptive_steps['right'])
                
                try:
                    result = super().compute_trajectory(**kwargs)
                    
                    return result
                    
                finally:
                    self.set_denoise_steps(original_left_steps, original_right_steps)
        
        return super().compute_trajectory(**kwargs)
    
    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio=None,
        left_proprioception=None,
        right_proprioception=None,
        run_inference=False
    ):
        """
        The forward method with the same signature as original BimanualComposer.
        For bimanual inference, it processes the inputs and calls compute_trajectory.
        
        Supports two calling modes:
        1. Legacy: pass proprio (B, nhist, 2, 8), will be split automatically
        2. New: pass left_proprioception and right_proprioception, each (B, nhist, 8)
        """
        if run_inference:
            if left_proprioception is None or right_proprioception is None:
                if proprio is None:
                    raise ValueError("Must provide either 'proprio' or both 'left_proprioception' and 'right_proprioception'")
                left_proprio = proprio[:, :, 0, :]
                right_proprio = proprio[:, :, 1, :]
            else:
                left_proprio = left_proprioception
                right_proprio = right_proprioception
            
            return self.compute_trajectory(
                trajectory_mask=trajectory_mask,
                rgb3d=rgb3d,
                rgb2d=rgb2d,
                pcd=pcd,
                instruction=instruction,
                left_proprioception=left_proprio,
                right_proprioception=right_proprio
            )
        else:
            raise NotImplementedError("Training mode is not available in this release.")
    
    def train(self, mode: bool = True):
        """Override train mode to handle EBM components."""
        super().train(mode)
        self._ebm_training_mode = mode
        
        if self.energy_composer:
            if mode:
                self.energy_composer.enable_training()
                # Disable inference mode during training
                if hasattr(self.energy_composer, 'set_inference_mode'):
                    self.energy_composer.set_inference_mode(False)
            else:
                self.energy_composer.enable_inference()
                # Enable inference mode for evaluation
                if hasattr(self.energy_composer, 'set_inference_mode'):
                    self.energy_composer.set_inference_mode(True)
        
        # Note: energy_sampler removed - not used in current implementation
        
        return self
    
    def eval(self):
        """Override eval mode to handle EBM components."""
        return self.train(False)
    
    
    def compute_adaptive_denoise_steps(self, observation: Dict[str, Any]) -> Dict[str, int]:
        """
        Compute adaptive denoise steps based on energy magnitude.
        
        Args:
            observation: Observation data for energy computation
            
        Returns:
            Dictionary containing adaptive denoise steps for left and right actors
        """
        if not self.enable_adaptive_denoising or not self.enable_energy_in_eval:
            return {
                'left': self.original_denoise_steps['left'],
                'right': self.original_denoise_steps['right']
            }
        
        try:
            with torch.no_grad():
                try:
                    baseline_left, baseline_right = self.energy_composer.sample_individual_actions(observation)
                    
                    if baseline_left is None or baseline_right is None:
                        raise ValueError("Failed to sample baseline actions")
                    
                    if baseline_left.dim() == 3:
                        baseline_left = baseline_left.unsqueeze(2)
                    if baseline_right.dim() == 3:
                        baseline_right = baseline_right.unsqueeze(2)
                    
                    gt_trajectory = torch.cat([baseline_left, baseline_right], dim=2)
                    
                except Exception as sampling_error:
                    print(f"⚠️ Error sampling baseline actions: {sampling_error}")
                    print("Falling back to original denoise steps.")
                    return {
                        'left': self.original_denoise_steps['left'],
                        'right': self.original_denoise_steps['right']
                    }
                
                batch_data = {
                    'gt_trajectory': gt_trajectory,
                    **{k: v for k, v in observation.items() if k != 'gt_trajectory'}
                }
                
                try:
                    energy = self.energy_composer.compute_energy(batch_data)
                    
                    if energy.numel() == 0:
                        raise ValueError("Energy tensor is empty")
                    
                    if energy.dim() > 1:
                        energy = energy.view(-1)
                    
                    if energy.numel() == 1:
                        avg_energy = energy.item()
                    else:
                        avg_energy = energy.mean().item()
                        
                except Exception as energy_error:
                    import traceback
                    print(f"⚠️ Error computing energy: {energy_error}")
                    print("Full traceback:")
                    traceback.print_exc()
                    raise energy_error
                
                if avg_energy <= self.energy_threshold_low:
                    denoise_steps = self.min_denoise_steps
                elif avg_energy >= self.energy_threshold_high:
                    denoise_steps = self.max_denoise_steps
                else:
                    energy_ratio = (avg_energy - self.energy_threshold_low) / (
                        self.energy_threshold_high - self.energy_threshold_low
                    )
                    denoise_steps = int(
                        self.min_denoise_steps + 
                        energy_ratio * (self.max_denoise_steps - self.min_denoise_steps)
                    )
                
                print(f"🔋 Energy: {avg_energy:.3f}, Adaptive denoise steps: {denoise_steps}")
                
                return {
                    'left': denoise_steps,
                    'right': denoise_steps
                }
                
        except Exception as e:
            print(f"⚠️ Error computing adaptive denoise steps: {e}")
            print("Falling back to original denoise steps")
            return {
                'left': self.original_denoise_steps['left'],
                'right': self.original_denoise_steps['right']
            }
    
    def set_denoise_steps(self, left_steps: int, right_steps: int):
        """
        Dynamically set denoise steps.
        
        Args:
            left_steps: Left arm denoise steps
            right_steps: Right arm denoise steps
        """
        if hasattr(self.left_actor, 'n_steps'):
            self.left_actor.n_steps = left_steps
        if hasattr(self.right_actor, 'n_steps'):
            self.right_actor.n_steps = right_steps
    
    def _inference_with_early_stopping(self, observation: Dict[str, Any]):
        """
        Inference with early stopping strategy.
        
        Algorithm:
        Start from 1 step and denoise incrementally. Compute energy at each step.
        Stop early when energy < energy_threshold_low.
        If all steps (1-max_denoise_steps) don't reach threshold, return last step result.
        
        Args:
            observation: Observation data
            
        Returns:
            Dict with 'left' and 'right' actions
        """
        energy_threshold = self.energy_threshold_low
        
        original_left_steps = getattr(self.left_actor, 'n_steps', 10)
        original_right_steps = getattr(self.right_actor, 'n_steps', 10)
        
        try:
            with torch.no_grad():
                for step in range(1, self.max_denoise_steps + 1):
                    self.set_denoise_steps(step, step)
                    current_result = super().compute_trajectory(**observation)
                    
                    bimanual_current = current_result
                    
                    batch_data_current = {
                        'gt_trajectory': bimanual_current,
                        **{k: v for k, v in observation.items() if k not in ['gt_trajectory', 'trajectory']}
                    }
                    
                    energy_current = self.energy_composer.compute_energy(batch_data_current)
                    if energy_current.dim() > 1:
                        energy_current = energy_current.view(-1)
                    avg_energy_current = energy_current.mean().item() if energy_current.numel() > 1 else energy_current.item()
                    
                    print(f"🔍 Step {step}/{self.max_denoise_steps}: energy={avg_energy_current:.3f}")
                    
                    if avg_energy_current < energy_threshold:
                        print(f"✅ Early stopped at step {step}/{self.max_denoise_steps}, energy: {avg_energy_current:.3f} < {energy_threshold:.3f}")
                        return current_result
                
                print(f"📊 Completed all {self.max_denoise_steps} steps, final energy: {avg_energy_current:.3f}")
                return current_result
                
        finally:
            self.set_denoise_steps(original_left_steps, original_right_steps)
    
    # def enable_performance_guarantee(self, enable: bool = True):
    #     """
    #     Enable or disable performance guarantee.
        
    #     When enabled, EBM optimization will never produce actions worse than 
    #     the baseline flow model output.
        
    #     Args:
    #         enable: Whether to enable performance guarantee
    #     """
    #     self.use_performance_guarantee = enable
    
    def get_energy_statistics(self, batch: dict) -> dict:
        """
        Get energy statistics for analysis.
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary with energy statistics
        """
        if not self.energy_composer:
            return {}
        
        with torch.no_grad():
            left_energy, right_energy = self.energy_composer.compute_individual_energies(batch)
            joint_energy = self.energy_composer.compute_joint_energy(batch)
            
            return {
                'left_energy_mean': left_energy.mean().item(),
                'left_energy_std': left_energy.std().item(),
                'right_energy_mean': right_energy.mean().item(), 
                'right_energy_std': right_energy.std().item(),
                'joint_energy_mean': joint_energy.mean().item(),
                'joint_energy_std': joint_energy.std().item(),
            }