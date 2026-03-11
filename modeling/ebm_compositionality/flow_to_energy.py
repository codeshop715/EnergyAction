"""
Flow to Energy Converter

Converts Flow Matching model outputs to energy functions for EBM-based composition.
Uses negative log-likelihood as energy for inference and evaluation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import math


class FlowToEnergyConverter:
    """
    Converts a pre-trained Flow Matching DenoiseActor into an energy function.
    
    The energy is computed as:
    E(action | observation) = flow_matching_loss(action | observation)
    
    This preserves the learned distribution while allowing energy-based composition.
    """
    
    def __init__(self, flow_actor, 
                 training_mode: str = "frozen",
                 partial_layers: Optional[list] = None,
                 energy_temperature: float = 1.0,
                 inference_mode: bool = False):
        """
        Initialize the converter.
        
        Args:
            flow_actor: Pre-trained DenoiseActor model
            training_mode: "frozen" (freeze all), "partial" (selective), "full" (unfreeze all)
            partial_layers: List of layer names to unfreeze in partial mode
            energy_temperature: Temperature scaling for energy values
            inference_mode: If True, disable gradients for pure inference
        """
        self.flow_actor = flow_actor
        self.training_mode = training_mode
        self.partial_layers = partial_layers or ["prediction_head"]  # Default: only unfreeze head
        self.energy_temperature = energy_temperature
        self.inference_mode = inference_mode
        
        # Store original requires_grad state for parameters
        self._original_requires_grad = {}
        for name, param in self.flow_actor.named_parameters():
            self._original_requires_grad[name] = param.requires_grad
            
        # Apply training mode
        self._apply_training_mode()
    
    def _apply_training_mode(self):
        """Apply the specified training mode to parameters."""
        if self.training_mode == "frozen":
            # Freeze all parameters
            for param in self.flow_actor.parameters():
                param.requires_grad = False
                
        elif self.training_mode == "partial":
            # Freeze all, then unfreeze specified layers
            for param in self.flow_actor.parameters():
                param.requires_grad = False
                
            for name, param in self.flow_actor.named_parameters():
                for layer_name in self.partial_layers:
                    if layer_name in name:
                        param.requires_grad = True
                        break
                        
        elif self.training_mode == "full":
            # Keep original requires_grad settings
            for name, param in self.flow_actor.named_parameters():
                param.requires_grad = self._original_requires_grad[name]
    
    def compute_log_likelihood(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute log-likelihood p_flow(action | observation) using Flow Matching.
        
        For Flow Matching models with L1 loss (Laplace distribution assumption):
        log p(x) = -|loss(x)| / scale - log(2 * scale)
        
        Note: This assumes the flow matching loss represents reconstruction error
        under a Laplace noise model. The temperature parameter acts as the scale.
        
        Args:
            batch_data: Dictionary containing observations and gt_trajectory
            
        Returns:
            Log-likelihood tensor of shape (B,)
        """
        # Compute flow matching loss (reconstruction error)
        # Use no_grad for inference mode or frozen parameters to save memory
        
        # use_no_grad = self.inference_mode or self.training_mode == "frozen"
        
        # if use_no_grad:
        #     with torch.no_grad():
        #         flow_loss = self.flow_actor.compute_loss(**batch_data)
        # else:
        flow_loss = self.flow_actor.compute_loss(**batch_data)
        
        # Convert flow loss to log-likelihood using Laplace distribution
        # For EBM compositionality, we NEED a proper log-likelihood:
        # p(x|c1 ∩ c2) ∝ p(x|c1) * p(x|c2)
        # => log p(x|c1 ∩ c2) = log p(x|c1) + log p(x|c2) + const
        # => E(x|c1 ∩ c2) = E(x|c1) + E(x|c2) - const
        
        scale = self.energy_temperature
        # scale = 1
        
        # Base log-likelihood from Laplace distribution
        base_log_likelihood = -torch.abs(flow_loss) / scale
        
        # Normalization constant
        normalization = torch.log(torch.tensor(2 * scale, device=flow_loss.device, dtype=flow_loss.dtype))
        # normalization = 0
        
        # Add offset to ensure energy > 0 (equivalent to log_likelihood < 0)
        # This preserves the relative ordering while ensuring positive energies
        offset = 1.0  # Minimum energy will be 1.0
        
        log_likelihood = base_log_likelihood - normalization - offset

        # res = -torch.abs(flow_loss)
        
        # Now: energy = -log_likelihood = |loss|/scale + log(2*scale) + offset ≥ offset > 0
        # return res
        
        return log_likelihood
    
    def compute_energy(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute energy E(action | observation) = -log p_flow(action | observation).
        
        This is the correct energy definition for EBM composition:
        - First compute log-likelihood: log p(x) = -|loss(x)|/scale - log(2*scale)  
        - Then compute energy: E(x) = -log p(x) = |loss(x)|/scale + log(2*scale)
        
        Args:
            batch_data: Dictionary containing:
                - gt_trajectory: Ground truth actions (B, T, nhand, action_dim)
                - Other observation data (proprio, rgb3d, pcd, etc.)
            
        Returns:
            Energy tensor of shape (B,) representing E(action | observation)
        """
        # Compute log-likelihood first (this is the mathematically correct approach)
        log_likelihood = self.compute_log_likelihood(batch_data)
        
        # Energy is negative log-likelihood: E = -log p
        energy = -log_likelihood
        
        # No clamping applied to preserve gradients for optimization
        # This allows the model to learn from arbitrarily large errors during training
        
        return energy
    
    def compute_energy_per_sample(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute per-sample energy values.
        
        For Flow Matching, the energy is equivalent to the flow matching loss.
        Since flow matching loss is typically computed per sample and then averaged,
        we can compute it directly for the batch.
        
        Args:
            batch_data: Same as compute_energy
            
        Returns:
            Energy tensor of shape (B,) with per-sample energies
        """
        # The flow model's compute_loss method typically returns averaged loss
        # For energy computation, we want per-sample values
        batch_size = batch_data['gt_trajectory'].shape[0]
        
        if batch_size == 1:
            # Single sample case
            return self.compute_energy(batch_data).unsqueeze(0)
        else:
            # For multiple samples, compute individually to get per-sample energies
            energies = []
            for i in range(batch_size):
                sample_batch = {}
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        sample_batch[key] = value[i:i+1]  # Keep batch dimension
                    else:
                        sample_batch[key] = value
                
                energy = self.compute_energy(sample_batch)
                energies.append(energy)
            
            return torch.stack(energies)
    
    def compute_trajectory_energy(self, observation: Dict[str, Any], 
                                action: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for a given action trajectory under observation.
        
        Args:
            observation: Observation data (without gt_trajectory)
            action: Action trajectory to evaluate (B, T, nhand, action_dim)
            
        Returns:
            Energy tensor of shape (B,)
        """
        # Create batch data by combining observation and action
        batch_data = observation.copy()
        batch_data['gt_trajectory'] = action
        
        return self.compute_energy(batch_data)
    
    def sample_from_flow(self, observation: Dict[str, Any], early_stop_callback=None) -> torch.Tensor:
        """
        Sample actions from the underlying flow model.
        This provides a baseline sampling method.
        
        Args:
            observation: Observation data for conditioning
            early_stop_callback: Optional callback for early stopping
            
        Returns:
            Sampled action trajectory (B, T, nhand, action_dim)
        """
        # Ensure proper gradient settings based on training mode and inference mode
        gradient_enabled = not self.inference_mode and (self.training_mode != "frozen")
        
        with torch.set_grad_enabled(gradient_enabled):
            # Pass early_stop_callback to compute_trajectory
            if early_stop_callback is not None:
                return self.flow_actor.compute_trajectory(**observation, early_stop_callback=early_stop_callback)
            else:
                return self.flow_actor.compute_trajectory(**observation)
    
    def enable_training(self):
        """Enable training mode (sets inference_mode to False)."""
        self.inference_mode = False
    
    def set_inference_mode(self, inference_mode: bool = True):
        """
        Set inference mode to optimize performance.
        
        Args:
            inference_mode: If True, disable gradients for pure inference
        """
        self.inference_mode = inference_mode
        if inference_mode:
            self.flow_actor.eval()
        else:
            self.enable_training()
    
    def enable_inference(self):
        """Enable inference mode."""
        self.set_inference_mode(True)
    
    def compute_energy_inference(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Optimized energy computation for inference (no gradients).
        
        Args:
            batch_data: Dictionary containing observations and gt_trajectory
            
        Returns:
            Energy tensor of shape (B,)
        """
        with torch.no_grad():
            return self.compute_energy(batch_data)
    
    def compute_trajectory_energy_inference(self, observation: Dict[str, Any], 
                                          action: torch.Tensor) -> torch.Tensor:
        """
        Optimized trajectory energy computation for inference.
        
        Args:
            observation: Observation data (without gt_trajectory)
            action: Action trajectory to evaluate (B, T, nhand, action_dim)
            
        Returns:
            Energy tensor of shape (B,)
        """
        with torch.no_grad():
            return self.compute_trajectory_energy(observation, action)
    
    def get_parameter_stats(self) -> Dict[str, int]:
        """Get statistics about parameter freezing."""
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        
        for param in self.flow_actor.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        return {
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
        }
