"""
Bimanual Coordination Constraints

Implements energy-based regularization constraints for bimanual manipulation:
1. Temporal smoothness (jerk minimization, velocity consistency)
2. Spatial safety (collision avoidance for end-effectors and joints)
3. Temporal coordination (synchronized motion between arms)

Features:
- Fully differentiable for gradient-based optimization
- Learnable constraint weights
- Uses inverse kinematics for accurate joint position estimation
- No dependency on RLBench environment during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .panda_kinematics import PandaIK


class ConstraintWeightPredictor(nn.Module):
    """
    Neural network that dynamically predicts constraint weights.
    
    Adaptively adjusts constraint importance based on current action state.
    For example: when arms are close, increase collision avoidance weight.
    """
    
    def __init__(self, action_dim: int = 8, hidden_dim: int = 64):
        """
        Args:
            action_dim: Dimension of single arm action (default: 8 for pos+quat+gripper)
            hidden_dim: Hidden dimension of the network
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),  # Input: left + right actions
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # Output: 5 constraint logits
        )
        
        # Learnable temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, combined_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            combined_actions: (B, T, 16) [left_action + right_action concatenated]
        
        Returns:
            weights: (B, T, 5) [softmax normalized weights for 5 constraints]
        """
        logits = self.network(combined_actions)  # (B, T, 5)
        
        # Temperature-scaled softmax for smooth weight distribution
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return weights


class BimanualCoordinationConstraints(nn.Module):
    """
    Coordination constraints for dual-arm manipulation.
    
    Implements energy-based regularization to ensure generated bimanual actions are:
    1. Temporally smooth (minimize jerk)
    2. Spatially safe (avoid collisions)
    3. Coordinated (synchronized motion patterns)
    
    All constraints are differentiable and suitable for gradient-based optimization.
    """
    
    def __init__(self, 
                 action_dim: int = 8,
                 min_ee_distance: float = 0.15,
                 min_joint_distance: float = 0.12,
                 weight_hidden_dim: int = 64):
        """
        Args:
            action_dim: Dimension of single arm action (pos(3) + quat(4) + gripper(1))
            min_ee_distance: Minimum safe distance between end-effectors (meters)
            min_joint_distance: Minimum safe distance between joint positions (meters)
            weight_hidden_dim: Hidden dimension for weight predictor network
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.min_ee_distance = min_ee_distance
        self.min_joint_distance = min_joint_distance
        
        # Learnable weight predictor
        self.weight_predictor = ConstraintWeightPredictor(
            action_dim=action_dim,
            hidden_dim=weight_hidden_dim
        )
        
        # Panda robot base positions (dual_panda setup in RLBench)
        # These will be converted to tensors on first forward pass
        self.base_left = [0.0, 0.37, 0.61]   # Left arm base
        self.base_right = [0.0, -0.37, 0.61]  # Right arm base
        
        self._base_left_tensor = None
        self._base_right_tensor = None
        
        # Initialize Panda IK solvers for accurate joint position estimation
        self.ik_left = PandaIK(base_position=self.base_left)
        self.ik_right = PandaIK(base_position=self.base_right)
    
    def _ensure_base_tensors(self, device: torch.device, dtype: torch.dtype):
        """Lazily initialize base position tensors on correct device."""
        if self._base_left_tensor is None:
            self._base_left_tensor = torch.tensor(
                self.base_left, device=device, dtype=dtype
            )
            self._base_right_tensor = torch.tensor(
                self.base_right, device=device, dtype=dtype
            )
    
    def forward(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all constraint energies with learnable weights.
        
        Args:
            left_actions: (B, T, 8) [pos(3), quat(4), gripper(1)]
            right_actions: (B, T, 8) [pos(3), quat(4), gripper(1)]
        
        Returns:
            Dictionary containing:
                - 'jerk': Weighted jerk energy (B, T)
                - 'velocity_smoothness': Weighted velocity smoothness energy (B, T)
                - 'temporal_sync': Weighted temporal synchronization energy (B, T)
                - 'ee_collision': Weighted end-effector collision energy (B, T)
                - 'joint_collision': Weighted joint collision energy (B, T)
                - 'total': Total constraint energy (B, T)
                - 'weights': Predicted weights (B, T, 5)
                - 'raw_energies': Dictionary of unweighted energies for debugging
        """
        # Ensure base position tensors are on correct device
        self._ensure_base_tensors(left_actions.device, left_actions.dtype)
        
        # 1. Compute raw constraint energies
        E_jerk = self._compute_jerk_energy(left_actions, right_actions)
        E_velocity = self._compute_velocity_smoothness(left_actions, right_actions)
        E_temporal_sync = self._compute_temporal_sync(left_actions, right_actions)
        E_ee_collision = self._compute_ee_collision(left_actions, right_actions)
        E_joint_collision = self._compute_joint_collision(left_actions, right_actions)
        
        # 2. Predict adaptive weights using neural network
        combined_actions = torch.cat([left_actions, right_actions], dim=-1)  # (B, T, 16)
        weights = self.weight_predictor(combined_actions)  # (B, T, 5)
        
        # 3. Apply learned weights to energies
        weighted_energies = {
            'jerk': weights[..., 0] * E_jerk,
            'velocity_smoothness': weights[..., 1] * E_velocity,
            'temporal_sync': weights[..., 2] * E_temporal_sync,
            'ee_collision': weights[..., 3] * E_ee_collision,
            'joint_collision': weights[..., 4] * E_joint_collision,
        }
        
        # 4. Compute total constraint energy
        total_energy = sum(weighted_energies.values())
        
        return {
            **weighted_energies,
            'total': total_energy,
            'weights': weights,
            'raw_energies': {
                'jerk': E_jerk,
                'velocity_smoothness': E_velocity,
                'temporal_sync': E_temporal_sync,
                'ee_collision': E_ee_collision,
                'joint_collision': E_joint_collision,
            }
        }
    
    # ==================== Temporal Smoothness Constraints ====================
    
    def _compute_jerk_energy(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        """
        Jerk energy: Minimizes change in acceleration (third derivative of position).
        
        Ensures smooth trajectories that are physically plausible and comfortable.
        
        Args:
            left_actions: (B, T, 8)
            right_actions: (B, T, 8)
        
        Returns:
            jerk_energy: (B, T) Jerk magnitude at each timestep
        """
        B, T = left_actions.shape[:2]
        
        # If sequence is too short for jerk computation, return zeros
        if T < 4:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)
        
        # Extract positions
        left_pos = left_actions[..., :3]   # (B, T, 3)
        right_pos = right_actions[..., :3]
        
        # First-order difference: velocity
        left_vel = left_pos[:, 1:] - left_pos[:, :-1]  # (B, T-1, 3)
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]
        
        # Second-order difference: acceleration
        left_acc = left_vel[:, 1:] - left_vel[:, :-1]  # (B, T-2, 3)
        right_acc = right_vel[:, 1:] - right_vel[:, :-1]
        
        # Third-order difference: jerk
        left_jerk = left_acc[:, 1:] - left_acc[:, :-1]  # (B, T-3, 3)
        right_jerk = right_acc[:, 1:] - right_acc[:, :-1]
        
        # Jerk energy = ||jerk||²
        jerk_norm = (left_jerk.pow(2).sum(dim=-1) + 
                     right_jerk.pow(2).sum(dim=-1))  # (B, T-3)
        
        # Pad to original length (first 3 timesteps have zero jerk)
        pad_size = T - jerk_norm.shape[1]
        jerk_energy = F.pad(jerk_norm, (pad_size, 0), value=0.0)  # (B, T)
        
        return jerk_energy
    
    def _compute_velocity_smoothness(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        """
        Velocity smoothness: Penalizes abrupt changes in velocity.
        
        Encourages continuous velocity profiles.
        
        Args:
            left_actions: (B, T, 8)
            right_actions: (B, T, 8)
        
        Returns:
            velocity_energy: (B, T) Velocity change magnitude
        """
        B, T = left_actions.shape[:2]
        
        # If sequence is too short for velocity smoothness computation, return zeros
        if T < 3:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)
        
        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]
        
        # Velocity
        left_vel = left_pos[:, 1:] - left_pos[:, :-1]  # (B, T-1, 3)
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]
        
        # Velocity change (acceleration)
        left_vel_change = left_vel[:, 1:] - left_vel[:, :-1]  # (B, T-2, 3)
        right_vel_change = right_vel[:, 1:] - right_vel[:, :-1]
        
        # Energy = ||velocity_change||²
        velocity_change_norm = (left_vel_change.pow(2).sum(dim=-1) + 
                               right_vel_change.pow(2).sum(dim=-1))  # (B, T-2)
        
        # Pad to original length
        pad_size = T - velocity_change_norm.shape[1]
        velocity_energy = F.pad(velocity_change_norm, (pad_size, 0), value=0.0)  # (B, T)
        
        return velocity_energy
    
    def _compute_temporal_sync(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        """
        Temporal synchronization: Encourages coordinated motion patterns.
        
        The two arms should move with similar speed and direction patterns.
        
        Args:
            left_actions: (B, T, 8)
            right_actions: (B, T, 8)
        
        Returns:
            sync_energy: (B, T) Synchronization penalty
        """
        B, T = left_actions.shape[:2]
        
        # If sequence is too short for temporal sync computation, return zeros
        if T < 2:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)
        
        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]
        
        # Compute velocities
        left_vel = left_pos[:, 1:] - left_pos[:, :-1]  # (B, T-1, 3)
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]
        
        # Speed (velocity magnitude)
        left_speed = left_vel.norm(dim=-1)  # (B, T-1)
        right_speed = right_vel.norm(dim=-1)
        
        # Speed difference
        speed_diff = (left_speed - right_speed).abs()
        
        # Velocity direction similarity (cosine similarity)
        # Normalize velocities to unit vectors (avoid division by zero)
        left_vel_norm = F.normalize(left_vel + 1e-8, dim=-1)
        right_vel_norm = F.normalize(right_vel + 1e-8, dim=-1)
        
        vel_cosine = (left_vel_norm * right_vel_norm).sum(dim=-1)  # (B, T-1)
        direction_diff = 1.0 - vel_cosine.abs()  # Encourage parallel or anti-parallel
        
        # Combined synchronization energy
        sync_energy = speed_diff + direction_diff  # (B, T-1)
        
        # Pad to original length
        pad_size = T - sync_energy.shape[1]
        sync_energy = F.pad(sync_energy, (pad_size, 0), value=0.0)  # (B, T)
        
        return sync_energy
    
    # ==================== Spatial Safety Constraints ====================
    
    def _compute_ee_collision(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        """
        End-effector collision avoidance.
        
        Penalizes configurations where end-effectors are too close.
        
        Args:
            left_actions: (B, T, 8)
            right_actions: (B, T, 8)
        
        Returns:
            collision_energy: (B, T) Collision penalty
        """
        left_pos = left_actions[..., :3]  # (B, T, 3)
        right_pos = right_actions[..., :3]
        
        # Euclidean distance between end-effectors
        distance = (left_pos - right_pos).norm(dim=-1)  # (B, T)
        
        # Exponential collision energy (increases as distance decreases)
        collision_energy = torch.exp(-distance / self.min_ee_distance)
        
        # Hard constraint penalty for distances below threshold
        violation = F.relu(self.min_ee_distance - distance)
        collision_energy = collision_energy + violation * 20.0
        
        return collision_energy
    
    def _compute_joint_collision(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        """
        Joint collision avoidance using inverse kinematics.
        
        Uses Panda IK solver to compute accurate joint positions from end-effector poses,
        then checks for collisions between all joint pairs.
        
        Args:
            left_actions: (B, T, 8) [pos(3), quat(4), gripper(1)]
            right_actions: (B, T, 8) [pos(3), quat(4), gripper(1)]
        
        Returns:
            joint_collision_energy: (B, T) Joint collision penalty
        """
        B, T = left_actions.shape[:2]
        
        left_pos = left_actions[..., :3]  # (B, T, 3)
        left_quat = left_actions[..., 3:7]  # (B, T, 4)
        right_pos = right_actions[..., :3]
        right_quat = right_actions[..., 3:7]
        
        # Reshape for batch IK computation
        left_pos_flat = left_pos.reshape(B * T, 3)
        left_quat_flat = left_quat.reshape(B * T, 4)
        right_pos_flat = right_pos.reshape(B * T, 3)
        right_quat_flat = right_quat.reshape(B * T, 4)
        
        # Compute joint positions using IK
        # joint_positions: (B*T, num_joints, 3)
        left_joint_pos = self.ik_left.compute_joint_positions(left_pos_flat, left_quat_flat)
        right_joint_pos = self.ik_right.compute_joint_positions(right_pos_flat, right_quat_flat)
        
        # Reshape back to (B, T, num_joints, 3)
        num_joints = left_joint_pos.shape[1]
        left_joint_pos = left_joint_pos.reshape(B, T, num_joints, 3)
        right_joint_pos = right_joint_pos.reshape(B, T, num_joints, 3)
        
        # 1. Compute pairwise distances between all left and right joints
        # Expand dimensions for broadcasting: (B, T, num_joints_left, 1, 3) vs (B, T, 1, num_joints_right, 3)
        left_expanded = left_joint_pos.unsqueeze(3)  # (B, T, num_joints, 1, 3)
        right_expanded = right_joint_pos.unsqueeze(2)  # (B, T, 1, num_joints, 3)
        
        # Pairwise distances: (B, T, num_joints, num_joints)
        pairwise_distances = (left_expanded - right_expanded).norm(dim=-1)
        
        # Collision energy increases exponentially as joints get closer
        collision_energy = torch.exp(-pairwise_distances / self.min_joint_distance)
        
        # Take maximum collision energy across all joint pairs (worst case)
        max_collision_per_timestep = collision_energy.max(dim=-1)[0].max(dim=-1)[0]  # (B, T)
        
        # 2. Hard constraint penalty for critical joint pairs
        # Focus on elbow (joint 3) and wrist (joint 6) as they're most likely to collide
        elbow_idx = 3
        wrist_idx = 6
        
        critical_pairs_energy = 0.0
        if num_joints > max(elbow_idx, wrist_idx):
            # Elbow-to-elbow distance
            elbow_distance = (left_joint_pos[:, :, elbow_idx] - 
                            right_joint_pos[:, :, elbow_idx]).norm(dim=-1)
            elbow_violation = F.relu(self.min_joint_distance - elbow_distance)
            
            # Wrist-to-wrist distance
            wrist_distance = (left_joint_pos[:, :, wrist_idx] - 
                            right_joint_pos[:, :, wrist_idx]).norm(dim=-1)
            wrist_violation = F.relu(self.min_joint_distance - wrist_distance)
            
            critical_pairs_energy = 20.0 * (elbow_violation + wrist_violation)
        
        # 3. Workspace separation constraint (soft boundary)
        # Left arm should stay in y > 0 region, right arm in y < 0 region
        workspace_penalty = (
            F.relu(-left_pos[..., 1]) +   # Penalize left arm in y < 0
            F.relu(right_pos[..., 1])      # Penalize right arm in y > 0
        )
        
        # Combined joint collision energy
        joint_collision_energy = (
            max_collision_per_timestep + 
            critical_pairs_energy + 
            workspace_penalty
        )
        
        return joint_collision_energy

