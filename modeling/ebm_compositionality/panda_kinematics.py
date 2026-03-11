"""
Panda Robot Kinematics

Simplified kinematic model for Panda dual-arm robot.
Used for estimating joint positions without full IK computation.

Reference: Franka Emika Panda specifications
- Link 1-3 (shoulder to elbow): 0.316m
- Link 4-6 (elbow to wrist): 0.384m  
- Link 7 + hand (wrist to EE): 0.107m
- Total reach: ~0.807m
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class PandaKinematics:
    """
    Simplified kinematic model for Panda robot arm.
    
    Provides geometric parameters and utility functions for
    estimating joint positions based on end-effector poses.
    """
    
    # Panda DH parameters (simplified)
    LINK_LENGTHS = {
        'shoulder_to_elbow': 0.316,  # Links 1-3
        'elbow_to_wrist': 0.384,     # Links 4-6
        'wrist_to_ee': 0.107,        # Link 7 + hand
        'total_reach': 0.807,        # Maximum reach from base
    }
    
    # Base positions for dual Panda setup in RLBench
    # These match the dual_panda configuration
    BASE_LEFT = [0.0, 0.37, 0.61]    # Left arm base (x, y, z)
    BASE_RIGHT = [0.0, -0.37, 0.61]  # Right arm base (x, y, z)
    
    # Typical workspace bounds for Panda
    WORKSPACE_BOUNDS = {
        'x': (-0.5, 0.8),   # Front-back
        'y': (-0.6, 0.6),   # Left-right  
        'z': (0.0, 1.2),    # Up-down
    }
    
    @staticmethod
    def estimate_elbow_position(ee_pos: torch.Tensor, 
                                base_pos: torch.Tensor,
                                elbow_ratio: float = 0.45,
                                z_offset: float = -0.08) -> torch.Tensor:
        """
        Estimate elbow position using simplified geometric model.
        
        Assumption: The elbow lies approximately along the line from base to end-effector,
        at a certain ratio of the total distance, with a downward offset due to gravity.
        
        Args:
            ee_pos: (B, T, 3) or (..., 3) End-effector positions
            base_pos: (3,) or (..., 3) Base position
            elbow_ratio: Ratio along base-to-EE direction (default: 0.45)
            z_offset: Vertical offset in meters (default: -0.08 for 8cm downward)
        
        Returns:
            elbow_pos: (..., 3) Estimated elbow positions
        """
        # Direction vector from base to end-effector
        direction = ee_pos - base_pos
        
        # Elbow position along the direction
        elbow_pos = base_pos + elbow_ratio * direction
        
        # Apply downward offset (typical configuration)
        elbow_offset = torch.zeros_like(elbow_pos)
        elbow_offset[..., 2] = z_offset
        elbow_pos = elbow_pos + elbow_offset
        
        return elbow_pos
    
    @staticmethod
    def estimate_wrist_position(ee_pos: torch.Tensor,
                               base_pos: torch.Tensor,
                               wrist_ratio: float = 0.85) -> torch.Tensor:
        """
        Estimate wrist position using simplified geometric model.
        
        Args:
            ee_pos: (..., 3) End-effector positions
            base_pos: (..., 3) Base position
            wrist_ratio: Ratio along base-to-EE direction (default: 0.85)
        
        Returns:
            wrist_pos: (..., 3) Estimated wrist positions
        """
        direction = ee_pos - base_pos
        wrist_pos = base_pos + wrist_ratio * direction
        
        return wrist_pos
    
    @staticmethod
    def check_workspace_bounds(pos: torch.Tensor) -> torch.Tensor:
        """
        Check if positions are within typical Panda workspace.
        
        Args:
            pos: (..., 3) Positions to check
        
        Returns:
            in_bounds: (...,) Boolean tensor indicating if positions are in workspace
        """
        x_in = (pos[..., 0] >= PandaKinematics.WORKSPACE_BOUNDS['x'][0]) & \
               (pos[..., 0] <= PandaKinematics.WORKSPACE_BOUNDS['x'][1])
        y_in = (pos[..., 1] >= PandaKinematics.WORKSPACE_BOUNDS['y'][0]) & \
               (pos[..., 1] <= PandaKinematics.WORKSPACE_BOUNDS['y'][1])
        z_in = (pos[..., 2] >= PandaKinematics.WORKSPACE_BOUNDS['z'][0]) & \
               (pos[..., 2] <= PandaKinematics.WORKSPACE_BOUNDS['z'][1])
        
        in_bounds = x_in & y_in & z_in
        
        return in_bounds
    
    @staticmethod
    def compute_reach_distance(ee_pos: torch.Tensor, 
                               base_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from base to end-effector.
        
        Useful for checking if configuration is within arm reach.
        
        Args:
            ee_pos: (..., 3) End-effector positions
            base_pos: (..., 3) Base position
        
        Returns:
            distance: (...,) Euclidean distance
        """
        distance = (ee_pos - base_pos).norm(dim=-1)
        return distance
    
    @staticmethod
    def is_reachable(ee_pos: torch.Tensor,
                    base_pos: torch.Tensor,
                    margin: float = 0.05) -> torch.Tensor:
        """
        Check if end-effector position is reachable.
        
        Args:
            ee_pos: (..., 3) End-effector positions
            base_pos: (..., 3) Base position  
            margin: Safety margin from maximum reach (meters)
        
        Returns:
            reachable: (...,) Boolean tensor
        """
        distance = PandaKinematics.compute_reach_distance(ee_pos, base_pos)
        max_reach = PandaKinematics.LINK_LENGTHS['total_reach'] - margin
        
        reachable = distance <= max_reach
        
        return reachable


class PandaIK(nn.Module):
    """
    Panda Inverse Kinematics with Forward Kinematics for joint position computation.
    
    Uses simplified geometric IK to estimate joint angles, then computes
    joint positions using forward kinematics.
    """
    
    def __init__(self, base_position: List[float] = None):
        """
        Args:
            base_position: [x, y, z] base position of the robot arm
        """
        super().__init__()
        
        if base_position is None:
            base_position = [0.0, 0.0, 0.0]
        
        self.register_buffer('base_pos', torch.tensor(base_position, dtype=torch.float32))
        
        # Panda DH parameters (simplified)
        # Joint positions relative to base
        self.joint_z_offsets = [
            0.333,   # Joint 1 (shoulder yaw)
            0.333,   # Joint 2 (shoulder pitch)  
            0.649,   # Joint 3 (elbow pitch) = 0.333 + 0.316
            0.649,   # Joint 4 (elbow yaw)
            1.033,   # Joint 5 (wrist pitch) = 0.649 + 0.384
            1.033,   # Joint 6 (wrist roll)
            1.140,   # Joint 7 (flange) = 1.033 + 0.107
        ]
    
    def compute_joint_positions(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D joint positions from end-effector pose using simplified FK.
        
        This is a geometric approximation that distributes joints along
        the path from base to end-effector based on Panda's kinematic structure.
        
        Args:
            ee_pos: (B, 3) End-effector positions
            ee_quat: (B, 4) End-effector orientations (not used in simplified version)
        
        Returns:
            joint_positions: (B, 7, 3) Positions of all 7 joints in 3D space
        """
        B = ee_pos.shape[0]
        device = ee_pos.device
        dtype = ee_pos.dtype
        
        # Ensure base_pos is on the same device
        base_pos = self.base_pos.to(device=device, dtype=dtype)
        
        # Direction from base to end-effector
        ee_direction = ee_pos - base_pos  # (B, 3)
        ee_distance = ee_direction.norm(dim=-1, keepdim=True)  # (B, 1)
        ee_direction_normalized = ee_direction / (ee_distance + 1e-8)  # (B, 3)
        
        # Initialize joint positions
        joint_positions = []
        
        # Total arm length (base to EE)
        total_length = 0.807  # Approximate Panda reach
        
        for i, z_offset in enumerate(self.joint_z_offsets):
            # Interpolation factor based on joint's position in kinematic chain
            # Earlier joints (shoulder) are closer to base, later joints (wrist) closer to EE
            if i < 2:
                # Shoulder joints - close to base
                ratio = 0.1 + (i / 7.0) * 0.2
            elif i < 4:
                # Elbow joints - middle region
                ratio = 0.35 + ((i - 2) / 7.0) * 0.15
            else:
                # Wrist joints - close to EE
                ratio = 0.65 + ((i - 4) / 7.0) * 0.25
            
            # Linear interpolation from base to EE
            joint_pos = base_pos + ratio * ee_direction  # (B, 3)
            
            # Add vertical offset based on kinematic structure
            joint_pos_with_offset = joint_pos.clone()
            # Adjust z-coordinate relative to base
            joint_pos_with_offset[:, 2] += z_offset * 0.15  # Scale down z-offset for stability
            
            # Add small lateral offset for elbow (makes it more realistic)
            if i == 2 or i == 3:  # Elbow joints
                # Offset perpendicular to the direction
                # Create a perpendicular vector in xy-plane
                perp_xy = torch.zeros_like(ee_direction)
                perp_xy[:, 0] = -ee_direction[:, 1]
                perp_xy[:, 1] = ee_direction[:, 0]
                perp_xy_norm = perp_xy / (perp_xy.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Add 8cm lateral offset for elbow
                joint_pos_with_offset = joint_pos_with_offset + 0.08 * perp_xy_norm
                # Add downward offset for elbow
                joint_pos_with_offset[:, 2] -= 0.05
            
            joint_positions.append(joint_pos_with_offset)
        
        # Stack all joint positions: (B, 7, 3)
        joint_positions = torch.stack(joint_positions, dim=1)
        
        return joint_positions


class DualPandaKinematics:
    """
    Kinematic utilities for dual Panda robot setup.
    """
    
    @staticmethod
    def get_base_positions(device: torch.device = None, 
                          dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get base positions for left and right arms.
        
        Args:
            device: Target device for tensors
            dtype: Target dtype for tensors
        
        Returns:
            Tuple of (base_left, base_right) tensors of shape (3,)
        """
        base_left = torch.tensor(PandaKinematics.BASE_LEFT, device=device, dtype=dtype)
        base_right = torch.tensor(PandaKinematics.BASE_RIGHT, device=device, dtype=dtype)
        
        return base_left, base_right
    
    @staticmethod
    def compute_arm_separation(left_pos: torch.Tensor,
                              right_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between left and right end-effectors.
        
        Args:
            left_pos: (..., 3) Left arm positions
            right_pos: (..., 3) Right arm positions
        
        Returns:
            separation: (...,) Distance between arms
        """
        separation = (left_pos - right_pos).norm(dim=-1)
        return separation
    
    @staticmethod
    def check_collision_risk(left_pos: torch.Tensor,
                            right_pos: torch.Tensor,
                            min_distance: float = 0.15) -> torch.Tensor:
        """
        Check if arms are at risk of collision.
        
        Args:
            left_pos: (..., 3) Left arm end-effector positions
            right_pos: (..., 3) Right arm end-effector positions
            min_distance: Minimum safe distance (meters)
        
        Returns:
            at_risk: (...,) Boolean tensor indicating collision risk
        """
        separation = DualPandaKinematics.compute_arm_separation(left_pos, right_pos)
        at_risk = separation < min_distance
        
        return at_risk

