"""Custom termination functions for Realman RM75 pick-and-place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Terminate episode if the object moves beyond a maximum distance from initial position.
    
    Args:
        env: The learning environment.
        maximum_distance: Maximum allowed distance from initial spawn position.
        asset_cfg: The scene entity configuration for the object.
    
    Returns:
        A boolean tensor indicating which environments should terminate.
    """
    # Get the object asset
    asset = env.scene[asset_cfg.name]
    
    # Get current object position (root body position)
    current_pos = asset.data.root_pos_w[:, :2]  # Only XY coordinates, shape: (num_envs, 2)
    
    # Get initial spawn position from the object's default state
    # default_root_state contains [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    initial_pos = asset.data.default_root_state[:, :2] + env.scene.env_origins[:,:2] # Only XY coordinates from spawn position
    
    # Calculate distance from initial position
    distance = torch.norm(current_pos - initial_pos, dim=1)
    
    # Return True for environments where object is too far
    return distance > maximum_distance
