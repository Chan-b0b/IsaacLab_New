"""Event terms for locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_constant_force_to_torso(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_range: dict[str, tuple[float, float]],
    zero_force_percentage: float = 0.0,
):
    """Apply a constant force to the torso for a random duration.
    
    This event term applies a constant force to the robot's torso body. The force is randomized
    within the specified range and applied for the duration of the event interval.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to apply the force to.
        asset_cfg: The scene entity configuration for the robot asset.
        force_range: Dictionary with keys 'x', 'y', 'z' specifying force ranges in Newton.
        zero_force_percentage: Percentage (0.0 to 1.0) of environments that should receive zero force.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Sample random forces for each environment
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    
    # Apply random forces in each direction
    forces[:, :, 0].uniform_(force_range["x"][0], force_range["x"][1])
    forces[:, :, 1].uniform_(force_range["y"][0], force_range["y"][1])
    forces[:, :, 2].uniform_(force_range["z"][0], force_range["z"][1])
    
    # Zero out forces for selected percentage of environments
    if zero_force_percentage > 0.0:
        zero_mask = torch.rand(len(env_ids), device=asset.device) < zero_force_percentage
        forces[zero_mask] *= 0
    
    # Apply the forces to the torso
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=None,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )

def reset_joints_default_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    current_default_joint_pos_all_dofs = asset.data.default_joint_pos[env_ids].clone()

    joint_indices_to_modify = asset_cfg.joint_ids

    selected_joint_pos = current_default_joint_pos_all_dofs[:, joint_indices_to_modify]

    biased_joint_pos = selected_joint_pos + math_utils.sample_uniform(*position_range, selected_joint_pos.shape, selected_joint_pos.device)

    joint_pos_limits_all_dofs = asset.data.soft_joint_pos_limits[env_ids]
    
    selected_joint_pos_limits = joint_pos_limits_all_dofs[:, joint_indices_to_modify]

    biased_joint_pos = biased_joint_pos.clamp_(selected_joint_pos_limits[..., 0], selected_joint_pos_limits[..., 1])

    # current_default_joint_pos_all_dofs[:, joint_indices_to_modify] = biased_joint_pos
    # asset.data.default_joint_pos[env_ids] = current_default_joint_pos_all_dofs

    asset.set_joint_position_target(biased_joint_pos, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)