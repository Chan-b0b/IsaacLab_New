"""Event terms for locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_constant_force_to_torso(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_range: dict[str, tuple[float, float]],
):
    """Apply a constant force to the torso for a random duration.
    
    This event term applies a constant force to the robot's torso body. The force is randomized
    within the specified range and applied for the duration of the event interval.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to apply the force to.
        asset_cfg: The scene entity configuration for the robot asset.
        force_range: Dictionary with keys 'x', 'y', 'z' specifying force ranges in Newton.
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
    
    # Apply the forces to the torso
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=None,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )
