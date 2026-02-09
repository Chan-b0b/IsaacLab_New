from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def applied_external_wrench(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """External forces and torques being applied to bodies via the permanent wrench composer.
    
    This returns the 6-D wrench (force xyz and torque xyz) that is actively being applied
    to the specified bodies through events like apply_constant_force_to_torso.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying the asset and body names to get wrench from.
        
    Returns:
        Tensor of shape (num_envs, num_bodies * 6) containing [fx, fy, fz, tx, ty, tz] for each body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the body indices
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        body_ids = slice(None)
    
    # Access the permanent wrench composer's composed forces and torques
    composed_forces = asset._permanent_wrench_composer.composed_force_as_torch[:, body_ids, :]
    composed_torques = asset._permanent_wrench_composer.composed_torque_as_torch[:, body_ids, :]
    
    # Stack forces and torques: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    wrench = torch.cat([composed_forces, composed_torques], dim=-1)
    
    return wrench.view(env.num_envs, -1)
