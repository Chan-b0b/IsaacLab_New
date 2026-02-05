"""Symmetry functions for humanoid robots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states_h1_2"]


@torch.no_grad()
def compute_symmetric_states_h1_2(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments observations and actions by applying left-right symmetry for H1-2 humanoid.

    This function creates augmented versions by applying left-right symmetrical transformation.
    The symmetry swaps left and right body parts while negating appropriate velocity components.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # we have 2 different symmetries (original + left-right), so augment batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # we have 2 different symmetries (original + left-right), so augment batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor for H1-2.

    This transforms the observation by:
    - Negating y-components of linear velocity, angular velocity (x,z), and projected gravity
    - Negating y-component of velocity command and negating yaw command
    - Swapping left/right joint positions, velocities, and actions

    Args:
        env: The environment instance.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    
    # Observation structure for H1-2 velocity task (policy):
    # 0-2:   base_ang_vel (x, y, z)
    # 3-5:   projected_gravity (x, y, z)
    # 6-8:   velocity_commands (vx, vy, vyaw)
    # 9-35:  joint_pos (27 joints - all joints in h1_2_handless)
    # 36-62: joint_vel (27 joints)
    # 63-74: last_action (12 joints - legs only)
    # 75:    gait_phase
    
    # ang vel: negate x and z
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity: negate y
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command: negate vy and vyaw
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    
    # joint positions, velocities, and actions: swap left/right
    obs[:, 9:36] = _switch_h1_2_joints_left_right_obs(obs[:, 9:36])  # joint_pos (27 joints)
    obs[:, 36:63] = _switch_h1_2_joints_left_right_obs(obs[:, 36:63])  # joint_vel (27 joints)
    obs[:, 63:75] = _switch_h1_2_joints_left_right_actions(obs[:, 63:75])  # last_action (12 joints - legs only)
    
    # gait phase: shift by 0.5 (half cycle) for left-right symmetry
    obs[:, 75] = (obs[:, 75] + 0.5) % 1.0
    
    # If height scan is present, flip it along y-axis
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        # Assuming height scan starts at index 76 with grid pattern
        # This would need to be adjusted based on actual grid size
        height_scan_start = 76
        height_scan_end = obs.shape[1]
        if height_scan_end > height_scan_start:
            # Assuming a grid of shape (rows, cols) - flip along cols (y-axis)
            # You may need to adjust the grid dimensions based on your configuration
            pass  # TODO: implement height scan flipping if needed

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the action tensor for H1-2.

    Args:
        actions: The action tensor to be transformed (batch_size, 12) - legs only.

    Returns:
        The transformed action tensor with left-right symmetry applied.
    """
    return _switch_h1_2_joints_left_right_actions(actions)


def _switch_h1_2_joints_left_right_obs(tensor: torch.Tensor) -> torch.Tensor:
    """Swap left and right joints for H1-2 humanoid observations (all 27 joints).

    H1-2 has 27 joints in the following order (from h1_2_handless.xml):
    0:  left_hip_yaw
    1:  left_hip_pitch
    2:  left_hip_roll
    3:  left_knee
    4:  left_ankle_pitch
    5:  left_ankle_roll
    6:  right_hip_yaw
    7:  right_hip_pitch
    8:  right_hip_roll
    9:  right_knee
    10: right_ankle_pitch
    11: right_ankle_roll
    12: torso
    13: left_shoulder_pitch
    14: left_shoulder_roll
    15: left_shoulder_yaw
    16: left_elbow
    17: left_wrist_roll
    18: left_wrist_pitch
    19: left_wrist_yaw
    20: right_shoulder_pitch
    21: right_shoulder_roll
    22: right_shoulder_yaw
    23: right_elbow
    24: right_wrist_roll
    25: right_wrist_pitch
    26: right_wrist_yaw

    Args:
        tensor: Joint tensor to swap (batch_size, 27).

    Returns:
        The tensor with left and right joints swapped and appropriate signs flipped.
    """
    tensor = tensor.clone()
    device = tensor.device
    
    # Define the swapping indices and sign flips
    # Legs: swap left (0-5) with right (6-11)
    left_leg = tensor[:, 0:6].clone()
    right_leg = tensor[:, 6:12].clone()
    
    # Swap and negate yaw/roll joints
    tensor[:, 0] = -right_leg[:, 0]  # left_hip_yaw = -right_hip_yaw
    tensor[:, 1] = right_leg[:, 1]   # left_hip_pitch = right_hip_pitch
    tensor[:, 2] = -right_leg[:, 2]  # left_hip_roll = -right_hip_roll
    tensor[:, 3] = right_leg[:, 3]   # left_knee = right_knee
    tensor[:, 4] = right_leg[:, 4]   # left_ankle_pitch = right_ankle_pitch
    tensor[:, 5] = -right_leg[:, 5]  # left_ankle_roll = -right_ankle_roll
    
    tensor[:, 6] = -left_leg[:, 0]   # right_hip_yaw = -left_hip_yaw
    tensor[:, 7] = left_leg[:, 1]    # right_hip_pitch = left_hip_pitch
    tensor[:, 8] = -left_leg[:, 2]   # right_hip_roll = -left_hip_roll
    tensor[:, 9] = left_leg[:, 3]    # right_knee = left_knee
    tensor[:, 10] = left_leg[:, 4]   # right_ankle_pitch = left_ankle_pitch
    tensor[:, 11] = -left_leg[:, 5]  # right_ankle_roll = -left_ankle_roll
    
    # Torso: unchanged
    # tensor[:, 12] stays the same
    
    # Arms: swap left (13-19) with right (20-26)
    left_arm = tensor[:, 13:20].clone()
    right_arm = tensor[:, 20:27].clone()
    
    tensor[:, 13] = right_arm[:, 0]   # left_shoulder_pitch = right_shoulder_pitch
    tensor[:, 14] = -right_arm[:, 1]  # left_shoulder_roll = -right_shoulder_roll
    tensor[:, 15] = -right_arm[:, 2]  # left_shoulder_yaw = -right_shoulder_yaw
    tensor[:, 16] = right_arm[:, 3]   # left_elbow = right_elbow
    tensor[:, 17] = -right_arm[:, 4]  # left_wrist_roll = -right_wrist_roll
    tensor[:, 18] = right_arm[:, 5]   # left_wrist_pitch = right_wrist_pitch
    tensor[:, 19] = -right_arm[:, 6]  # left_wrist_yaw = -right_wrist_yaw
    
    tensor[:, 20] = left_arm[:, 0]    # right_shoulder_pitch = left_shoulder_pitch
    tensor[:, 21] = -left_arm[:, 1]   # right_shoulder_roll = -left_shoulder_roll
    tensor[:, 22] = -left_arm[:, 2]   # right_shoulder_yaw = -left_shoulder_yaw
    tensor[:, 23] = left_arm[:, 3]    # right_elbow = left_elbow
    tensor[:, 24] = -left_arm[:, 4]   # right_wrist_roll = -left_wrist_roll
    tensor[:, 25] = left_arm[:, 5]    # right_wrist_pitch = left_wrist_pitch
    tensor[:, 26] = -left_arm[:, 6]   # right_wrist_yaw = -left_wrist_yaw
    
    return tensor


def _switch_h1_2_joints_left_right_actions(tensor: torch.Tensor) -> torch.Tensor:
    """Swap left and right leg joints for H1-2 actions (12 leg joints only).

    Actions control only the legs:
    0:  left_hip_yaw
    1:  left_hip_pitch
    2:  left_hip_roll
    3:  left_knee
    4:  left_ankle_pitch
    5:  left_ankle_roll
    6:  right_hip_yaw
    7:  right_hip_pitch
    8:  right_hip_roll
    9:  right_knee
    10: right_ankle_pitch
    11: right_ankle_roll

    Args:
        tensor: Action tensor to swap (batch_size, 12).

    Returns:
        The tensor with left and right leg joints swapped and appropriate signs flipped.
    """
    tensor = tensor.clone()
    device = tensor.device
    
    # Legs: swap left (0-5) with right (6-11)
    left_leg = tensor[:, 0:6].clone()
    right_leg = tensor[:, 6:12].clone()
    
    # Swap and negate yaw/roll joints
    tensor[:, 0] = -right_leg[:, 0]  # left_hip_yaw = -right_hip_yaw
    tensor[:, 1] = right_leg[:, 1]   # left_hip_pitch = right_hip_pitch
    tensor[:, 2] = -right_leg[:, 2]  # left_hip_roll = -right_hip_roll
    tensor[:, 3] = right_leg[:, 3]   # left_knee = right_knee
    tensor[:, 4] = right_leg[:, 4]   # left_ankle_pitch = right_ankle_pitch
    tensor[:, 5] = -right_leg[:, 5]  # left_ankle_roll = -right_ankle_roll
    
    tensor[:, 6] = -left_leg[:, 0]   # right_hip_yaw = -left_hip_yaw
    tensor[:, 7] = left_leg[:, 1]    # right_hip_pitch = left_hip_pitch
    tensor[:, 8] = -left_leg[:, 2]   # right_hip_roll = -left_hip_roll
    tensor[:, 9] = left_leg[:, 3]    # right_knee = left_knee
    tensor[:, 10] = left_leg[:, 4]   # right_ankle_pitch = left_ankle_pitch
    tensor[:, 11] = -left_leg[:, 5]  # right_ankle_roll = -left_ankle_roll
    
    return tensor

