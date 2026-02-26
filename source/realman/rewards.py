"""Custom reward functions for Realman RM75 pick-and-place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_height_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_7")
) -> torch.Tensor:
    """Penalize the end-effector for going below a minimum height.
    
    Args:
        env: The learning environment.
        minimum_height: The minimum allowed height for the gripper.
        asset_cfg: The scene entity configuration for the robot.
    
    Returns:
        A tensor of penalties (negative rewards) for each environment where the gripper is below the minimum height.
    """
    # Extract the asset and body indices from the scene entity
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    
    # Get the body positions in world frame
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]  # Shape: (num_envs, num_bodies, 3)
    
    # Extract z-coordinates (height)
    heights = body_pos_w[:, :, 2]  # Shape: (num_envs, num_bodies)
    
    # Check if any body is below the minimum height
    below_minimum = heights < minimum_height
    
    # Return -1.0 for environments where the gripper is too low, 0.0 otherwise
    penalty = torch.where(below_minimum.any(dim=1), torch.tensor(1.0, device=env.device), torch.tensor(0.0, device=env.device))
    
    return penalty


def object_goal_tracking_with_grasp(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    grasp_distance_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_outer_knuckle", "right_outer_knuckle"]),
) -> torch.Tensor:
    """Reward tracking goal position only when object is being held by gripper.
    
    Args:
        env: The learning environment.
        std: Standard deviation for tanh kernel.
        minimal_height: Minimum height for object to be considered lifted.
        grasp_distance_threshold: Maximum distance between gripper and object to consider it grasped.
        command_name: Name of the command for goal position.
        robot_cfg: Scene entity configuration for robot.
        object_cfg: Scene entity configuration for object.
        gripper_cfg: Scene entity configuration for gripper knuckles.
    
    Returns:
        Reward for tracking goal when object is grasped.
    """
    # Extract the used quantities
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    gripper = env.scene[gripper_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
    # Distance of object to goal position
    distance_to_goal = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    
    # Calculate gripper center as midpoint between left and right outer knuckles
    gripper_body_ids = gripper_cfg.body_ids
    gripper_positions = gripper.data.body_pos_w[:, gripper_body_ids, :]  # Shape: (num_envs, 2, 3)
    gripper_center = gripper_positions.mean(dim=1)  # Shape: (num_envs, 3)
    
    # Distance between gripper center and object
    object_pos = object.data.root_pos_w
    gripper_object_distance = torch.norm(gripper_center - object_pos, dim=1)
    
    # Check if object is being held: lifted AND close to gripper
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    is_grasped = gripper_object_distance < grasp_distance_threshold
    is_held = is_lifted & is_grasped
    
    # Reward only when object is held
    return (1 - torch.tanh(distance_to_goal / std)) * is_held.float()

def object_gripper_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward based on the distance between the gripper midpoint and the object.

    The gripper midpoint is computed from the two inner finger body positions
    (left_inner_finger, right_inner_finger) on the robot. Positions are made
    relative to the environment origins before computing Euclidean distance.
    The returned reward uses a tanh-kernel: r = 1 - tanh(distance / std).
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[asset_cfg.name]

    # Find body indices for the inner fingers (robust to list or single int)
    left_finger_id, _ = robot.find_bodies("left_inner_finger")
    right_finger_id, _ = robot.find_bodies("right_inner_finger")
    left_finger_idx = left_finger_id[0] if isinstance(left_finger_id, (list, tuple)) else left_finger_id
    right_finger_idx = right_finger_id[0] if isinstance(right_finger_id, (list, tuple)) else right_finger_id

    # Body positions in world frame: (num_envs, 3)
    left_finger_pos_w = robot.data.body_pos_w[:, left_finger_idx, :]
    right_finger_pos_w = robot.data.body_pos_w[:, right_finger_idx, :]

    # Gripper midpoint in world frame, then make relative to env_origins
    env_origins = getattr(env.scene, "env_origins", 0.0)
    tcp_position = (left_finger_pos_w + right_finger_pos_w) / 2.0 - env_origins

    # Object position (relative to env origins)
    object_position = object.data.root_pos_w - env_origins

    # Distance of gripper midpoint to object: (num_envs,)
    dist = torch.norm(object_position - tcp_position, dim=1)

    return 1 - torch.tanh(dist / std)


def grasping_reward(
    env: "ManagerBasedRLEnv",
    grasp_distance_threshold: float = 0.08,
    gripper_closed_threshold: float = 0.5,
    action_term_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
) -> torch.Tensor:
    """Reward when gripper is commanded closed and the gripper is near the object.

    Returns 1.0 when both conditions are met, 0.0 otherwise. This is intended as
    a sparse shaping reward to reinforce successful grasp commands.
    """
    # object and gripper scenes
    object = env.scene[object_cfg.name]
    gripper = env.scene[gripper_cfg.name]

    # compute gripper midpoint
    gripper_body_ids = gripper_cfg.body_ids
    gripper_positions = gripper.data.body_pos_w[:, gripper_body_ids, :]
    gripper_center = gripper_positions.mean(dim=1)

    object_pos = object.data.root_pos_w
    gripper_object_distance = torch.norm(gripper_center - object_pos, dim=1)

    # check distance condition
    close = gripper_object_distance < float(grasp_distance_threshold)

    # attempt to read raw gripper action (fallback to env.action_manager.action last column)
    try:
        term = env.action_manager.get_term(action_term_name)
        gripper_raw = term.raw_actions.squeeze(-1)
    except Exception:
        gripper_raw = env.action_manager.action[:, -1]

    # closed if raw action exceeds threshold
    closed = gripper_raw >= float(gripper_closed_threshold)

    return (close & closed).float()
