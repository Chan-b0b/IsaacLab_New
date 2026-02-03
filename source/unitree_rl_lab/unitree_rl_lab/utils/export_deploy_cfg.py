import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    
    # Get action joint names from the JointPositionAction configuration
    action_term = env.action_manager._terms.get("JointPositionAction")
    if action_term is not None:
        action_joint_names = action_term.cfg.joint_names
        # Map action joint names to indices in joint_sdk_names (SDK order)
        action_joint_ids_in_sdk, _ = resolve_matching_names(action_joint_names, joint_sdk_names, preserve_order=True)
        # Map action joint names to indices in Isaac Sim order
        action_joint_ids_in_sim, _ = resolve_matching_names(action_joint_names, asset.data.joint_names, preserve_order=True)
    else:
        # Fallback if no JointPositionAction
        action_joint_ids_in_sdk, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
        action_joint_ids_in_sim = action_joint_ids_in_sdk

    cfg = {}  # noqa: SIM904
    # joint_ids_map: Sequential indices for action joints (0, 1, 2, ..., num_action_joints-1)
    cfg["joint_ids_map"] = list(range(len(action_joint_ids_in_sdk)))
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    
    # Extract stiffness, damping, default_joint_pos for ALL joints in SDK order
    # Map all SDK joints to Isaac Sim order
    all_joint_ids_in_sim, _ = resolve_matching_names(joint_sdk_names, asset.data.joint_names, preserve_order=True)
    
    stiffness_sim = asset.data.default_joint_stiffness[0].detach().cpu().numpy()
    damping_sim = asset.data.default_joint_damping[0].detach().cpu().numpy()
    default_joint_pos_sim = asset.data.default_joint_pos[0].detach().cpu().numpy()
    joint_pos_limits_sim = asset.data.soft_joint_pos_limits[0].detach().cpu().numpy()
    
    # Reorder from Isaac Sim order to SDK order for all joints
    cfg["stiffness"] = [float(stiffness_sim[all_joint_ids_in_sim[i]]) for i in range(len(joint_sdk_names))]
    cfg["damping"] = [float(damping_sim[all_joint_ids_in_sim[i]]) for i in range(len(joint_sdk_names))]
    cfg["default_joint_pos"] = [float(default_joint_pos_sim[all_joint_ids_in_sim[i]]) for i in range(len(joint_sdk_names))]
    cfg["joint_pos_limits"] = [[float(joint_pos_limits_sim[all_joint_ids_in_sim[i], 0]), 
                                  float(joint_pos_limits_sim[all_joint_ids_in_sim[i], 1])] 
                                 for i in range(len(joint_sdk_names))]

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
