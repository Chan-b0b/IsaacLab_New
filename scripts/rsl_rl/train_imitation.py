"""Train with expert demonstrations that gradually fade out.

This script combines expert trajectories from a state machine with RL training.
The expert percentage decreases over training via curriculum learning.

Usage:
    python scripts/rsl_rl/train_with_expert.py --task Realman-RM75-PickPlace-Train-v0 --num_envs 256
"""

# """Launch Isaac Sim Simulator first."""
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach on port 5678...")
debugpy.wait_for_client()
print("Debugger attached!")

import argparse
import argcomplete

    
# Defer isaaclab_tasks imports until after SimulationApp is created

# Ensure project source and scripts are importable
import sys
import pathlib

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
src_dir = str(pathlib.Path(__file__).parent.parent.parent / "source")
repo_root = str(pathlib.Path(__file__).parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# add repo `source/` for project packages and top-level workspace for local imports

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
from datetime import datetime
import cli_args  # isort: skip

def main():
    """Train RL agent with expert demonstrations."""
    # Parse args and start SimulationApp (deferred to avoid pre-loading Omni modules)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Train RL agent with expert bootstrapping.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum iterations for training.")
    parser.add_argument("--disable_fabric", type=bool, default=None, help="Disable fabric simulation.")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    # Create the SimulationApp
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    from list_envs import import_packages  # noqa: F401
    import realman  # noqa: F401

    from scripts.realman_pick_place_sm import PickPlaceSm
    from isaaclab.sensors import FrameTransformer
    from isaaclab.assets import RigidObject
    
    # Parse configuration
    # Defer isaaclab_tasks imports until after SimulationApp has been created
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
    from isaaclab_tasks.utils import get_checkpoint_path
    from rsl_rl.runners import OnPolicyRunner
    from scripts.rsl_rl.imitation_runner import ExpertImitationRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Get agent configuration
    # Extract runner cfg entry point from the gym spec kwargs (preferred)
    spec = gym.spec(args_cli.task)
    agent_cfg_entry_point = None
    try:
        agent_cfg_entry_point = spec.kwargs.get("rsl_rl_cfg_entry_point")
    except Exception:
        agent_cfg_entry_point = None

    # Fallback: try a heuristic replace of the entry_point
    if not agent_cfg_entry_point:
        env_entry_point = spec.entry_point
        agent_cfg_entry_point = env_entry_point.replace("_env_cfg", "_ppo_runner_cfg")

    # Load agent config using the project's utilities; fall back to importlib resolver
    runner_cfg_obj = None
    try:
        runner_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    except Exception:
        # try a stripped-vX variant (common registry naming)
        import re
        stripped = re.sub(r"-v\\d+$", "", args_cli.task or "")
        if stripped and stripped != args_cli.task:
            try:
                runner_cfg_obj = load_cfg_from_registry(stripped, "rsl_rl_cfg_entry_point")
            except Exception:
                runner_cfg_obj = None

    if isinstance(runner_cfg_obj, dict):
        raise RuntimeError("Runner config resolved to dict; expected a config class.")

    if runner_cfg_obj is not None:
        agent_cfg = runner_cfg_obj() if callable(runner_cfg_obj) else runner_cfg_obj
    else:
        # Fallback: resolve via import-based resolver using the spec-derived entry point
        import importlib
        if isinstance(agent_cfg_entry_point, str) and ":" in agent_cfg_entry_point:
            mod_name, attr_name = agent_cfg_entry_point.split(":", 1)
            mod = importlib.import_module(mod_name)
            agent_cfg = getattr(mod, attr_name)()
        else:
            raise RuntimeError(f"Unable to resolve agent config for task {args_cli.task!r}")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Wrap environment for rsl-rl compatibility (vectorization/clip actions)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", True))
    
    # Create a log directory for the runner so store_code_state() has a valid path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_task_name = "realman_ikRL_hybrid"
    log_root_path = os.path.join(os.getcwd(), "logs", "rsl_rl", safe_task_name)
    log_dir = os.path.join(log_root_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


    runner = ExpertImitationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)
    
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Create state machine for expert
    print("[INFO] Initializing expert state machine...")
    expert_sm = PickPlaceSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        robot = env.unwrapped.scene["robot"]
    )
    
    # Get number of environments for expert based on curriculum
    def get_expert_env_count():
        """Get number of environments to use expert based on curriculum."""
        expert_percentage = getattr(env.unwrapped, "_expert_percentage", 1.0)
        num_expert_envs = int(env.unwrapped.num_envs * expert_percentage)
        return num_expert_envs
    
    # Training loop with expert integration
    print("\n" + "="*80)
    print("[INFO] Starting hybrid expert-RL training...")
    print("[INFO] Expert will gradually fade from 100% → 0% over 25k steps")
    print("[INFO] Early steps: Learn from expert demonstrations")
    print("[INFO] Later steps: Rely on task rewards from environment")
    print("="*80 + "\n")
    
    # Expert provider for the runner: returns (indices, actions) to overwrite
    # sampled actions for the given step. Indices can be a LongTensor or a
    # boolean mask. Actions should match the action shape for the selected envs.
    
    def expert_provider(obs, step_idx, tot_timesteps):
        num_expert_envs = get_expert_env_count()
        if num_expert_envs <= 0:
            return None, None, None, None
        with torch.inference_mode():
            robot = env.unwrapped.scene["robot"]
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()

            left_finger_id, _ = robot.find_bodies("left_inner_finger")
            right_finger_id, _ = robot.find_bodies("right_inner_finger")
            left_finger_idx = left_finger_id[0] if isinstance(left_finger_id, list) else left_finger_id
            right_finger_idx = right_finger_id[0] if isinstance(right_finger_id, list) else right_finger_id

            left_finger_pos_w = robot.data.body_pos_w[:, left_finger_idx, :]
            right_finger_pos_w = robot.data.body_pos_w[:, right_finger_idx, :]
            tcp_position = (left_finger_pos_w + right_finger_pos_w) / 2.0 - env.unwrapped.scene.env_origins

            object: RigidObject = env.unwrapped.scene["object"]
            object_position = object.data.root_pos_w.clone() - env.unwrapped.scene.env_origins
            object_orientation = object.data.root_quat_w.clone()

            target_position = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
            target_position += env.unwrapped.scene["robot"].data.root_pos_w - env.unwrapped.scene.env_origins
            ao = expert_sm.default_orientation_quat.to(env.unwrapped.device)
            target_orientation = ao[:, [3, 0, 1, 2]]

            expert_actions = expert_sm.compute(
                torch.cat([tcp_position, tcp_orientation], dim=-1),
                torch.cat([object_position, object_orientation], dim=-1),
                torch.cat([target_position, target_orientation], dim=-1),
            )
            # Select expert envs based on the expert percentage per-environment
            expert_pct = getattr(env.unwrapped, "_expert_percentage", 1.0)
            if expert_pct >= 1.0:
                idx = torch.arange(env.unwrapped.num_envs, device=expert_actions.device, dtype=torch.long)
                expert_idx = idx
                expert_joint_actions = expert_actions[idx]
                expert_target_pos = target_position[idx]
                expert_target_ori = target_orientation[idx]
                return expert_idx, expert_joint_actions, expert_target_pos, expert_target_ori

            if expert_pct <= 0.0:
                return None, None, None, None

            # Bernoulli selection per environment (on the action tensor device)
            # n_envs = env.unwrapped.num_envs
            # rand_vals = torch.rand(n_envs, device=expert_actions.device)
            # mask = rand_vals < float(expert_pct)
            # idx = mask.nonzero(as_tuple=False).squeeze(-1).long()
            # if idx.numel() == 0:
            #     # ensure at least one expert when pct>0 by sampling one index
            #     idx = torch.randint(0, n_envs, (1,), device=expert_actions.device, dtype=torch.long)
            # return idx, expert_actions[idx]

            return torch.arange(num_expert_envs), expert_actions[:num_expert_envs], target_position[:num_expert_envs], target_orientation[:num_expert_envs]
    
    # Log expert percentage
    def log_expert_percentage():
        expert_pct = getattr(env.unwrapped, "_expert_percentage", 1.0) * 100
        num_expert = get_expert_env_count()
        num_rl = env.unwrapped.num_envs - num_expert
        runner.writer.add_scalar("Expert/percentage", expert_pct, runner.tot_timesteps)
        runner.writer.add_scalar("Expert/num_expert_envs", num_expert, runner.tot_timesteps)
        runner.writer.add_scalar("Expert/num_rl_envs", num_rl, runner.tot_timesteps)
        
        if runner.current_learning_iteration % 10 == 0:
            print(f"[Expert] {expert_pct:.1f}% ({num_expert}/{env.unwrapped.num_envs} envs) | "
                  f"RL: {100-expert_pct:.1f}% ({num_rl}/{env.unwrapped.num_envs} envs)")
    
    # Override learn to add logging
    original_learn = runner.learn
    
    def learn_with_logging(*args, **kwargs):
        # Inject logging callback
        # note: OnPolicyRunner may not define `log_interval`; avoid accessing it
        
        def log_with_expert():
            log_expert_percentage()
        
        # Store original log function
        runner._log_expert = log_with_expert
        
        # Call original learn (forward all args/kwargs)
        return original_learn(*args, **kwargs)
    
    # Monkey-patch learn function
    runner.learn = learn_with_logging
    
    # Add expert logging to the runner's log function
    original_log = runner.log
    
    def log_with_expert_metrics(locs, width=80, pad=35):
        original_log(locs, width, pad)
        if hasattr(runner, "_log_expert"):
            runner._log_expert()
    
    runner.log = log_with_expert_metrics
    
    # Run training — pass the expert provider so the runner can inject expert actions
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
        expert_provider=expert_provider,
        expert_reset=lambda idx: expert_sm.reset_idx(idx),
    )
    # Close environment and SimulationApp
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
