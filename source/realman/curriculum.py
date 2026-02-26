"""Custom curriculum functions for Realman RM75 pick-and-place task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_expert_percentage(ManagerTermBase):
    """Curriculum that decreases expert demonstration percentage over training.
    
    Starts at 100% expert demonstrations and decreases to 0% over a specified number of steps.
    This allows the RL agent to bootstrap learning from expert trajectories before taking over.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Store curriculum parameters
        self._start_percentage = cfg.params.get("start_percentage", 1.0)
        self._end_percentage = cfg.params.get("end_percentage", 0.0)
        self._num_steps = cfg.params["num_steps"]
        
        # Initialize expert percentage for each environment
        if not hasattr(env, "_expert_percentage"):
            env._expert_percentage = self._start_percentage

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_percentage: float = 1.0,
        end_percentage: float = 0.0,
        num_steps: int = 25000,
    ):
        """Linearly decrease expert percentage based on training progress."""
        
        # Calculate interpolation factor (0.0 to 1.0)
        progress = min(env.common_step_counter / num_steps, 1.0)
        
        # Linear decay from start to end percentage
        current_percentage = start_percentage - progress * (start_percentage - end_percentage)
        
        # Store in environment for access by training script
        env._expert_percentage = float(current_percentage)
        
        return current_percentage


class modify_object_spawn_range(ManagerTermBase):
    """Curriculum that gradually increases the object spawn range over time."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get the event term configuration
        event_term_name = cfg.params["event_term_name"]
        self._event_term_cfg = env.event_manager.get_term_cfg(event_term_name)
        
        # Store start and end ranges
        self._start_range = cfg.params["start_range"]
        self._end_range = cfg.params["end_range"]
        self._num_steps = cfg.params["num_steps"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        event_term_name: str,
        start_range: dict,
        end_range: dict,
        num_steps: int,
    ):
        """Linearly interpolate spawn range based on training progress."""
        
        # Calculate interpolation factor (0.0 to 1.0)
        progress = min(env.common_step_counter / num_steps, 1.0)
        
        # Interpolate each dimension
        new_pose_range = {}
        for key in start_range.keys():
            if key in end_range:
                start_min, start_max = start_range[key]
                end_min, end_max = end_range[key]
                
                # Linear interpolation
                current_min = start_min + progress * (end_min - start_min)
                current_max = start_max + progress * (end_max - start_max)
                new_pose_range[key] = (current_min, current_max)
        
        # Update the event term configuration
        self._event_term_cfg.params["pose_range"] = new_pose_range
        env.event_manager.set_term_cfg(event_term_name, self._event_term_cfg)
        
        return new_pose_range
