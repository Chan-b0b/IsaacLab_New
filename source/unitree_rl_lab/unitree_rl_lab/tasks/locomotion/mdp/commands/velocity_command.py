from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityHeightCommand(UniformVelocityCommand):
    """Command generator for velocity with height and waist pitch.
    
    This generates a command with 6 dimensions:
    - Linear velocity x, y, z (m/s)
    - Walking binary flag (0 or 1)
    - Target height (m)
    - Waist pitch (rad) - fixed value
    """

    cfg: UniformVelocityHeightCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # create buffer for height command
        self.height_command = torch.zeros(self.num_envs, device=self.device)
        
        # create buffer for walking binary (separate from standing)
        self.is_walking_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # create full command buffer: [x_vel, y_vel, z_vel, walking_binary, height, waist_pitch]
        self._full_command = torch.zeros(self.num_envs, 6, device=self.device)
        # set fixed waist pitch
        self._full_command[:, 5] = self.cfg.waist_pitch

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tWalking probability: {self.cfg.rel_walking_envs}\n"
        msg += f"\tHeight range: {self.cfg.ranges.height}\n"
        msg += f"\tWaist pitch (fixed): {self.cfg.waist_pitch}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The full command. Shape is (num_envs, 6).
        
        Returns:
            Command tensor with [x_vel, y_vel, z_vel, walking_binary, height, waist_pitch]
        """
        return self._full_command

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity and height commands for the specified environments."""
        # call parent to resample velocity
        super()._resample_command(env_ids)
        
        # sample height commands
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids] = r.uniform_(*self.cfg.ranges.height)
        
        # sample walking binary (independent from standing)
        self.is_walking_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_walking_envs

    def _update_command(self):
        """Update the full command buffer."""
        # call parent to update velocity (this handles standing envs setting velocities to 0)
        super()._update_command()
        
        # update full command buffer
        # [0:2] - x, y velocity (already handled by parent for standing envs)
        self._full_command[:, 0:2] = self.vel_command_b[:, 0:2]
        # [2] - z velocity (angular velocity, already handled by parent for standing envs)
        self._full_command[:, 2] = self.vel_command_b[:, 2]
        # [3] - walking binary (independently sampled, 1 = use gait, 0 = no gait)
        self._full_command[:, 3] = self.is_walking_env.float()
        # [4] - height
        self._full_command[:, 4] = self.height_command
        # [5] - waist pitch (already set in __init__, fixed value)


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class UniformVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Configuration for velocity command with height and waist pitch.
    
    This command outputs: (x_vel, y_vel, z_vel, walking_binary, height, waist_pitch)
    where waist_pitch is fixed to -0.2.
    """
    
    class_type: type = UniformVelocityHeightCommand
    """The class type for the command generator."""

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Extended ranges for velocity and height commands."""
        
        height: tuple[float, float] = MISSING
        """Range for the target height command (in m)."""
    
    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and height commands."""
    
    waist_pitch: float = -0.2
    """Fixed waist pitch angle (in rad). Defaults to -0.2."""
    
    rel_walking_envs: float = 0.2
    """The sampled probability of environments that should use walking gait. Defaults to 0.8."""
    
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    