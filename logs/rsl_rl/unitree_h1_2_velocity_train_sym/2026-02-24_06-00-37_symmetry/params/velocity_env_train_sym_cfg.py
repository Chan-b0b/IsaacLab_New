"""Training configuration for H1-2 velocity tracking task with symmetry."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from .velocity_env_train_cfg import RobotTrainEnvCfg, CommandsTrainCfg, RewardsTrainCfg


@configclass
class RewardsTrainSymCfg(RewardsTrainCfg):
    """Reward terms for training with symmetry - inherits all from base and adds symmetry."""
    
    # -- symmetry reward
    # joint_mirror = RewTerm(
    #     func=mdp.joint_mirror,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [
    #             ["left_hip_yaw_joint", "right_hip_yaw_joint"],
    #             ["left_hip_roll_joint", "right_hip_roll_joint"],
    #             ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    #             ["left_knee_joint", "right_knee_joint"],
    #             ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
    #             ["left_ankle_roll_joint", "right_ankle_roll_joint"],
    #             ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
    #             ["left_shoulder_roll_joint", "right_shoulder_roll_joint"],
    #             ["left_shoulder_yaw_joint", "right_shoulder_yaw_joint"],
    #             ["left_elbow_pitch_joint", "right_elbow_pitch_joint"],
    #         ],
    #     },
    # )


@configclass
class RobotTrainSymEnvCfg(RobotTrainEnvCfg):
    """Training configuration for H1-2 velocity tracking environment with symmetry."""
    
    # Override rewards with symmetry-specific rewards
    rewards: RewardsTrainSymCfg = RewardsTrainSymCfg()


@configclass
class RobotTrainSymPlayEnvCfg(RobotTrainSymEnvCfg):
    """Play configuration for H1-2 velocity training environment with symmetry."""
    
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        # Play-specific settings
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
