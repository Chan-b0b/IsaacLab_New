"""Training configuration for H1-2 velocity tracking task."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from .velocity_env_cfg import RobotEnvCfg


@configclass
class CommandsTrainCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0,
        rel_walking_envs=0.95,
        heading_command=False,
        debug_vis=True,
        waist_pitch=-0.2,
        ranges=mdp.UniformVelocityHeightCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            height=(0.6, 0.95),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.4, 0.4)
        )
    )


@configclass
class RewardsTrainCfg:
    """Reward terms for training."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch.*",
    #                 ".*_shoulder_roll.*",
    #                 ".*_shoulder_yaw.*",
    #                 ".*_elbow.*"
    #                 ],
    #         )
    #     },
    # )
    
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])},
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )


    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    base_height = RewTerm(
        func=mdp.track_height_exp, 
        weight=1.0, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25),
            "sensor_cfg": SceneEntityCfg("height_scanner")
        }
    )

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=3,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={
            "threshold": 500,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    
    # -- gait control
    # maintain_default_pose_when_no_gait = RewTerm(
    #     func=mdp.maintain_default_pose_when_no_gait,
    #     weight=-2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #     },
    # )


@configclass
class RobotTrainEnvCfg(RobotEnvCfg):
    """Training configuration for H1-2 velocity tracking environment."""

    # Override commands with training-specific commands
    commands: CommandsTrainCfg = CommandsTrainCfg()
    
    # Override rewards with training-specific rewards
    rewards: RewardsTrainCfg = RewardsTrainCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        # Training-specific settings
        self.scene.num_envs = 4096
        self.episode_length_s = 20.0
        
        # Override push_robot event parameters for training
        self.events.push_robot.interval_range_s = (2.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}


@configclass
class RobotTrainPlayEnvCfg(RobotTrainEnvCfg):
    """Play configuration for H1-2 velocity training environment."""
    
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

