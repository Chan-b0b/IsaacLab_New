# Pick-and-place task configuration for Realman RM75-6FB-V (with camera and two-finger gripper)
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from realman.robot_rm75 import RM75_6FB_V_CFG


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for actor: visual features + minimal state."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint", 
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint", 
                                                                       "right_inner_finger_joint"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint",
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint",
                                                                       "right_inner_finger_joint"])},
        )
        # Visual features from wrist camera using pretrained ResNet18
        wrist_cam_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "model_device": "cuda:0",
            },
        )
        
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic: privileged state information."""

        # Full joint state
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint", 
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint", 
                                                                       "right_inner_finger_joint"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint",
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint",
                                                                       "right_inner_finger_joint"])},
        )
        wrist_cam_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "model_device": "cuda:0",
            },
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RealmanPickPlaceEnvCfg(LiftEnvCfg):
    # Override observations with camera-enabled configuration
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Use Realman RM75-6FB-V robot (includes camera links)
        self.scene.robot = RM75_6FB_V_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Joint-space control over 7 arm joints
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )

        # Binary gripper control following original Robotiq mimic multipliers
        # Mimic multipliers already account for opposite axis directions
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint", 
                         "right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint"],
            open_command_expr={
                "finger_joint": 0.725,                    # Master joint
                "left_inner_knuckle_joint": 0.725,        # mimic ×1
                "right_inner_knuckle_joint": -0.725,      # mimic ×-1
                "right_outer_knuckle_joint": -0.725,      # mimic ×-1
                "left_inner_finger_joint": -0.725,        # mimic ×-1
                "right_inner_finger_joint": 0.725,        # mimic ×1
            },
            close_command_expr={
                "finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
            },
        )

        # End-effector reference
        self.commands.object_pose.body_name = "link_7"
        self.commands.object_pose.debug_vis = False  # Disable arrow visualization
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.55),
            pos_y=(-0.1, 0.1),
            pos_z=(0.20, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )

        # Cube object on the table
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Camera attached to camera_link with adjusted view
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link_7/camera_rolink/camera_link/cam",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                    pos=(0, 0.0, 0.01), 
                    rot=(0.6408,0.2988,-0.2988,0.6408),
                convention="world"
            ),
        )

        # Fix reward joint velocity regex to match RM75 joints instead of openarm
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["joint_[1-7]"]
        )
        
        # Add penalty for low end-effector height to prevent ground contact
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        import realman.rewards as realman_rewards
        import realman.termination as realman_termination
        
        self.rewards.gripper_height_penalty = RewTerm(
            func=realman_rewards.gripper_height_penalty,
            weight=-10.0,  # Negative weight for penalty
            params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names="left_outer_knuckle")},  # Use left_outer_knuckle_link as gripper reference
        )
        
        # Replace default object_goal_tracking with grasp-conditioned version
        self.rewards.object_goal_tracking = RewTerm(
            func=realman_rewards.object_goal_tracking_with_grasp,
            params={
                "std": 0.3,
                "minimal_height": 0.04,
                "grasp_distance_threshold": 0.08,  # Object must be within 8cm of gripper
                "command_name": "object_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=16.0,
        )
        
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=realman_rewards.object_goal_tracking_with_grasp,
            params={
                "std": 0.05,
                "minimal_height": 0.04,
                "grasp_distance_threshold": 0.08,
                "command_name": "object_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=5.0,
        )
        
        # Add termination for object falling out of bounds
        self.terminations.object_out_of_bounds = DoneTerm(
            func=realman_termination.object_out_of_bounds,
            params={"maximum_distance": 0.4, "asset_cfg": SceneEntityCfg("object")},  # Terminate if object moves > 0.5m from spawn
        )


        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from realman.curriculum import modify_object_spawn_range
        self.curriculum.object_spawn_range = CurrTerm(
            func=modify_object_spawn_range,
            params={
                "event_term_name": "reset_object_position",
                "start_range": {"x": (0.1, 0.1), "y": (-0, 0)},
                "end_range": {"x": (0, 0.2), "y": (-0.25, 0.25)},
                "num_steps": 20000
            }
        )



        # Camera rendering settings (optimized for headless training)
        self.num_rerenders_on_reset = 3  # Ensure textures are loaded
        self.sim.render.antialiasing_mode = "FXAA"  # Use FXAA for headless
        
        # Enable ray-tracing for proper headless camera rendering
        self.sim.enable_scene_query_support = True
        
        # Add curriculum to gradually increase object spawn range

        # Increase PhysX GPU memory for large-scale training (9096+ envs)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**15  # 32768 (was default ~16k)
        self.sim.physx.gpu_max_rigid_contact_count = 2**23  # 8.4M contacts
        self.sim.physx.gpu_max_rigid_patch_count = 2**21  # 2M patches
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21  # 2M pairs
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25  # 33M
        self.sim.physx.gpu_collision_stack_size = 2**28  # 268MB
        
        # Improve velocity accuracy for manipulation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solver_velocity_iteration_count = 1  # Increase from 0 to 1

        # Add termination for gripper ground contact

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link_7",
            debug_vis=False,
            visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link_7",
                    name="end_effector",
                ),
            ],
        )


@configclass
class RealmanPickPlaceEnvCfg_PLAY(RealmanPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Keep training-scale layout unless adjusted by the user
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        # Disable observation corruption for evaluation
        self.observations.policy.enable_corruption = False

#python scripts/rsl_rl/train.py --task Realman-RM75-PickPlace-Train-v0 --num_envs 24 --max_iterations 50000 --enable_cameras --headless