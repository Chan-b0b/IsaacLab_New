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

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
import realman.rewards as realman_rewards
import realman.termination as realman_termination
# import debugpy

# # Initialize debugpy for remote debugging
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached!")

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ImageCfg(ObsGroup):

        wrist_cam_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "model_device": "cuda:0",
            },
        # Visual features from wrist camera using pretrained ResNet18
        )
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for actor: visual features + minimal state."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint", 
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint", 
                                                                       "right_inner_finger_joint"])}, #13
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]", "finger_joint",
                                                                       "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                                                                       "right_outer_knuckle_joint", "left_inner_finger_joint",
                                                                       "right_inner_finger_joint"])}, #13
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"}) #7
        actions = ObsTerm(func=mdp.last_action) #8
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = True
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

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    Image: ImageCfg = ImageCfg()

@configclass
class RealmanPickPlaceEnvCfg(LiftEnvCfg):
    # Override observations with camera-enabled configuration
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2

        # Use Realman RM75-6FB-V robot (includes camera links)
        self.scene.robot = RM75_6FB_V_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Joint-space control over 7 arm joints
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
            clip={"joint_[1-7]": (-3.14, 3.14)},
        )

        # Binary gripper control following original Robotiq mimic multipliers
        # Mimic multipliers already account for opposite axis directions
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint", 
                         "right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint"],
            open_command_expr={
                "finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
            },
            close_command_expr={
                "finger_joint": 0.5,                    # Master joint
                "left_inner_knuckle_joint": 0.5,        # mimic ×1
                "right_inner_knuckle_joint": -0.5,      # mimic ×-1
                "right_outer_knuckle_joint": -0.5,      # mimic ×-1
                "left_inner_finger_joint": -0.2,        # mimic ×-1
                "right_inner_finger_joint": 0.2,        # mimic ×1
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
                scale=(0.7, 0.7, 0.7),
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
        self.rewards.action_rate.weight = -0.001  # Penalize large actions to encourage smoothness

        self.rewards.joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", 
                    joint_names=["joint_[1-7]"])},
    )
        self.rewards.reaching_object = RewTerm(
            func=realman_rewards.object_gripper_distance, 
            params={"asset_cfg": SceneEntityCfg("robot", 
                    body_names=["left_inner_finger", "right_inner_finger"]), 
                    "std": 0.1}, 
            weight=3)

        # Sparse grasping reward: when gripper commanded closed and near the object
        self.rewards.grasp_reward = RewTerm(
            func=realman_rewards.grasping_reward,
            params={
                "grasp_distance_threshold": 0.06,
                "gripper_closed_threshold": 0,
                "action_term_name": "gripper_action",
                "gripper_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            },
            weight=5.0,
        )
        
        self.rewards.gripper_height_penalty = RewTerm(
            func=realman_rewards.gripper_height_penalty,
            weight=-10.0,  # Negative weight for penalty
            params={"minimum_height": 0.03, 
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=["left_inner_finger",
                                             "right_inner_finger"])}
        )
        
        self.rewards.object_goal_tracking = RewTerm(
            func=realman_rewards.object_goal_tracking_with_grasp,
            params={
                "std": 0.05,
                "minimal_height": 0.04,
                "grasp_distance_threshold": 0.08,
                "command_name": "object_pose",
                "gripper_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            },
            weight=3.0,
        )
        
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.lifting_object = None
        
        # RewTerm(
        #     func=mdp.object_is_lifted, 
        #     params={"minimal_height": 0.1}, 
        #     weight=0.5
        # )

        # Add termination for object falling out of bounds
        self.terminations.object_out_of_bounds = DoneTerm(
            func=realman_termination.object_out_of_bounds,
            params={"maximum_distance": 0.15, "asset_cfg": SceneEntityCfg("object")},  # Terminate if object moves > 0.18m from spawn
        )


        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab.managers import EventTermCfg as EventTerm
        from realman.curriculum import modify_object_spawn_range, modify_expert_percentage
        from realman.physics import ApplyGravityTerm
        
        # Curriculum: gradually expand object spawn area
        self.curriculum.object_spawn_range = CurrTerm(
            func=modify_object_spawn_range,
            params={
                "event_term_name": "reset_object_position",
                "start_range": {"x": (-0.1,0.1), "y": (-0.15, 0.15)},
                "end_range": {"x": (-0.08,0.08), "y": (-0.15, 0.15)},
                "num_steps": 2000000
            }
        )
        
        # Curriculum: decrease expert percentage from 100% to 0% over 25k steps
        self.curriculum.expert_percentage = CurrTerm(
            func=modify_expert_percentage,
            params={
                "start_percentage": 0.99,  # Start with 99% expert
                "end_percentage": 0.0,     # Fade to 0% expert
                "num_steps": 2000000         # Complete fade-out by 1M steps
            }
        )
        self.curriculumaction_rate = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate", "weight": 2, "num_steps": 50000},
        )

        self.curriculumjoint_vel = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "joint_vel", "weight": 2, "num_steps": 50000},
        )

        self.events.finger_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
                "static_friction_range": (10.0, 10.0),
                "dynamic_friction_range": (10.0, 10.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        
        # Apply gravity compensation each step via an event term scheduled at the
        # simulation interval. Using an EventTerm with mode="interval" and
        # interval_range_s=(sim.dt, sim.dt) ensures the term is invoked every
        # physics step (when the environment calls event_manager.apply with dt).
        
        self.events.apply_gravity = EventTerm(
            func=ApplyGravityTerm,
            mode="interval",
            interval_range_s=(self.sim.dt, self.sim.dt),
            is_global_time=False,
        )

        # This uses a ManagerTerm so it will be executed by the env's managers
        # on their regular schedule. It is guarded and will be a no-op if the
        # simulation/robot does not expose the expected interfaces.

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

        # Optionally disable scene gravity (set to zero): persistent in env config
        # Uncomment to run gravity-free simulations by default.
        # self.sim.gravity = (0.0, 0.0, 0.0)

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

        tcp_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TCP_Marker",
            markers={
                "tcp": sim_utils.SphereCfg(
                    radius=0.05, # 1cm sphere
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        # Do not instantiate VisualizationMarkers here (stage may not exist during config parsing).
        # Store the config on the environment cfg; the visualizer should be created at runtime when the stage exists.
        self.tcp_visualizer_cfg = tcp_marker_cfg

        des_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EE_Desired",
            markers={
                "tcp": sim_utils.SphereCfg(
                    radius=0.05, # 1cm sphere
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                )
            },
        )
        # Do not instantiate VisualizationMarkers here (stage may not exist during config parsing).
        # Store the config on the environment cfg; the visualizer should be created at runtime when the stage exists.
        self.des_visualizer_cfg = des_marker_cfg
        
@configclass
class RealmanPickPlaceEnvCfg_PLAY(RealmanPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Reduce number of environments for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for evaluation
        self.observations.policy.enable_corruption = False
        # Make the UI interactive for playing
        self.sim.device = "cuda:0"
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

#python scripts/rsl_rl/train.py --task Realman-RM75-PickPlace-Train-v0 --num_envs 24 --max_iterations 50000 --enable_cameras --headless