# Realman RM75-6FB-V articulation configuration (with two-finger gripper)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.materials import PhysicsMaterialCfg
RM75_6FB_V_URDF_PATH = "/home/maverick/Humanoid/IsaacLab_New/source/realman/rm_models/RM75/urdf/RM75-6FB-V/urdf/RM75-6FB-V-handE.urdf"

# 7-DOF arm with vision add-on and EG2-4C2 two-finger gripper
RM75_6FB_V_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=RM75_6FB_V_URDF_PATH,
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Pose with camera facing down toward table
        joint_pos={
            "joint_1": 0.0,         # Base rotation: neutral
            "joint_2": 0.2,        # Shoulder: tilt forward
            "joint_3": 0,         # Elbow: bend up
            "joint_4": 1.2,         # Wrist pitch 1: adjust angle
            "joint_5": 0,        # Wrist pitch 2: point downward
            "joint_6": 1.2,         # Wrist roll: neutral
            "joint_7": 0.0,         # End-effector roll: neutral
            # Robotiq Hand-E finger joints (prismatic)
            "robotiq_hande_left_finger_joint": 0.02,
            "robotiq_hande_right_finger_joint": 0.02,
        },
    ),
    actuators={
        "rm75_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            # Conservative limits; can be tuned per joint spec
            velocity_limit_sim=3.14,
            effort_limit_sim={
                "joint_[1-2]": 60.0,
                "joint_[3-4]": 30.0,
                "joint_[5-7]": 10.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_hande_.*"],
            velocity_limit_sim=2.0,
            effort_limit_sim=1000.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
