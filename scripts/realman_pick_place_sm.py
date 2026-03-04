"""
Script to run Realman RM75 pick-and-place with state machine expert.

The state machine provides expert demonstrations that gradually decrease as RL improves.

Usage:
    python scripts/realman_pick_place_sm.py --num_envs 32
"""

"""Launch Isaac Sim Simulator first."""


# Note: AppLauncher and argument parsing are performed only when running
# this file as a script to avoid importing Omniverse modules at import-time
# (which can interfere when this module is imported by other scripts).

"""Rest everything else."""

# Debugger attach removed to avoid blocking imports and runtime.

from collections.abc import Sequence
import gymnasium as gym
import torch
import warp as wp
RigidObject = None
FrameTransformer = None
VisualizationMarkers = None
import copy
import math
quat_inv = None
matrix_from_quat = None
subtract_frame_transforms = None


def _ensure_isaac_imports():
    """Ensure isaaclab imports are available (lazy import)."""
    global RigidObject, FrameTransformer, VisualizationMarkers
    global quat_inv, matrix_from_quat, subtract_frame_transforms
    if RigidObject is None or FrameTransformer is None or VisualizationMarkers is None:
        from isaaclab.assets import RigidObject as _RigidObject
        from isaaclab.sensors import FrameTransformer as _FrameTransformer
        from isaaclab.markers import VisualizationMarkers as _VisualizationMarkers
        RigidObject, FrameTransformer, VisualizationMarkers = _RigidObject, _FrameTransformer, _VisualizationMarkers
    if quat_inv is None or matrix_from_quat is None or subtract_frame_transforms is None:
        from isaaclab.utils.math import quat_inv as _quat_inv, matrix_from_quat as _matrix_from_quat, subtract_frame_transforms as _subtract_frame_transforms
        quat_inv, matrix_from_quat, subtract_frame_transforms = _quat_inv, _matrix_from_quat, _subtract_frame_transforms
import numpy as np

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickPlaceSmState:
    """States for the pick and place state machine."""
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    MOVE_TO_TARGET = wp.constant(5)


class PickPlaceSmWaitTime:
    """Additional wait times (in s) for states before switching."""
    REST = wp.constant(0.01)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.5)
    GRASP_OBJECT = wp.constant(1)
    LIFT_OBJECT = wp.constant(0.5)
    MOVE_TO_TARGET = wp.constant(0.5)
    MOVE_INTERVAL = wp.constant(0.5)

@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    target_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    approach_above_offset: wp.array(dtype=wp.transform),
    default_orientation: wp.array(dtype=wp.quat),
    grasp_offset: wp.array(dtype=wp.transform),
    lift_height: float,
    max_grasp_distance: float,
    position_threshold: float,
    vertical_orientation: wp.array(dtype=wp.quat),
    horizontal_orientation: wp.array(dtype=wp.quat),
    wait_thresholds: wp.array(dtype=float),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    # If we've passed the grasp state, ensure EE hasn't drifted too far from the object.
    # If it has, reset to approach-above-object so the SM can re-acquire the object.
    ee_pos_check = wp.transform_get_translation(ee_pose[tid])
    obj_pos_check = wp.transform_get_translation(object_pose[tid])
    dist_check = wp.length(ee_pos_check - obj_pos_check)
    state = sm_state[tid]
    if state > PickPlaceSmState.GRASP_OBJECT:
        if dist_check > max_grasp_distance:
            sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    state = sm_state[tid]

    # decide next state
    if state == PickPlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= wait_thresholds[tid * 6 + 0]:
            sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
            
    elif state == PickPlaceSmState.APPROACH_ABOVE_OBJECT:
        # Move above the object using the positional offset and the saved orientation

        object_pos = wp.transform_get_translation(object_pose[tid])
        grasp_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2]+0.1) 
        des_ee_pose[tid] = wp.transform(grasp_pos, vertical_orientation[tid])
        grasp_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2]) 

        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= wait_thresholds[tid * 6 + 1]:
                sm_state[tid] = PickPlaceSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
                
    elif state == PickPlaceSmState.APPROACH_OBJECT:
        # Move to grasp position (use saved orientation)
        object_pos = wp.transform_get_translation(object_pose[tid])
        grasp_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2] + 0.04) 
        des_ee_pose[tid] = wp.transform(grasp_pos, vertical_orientation[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            grasp_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= wait_thresholds[tid * 6 + 2]:
                sm_state[tid] = PickPlaceSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
                
    elif state == PickPlaceSmState.GRASP_OBJECT:
        # Close gripper (maintain saved orientation)
        object_pos = wp.transform_get_translation(object_pose[tid])
        grasp_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2] + 0.04) 
        des_ee_pose[tid] = wp.transform(grasp_pos, vertical_orientation[tid])
        grasp_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2]) 
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= wait_thresholds[tid * 6 + 3]:
            sm_state[tid] = PickPlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
            
    elif state == PickPlaceSmState.LIFT_OBJECT:
        # Lift object up (maintain saved orientation)
        object_pos = wp.transform_get_translation(object_pose[tid])
        lifted_pos = wp.vec3(object_pos[0], object_pos[1], object_pos[2] + lift_height)
        gripper_state[tid] = GripperState.CLOSE

        t = sm_wait_time[tid] / wait_thresholds[tid * 6 + 4]
        if t > 1.0:
            t = 1.0
        ee_pos = wp.transform_get_translation(ee_pose[tid])
        orientation = (t * horizontal_orientation[tid] + (1.0 - t) * vertical_orientation[tid])

        des_ee_pose[tid] = wp.transform(lifted_pos, orientation)

        if sm_wait_time[tid] >= wait_thresholds[tid * 6 + 4]:
            sm_state[tid] = PickPlaceSmState.MOVE_TO_TARGET
            sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.MOVE_TO_TARGET:
        # Move to above target position (maintain saved orientation)
        target_pos = wp.transform_get_translation(target_pose[tid])
        above_target = wp.vec3(target_pos[0], target_pos[1], target_pos[2])
        # For transit to target, use horizontal orientation
        gripper_state[tid] = GripperState.CLOSE
        # Interpolate desired EE pose over the MOVE_TO_TARGET interval using sm_wait_time
        # t in [0,1] where 0 => above_target, 1 => current ee_pose (user requested order)
        t = sm_wait_time[tid] / wait_thresholds[tid * 6 + 5]
        if t > 1.0:
            t = 1.0
        ee_pos = wp.transform_get_translation(ee_pose[tid])
        interp_pos = (t * above_target + (1.0 - t) * ee_pos)
        des_ee_pose[tid] = wp.transform(interp_pos, horizontal_orientation[tid])

        # If EE is already very close to the interpolated target, snap desired pose to above_target
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            interp_pos,
            0.01,
        ):
            des_ee_pose[tid] = wp.transform(above_target, horizontal_orientation[tid])
                
    # (LOWER_TO_TARGET and RELEASE_OBJECT removed) -- lowering and release skipped

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickPlaceSm:
    """A simple state machine in a robot's task space to pick and place an object.
    
    States:
    1. REST: Robot at rest
    2. APPROACH_ABOVE_OBJECT: Move above object
    3. APPROACH_OBJECT: Move to grasp position
    4. GRASP_OBJECT: Close gripper
    5. LIFT_OBJECT: Lift object up
    6. MOVE_TO_TARGET: Move to target position
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.05, robot=None, randomize_wait: bool = True, max_grasp_distance: float = 0.15):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            position_threshold: Distance threshold for reaching waypoints (m).
            robot: Robot articulation for IK computation.
        """
        # save parameters
        self.dt = float(dt)
        # Ensure isaaclab imports are available (lazy import)
        try:
            _ensure_isaac_imports()
        except Exception:
            pass
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.robot = robot
        self.randomize_wait = bool(randomize_wait)
    

        
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # store last commanded target pose to detect updates and reset interpolation timer
        self._last_target_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # approach above object (10cm above) -- positional offset
        self.approach_above_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.approach_above_offset[:, 2] = 0.2  # 10cm above
        self.approach_above_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # default orientation (saved per-env) -- store only quaternion (x,y,z,w)
        self.default_orientation_quat = torch.zeros((self.num_envs, 4), device=self.device)
        # initialize to identity quaternion (x,y,z,w) = (0,0,0,1)
        self.default_orientation_quat[:, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # grasp offset (slightly above object surface)
        self.grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.grasp_offset[:, 2] = 0.02  # 2cm above object center
        self.grasp_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # lift height
        self.lift_height = 0.1  # 30cm lift
        # maximum allowed EE-object distance after grasp before resetting state (m)
        self.max_grasp_distance = float(max_grasp_distance)

        # Per-environment wait thresholds for state transitions.
        # Columns: [REST, APPROACH_ABOVE_OBJECT, APPROACH_OBJECT, GRASP_OBJECT, LIFT_OBJECT, MOVE_INTERVAL]
        base_waits = torch.tensor([0.01, 0.1, 0.5, 0.3, 0.1, 0.1], device=self.device)
        if hasattr(self, 'randomize_wait') and self.randomize_wait:
            # randomize ±25%
            rnd = 1.0 + (torch.rand((self.num_envs, 6), device=self.device) - 0.5) * 0.5
            self.wait_thresholds = base_waits.unsqueeze(0) * rnd
        else:
            self.wait_thresholds = base_waits.unsqueeze(0).repeat(self.num_envs, 1)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.approach_above_offset_wp = wp.from_torch(self.approach_above_offset, wp.transform)
        self.default_orientation_wp = wp.from_torch(self.default_orientation_quat, wp.quat)
        self.grasp_offset_wp = wp.from_torch(self.grasp_offset, wp.transform)
        # Flatten to 1D so kernel can index via (tid * 6 + idx)
        self.wait_thresholds_wp = wp.from_torch(self.wait_thresholds.reshape(-1).contiguous(), wp.float32)
        # Prepare vertical (downward) and horizontal orientation presets computed from default orientation
        # We'll compute simple Euler-based presets: keep roll/yaw from default, set pitch to -90deg for vertical

        # default_orientation_quat stored as (x,y,z,w) -- convert to (w,x,y,z)
        default_wxyz = torch.stack([self.default_orientation_quat[:, 3],
                                    self.default_orientation_quat[:, 0],
                                    self.default_orientation_quat[:, 1],
                                    self.default_orientation_quat[:, 2]], dim=1)

        r, p, y = self._quat_wxyz_to_euler(default_wxyz)
        # vertical: set pitch to +pi/2 (point down in robot frame)
        vertical_pitch = torch.full_like(p, math.pi)
        vertical_wxyz = self._euler_to_quat_wxyz(r, vertical_pitch, y)
        # horizontal: set pitch to 0 (level)
        horizontal_pitch = torch.full_like(p, math.pi*3/4)
        horizontal_wxyz = self._euler_to_quat_wxyz(r, horizontal_pitch, y)

        # convert back to (x,y,z,w) for warp (matches earlier convention)
        vertical_xyzw = torch.stack([vertical_wxyz[:, 1], vertical_wxyz[:, 2], vertical_wxyz[:, 3], vertical_wxyz[:, 0]], dim=1)
        horizontal_xyzw = torch.stack([horizontal_wxyz[:, 1], horizontal_wxyz[:, 2], horizontal_wxyz[:, 3], horizontal_wxyz[:, 0]], dim=1)

        self.vertical_orientation_quat = vertical_xyzw
        self.horizontal_orientation_quat = horizontal_xyzw
        self.vertical_orientation_wp = wp.from_torch(self.vertical_orientation_quat, wp.quat)
        self.horizontal_orientation_wp = wp.from_torch(self.horizontal_orientation_quat, wp.quat)
        # Setup IK solver if robot provided

        if self.robot is not None:
            from isaaclab.controllers.differential_ik import DifferentialIKController
            from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

            # IK configuration for Cartesian to joint conversion
            # use_relative_mode=False makes it return absolute joint positions
            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,  # Return absolute joint positions
                ik_method="dls",
                ik_params={"lambda_val": 0.2, "k_val": 0.5}
            )

            # Get arm joint indices and ee body id
            self.arm_joint_ids, _ = self.robot.find_joints("joint_[1-7]")
            self.ee_body_id, _ = self.robot.find_bodies("link_7")

            # Instantiate the baseline differential IK controller
            # self.ik_controller = DifferentialIKController(ik_cfg, num_envs, device)

            from differential_ik_weight import DifferentialIKControllerWithWeight
            
            self.ik_controller = DifferentialIKControllerWithWeight(ik_cfg, num_envs, device, robot=self.robot, arm_joint_ids=self.arm_joint_ids)


            # capture initial pose and home positions
            self._capture_initial_pose()

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine."""
        # reset state machine for given env ids
        self.sm_state[env_ids if env_ids is not None else slice(None)] = 0
        self.sm_wait_time[env_ids if env_ids is not None else slice(None)] = 0.0
        
        # Reset IK controller for these environments
        if hasattr(self, 'ik_controller'):
            self.ik_controller.reset(env_ids)

        if hasattr(self, 'home_joint_pos') and self.home_joint_pos is not None:
            idx = self._normalize_env_idx(env_ids)
            try:
                # env_ids may be a slice or tensor/list
                if idx is slice(None):
                    self.robot.set_joint_position_target(self.home_joint_pos, joint_ids=self.arm_joint_ids)
                else:
                    self.robot.set_joint_position_target(self.home_joint_pos[idx], joint_ids=self.arm_joint_ids)
            except Exception:
                pass

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, target_pose: torch.Tensor):
        """Compute the desired joint angles and gripper state.
        
        Args:
            ee_pose: Current end-effector pose (N, 7) [xyz, wxyz]
            object_pose: Object pose (N, 7) [xyz, wxyz]
            target_pose: Target pose (N, 7) [xyz, wxyz]
            
        Returns:
            Desired joint angles (N, 8) [joint1-7, gripper_state]
        """
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose_warp = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose_warp = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        target_pose_warp = target_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose_warp.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose_warp.contiguous(), wp.transform)
        target_pose_wp = wp.from_torch(target_pose_warp.contiguous(), wp.transform)

        # run state machine to get desired Cartesian pose
        # If the external command (target_pose) changed since last call, reset the interpolation timer
        try:
            # target_pose is in (x,y,z,w) order already at this stage
            delta = torch.norm(target_pose - self._last_target_pose, dim=1)
            changed = delta > 1e-6
            if changed.any():
                # Only reset interpolation timer for envs that are currently in MOVE_TO_TARGET
                # PickPlaceSmState.MOVE_TO_TARGET corresponds to integer 5
                try:
                    move_mask = (self.sm_state == 5) & changed
                except Exception:
                    move_mask = changed
                if move_mask.any():
                    self.sm_wait_time[move_mask] = 0.0
                    # update warp buffer only if we changed sm_wait_time
                    self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
                # always update last command cache for changed envs
                self._last_target_pose[changed] = target_pose[changed]
        except Exception:
            pass

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                target_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.approach_above_offset_wp,
                self.default_orientation_wp,
                self.grasp_offset_wp,
                self.lift_height,
                self.max_grasp_distance,
                self.position_threshold,
                self.vertical_orientation_wp,
                self.horizontal_orientation_wp,
                self.wait_thresholds_wp,
            ],
            device=self.device,
        )

        try:
            wp.synchronize()
        except Exception:
            pass

        # convert transformations back to (w, x, y, z)
        des_ee_pose_cart = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        
        # Convert Cartesian pose to joint angles using IK
        if self.robot is not None and hasattr(self, 'ik_controller'):
            # Set IK target - already in robot base frame since inputs are relative to env_origins
            # and robot bases are at env_origins with identity orientation
            self.ik_controller.set_command(des_ee_pose_cart)
            joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]

            # 2. Identify body indices (handling potential list format)
            def get_idx(id_val): return id_val[0] if isinstance(id_val, (list, torch.Tensor)) else id_val

            left_idx = get_idx(self.robot.find_bodies("left_inner_finger")[0])
            right_idx = get_idx(self.robot.find_bodies("right_inner_finger")[0])
            ee_idx = get_idx(self.ee_body_id)

            # 3. Get World Frame positions
            left_pos_w = self.robot.data.body_pos_w[:, left_idx, :]
            right_pos_w = self.robot.data.body_pos_w[:, right_idx, :]
            ee_body_pos_w = self.robot.data.body_pos_w[:, ee_idx, :]
            ee_quat_w = self.robot.data.body_quat_w[:, ee_idx, :]

            # TCP is the midpoint between fingers
            tcp_pos_w = (left_pos_w + right_pos_w) / 2.0

            # 4. Convert current TCP position/orientation to Robot Base Frame for the IK input
            tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, tcp_pos_w, ee_quat_w
            )

            body_pos_b, body_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, ee_body_pos_w, ee_quat_w)
            
            # 5. Jacobian Math (Shift to TCP and Rotate to Base Frame)
            # Get Jacobian for the EE body origin (expressed in World Frame)
            jacobian_full = self.robot.root_physx_view.get_jacobians()
            jacobian_body_w = jacobian_full[:, ee_idx, :, self.arm_joint_ids]

            Jv_w = jacobian_body_w[:, 0:3, :] 
            Jw_w = jacobian_body_w[:, 3:6, :]

            # Shift linear part from body origin to TCP in World Frame: Jp = Jv - skew(r) @ Jw
            r_w = tcp_pos_w - ee_body_pos_w 

            def skew_matrix(vec):
                N = vec.shape[0]
                S = torch.zeros((N, 3, 3), device=vec.device, dtype=vec.dtype)
                S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = -vec[:, 2], vec[:, 1], vec[:, 2]
                S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = -vec[:, 0], -vec[:, 1], vec[:, 0]
                return S

            Jp_tcp_w = Jv_w - torch.matmul(skew_matrix(r_w), Jw_w)

            # Rotate Jacobian parts from World to Base Frame
            R_w2b = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
            J_linear_b = torch.matmul(R_w2b, Jp_tcp_w)
            J_angular_b = torch.matmul(R_w2b, Jw_w)
            jacobian_tcp = torch.cat([J_linear_b, J_angular_b], dim=1)

            # 6. Compute IK - returns target joint positions
            # joint_pos_des = self.ik_controller.compute(
            #     tcp_pos_b, tcp_quat_b, jacobian_tcp, joint_pos
            # )

            joint_pos_des = self.ik_controller.compute(
                tcp_pos_b, tcp_quat_b, jacobian_tcp, joint_pos
            )
            
            gravity_forces = self.robot.root_physx_view.get_gravity_compensation_forces()

            self.robot.set_joint_effort_target(gravity_forces[:, self.arm_joint_ids], joint_ids=self.arm_joint_ids)

            # Set current EE pose for IK to TCP (position) and use link_7 orientation
            # ee_pos_b = tcp_pos_b
            # ee_quat_b = body_quat_b

            # # Compute IK - returns absolute joint positions (target)
            # joint_pos_des = self.ik_controller.compute(
            #     ee_pos_b, ee_quat_b, jacobian_tcp, joint_pos
            # )

            # The environment's `JointPositionAction` applies: processed = raw * scale + offset
            # Our IK returned absolute desired joint positions (processed). Convert back to raw:
            # raw = (processed - offset) / scale
            # Use robot default joint positions as offset and the known scale (0.5 in env cfg)
            scale = 0.5
            offset = self.robot.data.default_joint_pos[:, self.arm_joint_ids]
            raw_arm_action = (joint_pos_des - offset) / scale 

            # Gripper raw action: positive->open, negative->close (BinaryJointAction expects scalar)
            gripper_raw = self.des_gripper_state.unsqueeze(-1)
            # Return raw actions matching the environment's expected input
            return torch.cat([raw_arm_action, gripper_raw], dim=-1)
        else:
            # Fallback: return Cartesian pose (for debugging)
            return torch.cat([des_ee_pose_cart, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    def compute_ik_actions(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Compute raw joint actions (arm + gripper) for the given target Cartesian poses.

        Args:
            target_pose: Tensor of shape (N,7) in format [px,py,pz,w,x,y,z] (w first in quaternion).

        Returns:
            raw_actions: Tensor of shape (N,8) containing raw arm joint commands and gripper scalar.
        """
        # Expect target_pose in same (x,y,z,w) ordering as used by compute()

        # Convert to warp-compatible ordering (x,y,z,w -> x,y,z,w) handled below
        # target_pose is expected as [px,py,pz,w,x,y,z] (w first), convert to (x,y,z,w)
        des_ee_pose_cart = target_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        if self.robot is not None and hasattr(self, 'ik_controller'):
            # Set IK target
            self.ik_controller.set_command(des_ee_pose_cart)
            joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]

            def get_idx(id_val):
                return id_val[0] if isinstance(id_val, (list, torch.Tensor)) else id_val

            left_idx = get_idx(self.robot.find_bodies("left_inner_finger")[0])
            right_idx = get_idx(self.robot.find_bodies("right_inner_finger")[0])
            ee_idx = get_idx(self.ee_body_id)

            left_pos_w = self.robot.data.body_pos_w[:, left_idx, :]
            right_pos_w = self.robot.data.body_pos_w[:, right_idx, :]
            ee_body_pos_w = self.robot.data.body_pos_w[:, ee_idx, :]
            ee_quat_w = self.robot.data.body_quat_w[:, ee_idx, :]

            tcp_pos_w = (left_pos_w + right_pos_w) / 2.0

            tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, tcp_pos_w, ee_quat_w
            )

            body_pos_b, body_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, ee_body_pos_w, ee_quat_w
            )

            jacobian_full = self.robot.root_physx_view.get_jacobians()
            jacobian_body_w = jacobian_full[:, ee_idx, :, self.arm_joint_ids]

            Jv_w = jacobian_body_w[:, 0:3, :]
            Jw_w = jacobian_body_w[:, 3:6, :]

            def skew_matrix(vec):
                N = vec.shape[0]
                S = torch.zeros((N, 3, 3), device=vec.device, dtype=vec.dtype)
                S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = -vec[:, 2], vec[:, 1], vec[:, 2]
                S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = -vec[:, 0], -vec[:, 1], vec[:, 0]
                return S

            r_w = tcp_pos_w - ee_body_pos_w
            Jp_tcp_w = Jv_w - torch.matmul(skew_matrix(r_w), Jw_w)

            R_w2b = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
            J_linear_b = torch.matmul(R_w2b, Jp_tcp_w)
            J_angular_b = torch.matmul(R_w2b, Jw_w)
            jacobian_tcp = torch.cat([J_linear_b, J_angular_b], dim=1)

            joint_pos_des = self.ik_controller.compute(
                tcp_pos_b, tcp_quat_b, jacobian_tcp, joint_pos
            )

            gravity_forces = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(gravity_forces[:, self.arm_joint_ids], joint_ids=self.arm_joint_ids)

            scale = 0.5
            offset = self.robot.data.default_joint_pos[:, self.arm_joint_ids]
            raw_arm_action = (joint_pos_des - offset) / scale

            gripper_raw = torch.ones((raw_arm_action.shape[0], 1), device=raw_arm_action.device)
            return torch.cat([raw_arm_action, gripper_raw], dim=-1)
        else:
            # Fallback: return Cartesian target + open gripper
            gripper_raw = torch.ones((des_ee_pose_cart.shape[0], 1), device=des_ee_pose_cart.device)
            return torch.cat([des_ee_pose_cart, gripper_raw], dim=-1)

    def _quat_wxyz_to_euler(self, qwxyz: torch.Tensor):
        # qwxyz: (N,4) in (w,x,y,z)
        w = qwxyz[:, 0]
        x = qwxyz[:, 1]
        y = qwxyz[:, 2]
        z = qwxyz[:, 3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)
        return roll, pitch, yaw

    def _euler_to_quat_wxyz(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor):
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack([w, x, y, z], dim=1)

    def _capture_initial_pose(self):
        """Capture the initial end-effector orientation (stored as x,y,z,w) and
        the home joint positions / TCP pose for reset.
        """
        if self.robot is None:
            return
        try:
            ee_body_idx = self.ee_body_id[0] if isinstance(self.ee_body_id, list) else self.ee_body_id
            # get body quaternion in world frame (w,x,y,z)
            ee_quat_w = self.robot.data.body_quat_w[:, ee_body_idx, :]
            # convert to robot root frame (returns (pos, quat) where quat is (w,x,y,z))
            from isaaclab.utils.math import subtract_frame_transforms
            _, ee_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w,
                self.robot.data.body_pos_w[:, ee_body_idx, :], ee_quat_w
            )
            # reorder (w,x,y,z) -> (x,y,z,w) for warp offset
            ee_quat_b_xyzw = torch.stack([ee_quat_b[:, 1], ee_quat_b[:, 2], ee_quat_b[:, 3], ee_quat_b[:, 0]], dim=1)
            self.default_orientation_quat[:, :] = ee_quat_b_xyzw
            self.default_orientation_wp = wp.from_torch(self.default_orientation_quat, wp.quat)
        except Exception:
            pass

        try:
            self.home_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids].clone()
            # compute TCP (midpoint between knuckles) in robot root frame
            left_knuckle_id, _ = self.robot.find_bodies("left_outer_knuckle")
            right_knuckle_id, _ = self.robot.find_bodies("right_outer_knuckle")
            lk_idx = left_knuckle_id[0] if isinstance(left_knuckle_id, list) else left_knuckle_id
            rk_idx = right_knuckle_id[0] if isinstance(right_knuckle_id, list) else right_knuckle_id
            left_knuckle_pos_w = self.robot.data.body_pos_w[:, lk_idx, :]
            right_knuckle_pos_w = self.robot.data.body_pos_w[:, rk_idx, :]
            tcp_pos_w = (left_knuckle_pos_w + right_knuckle_pos_w) / 2.0
            ee_body_idx = self.ee_body_id[0] if isinstance(self.ee_body_id, list) else self.ee_body_id
            ee_quat_w = self.robot.data.body_quat_w[:, ee_body_idx, :]
            from isaaclab.utils.math import subtract_frame_transforms
            tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, tcp_pos_w, ee_quat_w
            )
            self.home_tcp_pos_b = tcp_pos_b.clone()
            self.home_ee_quat_b = tcp_quat_b.clone()
        except Exception:
            self.home_joint_pos = None
            self.home_tcp_pos_b = None
            self.home_ee_quat_b = None

    def _normalize_env_idx(self, env_ids: Sequence[int] | None):
        """Normalize env_ids into a slice or a long tensor on the class device.

        Returns:
            slice(None) when selecting all envs, or a `torch.LongTensor` of indices.
        """
        if env_ids is None:
            return slice(None)
        if isinstance(env_ids, slice):
            return slice(None)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(dtype=torch.long, device=self.device)
        if isinstance(env_ids, (list, tuple)):
            return torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
        # single integer
        return torch.tensor([int(env_ids)], dtype=torch.long, device=self.device)
    
def main(args_cli):
    """Run the pick and place state machine demonstration.

    Args:
        args_cli: Parsed CLI arguments (from AppLauncher.add_app_launcher_args).
    """
    # Import realman tasks
    import sys
    import pathlib
    
    # Add source directory to path
    source_dir = pathlib.Path(__file__).parent.parent / "source"
    sys.path.insert(0, str(source_dir))
    
    import realman  # noqa: F401
    
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    
    # Ensure isaaclab imports available now that SimulationApp is running
    try:
        _ensure_isaac_imports()
    except Exception:
        pass

    # parse configuration
    env_cfg = parse_env_cfg(
        "Realman-RM75-PickPlace-Train",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # create environment
    env = gym.make("Realman-RM75-PickPlace-Train", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # Instantiate visualization markers at runtime (stage exists after reset)
    try:
        # visualizer config stored on env cfg to avoid scene asset parsing errors
        if hasattr(env_cfg, 'tcp_visualizer_cfg'):
            scene = env.unwrapped.scene
            if not hasattr(scene, 'tcp_visualizer'):
                scene.tcp_visualizer = VisualizationMarkers(env_cfg.tcp_visualizer_cfg)
            if not hasattr(scene, 'des_visualizer'):
                # Prefer a dedicated des_visualizer config on the env cfg if provided
                scene.des_visualizer = VisualizationMarkers(env_cfg.des_visualizer_cfg)

    except Exception:
        pass

    # create action buffers
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    
    # create state machine with robot for IK
    pick_place_sm = PickPlaceSm(
        env_cfg.sim.dt * env_cfg.decimation, 
        env.unwrapped.num_envs, 
        env.unwrapped.device,
        robot=env.unwrapped.scene["robot"]
    )

    print("[INFO] Starting pick and place state machine demonstration...")
    print("[INFO] The robot will:")
    print("       1. Approach above the object")
    print("       2. Move down to grasp")
    print("       3. Close gripper")
    print("       4. Lift object")
    print("       5. Move to target position")
    print("       6. Lower and release")

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            step_ret = env.step(actions)
            dones = step_ret[-2]
            
            # (debug prints removed)

            # observations
            # -- end-effector frame (gripper midpoint between knuckles)
            robot = env.unwrapped.scene["robot"]
            left_finger_id, _ = robot.find_bodies("left_inner_finger")
            right_finger_id, _ = robot.find_bodies("right_inner_finger")
            left_finger_idx = left_finger_id[0] if isinstance(left_finger_id, list) else left_finger_id
            right_finger_idx = right_finger_id[0] if isinstance(right_finger_id, list) else right_finger_id
            
            left_finger_pos_w = robot.data.body_pos_w[:, left_finger_idx, :]
            right_finger_pos_w = robot.data.body_pos_w[:, right_finger_idx, :]
            
            # Gripper midpoint in world frame, then make relative to env_origins
            tcp_position = (left_finger_pos_w + right_finger_pos_w) / 2.0 - env.unwrapped.scene.env_origins
            
            # Use link_7 orientation
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            
            # -- object position (privileged)
            object: RigidObject = env.unwrapped.scene["object"]
            object_position = object.data.root_pos_w.clone() - env.unwrapped.scene.env_origins
            object_orientation = object.data.root_quat_w.clone()
            
            # -- target position from command
            target_position = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
            target_position += env.unwrapped.scene["robot"].data.root_pos_w - env.unwrapped.scene.env_origins
            # Use the state's default_orientation as the target orientation
            # default_orientation_quat is stored as (x, y, z, w) -> convert to (w, x, y, z)
            ao = pick_place_sm.default_orientation_quat.to(env.unwrapped.device)
            target_orientation = ao[:, [3, 0, 1, 2]]

            # advance state machine
            actions = pick_place_sm.compute(
                torch.cat([tcp_position, tcp_orientation], dim=-1),
                torch.cat([object_position, object_orientation], dim=-1),
                torch.cat([target_position, target_orientation], dim=-1),
            )

            # Visualize tcp_position and desired EE pose (no orientation needed)
            try:
                scene = env.unwrapped.scene
                if hasattr(scene, 'tcp_visualizer'):
                    # tcp_position is relative to env origins; convert to world
                    env_origins_np = scene.env_origins.detach().cpu().numpy()
                    tcp_np = tcp_position.detach().cpu().numpy() + env_origins_np
                    scales = np.ones((tcp_np.shape[0], 3), dtype=float) * 0.2
                    scene.tcp_visualizer.visualize(translations=tcp_np, scales=scales)
                if hasattr(scene, 'des_visualizer'):
                    des_np = pick_place_sm.des_ee_pose.detach().cpu().numpy()
                    des_trans = des_np[:, :3] + scene.env_origins.detach().cpu().numpy()
                    scales = np.ones((des_trans.shape[0], 3), dtype=float) * 0.2
                    scene.des_visualizer.visualize(translations=des_trans, scales=scales)
            except Exception:
                pass

            # reset state machine when episode ends
            if dones.any():
                pick_place_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # Parse CLI and start SimulationApp here to avoid side-effects on import
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Pick and place state machine for Realman RM75.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum iterations for training.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        main(args_cli)
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
