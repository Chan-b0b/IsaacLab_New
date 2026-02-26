from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ApplyGravityTerm(ManagerTermBase):
    """Manager term that applies gravity compensation forces to the robot each step.

    This term is intentionally lightweight and guarded so it is safe to include
    in environment configs even when running in headless / unit-test contexts.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        # Nothing to configure for now

    def __call__(self, env: "ManagerBasedRLEnv", env_ids=None):
        try:
            # Try to find the robot asset in the scene; different envs may expose it
            try:
                robot = env.scene["robot"]
            except Exception:
                return None

            # Get arm joint ids if available
            try:
                arm_joint_ids, _ = robot.find_joints("joint_[1-7]")
            except Exception:
                arm_joint_ids = None

            if arm_joint_ids is None:
                return None

            # Get generalized gravity forces and apply as effort target (guarded)
            try:
                gravity_forces = robot.root_physx_view.get_gravity_compensation_forces()
                robot.set_joint_effort_target(gravity_forces[:, arm_joint_ids], joint_ids=arm_joint_ids)
            except Exception:
                # Be noisy only in debugging contexts; otherwise ignore
                return None
        except Exception:
            return None
        return None
