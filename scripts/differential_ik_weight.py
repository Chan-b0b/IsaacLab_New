"""Helpers for inertia-weighted Differential IK (script-local copy).

This is a copy of the controller helper placed under `scripts/` so
runtime scripts can import it without touching controller sources.
Use this module for any future changes instead of modifying
`IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik_weight.py`.
"""
from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.utils.math import compute_pose_error
from isaaclab.controllers.differential_ik import DifferentialIKController


class DifferentialIKControllerWithWeight(DifferentialIKController):
    """Subclass of `DifferentialIKController` that adds inertia-based joint weights.

    Overrides the `compute` and `_compute_delta_joint_pos` methods to optionally
    use per-joint weights computed from the robot's generalized mass matrices.
    """

    def __init__(self, cfg, num_envs: int, device: str | torch.device, robot=None, arm_joint_ids: Sequence[int] | None = None, eps: float = 1e-6, normalize: bool = False):
        """Initialize the subclass and optionally compute initial weights.

        Args:
            cfg: IK controller config passed to the base class.
            num_envs: Number of environments.
            device: Device for computation.
            robot: Robot articulation providing mass matrices (optional).
            arm_joint_ids: Indices of the arm joints.
            eps: Small regularizer when inverting inertia diagonals.
            normalize: Whether to normalize weights per-env mean to 1.
        """
        super().__init__(cfg, num_envs, device)
        self.robot = robot
        self.arm_joint_ids = list(arm_joint_ids) if arm_joint_ids is not None else None
        self.eps = eps
        self.normalize = normalize
        self.joint_weights = None
        if self.robot is not None and self.arm_joint_ids is not None:
            try:
                self.joint_weights = self.compute_inertia_based_joint_weights()
            except Exception:
                self.joint_weights = None

    def update_joint_weights(self):
        """Recompute joint weights from the current robot mass matrices."""
        if self.robot is None or self.arm_joint_ids is None:
            self.joint_weights = None
            return
        try:
            self.joint_weights = self.compute_inertia_based_joint_weights()
        except Exception:
            self.joint_weights = None

    def compute_inertia_based_joint_weights(self, device: str | torch.device | None = None, eps: float | None = None, normalize: bool | None = None) -> torch.Tensor | None:
        """Instance method: compute per-env inertia-based joint weights using stored robot and joint ids.

        Args:
            device: Device for the returned tensor. Defaults to the base controller device.
            eps: Regularization to add to diagonal inertia. Defaults to the instance `eps`.
            normalize: Whether to normalize per-env mean. Defaults to the instance `normalize`.

        Returns:
            Tensor of shape (num_envs, n_joints) or None on failure.
        """


        if self.robot is None or self.arm_joint_ids is None:
            return None
        if device is None:
            device = self._device
        if eps is None:
            eps = self.eps
        if normalize is None:
            normalize = self.normalize
        try:
            mass_matrices = self.robot.root_physx_view.get_generalized_mass_matrices()
            arm_mass = mass_matrices[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids]
            M_diag = torch.diagonal(arm_mass, dim1=1, dim2=2)

            J_pos = self.jocobian[:, 0:3, :] 
            
            payload_mass = 0.473
            # The diagonal contribution: m * sum(J_rows^2)
            # This calculates the diagonal of J.T @ (m*I) @ J efficiently
            payload_inertia_diag = payload_mass * torch.sum(J_pos**2, dim=1)
            M_diag = M_diag + payload_inertia_diag

            weights = 1.0 / (M_diag + eps)
            if normalize:
                denom = weights.mean(dim=1, keepdim=True) + 1e-8
                weights = weights / denom
            return weights.to(device)
        except Exception:
            return None

    def ik_with_inertia_weights(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor, recompute_weights: bool = False) -> torch.Tensor:
        """Compute IK using inertia weights (instance method).

        If `recompute_weights` is True, recompute weights from the robot before solving.
        This method delegates to `self.compute(...)` which uses the internal
        weighted solver implementation.
        """
        self.jocobian = jacobian

        if recompute_weights:
            self.update_joint_weights()
        return self.compute(ee_pos, ee_quat, jacobian, joint_pos)

    def compute(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        """Compute IK using the wrapped controller but supplying inertia-based weights.

        The wrapped controller's API is preserved; this method mirrors the
        `DifferentialIKController.compute(...)` behavior but calls an internal
        weighted solver when available.
        """
        # Compute desired pose error using the wrapped controller's internals
        # We mirror the compute() logic but route the delta computation to a
        # weighted helper below.
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos, joint_weights=self.joint_weights)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            # print("[DEBUG IK] position_error:", position_error)
            # print("[DEBUG IK] axis_angle_error:", axis_angle_error)
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian, joint_weights=self.joint_weights)

        joint_pos_des = joint_pos + delta_joint_pos
        return joint_pos_des

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor, joint_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Weighted version of delta joint computation.

        Supports the same `ik_method` choices as the wrapped controller.
        """
        cfg = self.cfg
        device = self._device

        if cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{cfg.ik_method}' is not defined!")

        if cfg.ik_method == "pinv":
            k_val = cfg.ik_params.get("k_val", 1.0)
            if joint_weights is None:
                jacobian_pinv = torch.linalg.pinv(jacobian)
                delta_joint_pos = (jacobian_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)
            else:
                W = torch.diag_embed(joint_weights)
                jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
                JWJt = jacobian @ (W @ jacobian_T)
                JWJt_pinv = torch.linalg.pinv(JWJt)
                delta_joint_pos = (W @ jacobian_T @ JWJt_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)
            return k_val * delta_joint_pos

        elif cfg.ik_method == "svd":
            k_val = cfg.ik_params.get("k_val", 1.0)
            min_singular_value = cfg.ik_params["min_singular_value"]
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(min_singular_value < S, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = (jacobian_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)
            return k_val * delta_joint_pos

        elif cfg.ik_method == "trans":
            k_val = cfg.ik_params.get("k_val", 1.0)
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = (jacobian_T @ delta_pose.unsqueeze(-1)).squeeze(-1)
            return k_val * delta_joint_pos

        elif cfg.ik_method == "dls":
            lambda_val = cfg.ik_params.get("lambda_val", 0.1)
            k_val = cfg.ik_params.get("k_val", 1.0)
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            m = jacobian.shape[1]
            if joint_weights is not None:
                W = torch.diag_embed(joint_weights)
                JWJt = jacobian @ (W @ jacobian_T)
                lambda_matrix = (lambda_val**2) * torch.eye(n=m, device=device).unsqueeze(0)
                if JWJt.dim() == 3:
                    lambda_matrix = lambda_matrix.expand(JWJt.shape[0], -1, -1)
                inv_term = torch.inverse(JWJt + lambda_matrix)
                delta_joint_pos = (W @ jacobian_T @ inv_term @ delta_pose.unsqueeze(-1)).squeeze(-1)
            else:
                lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
                delta_joint_pos = (
                    jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
                ).squeeze(-1)
            return k_val * delta_joint_pos

        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {cfg.ik_method}")


if __name__ == "__main__":
    pass

