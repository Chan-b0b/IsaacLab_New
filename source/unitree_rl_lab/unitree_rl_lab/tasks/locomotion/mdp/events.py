"""Event terms for locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from isaaclab.assets import Articulation, DeformableObject, RigidObject


def apply_constant_force_to_torso(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_range: dict[str, tuple[float, float]],
    zero_force_percentage: float = 0.0,
):
    """Apply a constant force to the torso for a random duration.
    
    This event term applies a constant force to the robot's torso body. The force is randomized
    within the specified range and applied for the duration of the event interval.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to apply the force to.
        asset_cfg: The scene entity configuration for the robot asset.
        force_range: Dictionary with keys 'x', 'y', 'z' specifying force ranges in Newton.
        zero_force_percentage: Percentage (0.0 to 1.0) of environments that should receive zero force.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Sample random forces for each environment
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    
    # Apply random forces in each direction
    forces[:, :, 0].uniform_(force_range["x"][0], force_range["x"][1])
    forces[:, :, 1].uniform_(force_range["y"][0], force_range["y"][1])
    forces[:, :, 2].uniform_(force_range["z"][0], force_range["z"][1])
    
    # Zero out forces for selected percentage of environments
    if zero_force_percentage > 0.0:
        zero_mask = torch.rand(len(env_ids), device=asset.device) < zero_force_percentage
        forces[zero_mask] *= 0
    
    # Apply the forces to the torso
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=None,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )

def reset_joints_default_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_limit_buffer: float = 1,
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    current_default_joint_pos_all_dofs = asset.data.joint_pos_target[env_ids].clone()

    joint_indices_to_modify = asset_cfg.joint_ids

    selected_joint_pos = current_default_joint_pos_all_dofs[:, joint_indices_to_modify]

    biased_joint_pos = selected_joint_pos + math_utils.sample_uniform(*position_range, selected_joint_pos.shape, selected_joint_pos.device)

    joint_pos_limits_all_dofs = asset.data.soft_joint_pos_limits[env_ids]
    
    selected_joint_pos_limits = joint_pos_limits_all_dofs[:, joint_indices_to_modify] * pose_limit_buffer

    biased_joint_pos = biased_joint_pos.clamp_(selected_joint_pos_limits[..., 0], selected_joint_pos_limits[..., 1])

    # current_default_joint_pos_all_dofs[:, joint_indices_to_modify] = biased_joint_pos
    # asset.data.default_joint_pos[env_ids] = current_default_joint_pos_all_dofs

    asset.set_joint_position_target(biased_joint_pos, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)
    
    

def randomize_rigid_body_mass_percentage(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    randomize_percentage: float = 1.0,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
):
    """Randomize the mass of bodies for only a percentage of environments.

    This function is similar to the randomize_rigid_body_mass class but applies the randomization
    to only a specified percentage of environments. This is useful for domain randomization where
    you want some environments to have perturbed masses while others remain at default values.

    Args:
        env: The environment instance.
        env_ids: The environment indices to potentially randomize. If None, considers all environments.
        asset_cfg: The asset configuration specifying which bodies to randomize.
        mass_distribution_params: The distribution parameters (min, max) for mass randomization.
        operation: The operation to perform: 'add', 'scale', or 'abs'.
        randomize_percentage: Percentage (0.0 to 1.0) of environments to randomize. Default is 1.0 (100%).
        distribution: The distribution type: 'uniform', 'log_uniform', or 'gaussian'.
        recompute_inertia: Whether to recompute inertia tensors after mass change.
        min_mass: Minimum allowed mass value to avoid physics errors.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization to only a percentage of environments
    if randomize_percentage < 1.0 and len(env_ids) > 0:
        # randomly select which environments to randomize
        num_to_randomize = max(1, int(len(env_ids) * randomize_percentage))
        # shuffle and select subset
        perm = torch.randperm(len(env_ids), device="cpu")[:num_to_randomize]
        env_ids = env_ids[perm]
    
    if len(env_ids) == 0:
        return  # skip if no environments selected

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    masses = torch.clamp(masses, min=min_mass)  # ensure masses are positive

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the ratios
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)
        
        

def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data