from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_base_height(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Reward for tracking the desired base height trajectory.
    Crucial for the takeoff and landing phases.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Get the desired height from the command generator
    command = env.command_manager.get_command(command_name)
    # INDEX 0 IS HEIGHT
    target_height = command[:, 0]
    assert target_height is not None, f"Command '{command_name}' not found."
    
    # Get current height (z-position)
    current_height = asset.data.root_link_pos_w[:, 2]
    
    # Calculate error
    # We assume command is a scalar [Batch, 1] or [Batch]
    if target_height.dim() > 1:
        target_height = target_height.squeeze(-1)
        
    error = torch.square(target_height - current_height)
    return torch.exp(-error / std**2)


def track_base_pitch(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Reward for tracking the desired pitch (orientation).
    This guides the robot through the rotation of the backflip.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # INDEX 1 IS PITCH
    target_pitch = command[:, 1]
    assert target_pitch is not None, f"Command '{command_name}' not found."
    
    # Get current pitch from quaternion
    quat = asset.data.root_link_quat_w
    # Convert to Euler (roll, pitch, yaw)
    _, pitch, _ = euler_xyz_from_quat(quat)
    
    # Handle wrapping if necessary, though for a single backflip 
    # tracking the raw pitch trajectory usually suffices if generated correctly.
    if target_pitch.dim() > 1:
        target_pitch = target_pitch.squeeze(-1)

    error = torch.square(target_pitch - pitch)
    return torch.exp(-error / std**2)


def penalize_xy_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Penalize horizontal drift. A backflip should be mostly vertical.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_link_lin_vel_w[:, :2]
    return torch.sum(torch.square(vel_xy), dim=1)


def penalize_yaw_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Penalize yaw rotation. The robot should only rotate in Pitch.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_link_ang_vel_w[:, 2]
    return torch.square(ang_vel_z)


# --- PRESERVED UTILITY FUNCTIONS ---

def default_joint_position(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Encourages the robot to stay near the default standing pose when possible."""
  asset: Entity = env.scene[asset_cfg.name]
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  # Using simple L1 or L2 norm here
  return torch.sum(torch.abs(current_joint_pos - desired_joint_pos), dim=1)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities (general regularization)."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  # We might want to penalize roll/yaw specifically, but this general term 
  # can be useful with a low weight.
  return torch.sum(torch.square(ang_vel), dim=1)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """
  Penalize high impact forces at landing.
  Even for a backflip, we want a controlled landing.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force
  force_magnitude = torch.norm(forces, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)
  
  # Tracking metrics
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  
  return cost


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(torch.sqrt(angmom_magnitude_sq))
  return angmom_magnitude_sq

def feet_airborne(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """
    Rewards the robot when all feet are off the ground (flight phase).
    This encourages the robot to jump high and tuck its legs.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    
    # Get contact forces [num_envs, num_feet, 3]
    forces = contact_sensor.data.force
    
    # Calculate force magnitude per foot
    force_mags = torch.norm(forces, dim=-1)
    
    # Check if feet are touching the ground (Force > 1.0 Newton)
    in_contact = force_mags > 1.0
    
    # Count how many feet are in contact
    num_contacts = torch.sum(in_contact.float(), dim=1)
    
    # Reward 1.0 if NO feet are touching the ground
    return (num_contacts == 0).float()
