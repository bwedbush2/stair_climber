"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  nconmax=35,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


def create_velocity_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  site_names: tuple[str, ...],
  feet_sensor_cfg: ContactSensorCfg,
  self_collision_sensor_cfg: ContactSensorCfg,
  foot_friction_geom_names: tuple[str, ...] | str,
  posture_std_standing: dict[str, float],
  posture_std_walking: dict[str, float],
  posture_std_running: dict[str, float],
) -> ManagerBasedRlEnvCfg:
  """Create a velocity locomotion task configuration.

  Args:
    robot_cfg: Robot configuration (with sensors).
    action_scale: Action scaling factor(s).
    viewer_body_name: Body for camera tracking.
    site_names: List of site names for foot height/clearance.
    feet_sensor_cfg: Contact sensor config for feet-ground contact.
    self_collision_sensor_cfg: Contact sensor config for self-collision.
    foot_friction_geom_names: Geometry names for friction randomization.
    posture_std_standing: Joint std devs for standing posture reward.
    posture_std_walking: Joint std devs for walking posture reward.
    posture_std_running: Joint std devs for running posture reward.
    body_ang_vel_weight: Weight for body angular velocity penalty.
    angular_momentum_weight: Weight for angular momentum penalty.
    self_collision_weight: Weight for self-collision cost.
    air_time_weight: Weight for feet air time reward.

  Returns:
    Complete ManagerBasedRlEnvCfg for velocity task.
  """
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": robot_cfg}
  scene.sensors = (feet_sensor_cfg, self_collision_sensor_cfg)

  # Enable curriculum mode for terrain generator.
  if scene.terrain is not None and scene.terrain.terrain_generator is not None:
    scene.terrain.terrain_generator.curriculum = True

  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = viewer_body_name

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=action_scale,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      asset_name="robot",
      resampling_time_range=(1e9, 1e9),
      rel_standing_envs=1.0,
    )
  }

  policy_terms = {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
  }

  critic_terms = {
    **policy_terms,
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-1.0, 1.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  rewards = {
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(viewer_body_name,)),
      },
    ),
    "base_z": RewardTermCfg(
      func=mdp.base_z,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(viewer_body_name,)),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(30.0)},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=4,
    episode_length_s=20.0,
  )
