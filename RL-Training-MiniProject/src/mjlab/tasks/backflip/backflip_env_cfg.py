import math
from copy import deepcopy
import torch

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
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
from mjlab.tasks.backflip import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

def z_vel_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_link_lin_vel_w[:, 2]

def ramp_pitch_air(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    base_height = robot.data.root_link_pos_w[:, 2]
    height_factor = torch.clamp((base_height - 0.3) / 0.3, 0.0, 1.0)
    pitch_vel = robot.data.root_link_ang_vel_b[:, 1]
    backflip_vel = torch.clamp(-pitch_vel, min=0.0)
    return backflip_vel * height_factor

def penalize_roll_heavy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return -1.0 * torch.square(robot.data.root_link_ang_vel_b[:, 0])

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
  ),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",
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

def create_backflip_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  site_names: tuple[str, ...],
  feet_sensor_cfg: ContactSensorCfg,
  self_collision_sensor_cfg: ContactSensorCfg,
) -> ManagerBasedRlEnvCfg:

  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": robot_cfg}
  scene.sensors = (feet_sensor_cfg, self_collision_sensor_cfg)

  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = viewer_body_name

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "backflip_ref": mdp.BackflipCommandCfg(
      asset_name="robot",
      cycle_time=2.0,
      resampling_time_range=(3.0, 5.0),
      ranges=mdp.BackflipCommandCfg.Ranges(jump_height=(0.7, 0.8)),
      debug_vis=True,
    )
  }

  policy_terms: dict[str, ObservationTermCfg] = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "backflip_ref"},
    )
  }

  critic_terms = {**policy_terms}

  observations = {
    "policy": ObservationGroupCfg(
        terms=policy_terms,
        concatenate_terms=True,
        enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
        "field": "geom_friction",
        "ranges": (0.5, 1.0),
      }
    ),
  }

  rewards = {
    "track_height": RewardTermCfg(
        func=mdp.track_base_height,
        weight=20.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),
    "track_pitch": RewardTermCfg(
        func=mdp.track_base_pitch,
        weight=30.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),
    "ramp_pitch": RewardTermCfg(
        func=ramp_pitch_air,
        weight=10.0,
        params={},
    ),
    "jump_velocity": RewardTermCfg(
        func=z_vel_reward,
        weight=5.0,
        params={},
    ),
    "prevent_roll": RewardTermCfg(
        func=penalize_roll_heavy,
        weight=5.0,
        params={},
    ),
    "penalize_xy_drift": RewardTermCfg(
        func=mdp.penalize_xy_velocity,
        weight=-2.0,
    ),
    "body_collision": RewardTermCfg(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    ),
    "action_rate": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.005,
    ),
    "penalize_yaw_spin": RewardTermCfg(
        func=mdp.penalize_yaw_velocity,
        weight=-0.5,
    ),
    "soft_landing": RewardTermCfg(
        func=mdp.soft_landing,
        weight=-1.0,
        params={"sensor_name": feet_sensor_cfg.name},
    ),
    "dof_pos_limits": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    "illegal_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": self_collision_sensor_cfg.name},
        time_out=False,
    )
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
    episode_length_s=2.5,
  )