"""Backflip task configuration for Unitree Go2."""

import math
from copy import deepcopy

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

# --- SCENE CONFIGURATION ---
SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="plane", 
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

  # --- ACTIONS ---
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      # Scale 0.8 is critical for explosive movement (Backflip needs this high)
      scale=0.8,  
      use_default_offset=True,
    )
  }

  # --- COMMANDS ---
  commands: dict[str, CommandTermCfg] = {
    "backflip_ref": mdp.BackflipCommandCfg( 
      asset_name="robot",
      # Note: Ensure your backflip_command.py has cycle_time=2.0 as discussed
      resampling_time_range=(3.0, 5.0), 
      ranges=mdp.BackflipCommandCfg.Ranges(jump_height=(0.5, 0.7)),
      debug_vis=True,
    )
  }

  # --- OBSERVATIONS ---
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
        # Disable noise so the robot can sense precise timing
        enable_corruption=False 
    ),
    "critic": ObservationGroupCfg(
        terms=critic_terms, 
        concatenate_terms=True, 
        enable_corruption=False
    ),
  }

  # --- EVENTS ---
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

  # --- REWARDS ---
  rewards = {
    # 1. Primary Task: High Weights to force the jump
    "track_height": RewardTermCfg(
        func=mdp.track_base_height, 
        weight=15.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),
    "track_pitch": RewardTermCfg(
        func=mdp.track_base_pitch, 
        weight=15.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),

    # 2. Collision Penalty (The Anti-Belly-Flop Reward)
    # FIX APPLIED: Removed 'threshold' parameter
    "body_collision": RewardTermCfg(
        func=mdp.illegal_contact, 
        weight=-5.0,
        params={"sensor_name": self_collision_sensor_cfg.name}, 
    ),

    "feet_airborne": RewardTermCfg(
        func=mdp.feet_airborne,
        weight=2.0,  # Positive weight to encourage jumping
        params={"sensor_name": feet_sensor_cfg.name},
    ),
    
    # 3. Stability / Orientation
    # Helps the robot land flat instead of sideways/tilted
    "flat_orientation": RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    ),
    "penalize_xy_drift": RewardTermCfg(
        func=mdp.penalize_xy_velocity, 
        weight=-0.5,
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
    
    # 4. Limits (Low penalty to allow full extension)
    "dof_pos_limits": RewardTermCfg(
        func=mdp.joint_pos_limits, 
        weight=-0.2, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  # --- TERMINATIONS ---
  terminations = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    # DISABLED: illegal_contact termination so we don't reset mid-flip
    # The 'body_collision' reward above handles the punishment now.
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
    episode_length_s=5.0, 
  )