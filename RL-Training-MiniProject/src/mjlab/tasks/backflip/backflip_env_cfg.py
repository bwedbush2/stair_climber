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
from mjlab.tasks.backflip.mdp import BackflipCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# --- SCENE CONFIGURATION ---
# Use flat terrain for backflip training to reduce complexity
SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="plane", # Changed from "generator" to "plane" for stable jumping
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

def create_backflip_env_cfg(  # Renamed function
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  site_names: tuple[str, ...],
  feet_sensor_cfg: ContactSensorCfg,
  self_collision_sensor_cfg: ContactSensorCfg,
  # Removed unused posture args for clarity, or you can keep them if you wish
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
      scale=action_scale,
      use_default_offset=True,
    )
  }

  # --- COMMANDS (Part 3b) ---
  # The robot needs to know the target height/pitch over time.
  # You must implement `BackflipCommandCfg` in your command generator file.
  commands: dict[str, CommandTermCfg] = {
    "backflip_ref": mdp.BackflipCommandCfg( 
      asset_name="robot",
      resampling_time_range=(3.0, 5.0), # Time between flips
      ranges=mdp.BackflipCommandCfg.Ranges(jump_height=(0.5, 0.7)),                  # Target height
      # The command generator should output: [target_height, target_pitch, phase]
      debug_vis=True,
    )
  }

  # --- OBSERVATIONS (Part 3b) ---
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
    # CRITICAL: The policy must know the target state (height/pitch) or phase
    "commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "backflip_ref"},
    )
  }

  # Critic sees the same (or privileged info if you want to add it like in Part 2)
  critic_terms = {**policy_terms}

  observations = {
    "policy": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
  }

  # --- EVENTS ---
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}, # Tighter reset for backflip
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
    # Reduce domain randomization initially to make learning easier
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
        "field": "geom_friction",
        "ranges": (0.5, 1.0), # Slightly higher min friction for jumping grip
      }
    ),
  }

  # --- REWARDS (Part 3b) ---
  rewards = {
    # 1. Trajectory Tracking (The "Core" task) [cite: 326, 328, 458]
    # Replaces track_linear_velocity/track_angular_velocity
    "track_height": RewardTermCfg(
        func=mdp.track_base_height, # From your updated rewards.py
        weight=2.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),
    "track_pitch": RewardTermCfg(
        func=mdp.track_base_pitch, # From your updated rewards.py
        weight=2.0,
        params={"std": 0.2, "command_name": "backflip_ref"},
    ),

    # 2. Constraints (Keep the flip vertical and safe)
    "penalize_xy_drift": RewardTermCfg(
        func=mdp.penalize_xy_velocity, # From your updated rewards.py
        weight=-0.5,
    ),
    "penalize_yaw_spin": RewardTermCfg(
        func=mdp.penalize_yaw_velocity, # From your updated rewards.py
        weight=-0.5,
    ),
    
    # 3. Regularization (Smoothness and Landing)
    "soft_landing": RewardTermCfg(
        func=mdp.soft_landing,
        weight=-1.0,
        params={"sensor_name": feet_sensor_cfg.name},
    ),
    "action_rate": RewardTermCfg(
        func=mdp.action_rate_l2, 
        weight=-0.05,
    ),
    "dof_pos_limits": RewardTermCfg(
        func=mdp.joint_pos_limits, 
        weight=-5.0, # Stronger penalty to prevent self-destruction
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    
    # REMOVED: upright (flat_orientation), default_joint_pos (conflicts with tucking), gait shaping.
  }

  # --- TERMINATIONS (Part 3e) ---
  terminations = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    # REMOVED: "fell_over" / bad_orientation. 
    # The robot MUST go upside down (pitch ~ 180 deg or 3.14 rad).
    # If you keep bad_orientation, the episode will end exactly when the backflip succeeds.
    
    # OPTIONAL: Terminate if body (not feet) touches ground.
    "illegal_contact": TerminationTermCfg(
        func=mdp.illegal_contact, # Assuming standard MDP function
        params={"sensor_name": self_collision_sensor_cfg.name, "threshold": 1.0},
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
    episode_length_s=5.0, # Short episodes for single flips
  )