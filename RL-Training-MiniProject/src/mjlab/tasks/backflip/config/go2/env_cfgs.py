"""Unitree Go2 velocity tracking environment configurations."""

from copy import deepcopy

# --- CHANGE 1: Update Imports for Go2 ---
# Ensure these exist in your mjlab.asset_zoo.robots file.
# If GO2_ACTION_SCALE is missing, you can temporarily use GO1_ACTION_SCALE
# or define a new one (typically around 0.25 - 0.5 for Go2).
from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import VIEWER_CONFIG, create_velocity_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO2_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity tracking configuration."""
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")

  # NOTE: Verify these geom names match your Go2 MJCF (XML) file.
  # Some Go2 models might use "foot" instead of "foot_collision".
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  # --- CHANGE 2: Use Go2 Robot Config and Action Scale ---
  cfg = create_velocity_env_cfg(
    robot_cfg=get_go2_robot_cfg(),
    action_scale=GO2_ACTION_SCALE,
    viewer_body_name="trunk",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=nonfoot_ground_cfg,
    foot_friction_geom_names=geom_names,
    # These noise parameters are generally safe to keep consistent between Go1/Go2
    # for initial training, as the morphologies are similar.
    posture_std_standing={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    },
    posture_std_walking={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
    posture_std_running={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
  )

  cfg.viewer = deepcopy(VIEWER_CONFIG)
  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  assert cfg.terminations is not None
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "nonfoot_ground_touch"},
  )
  return cfg


@retval
def UNITREE_GO2_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity tracking configuration."""
  # Start with rough terrain config.
  cfg = deepcopy(UNITREE_GO2_ROUGH_ENV_CFG)

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  return cfg