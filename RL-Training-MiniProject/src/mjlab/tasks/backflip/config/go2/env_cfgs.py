"""Unitree Go2 velocity tracking environment configurations."""

from copy import deepcopy


from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.backflip import mdp
from mjlab.tasks.backflip.backflip_env_cfg import VIEWER_CONFIG, create_backflip_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO2_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 Backflip configuration."""
  
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
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
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = create_backflip_env_cfg(
    robot_cfg=get_go2_robot_cfg(),
    action_scale=GO2_ACTION_SCALE,
    viewer_body_name="trunk",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=nonfoot_ground_cfg,
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