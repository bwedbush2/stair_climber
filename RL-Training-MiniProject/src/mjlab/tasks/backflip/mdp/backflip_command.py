from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.asset_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows.

    Note: Only visualizes the selected environment (visualizer.env_idx).
    """
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    # Helper to transform local to world coordinates.
    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command linear velocity arrow (blue).
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

    # Command angular velocity arrow (green).
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # Actual linear velocity arrow (cyan).
    act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
    )

    # Actual angular velocity arrow (light green).
    act_ang_from = act_lin_from
    act_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
    )
    visualizer.add_arrow(
      act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  asset_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  class_type: type[CommandTerm] = UniformVelocityCommand

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )

class BackflipCommand(CommandTerm):
    """
    Command generator for a backflip trajectory.
    Outputs a 2D command: [Target Base Height, Target Base Pitch]
    """
    cfg: BackflipCommandCfg

    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.asset_name]

        # Command buffer: [Height, Pitch]
        self.bf_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Internal state tracking
        self.time_in_cycle = torch.zeros(self.num_envs, device=self.device)
        self.target_jump_height = torch.zeros(self.num_envs, device=self.device)

        # Metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Returns the current command [Target Height, Target Pitch]."""
        return self.bf_command_b

    def _update_metrics(self) -> None:
        # Calculate tracking error for height and pitch
        current_height = self.robot.data.root_link_pos_w[:, 2]
        # Approximate pitch from z-axis of rotation matrix (simplification for metrics)
        # or use exact euler extraction if available. 
        # Here we just track height error as the primary metric.
        self.metrics["error_height"] += (
            torch.abs(self.bf_command_b[:, 0] - current_height) 
            / (self.cfg.cycle_time / self._env.step_dt)
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """
        Resets the backflip cycle for the specified environments.
        """
        # Reset timer
        self.time_in_cycle[env_ids] = 0.0

        # Sample a specific jump height for this attempt (if ranges provided)
        r = torch.empty(len(env_ids), device=self.device)
        self.target_jump_height[env_ids] = r.uniform_(*self.cfg.ranges.jump_height)

        # Initialize command to standing state
        self.bf_command_b[env_ids, 0] = 0.28  # Default standing height
        self.bf_command_b[env_ids, 1] = 0.0   # Default pitch

    def _update_command(self) -> None:
        """
        Computes the target height and pitch based on the current time in the cycle.
        """
        # Advance timer
        self.time_in_cycle += self._env.step_dt
        
        # Wrap timer for continuous training
        self.time_in_cycle = torch.remainder(self.time_in_cycle, self.cfg.cycle_time)
        
        # Normalize time t in [0, 1]
        t = self.time_in_cycle / self.cfg.cycle_time
        
        # --- Trajectory Generation ---
        # 1. Base Height Trajectory (Squat -> Jump -> Land)
        target_h = torch.zeros_like(t)
        
        # Phase 1: Squat (0.0 to 0.2)
        mask_squat = (t < 0.2)
        target_h[mask_squat] = 0.28 + (0.15 - 0.28) * (t[mask_squat] / 0.2)
        
        # Phase 2: Air/Jump (0.2 to 0.7)
        mask_air = (t >= 0.2) & (t < 0.7)
        t_air = (t[mask_air] - 0.2) / 0.5
        # Parabolic jump: peaks at target_jump_height
        h_peak = self.target_jump_height[mask_air]
        target_h[mask_air] = 0.15 + (h_peak - 0.15) * 4 * (t_air - t_air**2)
        
        # Phase 3: Land (0.7 to 1.0)
        mask_land = (t >= 0.7)
        t_land = (t[mask_land] - 0.7) / 0.3
        target_h[mask_land] = 0.15 + (0.28 - 0.15) * t_land

        self.bf_command_b[:, 0] = target_h

        # 2. Base Pitch Trajectory (Rotate 360)
        target_p = torch.zeros_like(t)
        
        # Only rotate during the jump phase (0.25 to 0.75)
        mask_spin = (t >= 0.25) & (t < 0.75)
        t_spin = (t[mask_spin] - 0.25) / 0.5
        target_p[mask_spin] = -2 * np.pi * t_spin
        
        self.bf_command_b[:, 1] = target_p

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """
        Visualize the target height and pitch.
        """
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        # Get current state
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_pos_w = base_pos_ws[batch]
        
        # Get command
        cmd = self.bf_command_b[batch].cpu().numpy() # [height, pitch]
        target_height = cmd[0]

        # Skip if robot uninitialized
        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        # Visualization Config
        scale = self.cfg.viz.scale
        color = self.cfg.viz.target_color

        # Draw Target Height (Sphere/Point)
        # We project the current XY position to the Target Z
        target_pos = base_pos_w.copy()
        target_pos[2] = target_height
        
        # Add sphere at target height
        # visualizer.add_point(
        #     target_pos, radius=0.05 * scale, color=color
        # )
        
        # Optional: Draw arrow indicating "Up" or Orientation if desired
        # For now, just the height target is most useful.


@dataclass(kw_only=True)
class BackflipCommandCfg(CommandTermCfg):
    asset_name: str
    cycle_time: float = 1.5
    class_type: type[CommandTerm] = BackflipCommand

    @dataclass
    class Ranges:
        jump_height: tuple[float, float] = (0.5, 0.7) # Min/Max jump height
    
    ranges: Ranges = field(default_factory=Ranges)

    @dataclass
    class VizCfg:
        scale: float = 1.0
        target_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)
