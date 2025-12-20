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


class BackflipCommand(CommandTerm):
    """
    Command generator for a backflip trajectory.
    Outputs a 2D command: [Target Base Height, Target Base Pitch]
    """
    cfg: BackflipCommandCfg

    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.asset_name]
        self.bf_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.time_in_cycle = torch.zeros(self.num_envs, device=self.device)
        self.target_jump_height = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.bf_command_b

    def _update_metrics(self) -> None:
        current_height = self.robot.data.root_link_pos_w[:, 2]
        self.metrics["error_height"] += (
                torch.abs(self.bf_command_b[:, 0] - current_height)
                / (self.cfg.cycle_time / self._env.step_dt)
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.time_in_cycle[env_ids] = 0.0
        r = torch.empty(len(env_ids), device=self.device)
        self.target_jump_height[env_ids] = r.uniform_(*self.cfg.ranges.jump_height)
        self.bf_command_b[env_ids, 0] = 0.28
        self.bf_command_b[env_ids, 1] = 0.0

    def _update_command(self) -> None:
        self.time_in_cycle += self._env.step_dt
        self.time_in_cycle = torch.remainder(self.time_in_cycle, self.cfg.cycle_time)
        t = self.time_in_cycle / self.cfg.cycle_time

        # --- 1. Base Height Trajectory ---
        target_h = torch.zeros_like(t)

        # Phase 1: Squat (0.0 to 0.25)
        mask_squat = (t < 0.25)
        target_h[mask_squat] = 0.28 + (0.15 - 0.28) * (t[mask_squat] / 0.25)

        # Phase 2: Air/Jump (0.25 to 0.75)
        mask_air = (t >= 0.25) & (t < 0.75)
        t_air = (t[mask_air] - 0.25) / 0.5
        h_peak = self.target_jump_height[mask_air]
        target_h[mask_air] = 0.15 + (h_peak - 0.15) * 4 * (t_air - t_air ** 2)

        # Phase 3: Land (0.75 to 1.0)
        mask_land = (t >= 0.75)
        t_land = (t[mask_land] - 0.75) / 0.25
        target_h[mask_land] = 0.15 + (0.28 - 0.15) * t_land

        self.bf_command_b[:, 0] = target_h

        # --- 2. Base Pitch Trajectory ---
        target_p = torch.zeros_like(t)

        # PHYSICS FIX: Start spinning AT START OF JUMP (0.25)
        # We start rotation exactly when the robot begins extending its legs (0.25).
        # This allows it to generate angular momentum against the ground.
        mask_spin = (t >= 0.25) & (t < 0.7)
        t_spin = (t[mask_spin] - 0.25) / 0.45
        target_p[mask_spin] = -2 * np.pi * t_spin

        # Landing Phase (Flat)
        mask_post_spin = (t >= 0.7)
        target_p[mask_post_spin] = -2 * np.pi

        self.bf_command_b[:, 1] = target_p

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs: return
        base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
        if np.linalg.norm(base_pos) < 1e-6: return
        # Visualization placeholder


@dataclass(kw_only=True)
class BackflipCommandCfg(CommandTermCfg):
    asset_name: str
    cycle_time: float = 2.0
    class_type: type[CommandTerm] = BackflipCommand

    @dataclass
    class Ranges:
        jump_height: tuple[float, float] = (0.5, 0.7)

    ranges: Ranges = field(default_factory=Ranges)

    @dataclass
    class VizCfg:
        scale: float = 1.0
        target_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)