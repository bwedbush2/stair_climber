"""Script to generate Deliverable 1.4 plots (Fixed Command Sequence)."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import numpy as np
#import matplotlib.pyplot as plt  # Added for plotting
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int = 1  # Forced to 1 for testing
  device: str | None = None
  
  # New flag to trigger the Deliverable 1.4 sequence
  test_tracking: bool = True 

def _apply_play_env_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply overrides for deterministic testing."""
  cfg.episode_length_s = int(1e9)
  # Disable observation corruption so we see clean data
  if "policy" in cfg.observations:
      cfg.observations["policy"].enable_corruption = False
  # Remove random pushes
  if cfg.events is not None:
      cfg.events.pop("push_robot", None)
  # Force terrain to flat for consistent testing
  if cfg.scene.terrain is not None:
      cfg.scene.terrain.terrain_type = "plane"
      cfg.scene.terrain.terrain_generator = None

def run_play(task: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # 1. Load & Config Env
  env_cfg = load_env_cfg(task)
  _apply_play_env_overrides(env_cfg)
  env_cfg.scene.num_envs = 1  # Ensure single robot
  
  agent_cfg = load_rl_cfg(task)

  # 2. Load Checkpoint
  if cfg.checkpoint_file is None:
      raise ValueError("You must provide a --checkpoint_file for this test!")
      
  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  print(f"[INFO]: Loading checkpoint: {resume_path.name}")
  log_dir = resume_path.parent

  # 3. Create Env
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # 4. Load Policy
  runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  # =========================================================================
  # DELIVERABLE 1.4 LOGIC: Fixed Command Sequence
  # =========================================================================
  print("\n[INFO] Starting Deliverable 1.4 Test Sequence...")
  
  # Phases: 125 steps each (~2.5s)
  phases = [
      {"steps": 125, "cmd": [0.6, 0.0, 0.0], "ramp": True},  # Forward
      {"steps": 125, "cmd": [0.0, 0.4, 0.0], "ramp": False}, # Lateral
      {"steps": 125, "cmd": [0.0, 0.0, 0.4], "ramp": False}, # Turn
      {"steps": 125, "cmd": [0.5, 0.0, 0.3], "ramp": False}, # Mixed
  ]

  obs, _ = env.reset()
  logs = {"cmd_x": [], "vel_x": [], "cmd_y": [], "vel_y": [], "cmd_yaw": [], "vel_yaw": []}

  with torch.no_grad():
      for phase_idx, phase in enumerate(phases):
          target_cmd = torch.tensor(phase["cmd"], device=env.unwrapped.device)
          
          for step in range(phase["steps"]):
              # A. Manual Command Injection
              current_cmd = target_cmd.clone()
              if phase["ramp"] and phase_idx == 0:
                  # Linear ramp 0 -> 0.6
                  current_cmd[0] = (step / phase["steps"]) * 0.6
              
              # Set command in the underlying env manager
              # Note: "twist" must match your config command name
              # 1. Get the reference to the command tensor
              twist_cmd = env.unwrapped.command_manager.get_command("twist")
              # 2. Overwrite the values in-place ([:] is crucial!)
              twist_cmd[:] = current_cmd.unsqueeze(0)

              # B. Step Policy
              action = policy(obs)
              obs, _, _, _ = env.step(action)

              # C. Log Data
              # Command
              logs["cmd_x"].append(current_cmd[0].item())
              logs["cmd_y"].append(current_cmd[1].item())
              logs["cmd_yaw"].append(current_cmd[2].item())
              
              # Actual Velocity (from Robot Data)
              # Access the underlying robot entity
              robot = env.unwrapped.scene["robot"]
              lin_vel = robot.data.root_link_lin_vel_b[0] # x, y, z
              ang_vel = robot.data.root_link_ang_vel_b[0] # x, y, z
              
              logs["vel_x"].append(lin_vel[0].item())
              logs["vel_y"].append(lin_vel[1].item())
              logs["vel_yaw"].append(ang_vel[2].item())

  env.close()

  # # =========================================================================
  # # PLOTTING
  # # =========================================================================
  # print("[INFO] Generating Plots...")
  # fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  # time_axis = np.arange(len(logs["cmd_x"]))

  # # Plot X
  # axs[0].plot(time_axis, logs["cmd_x"], 'r--', label="Command X")
  # axs[0].plot(time_axis, logs["vel_x"], 'b-', label="Actual X")
  # axs[0].set_title("Forward Velocity Tracking")
  # axs[0].set_ylabel("Velocity (m/s)")
  # axs[0].legend()
  # axs[0].grid(True)

  # # Plot Y
  # axs[1].plot(time_axis, logs["cmd_y"], 'r--', label="Command Y")
  # axs[1].plot(time_axis, logs["vel_y"], 'b-', label="Actual Y")
  # axs[1].set_title("Lateral Velocity Tracking")
  # axs[1].set_ylabel("Velocity (m/s)")
  # axs[1].grid(True)

  # # Plot Yaw
  # axs[2].plot(time_axis, logs["cmd_yaw"], 'r--', label="Command Yaw")
  # axs[2].plot(time_axis, logs["vel_yaw"], 'b-', label="Actual Yaw")
  # axs[2].set_title("Yaw Rate Tracking")
  # axs[2].set_ylabel("Rate (rad/s)")
  # axs[2].set_xlabel("Steps")
  # axs[2].grid(True)

  # plt.tight_layout()
  # output_file = "deliverable_1_4_plots.png"
  # plt.savefig(output_file)
  # print(f"✅ Plots saved to {output_file}")
  # print("You can also view them in the output above if running in a notebook.")
  # plt.show()

def main():
  import mjlab.tasks  # noqa: F401
  all_tasks = list_tasks()
  
  # Simple CLI parser specifically for this script
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  
  args = tyro.cli(PlayConfig, args=remaining_args)
  run_play(chosen_task, args)

if __name__ == "__main__":
  main()