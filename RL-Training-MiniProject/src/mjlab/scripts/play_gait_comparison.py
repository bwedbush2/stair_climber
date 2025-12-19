"""Script to generate Deliverable 2.1: Gait Shaping Foot Trajectory Plots."""

import os
import sys
import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import numpy as np
import matplotlib.pyplot as plt
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

@dataclass(frozen=True)
class GaitTestConfig:
    agent: Literal["trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int = 1
    device: str | None = None

    # Label for the plot legend (e.g., "No_Gait", "With_Gait")
    label: str = "Model"

    # Command velocity for gait testing (Forward walking)
    cmd_vx: float = 0.6

def _apply_play_env_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Setup environment for clean, deterministic gait testing."""
    cfg.episode_length_s = int(1e9)

    # 1. Disable Noise for clean trajectory plots
    if "policy" in cfg.observations:
        cfg.observations["policy"].enable_corruption = False

    # 2. Remove Random Pushes
    if cfg.events is not None:
        cfg.events.pop("push_robot", None)

    # 3. Force Flat Terrain (Plane)
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

def run_gait_test(task: str, cfg: GaitTestConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load & Config Env
    env_cfg = load_env_cfg(task)
    _apply_play_env_overrides(env_cfg)
    env_cfg.scene.num_envs = 1

    agent_cfg = load_rl_cfg(task)

    # 2. Load Checkpoint
    if cfg.checkpoint_file is None:
        raise ValueError("Please provide a --checkpoint_file!")

    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    log_dir = resume_path.parent
    print(f"[INFO] Loading checkpoint: {resume_path.name}")

    # 3. Create Env
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 4. Load Policy
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # 5. Locate Front-Left (FL) Foot
    # We need to find the specific site index for the FL foot to track its position.
    robot = env.unwrapped.scene["robot"]

    # Try finding "FL" or "FL_foot" in site names
    try:
        # Heuristic: Find first site containing "FL"
        all_sites = robot.site_names
        fl_site_name = next(name for name in all_sites if "FL" in name)
        fl_site_id = robot.find_sites(fl_site_name)[0][0]
        print(f"[INFO] Tracking Foot Site: '{fl_site_name}' (ID: {fl_site_id})")
    except StopIteration:
        print("[WARN] Could not auto-detect 'FL' site. Defaulting to index 0.")
        fl_site_id = 0

    # =========================================================================
    # TEST LOOP
    # =========================================================================
    print(f"\n[INFO] Running Gait Test (Cmd Vx={cfg.cmd_vx} m/s)...")

    # Run for 150 steps (~3 seconds) to capture multiple gait cycles
    num_steps = 150
    obs, _ = env.reset()

    foot_heights = []

    with torch.no_grad():
        for i in range(num_steps):
            # A. Inject Constant Forward Command
            twist_cmd = env.unwrapped.command_manager.get_command("twist")
            # [Vx, Vy, Yaw]
            twist_cmd[:] = torch.tensor([cfg.cmd_vx, 0.0, 0.0], device=device).unsqueeze(0)

            # B. Step
            action = policy(obs)
            obs, _, _, _ = env.step(action)

            # C. Log Foot Z-Position (World Frame)
            # site_pos_w is [num_envs, num_sites, 3]
            # We want [0, fl_site_id, 2] (Z-axis)
            z_pos = robot.data.site_pos_w[0, fl_site_id, 2].item()
            foot_heights.append(z_pos)

    env.close()

    # =========================================================================
    # SAVE & PLOT LOGIC
    # =========================================================================

    # 1. Save current run data
    save_filename = f"gait_data_{cfg.label}.npz"
    time_axis = np.arange(len(foot_heights)) * env.unwrapped.step_dt
    np.savez(save_filename, time=time_axis, height=np.array(foot_heights), label=cfg.label)
    print(f"[INFO] Saved data to {save_filename}")

    # 2. Find ALL saved gait data files to overlay
    data_files = glob.glob("gait_data_*.npz")

    print("\n[INFO] Generating Comparison Plot...")
    plt.figure(figsize=(10, 6))

    for file in data_files:
        data = np.load(file)
        lbl = str(data["label"])
        t = data["time"]
        z = data["height"]

        # Style distinction
        style = '-' if lbl == cfg.label else '--'
        alpha = 1.0 if lbl == cfg.label else 0.7
        linewidth = 2.5 if lbl == cfg.label else 1.5

        plt.plot(t, z, linestyle=style, alpha=alpha, linewidth=linewidth, label=lbl)

    # 3. Add Target Clearance Line (Reference)
    plt.axhline(y=0.1, color='k', linestyle=':', alpha=0.5, label="Target (0.1m)")

    plt.title(f"Deliverable 2.1: Front-Left Foot Trajectory Comparison\n(Cmd Vx = {cfg.cmd_vx} m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Foot Height (m)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.25) # Focus on the swing phase

    out_file = "deliverable_2_1_gait_plot.png"
    plt.savefig(out_file)
    print(f"✅ Comparison Plot saved to {out_file}")
    plt.show()

def main():
  import mjlab.tasks
  all_tasks = list_tasks()

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(GaitTestConfig, args=remaining_args)
  run_gait_test(chosen_task, args)

if __name__ == "__main__":
  main()