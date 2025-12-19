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
    label: str = "Model"
    cmd_vx: float = 0.6

    # NEW FLAG: If True, it removes the gait rewards/obs from the config
    disable_gait_features: bool = False


def _apply_play_env_overrides(cfg: ManagerBasedRlEnvCfg, disable_gait: bool) -> None:
    """Setup environment for clean, deterministic gait testing."""
    cfg.episode_length_s = int(1e9)

    # 1. Disable Noise
    if "policy" in cfg.observations:
        cfg.observations["policy"].enable_corruption = False

    # 2. Remove Random Pushes
    if cfg.events is not None:
        cfg.events.pop("push_robot", None)

    # 3. Force Flat Terrain
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

    # 4. (NEW) REMOVE GAIT FEATURES IF REQUESTED
    # This effectively reverts the config to the "No Gait" version
    if disable_gait:
        print("[INFO] --disable_gait_features set: Removing gait terms from config...")

        # Remove Rewards (so the env doesn't crash looking for sensors)
        gait_rewards = ["foot_clearance", "foot_swing_height", "foot_slip"]
        for r in gait_rewards:
            if r in cfg.rewards:
                print(f"   - Removing reward: {r}")
                cfg.rewards.pop(r)

        # Remove Critic Observations (so shapes match the old model)
        # Note: Usually old models don't care about critic obs during play,
        # but we remove them to be safe.
        gait_obs = ["feet_height", "feet_air_time", "feet_contact"]
        if "critic" in cfg.observations:
            for o in gait_obs:
                if o in cfg.observations["critic"].func_dict:  # Accessing the terms dict
                    print(f"   - Removing critic obs: {o}")
                    cfg.observations["critic"].func_dict.pop(o)
                # Also check standard dict access if implemented differently
                elif hasattr(cfg.observations["critic"], "terms") and o in cfg.observations["critic"].terms:
                    print(f"   - Removing critic obs: {o}")
                    cfg.observations["critic"].terms.pop(o)


def run_gait_test(task: str, cfg: GaitTestConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Config
    env_cfg = load_env_cfg(task)

    # 2. Apply Overrides (strips gait features if needed)
    _apply_play_env_overrides(env_cfg, cfg.disable_gait_features)

    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task)

    # 3. Load Checkpoint
    if cfg.checkpoint_file is None:
        raise ValueError("Please provide a --checkpoint_file!")
    resume_path = Path(cfg.checkpoint_file)
    log_dir = resume_path.parent
    print(f"[INFO] Loading checkpoint: {resume_path.name}")

    # 4. Create Env
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5. Load Policy
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # 6. Locate Front-Left (FL) Foot
    robot = env.unwrapped.scene["robot"]
    try:
        all_sites = robot.site_names
        fl_site_name = next(name for name in all_sites if "FL" in name)
        fl_site_id = robot.find_sites(fl_site_name)[0][0]
    except StopIteration:
        fl_site_id = 0

    # 7. Run Test
    print(f"\n[INFO] Running Gait Test (Cmd Vx={cfg.cmd_vx} m/s)...")
    num_steps = 150
    obs, _ = env.reset()
    foot_heights = []

    with torch.no_grad():
        for i in range(num_steps):
            twist_cmd = env.unwrapped.command_manager.get_command("twist")
            twist_cmd[:] = torch.tensor([cfg.cmd_vx, 0.0, 0.0], device=device).unsqueeze(0)
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            z_pos = robot.data.site_pos_w[0, fl_site_id, 2].item()
            foot_heights.append(z_pos)

    env.close()

    # 8. Save & Plot
    save_filename = f"gait_data_{cfg.label}.npz"
    time_axis = np.arange(len(foot_heights)) * env.unwrapped.step_dt
    np.savez(save_filename, time=time_axis, height=np.array(foot_heights), label=cfg.label)

    data_files = glob.glob("gait_data_*.npz")
    plt.figure(figsize=(10, 6))
    for file in data_files:
        data = np.load(file)
        lbl = str(data["label"])
        t = data["time"]
        z = data["height"]
        style = '-' if lbl == cfg.label else '--'
        alpha = 1.0 if lbl == cfg.label else 0.7
        linewidth = 2.5 if lbl == cfg.label else 1.5
        plt.plot(t, z, linestyle=style, alpha=alpha, linewidth=linewidth, label=lbl)

    plt.axhline(y=0.1, color='k', linestyle=':', alpha=0.5, label="Target (0.1m)")
    plt.title(f"Deliverable 2.1: Front-Left Foot Trajectory Comparison\n(Cmd Vx = {cfg.cmd_vx} m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Foot Height (m)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.25)
    plt.savefig("deliverable_2_1_gait_plot.png")
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