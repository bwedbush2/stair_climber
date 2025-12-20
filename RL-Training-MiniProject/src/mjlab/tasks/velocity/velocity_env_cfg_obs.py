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
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

@dataclass(frozen=True)
class GaitTestConfig:
    checkpoint_file: str | None = None
    label: str = "Model" 
    cmd_vx: float = 0.6 
    device: str = "cuda:0"

def run_gait_test(task: str, cfg: GaitTestConfig):
    configure_torch_backends()
    
    # Load Env Config (It will load whatever is currently in env_cfgs.py)
    env_cfg = load_env_cfg(task)
    
    # Apply standard overrides (Noise/Terrain)
    env_cfg.episode_length_s = int(1e9)
    if "policy" in env_cfg.observations: env_cfg.observations["policy"].enable_corruption = False
    if env_cfg.events: env_cfg.events.pop("push_robot", None)
    if env_cfg.scene.terrain: 
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    
    env_cfg.scene.num_envs = 1
    
    # Create Env
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load Policy
    resume_path = Path(cfg.checkpoint_file)
    print(f"[INFO] Loading checkpoint: {resume_path.name}")
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(resume_path.parent), device=cfg.device)
    runner.load(str(resume_path), map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    # Track Foot
    robot = env.unwrapped.scene["robot"]
    try:
        fl_id = robot.find_sites(next(n for n in robot.site_names if "FL" in n))[0][0]
    except: fl_id = 0

    # Run Loop
    print(f"[INFO] Running Gait Test (Cmd Vx={cfg.cmd_vx})...")
    obs, _ = env.reset()
    heights = []
    
    with torch.no_grad():
        for _ in range(150):
            cmd = env.unwrapped.command_manager.get_command("twist")
            cmd[:] = torch.tensor([cfg.cmd_vx, 0.0, 0.0], device=cfg.device).unsqueeze(0)
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            heights.append(robot.data.site_pos_w[0, fl_id, 2].item())

    env.close()

    # Save Data
    np.savez(f"gait_data_{cfg.label}.npz", time=np.arange(len(heights))*0.02, height=heights, label=cfg.label)
    
    # Plot
    files = glob.glob("gait_data_*.npz")
    plt.figure(figsize=(10,6))
    for f in files:
        d = np.load(f)
        plt.plot(d['time'], d['height'], label=str(d['label']))
    plt.axhline(0.1, color='k', linestyle=':')
    plt.legend()
    plt.savefig("deliverable_2_1_gait_plot.png")

def main():
    task, args = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()), return_unknown_args=True)
    cfg = tyro.cli(GaitTestConfig, args=args)
    run_gait_test(task, cfg)

if __name__ == "__main__":
    main()