import time
import numpy as np
from stable_baselines3 import PPO
from ray_stairs_env import RayProceduralEnv

def main():
    print("🎬 Initializing Visual Test...")

    # 1. Setup Env
    env = RayProceduralEnv(render_mode="human")
    
    # 2. Load Policy (Or use None to test random actions)
    try:
        model = PPO.load("ray_stairs_policy", device="cpu")
        print("🧠 Model Loaded.")
    except:
        print("⚠️ No model found. Running Random Actions.")
        model = None

    obs, _ = env.reset()
    
    # --- FIX FOR "FROZEN" START ---
    # Instead of time.sleep(1.0), we loop and sync the viewer.
    # This keeps the window alive while we wait.
    print("⏳ Waiting for viewer...")
    for _ in range(30): 
        env._render_frame()
        time.sleep(0.05)

    print("🚀 Running...")
    
    try:
        while True:
            # 1. Get Action
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(1) # Sit still if no model

            # 2. Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Render & Slow Down
            # Loop multiple times per step for smooth 60fps video
            # (Since physics runs at 20 steps per frame, we just wait a bit)
            time.sleep(0.02) 
            
            # 4. Handle Reset
            if terminated:
                print(f"🔄 Resetting... Reward: {reward:.1f}")
                
                # "Pause" on death without freezing window
                for _ in range(20): 
                    env._render_frame()
                    time.sleep(0.05)
                    
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("🛑 Stopped.")
        env.close()

if __name__ == "__main__":
    main()