import time
import numpy as np
import mujoco
import mujoco.viewer
from ray_stairs_env import RayProceduralEnv
from stable_baselines3 import PPO
import os

def main():
    print("🎬 Initializing Visual Test...")

    # 1. Setup Env (No internal render, we handle it manually)
    env = RayProceduralEnv(render_mode=None)
    
    # 2. Load Model (Optional)
    model = None
    if os.path.exists("ray_stairs_policy.zip"):
        print("🧠 Model Loaded.")
        model = PPO.load("ray_stairs_policy", device="cpu")
    else:
        print("⚠️ No model found. Using Random Actions.")

    # 3. MAIN LOOP
    # We loop FOREVER. Each iteration is a new Map/Episode.
    while True:
        try:
            # A. Generate New World
            obs, _ = env.reset()
            
            # B. Launch Viewer for THIS specific world
            # We must use 'launch_passive' inside the loop because 'env.model' changes every time!
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                
                # Configure Camera
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "car")
                viewer.cam.distance = 4.0
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 90
                
                # Sync once to show the spawn state
                viewer.sync()
                
                # Wait 1s so you can see the new map before action starts
                time.sleep(1.0)
                
                step_count = 0
                
                # C. Episode Loop
                while viewer.is_running():
                    step_start = time.time()

                    # 1. Action
                    if model:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        # Sit still if no model (Debug Physics)
                        action = np.zeros(1)

                    # 2. Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    # 3. Sync Viewer
                    viewer.sync()

                    # 4. Check Termination
                    if terminated:
                        print(f"❌ Terminated at Step {step_count} | Reward: {reward:.1f}")
                        
                        # Freeze for 1 second so you can see the crash
                        end_time = time.time()
                        while time.time() - end_time < 1.0:
                            viewer.sync()
                            time.sleep(0.02)
                        
                        # Break inner loop to trigger a Reset (and new window)
                        break

                    # 5. Frame Rate Cap (30 FPS)
                    # This prevents the window from freezing/ghosting
                    time_until_next_frame = 0.033 - (time.time() - step_start)
                    if time_until_next_frame > 0:
                        time.sleep(time_until_next_frame)
                        
        except KeyboardInterrupt:
            print("🛑 Stopped.")
            break
        except Exception as e:
            # If MuJoCo crashes on close, just print and continue
            print(f"⚠️ Warning: {e}")
            break

    env.close()

if __name__ == "__main__":
    main()