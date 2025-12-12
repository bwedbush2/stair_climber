import time
import numpy as np
import mujoco
import mujoco.viewer
from ray_stairs_env import RayProceduralEnv
from stable_baselines3 import PPO
import os

def main():
    print("Testing Intializing")

    env = RayProceduralEnv(render_mode=None)
    
    model = None
    if os.path.exists("ray_stairs_policy.zip"):
        print("Model Succesfully Loaded")
        model = PPO.load("ray_stairs_policy", device="cpu")
    else:
        print("No model found. ---------BIG ISSUE----------")

    while True:
        try:
            obs, _ = env.reset()
            
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "car")
                viewer.cam.distance = 4.0
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 90
                
                viewer.sync()
                
                time.sleep(1.0)
                
                step_count = 0
                
                while viewer.is_running():
                    step_start = time.time()

                    if model:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        action = np.zeros(1)

                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    viewer.sync()

                    if terminated:
                        print(f"Terminated at Step {step_count} with Reward: {reward:.1f}")
                        

                        end_time = time.time()
                        while time.time() - end_time < 1.0:
                            viewer.sync()
                            time.sleep(0.02)
                        
                        break

                    time_until_next_frame = 0.033 - (time.time() - step_start)
                    if time_until_next_frame > 0:
                        time.sleep(time_until_next_frame)
                        
        except KeyboardInterrupt:
            print("System Interrupted")
            break
        except Exception as e:
            print(f"Warning: {e}")
            break

    env.close()

if __name__ == "__main__":
    main()