from stable_baselines3 import PPO
from ray_stairs_env import RayProceduralEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def main():
    print("🛠️  Initializing Training Environment...")
    env = RayProceduralEnv() # CPU Mode, No Render
    
    # Checkpoint: Save model every 100,000 steps so you don't lose progress
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path='./ray_checkpoints/',
        name_prefix='ray_stairs'
    )
    
    # Load existing model if it exists, otherwise create new
    # if os.path.exists("ray_stairs_policy.zip"):
    #     print("🔄 Loading existing model to continue training...")
    #     model = PPO.load("ray_stairs_policy", env=env, device="cpu",
    #                      ent_coef=0.05, learning_rate=3e-4)
    if False:
        print("??")
    else:
        print("🆕 Creating new model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device="cpu",
            learning_rate=2e-4,
            ent_coef=0.03,
            batch_size=2048,
            n_steps=2048,
            gamma=0.999
        )

    print("🏋️‍♂️ Training Started (Press Ctrl+C to Stop & Save)...")
    try:
        # 5 Million steps is a good overnight target
        model.learn(total_timesteps=50_000_000, callback=checkpoint_callback)
        model.save("ray_stairs_policy")
        print("✅ DONE! Saved 'ray_stairs_policy.zip'")
    except KeyboardInterrupt:
        model.save("ray_stairs_policy")
        print("\n🛑 Training Interrupted. Model Saved.")

if __name__ == "__main__":
    main()