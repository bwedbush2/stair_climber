from stable_baselines3 import PPO
from ray_stairs_env import RayFocusedStairs
import os

def main():
    # 1. SETUP
    print("🛠️ Generating Scenario 3 XML...")
    # Make sure you run build_ray.py manually or we can call it here if organized
    # For now, assuming you ran: python build_ray.py -> 3
    
    env = RayFocusedStairs() # CPU Mode
    
    # 2. MODEL (Tuned Hyperparameters)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu", # Force CPU for stability
        learning_rate=3e-4,
        ent_coef=0.01,    # Encourage exploration (prevents static leg holding)
        batch_size=2048,  # Smoother updates
        n_steps=2048,
        gamma=0.99
    )

    print("🏋️‍♂️ Training Focused Stair Climber...")
    try:
        # 1 Million steps on this focused task is ALOT. It should solve it in 200k.
        model.learn(total_timesteps=35000000)
        model.save("ray_stairs_policy")
        print("✅ Saved 'ray_stairs_policy.zip'")
    except KeyboardInterrupt:
        model.save("ray_stairs_policy")
        print("🛑 Saved Interrupted Model")

    # 3. TEST
    print("🎬 Running Test...")
    env = RayFocusedStairs(render_mode="human")
    obs, _ = env.reset()
    
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()