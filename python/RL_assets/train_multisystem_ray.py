from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from ray_stairs_env import RayProceduralEnv


def make_env():
    return RayProceduralEnv()


if __name__ == "__main__":
    num_cpu = 8
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", batch_size=2048)

    model.learn(total_timesteps=5_000_000)