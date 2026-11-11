from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mujoco_plain.fr3_gymnasium import Fr3GymnasiumEnv

check_env(Fr3GymnasiumEnv())

env = Monitor(Fr3GymnasiumEnv())
try:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./runs/fr3_ppo",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",
    )
    model.learn(total_timesteps=1_000_000)  # short smoke test first
    model.save("fr3_ppo")
finally:
    env.close()