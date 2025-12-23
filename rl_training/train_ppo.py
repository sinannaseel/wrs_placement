from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from SICRLS.envs import DependencyWorldEnv

# Vectorized env (even n_envs=1 helps PPO internals)
env = make_vec_env(
    DependencyWorldEnv,
    n_envs=4,
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,

    # ---- key hyperparameters (important) ----
    learning_rate=3e-4,
    n_steps=256,          # rollout length
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # encourage exploration early
    tensorboard_log="./ppo_logs/",
)

model.learn(total_timesteps=15_000)

model.save("ppo_dependency_seq")

env.close()
