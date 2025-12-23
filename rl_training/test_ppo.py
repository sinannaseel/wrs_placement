from stable_baselines3 import PPO
from SICRLS.envs import DependencyWorldEnv

env = DependencyWorldEnv()
model = PPO.load("ppo_dependency_seq")

obs, _ = env.reset()
sequence = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)

    if "picked" in info:
        sequence.append(info["picked"])

    print("Action:", action, "Info:", info)

    if term or trunc:
        break

print("\nLearned sequence:", sequence)

env.close()
