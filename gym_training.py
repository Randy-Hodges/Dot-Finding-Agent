import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dot_environment import render
import dot_environment

# Parallel environments
env = dot_environment.dot_environment()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=60000)
# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

for _ in range(5):
  render(env, model)