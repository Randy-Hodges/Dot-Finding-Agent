import gym
import time

from stable_baselines3 import PPO
from env.bw_configs import FRAME_INTERVAL
# from stable_baselines3.common.env_util import make_vec_env
from dot_environment import render
import dot_environment as dot_environment

# Parallel environments
env = dot_environment.dot_environment()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# for _ in range(10000):
#     # action = model.predict(obs)
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         # env.close()
#         break
    

for _ in range(5):
  render(env, model)