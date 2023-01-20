import numpy as np
import gym
from gym.spaces import Box, Discrete
import tensorflow as tf
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class MultiAgentEnv(gym.Env):
    def __init__(self):
        # action space
        self.action_space = Discrete(3) # 8 directions plus no movement (so 9 options)
        self.observation_space = Box(low, high, dtype=np.float32)



    def step(self, action_n):
        obs_n    = list()
        reward_n = list()
        done_n   = list()
        info_n   = {'n': []}
        # ...
        return obs_n, reward_n, done_n, info_n