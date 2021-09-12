import numpy as np
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt 
import tensorflow as tf

from Interactive_Objects import Player, Reward
from bw_configs import xlower_bound, xupper_bound, ylower_bound, yupper_bound,  \
    FRAME_INTERVAL, ANIM_TIME, FRAMES, DOT_SIZE, REWARD_SIZE, BORDER_DELTA, \
    ZERO_LIMIT


score = 0
frame_number = 0 # used to count frames in the step function

time_template = '%.1fs'
render_preconfig = False

class dot_environment(gym.Env):

    def __init__(self):
        # action space
        self.action_space = Discrete(9) # 8 directions plus no movement (so 9 options)
        # observation space
        low = {
            'x': xlower_bound,
            'y': ylower_bound,
            'velx': 0,
            'vely': 0,
            'accx': 0,
            'accy': 0,
            'forcex': -1,
            'forcey': -1,
            'rewardx': xlower_bound,
            'rewardy': ylower_bound,
        }
        high = {
            'x': xupper_bound,
            'y': yupper_bound,
            'velx': np.inf,
            'vely': np.inf,
            'accx': np.inf,
            'accy': np.inf,
            'forcex': 1,
            'forcey': 1,
            'rewardx': xupper_bound,
            'rewardy': yupper_bound,
        }
        low = np.array(list(low.values()), dtype=np.float)
        high = np.array(list(high.values()), dtype=np.float)
        self.observation_space = Box(low, high, dtype=np.float) 
        # score
        self.score = 0
        # Plotting/rendering values 
        self.fig = None 
        self.ax = None
        self.player_marker = None  
        self.reward_marker = None
        self.score_text = None
        self.time_text = None


    def step(self, action, player: Player, reward_obj: Reward):
        'ayoo this is my custom step function'
        global frame_number

        frame_number += 1
        score_change = -.01
        # Player Position
        player.update_position(action=action)
        x, y = player.position
        # print(f'x:{x}, y:{y}, accel:{player.acceleration}, vel:{player.velocity}, framecount:{framecount}')

        reward = self.reward_function(player, reward_obj)
        #time_text.set_text(time_template % (FRAME_INTERVAL*framecount/1000))
        done = False
        if frame_number >= FRAMES:
            frame_number = 0
            done = True

        return np.array([player.position[0], player.position[1], player.velocity[0], player.velocity[1], \
                         player.acceleration[0], player.acceleration[1], player.force[0], player.force[1], \
                         reward_obj.position[0], reward_obj.position[1], \
                         ]), \
               np.array(reward), \
               np.array(done)
        

    def reward_function(self, player: Player, reward_obj: Reward):
        '''Gives the reward for each step in the environment'''
        # Check if reward received
        reward = 0

        # reward for contacting the dot
        if np.linalg.norm(player.position - reward_obj.position) <= ((REWARD_SIZE/2 + DOT_SIZE/2)):
            reward += 100
            self.score += 100
            reward_obj.random_position()

        # penalty for moving away from the dot
        # if np.dot(reward_obj.position - player.position, player.velocity) <= 0:
        #     reward -= 1

        # rewarding based on velocity towards target (clipped at -2 and 2)
        player_to_reward = np.array(reward_obj.position - player.position)
        reward += np.clip(float(np.dot(player.velocity, player_to_reward/np.linalg.norm(player_to_reward))), -2, 2)

        # penalty for getting stuck in corner
        if abs(player.position[0]) == abs(xlower_bound)-3:
            #reward -= 1
            if abs(player.position[1]) == abs(xlower_bound)-3:
                reward -= 2

        return reward


    def reset(self, player: Player, reward_obj: Reward):
        reward_obj.random_position()
        player.random_position()
        self.score = 0
        state = np.array([player.position[0], player.position[1], player.velocity[0], player.velocity[1], \
                         player.acceleration[0], player.acceleration[1], player.force[0], player.force[1], \
                         reward_obj.position[0], reward_obj.position[1], \
                         ])
        return state.astype(np.float32)


    def get_state(self, player: Player, reward_obj: Reward):
        state = np.array([player.position[0], player.position[1], player.velocity[0], player.velocity[1], \
                         player.acceleration[0], player.acceleration[1], player.force[0], player.force[1], \
                         reward_obj.position[0], reward_obj.position[1], \
                         ])
        return state
    
        

        
                         
if __name__ == '__main__':
    player = Player()
    reward = Reward()
    env = dot_environment()
    env.step(2, player, reward)