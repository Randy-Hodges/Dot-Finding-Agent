import time
import numpy as np
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import tensorflow as tf
# from stable_baselines3.common.env_checker import check_env
from ray.rllib.env.env_context import EnvContext

from interactive_objects import Player, Reward
from bw_configs import xlower_bound, xupper_bound, ylower_bound, yupper_bound,  \
    FRAME_INTERVAL, ANIM_TIME, FRAMES, DOT_SIZE, REWARD_SIZE, BORDER_DELTA, \
    ZERO_LIMIT


score = 0
frame_number = 0 # used to count frames in the step function

# time_template = '%.1fs'
# # preconfigure the black, square world
# plt.style.use('dark_background')
# plt.ion()
# fig = plt.figure(figsize=(5,5)) 
# ax = fig.add_subplot(xlim=(xlower_bound - BORDER_DELTA, xupper_bound + BORDER_DELTA), ylim=(ylower_bound - BORDER_DELTA, yupper_bound + BORDER_DELTA)) 
# bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# width, height = bbox.width, bbox.height

# point = 1/72 # inches
# inches_per_unit_width = width/(xupper_bound - xlower_bound) # one unit = this many inches | inches/units
# points_per_unit_width = inches_per_unit_width/point # one unit = this many points | (72 point / 1 inch)(inches_per_unit_width / 1 unit)

# DOT_SIZE2 = DOT_SIZE*points_per_unit_width
# REWARD_SIZE2 = REWARD_SIZE*points_per_unit_width

# player_marker, = ax.plot([], [], 'o-', markersize=DOT_SIZE2)  
# reward_marker, = ax.plot([], [], 'o-',color='red', markersize=REWARD_SIZE2)
# score_text = ax.text(xlower_bound + 5, yupper_bound - 5, "0")
# time_template = '%.1fs'
# time_text = ax.text(xupper_bound - 10, yupper_bound - 5, '0')

class dot_environment(gym.Env):

    def __init__(self, config: EnvContext = {}): 
        # action space
        self.action_space = Discrete(9) # 8 directions plus no movement (so 9 options)
        # observation space
        low = {
            'x': xlower_bound,
            'y': ylower_bound,
            'velx': np.NINF,
            'vely': np.NINF,
            'accx': np.NINF,
            'accy': np.NINF,
            'forcex': -1.0,
            'forcey': -1.0,
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
            'forcex': 1.0,
            'forcey': 1.0,
            'rewardx': xupper_bound,
            'rewardy': yupper_bound,
        }
        low = np.array(list(low.values()), dtype=np.float32)
        high = np.array(list(high.values()), dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32) 
        # score
        self.score = 0
        # Plotting/rendering values 
        self.fig = None 
        self.ax = None
        self.player_marker = None  
        self.reward_marker = None
        self.score_text = None
        self.time_text = None
        self.player = Player()
        self.player.random_position()
        self.reward_obj = Reward()
        self.reward_obj.random_position()
        

    def step(self, action):
        'ayoo this is my custom step function'
        global frame_number

        frame_number += 1
        score_change = -.01
        # Player Position
        self.player.update_position(action=action)
        x, y = self.player.position
        # print(f'x:{x}, y:{y}, accel:{self.player.acceleration}, vel:{self.player.velocity}, framecount:{framecount}')

        reward = self.reward_function()
        #time_text.set_text(time_template % (FRAME_INTERVAL*framecount/1000))
        done = False
        if frame_number >= FRAMES:
            frame_number = 0
            done = True
        state = self.get_state()

        return  state, \
                reward, \
                done, \
                {}
               

    def reward_function(self):
        '''Gives the reward for each step in the environment'''
        # Check if reward received
        reward = 0

        # reward for contacting the dot
        if np.linalg.norm(self.player.position - self.reward_obj.position) <= ((REWARD_SIZE/2 + DOT_SIZE/2)):
            reward += 100
            self.score += 100
            self.reward_obj.random_position()

        # penalty for moving away from the dot
        # if np.dot(self.reward_obj.position - self.player.position, self.player.velocity) <= 0:
        #     reward -= 1

        # rewarding based on velocity towards target (clipped at -2 and 2)
        player_to_reward = np.array(self.reward_obj.position - self.player.position)
        reward += np.clip(float(np.dot(self.player.velocity, player_to_reward/np.linalg.norm(player_to_reward))), -2, 2)

        # penalty for getting stuck in corner
        if abs(self.player.position[0]) == abs(xlower_bound)-3:
            #reward -= 1
            if abs(self.player.position[1]) == abs(xlower_bound)-3:
                reward -= 2

        return float(reward)


    def reset(self):
        self.reward_obj.random_position()
        self.player.random_position()
        self.score = 0
        state = np.array([self.player.position[0], self.player.position[1], self.player.velocity[0], self.player.velocity[1], \
                         self.player.acceleration[0], self.player.acceleration[1], self.player.force[0], self.player.force[1], \
                         self.reward_obj.position[0], self.reward_obj.position[1] \
                         ])
        return state.astype(np.float32)


    def get_state(self):
        state = np.array([self.player.position[0], self.player.position[1], self.player.velocity[0], self.player.velocity[1], \
                         self.player.acceleration[0], self.player.acceleration[1], self.player.force[0], self.player.force[1], \
                         self.reward_obj.position[0], self.reward_obj.position[1] \
                         ])
        state = state.astype(np.float32)
        return state
    
    
    def render(self):
        print(f"score:{self.score}")
        # player_marker.set_data(self.player.position)
        # reward_marker.set_data(self.reward_obj.position)
        # score_text.set_text(str(self.score))
        # time_text.set_text(time_template % (FRAME_INTERVAL*frame_number/1000))
        # fig.canvas.draw()
        # time.sleep(FRAME_INTERVAL/1000)


    def close(self):
        plt.close()


def render(env, model, print_states = False, rllib = False):
  # preconfigure the black, square world
  plt.style.use('dark_background')
  fig = plt.figure(figsize=(5,5)) 
  ax = fig.add_subplot(xlim=(xlower_bound - BORDER_DELTA, xupper_bound + BORDER_DELTA), ylim=(ylower_bound - BORDER_DELTA, yupper_bound + BORDER_DELTA)) 
  bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  width, height = bbox.width, bbox.height

  point = 1/72 # inches
  inches_per_unit_width = width/(xupper_bound - xlower_bound) # one unit = this many inches | inches/units
  points_per_unit_width = inches_per_unit_width/point # one unit = this many points | (72 point / 1 inch)(inches_per_unit_width / 1 unit)

  DOT_SIZE2 = DOT_SIZE*points_per_unit_width
  REWARD_SIZE2 = REWARD_SIZE*points_per_unit_width

  player_marker, = ax.plot([], [], 'o-', markersize=DOT_SIZE2)  
  reward_marker, = ax.plot([], [], 'o-',color='red', markersize=REWARD_SIZE2)
  score_text = ax.text(xlower_bound + 5, yupper_bound - 5, str(env.score))
  time_template = '%.1fs'
  time_text = ax.text(xupper_bound - 10, yupper_bound - 5, '0')

  def animate(framecount):
    obs = env.get_state()
    if rllib:
        action = model.compute_single_action(obs)
    else:
        action, _states = model.predict(obs)
        
    obs, _, done, _ = env.step(action)
    
    # stop animation if env is done
    if done:
      anim.event_source.stop()
      plt.close()

    player_marker.set_data(env.player.position)
    if print_states:
      tf.print(f'Pos: [{env.player.position[0]}, {env.player.position[1]}], Action: {action}')
    reward_marker.set_data(env.reward_obj.position)
    score_text.set_text(str(env.score))
    time_text.set_text(time_template % (FRAME_INTERVAL*framecount/1000))

    return player_marker, reward_marker, score_text, time_text
  
  
  def world_init(): 
    '''Initialization function for the square world'''
    env.reset()
    player_marker.set_data(env.player.position) 
    reward_marker.set_data(env.reward_obj.position)
    return player_marker, reward_marker, score_text, time_text

  anim = animation.FuncAnimation(fig, animate, init_func=world_init, 
                                frames=int(FRAMES), interval=FRAME_INTERVAL, blit=True, repeat=False) 
  plt.show()
        
                         
if __name__ == '__main__':
   
    env = dot_environment()
    # check_env(env)
