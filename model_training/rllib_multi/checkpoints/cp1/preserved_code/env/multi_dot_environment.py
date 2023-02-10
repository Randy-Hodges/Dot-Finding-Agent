#!/usr/bin/python3 
#
#
# Desc: 
# 
# Warnings: copied from dot_environment, not yet implemented
# 

import time
import numpy as np
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import tensorflow as tf
# from stable_baselines3.common.env_checker import check_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from interactive_objects import Player, Reward
from bw_configs import *
# ---------------------------------------------------------------------------------------
score = 0
frame_number = 0 # used to count frames in the step function


class MultiDotEnvironment(MultiAgentEnv):

    def __init__(self, config: EnvContext = {}): 
        super().__init__
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
        self.reward_obj = Reward()
        self.reward_obj.random_position()
        self.reward_touched = False
        # Multiple Agents
        self.agent_dict = {}
        for i in range(NUM_STARTING_AGENTS):
            player = Player()
            player.random_position()
            self.agent_dict[str(i)] = player
        

    def reset(self):
        agent_state_dict = {}
        self.reward_obj.random_position()
        for id, agent in self.agent_dict.items():
            agent.random_position()
            state = np.array([agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1], \
                         agent.acceleration[0], agent.acceleration[1], agent.force[0], agent.force[1], \
                         self.reward_obj.position[0], self.reward_obj.position[1] \
                         ])
            agent_state_dict[id] = state.astype(np.float32)
        self.score = 0
        return agent_state_dict


    def step(self, actions):
        'ayoo this is my custom step function'
        global frame_number
        frame_number += 1
        score_change = -.01
        all_done = False
        states = {}
        rewards = {}
        dones = {}
        infos = {}
        # Agent action
        for id, agent in self.agent_dict.items():
            agent.update_position(action=actions[id])
            states[id] = self.get_state(agent)
            rewards[id] = self.reward_function(agent)
            dones[id] = (frame_number >= FRAMES)
            infos[id] = {}
        # Update reward position
        if self.reward_touched:
            self.reward_obj.random_position()
            self.reward_touched = False
        # Update frame count
        if frame_number >= FRAMES:
                frame_number = 0
                all_done = True
        dones["__all__"] = all_done

        return states, \
               rewards, \
               dones, \
               infos
               

    def reward_function(self, agent):
        '''Gives the reward for each step in the environment'''
        # Initial Reward
        reward = 0

        # Reward for contacting the dot
        if np.linalg.norm(agent.position - self.reward_obj.position) <= ((REWARD_SIZE/2 + DOT_SIZE/2)):
            reward += 100
            self.score += 100
            self.reward_touched = True  

        # Penalty for moving away from the dot
        # if np.dot(self.reward_obj.position - agent.position, agent.velocity) <= 0:
        #     reward -= 1

        # Rewarding based on velocity towards target (clipped at -2 and 2)
        player_to_reward = np.array(self.reward_obj.position - agent.position)
        reward += np.clip(float(np.dot(agent.velocity, player_to_reward/np.linalg.norm(player_to_reward))), -2, 2)

        # Penalty for getting stuck in corner 
        if abs(agent.position[0]) >= abs(xlower_bound)-3:
            if abs(agent.position[1]) >= abs(xlower_bound)-3:
                reward -= 2

        return float(reward)


    def get_state(self, agent):
        state = np.array([agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1], \
                         agent.acceleration[0], agent.acceleration[1], agent.force[0], agent.force[1], \
                         self.reward_obj.position[0], self.reward_obj.position[1] \
                         ])
        state = state.astype(np.float32)
        return state
    
    
    def render(self):
        # player_marker.set_data(self.player.position)
        # reward_marker.set_data(self.reward_obj.position)
        # score_text.set_text(str(self.score))
        # time_text.set_text(time_template % (FRAME_INTERVAL*frame_number/1000))
        # fig.canvas.draw()
        # time.sleep(FRAME_INTERVAL/1000)
        print(f"score:{self.score}")


    def close(self):
        plt.close()



def render(env, model, print_states = False, rllib = False):
  # region preconfigure the black, square world
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
  # endregion

  def animate(framecount):
    """Runs one frame of the animation"""
    for id, agent in env.agent_dict.items():
        obs = env.get_state(agent)
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


def render_multi(env, model, print_states = False, rllib = False):
    # region preconfigure the black, square world
    # Figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(5,5)) 
    ax = fig.add_subplot(xlim=(xlower_bound - BORDER_DELTA, xupper_bound + BORDER_DELTA), ylim=(ylower_bound - BORDER_DELTA, yupper_bound + BORDER_DELTA)) 
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    # Unit Definitions
    point = 1/72 # inches
    inches_per_unit_width = width/(xupper_bound - xlower_bound) # one unit = this many inches | inches/units
    points_per_unit_width = inches_per_unit_width/point # one unit = this many points | (72 point / 1 inch)(inches_per_unit_width / 1 unit)
    DOT_SIZE2 = DOT_SIZE*points_per_unit_width
    REWARD_SIZE2 = REWARD_SIZE*points_per_unit_width
    # Marker Creation
    player_markers = {}
    for id, agent in env.agent_dict.items():
        player_marker, = ax.plot([], [], 'o-', markersize=DOT_SIZE2)  
        player_marker.set_data(agent.position) 
        player_markers[id] = player_marker
    reward_marker, = ax.plot([], [], 'o-',color='red', markersize=REWARD_SIZE2)
    reward_marker.set_data(env.reward_obj.position)
    # Text Creation
    score_text = ax.text(xlower_bound + 5, yupper_bound - 5, str(env.score))
    time_template = '%.1fs'
    time_text = ax.text(xupper_bound - 10, yupper_bound - 5, '0')
    # plt.show()
    # endregion

    env.reset()
    frame_count = 0
    while True:
        plt.pause(FRAME_INTERVAL/1000)
        frame_count += 1
        actions = {}
        for id, agent in env.agent_dict.items():
            player_markers[id].set_data(agent.position)
            obs = env.get_state(agent)
            action = model.compute_single_action(obs, policy_id = id)
            actions[id] = action

        _, _, dones, _ = env.step(actions)
        reward_marker.set_data(env.reward_obj.position)
        if print_states:
            tf.print(f'Pos: [{env.player.position[0]}, {env.player.position[1]}], Action: {action}')
        score_text.set_text(str(env.score))
        time_text.set_text(time_template % (FRAME_INTERVAL*frame_count/1000))
        if dones["__all__"]:
            plt.close()
            break
    
    plt.show()
   

                         
if __name__ == '__main__':
   
    env = MultiDotEnvironment()
    print(env.reset())
    new_obs, rewards, dones, infos = env.step(
        actions={"0": 1, "1": 2})
    print(f"rewards: {rewards}")
    print(dones)

