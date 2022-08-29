import functools
import numpy as np
from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import api_test

import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import tensorflow as tf

from Interactive_Objects import Player, Reward
from bw_configs import xlower_bound, xupper_bound, ylower_bound, yupper_bound,  \
    FRAME_INTERVAL, ANIM_TIME, FRAMES, DOT_SIZE, REWARD_SIZE, BORDER_DELTA, \
    ZERO_LIMIT

print("---------------------------------------------------------------------\n")
score = 0
frame_number = 0 # used to count frames in the step function 
time_template = '%.1fs'


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env():
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env()
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pettingzoo_parallel"}

    def __init__(self):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.my_num_agents = 3
        self.possible_agents = ["player_" + str(r) for r in range(self.my_num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.actspace = Discrete(9) # 8 directions plus no input (so 9 options)
        self._action_spaces = {agent: self.actspace for agent in self.possible_agents}
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
        self.obsspace = Box(low, high, dtype=np.float32) 
        self._observation_spaces = {
            agent: self.obsspace for agent in self.possible_agents
        }
        # score
        self.score = 0
        # Plotting/rendering values 
        # self.fig = None 
        # self.ax = None
        # self.player_marker = None  
        # self.reward_marker = None
        # self.score_text = None
        # self.time_text = None
        self.players = [Player() for _ in range(self.my_num_agents)]
        for player in self.players:
            player.random_position()
        self.reward_obj = Reward()
        self.reward_obj.random_position()
        self.frame_number = 0


    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.obsspace


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.actspace


    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        # if len(self.agents) == 2:
        #     string = "Current state: Agent1: {} , Agent2: {}".format(
        #         MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
        #     )
        # else:
        #     string = "Game over"
        # print(string)
        pass


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass


    def reset(self, seed=None, **kargs):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        # print(f"self.agents: {self.agents}")
        self.num_moves = 0
        observations = {agent: self.get_state(agent) for agent in self.agents}
        return observations


    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}
        
        self.frame_number += 1

        # perform actions
        for agent, action in actions.items():
            player = self.get_player(agent)
            player.update_position(action=action)

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        rewards = {agent: self.get_reward(agent) for agent in self.agents}

        # update dones
        # updates dones
        done = False
        if self.frame_number >= FRAMES:
            # self.frame_number = 0
            done = True
        dones = {agent: done for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {agent: self.get_state(agent) for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if done:
            self.agents = []

        return observations, rewards, dones, infos


    def get_reward(self, agent):
        '''Gives the reward for each step in the environment'''
        # Check if reward received
        reward = 0
        player = self.get_player(agent)

        # reward for contacting the dot
        if np.linalg.norm(player.position - self.reward_obj.position) <= ((REWARD_SIZE/2 + DOT_SIZE/2)):
            reward += 100
            self.score += 100
            self.reward_obj.random_position()

        # penalty for moving away from the dot
        # if np.dot(self.reward_obj.position - player.position, player.velocity) <= 0:
        #     reward -= 1

        # rewarding based on velocity towards target (clipped at -2 and 2)
        player_to_reward = np.array(self.reward_obj.position - player.position)
        reward += np.clip(float(np.dot(player.velocity, player_to_reward/np.linalg.norm(player_to_reward))), -2, 2)

        # penalty for getting stuck in corner 
        # (Looking back at this, idk what the 3 is. An offset of some kind but idk why the number 3)
        if abs(player.position[0]) == abs(xlower_bound)-3:
            #reward -= 1
            if abs(player.position[1]) == abs(xlower_bound)-3:
                reward -= 2

        return float(reward)

        
    def get_state(self, agent):
        player = self.get_player(agent)
        state = np.array([player.position[0], player.position[1], player.velocity[0], player.velocity[1], \
                         player.acceleration[0], player.acceleration[1], player.force[0], player.force[1], \
                         self.reward_obj.position[0], self.reward_obj.position[1] 
                         ])
        state = state.astype(np.float32)
        return state


    def get_player(self, agent_name):
        "Gets the Player object for a given agent name"
        agent_num = self.agent_name_mapping[agent_name]
        player = self.players[agent_num]
        return player



def render_parallel(env, model, print_states = False):
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
    reward_marker, = ax.plot([], [], 'o-',color='red', markersize=REWARD_SIZE2)
    player_markers, = {agent: ax.plot([], [], 'o-', markersize=DOT_SIZE2) for agent in env.agents}
    score_text = ax.text(xlower_bound + 5, yupper_bound - 5, str(env.score))
    time_template = '%.1fs'
    time_text = ax.text(xupper_bound - 10, yupper_bound - 5, '0')

    def animate(framecount):
        obs, rewards, dones, infos = env.step()
        action, _states = model.predict(obs)
        env.step(action)
        # obs = env.get_state()
        # obs, _, done, _ = env.step(action)
        
        # stop animation if env is done
        if done:
            anim.event_source.stop()
            plt.close()

        # setting data
        for agent in env.agents:
            player = env.get_player(agent)
            player_markers[agent].set_data(player.position)
        if print_states:
            tf.print(f'Pos: [{env.player.position[0]}, {env.player.position[1]}], Action: {action}')
        reward_marker.set_data(env.reward_obj.position)
        score_text.set_text(str(env.score))
        time_text.set_text(time_template % (FRAME_INTERVAL*framecount/1000))

        return player_markers, reward_marker, score_text, time_text
    
    def world_init(): 
        '''Initialization function for the square world'''
        env.reset()
        for agent in env.agents:
            player = env.get_player(agent)
            player_markers[agent].set_data(player.position)
        reward_marker.set_data(env.reward_obj.position)
        return player_markers, reward_marker, score_text, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=world_init, 
                                    frames=int(FRAMES), interval=FRAME_INTERVAL, blit=True, repeat=False) 
    plt.show()


if __name__ == '__main__':
    env = env()
    api_test(env, num_cycles=1000, verbose_progress=False)