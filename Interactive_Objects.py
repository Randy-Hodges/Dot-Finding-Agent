import keyboard
import random
import numpy as np
from bw_configs import xlower_bound, xupper_bound, ylower_bound, yupper_bound, BORDER_DELTA, ZERO_LIMIT, REWARD_SIZE


class Player(object):
    '''Container for the methods and data used for the Player in the black world'''
    id = 0
    def __init__(
                self, 
                position = np.array([0,0]), 
                velocity = np.array([0,0]),
                acceleration = np.array([0,0]),
                force = np.array([0,0]),
                mass = 10,
                force_scale = 5,
                max_speed = 20,
                friction = .1,
                is_human = False
                ):
        self.pid = self.id
        self.id += 1
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.force = force
        self.mass = mass if mass != 0 else 10
        self.force_scale = force_scale
        self.max_speed = max_speed
        self.friction = friction
        self.is_human = is_human

    def steer(self, action) -> None:
        '''Push the player in a given direction'''
        if self.is_human:
            direction = self.get_direction()
        else:
            direction = self.map_action(action)
        
        self.force = np.array([0,0])

        direction = direction.lower()
        if 'up' in direction:
            self.force[1] += 1
        if 'down' in direction:
            self.force[1] += -1
        if 'right' in direction:
            self.force[0] += 1
        if 'left' in direction:
            self.force[0] += -1
        # normalize so that diagonal speeds are the same as cardinal speeds
        force_norm = np.linalg.norm(self.force)
        if force_norm != 0:
            self.force = self.force/force_norm
        

    def get_direction(self):
        direction = ''
        if keyboard.is_pressed('left'):
            direction += 'left'
        if keyboard.is_pressed('right'):
            direction += 'right'
        if keyboard.is_pressed('up'):
            direction += 'up'
        if keyboard.is_pressed('down'):
            direction += 'down'
        return direction

    def map_action(self, action):
        direction = ''
        if action == 0 or action == 1 or action == 7:
            direction += 'up'
        if action == 1 or action == 2 or action == 3:
            direction += 'right'
        if action == 3 or action == 4 or action == 5:
            direction += 'down'
        if action == 5 or action == 6 or action == 7:
            direction += 'left'
        if direction == 8:
            direction += 'no_movement'
        #print('action:', action)
        return direction
    
    def update_position(self, action=9):
        '''Update the position (plus other physics values) of the player'''
        self.steer(action)
        # time, should represent an interval of time, not sure what yet though
        t = 1
        # When the player is on the edge of the map, reduce the force/velocity, might change later with position
        # x edges
        if self.position[0] <= xlower_bound:
            self.velocity[0] = 0
            if self.force[0] < 0:
                self.force[0] = 0
        if self.position[0] >= xupper_bound:
            self.velocity[0] = 0
            if self.force[0] > 0:
                self.force[0] = 0 
        # y edges
        if self.position[1] <= ylower_bound:
            self.velocity[1] = 0
            if self.force[1] < 0: 
                self.force[1] = 0
        if self.position[1] >= yupper_bound:
            self.velocity[1] = 0
            if self.force[1] > 0:
                self.force[1] = 0
            
        # Update friction force
        friction_force = self.mass*self.friction
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm == 0:
            friction_direction = np.array([0,0])
        else:
            friction_direction = self.velocity/vel_norm

        # Find new acceleration | dF = ma  --> a = dF/m
        self.acceleration = (self.force*self.force_scale - friction_force*friction_direction) / self.mass

        # Find new velocity | a = (vf - v0)/t  -->  vf = a*t + v0 
        self.velocity = self.acceleration*t + self.velocity

        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm < ZERO_LIMIT:
            self.velocity = [0,0]

        # Find new Position | xf = x0 + v0*t + 1/2*a*t^2
        self.position = self.position + self.velocity*t + .5*self.acceleration*t*t

        # Handling edges of world 
        if self.position[0] <= xlower_bound:
            self.position[0] = xlower_bound 
        if self.position[0] >= xupper_bound:
            self.position[0] = xupper_bound
        if self.position[1] <= ylower_bound:
            self.position[1] = ylower_bound 
        if self.position[1] >= yupper_bound:
            self.position[1] = yupper_bound

    def random_position(self):
        self.position = np.array([random.randrange(xlower_bound, xupper_bound), 
                         random.randrange(ylower_bound, yupper_bound)])



class Reward(object):
    '''Container for the methods and data used for the Reward in the black world'''
    global xlower_bound
    global xupper_bound
    global ylower_bound
    global yupper_bound
    global BORDER_DELTA

    def __init__(self,
                position = np.array([0,0]),
                radius = REWARD_SIZE
                ):
        self.position = position
        self.radius = radius

    def random_position(self):
        self.position = np.array([random.randrange(xlower_bound, xupper_bound), 
                         random.randrange(ylower_bound, yupper_bound)])
  