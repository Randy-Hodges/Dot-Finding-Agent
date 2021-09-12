#!/usr/bin/python3
#
# Desc: Creates a square world with a white dot (our agent) and a red dot (our target). The agent gets rewarded for 
#       touching the red dot.

import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np

from bw_configs import FRAMES, FRAME_INTERVAL, DOT_SIZE, REWARD_SIZE, BORDER_DELTA, xlower_bound, \
    xupper_bound, ylower_bound, yupper_bound
from Interactive_Objects import Player, Reward

score = 0
is_human = True
model = None

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
score_text = ax.text(xlower_bound + 5, yupper_bound - 5, str(score))
time_template = '%.1fs'
time_text = ax.text(xupper_bound - 10, yupper_bound - 5, '0')
      
# ---------------------------------------------------------
# Initializing player and rewards
# weird place to put this stuff but whatever
player = Player(is_human=is_human)
reward = Reward()

# -------------------- Plot Functions ---------------------- (maybe should put in another file)
def world_init(): 
    '''Initialization function for the square world'''
	# creating an empty plot/frame 
    player_marker.set_data([], []) 
    reward.random_position()
    reward_marker.set_data(reward.position)
    return player_marker, reward_marker, score_text, time_text

def animate(framecount):
    global player_marker
    global player
    global reward
    global score

    # Player Position
    player.update_position()
    x, y = player.position
    player_marker.set_data(player.position)
    # print(f'x:{x}, y:{y}, accel:{player.acceleration}, vel:{player.velocity}, framecount:{framecount}')

    # Check if reward received
    # if reward, update score and new reward position
    if np.linalg.norm(player.position - reward.position) <= ((REWARD_SIZE/2 + DOT_SIZE/2)):
        score += 100
        score_text.set_text(str(score))
        reward.random_position()
        reward_marker.set_data(reward.position)

    time_text.set_text(time_template % (FRAME_INTERVAL*framecount/1000))

    return player_marker, reward_marker, score_text, time_text

def customize_plot():
    # setting a title for the plot 
    plt.title('Lil Mini-Game') 

    # hiding the axis details 
    #plt.axis('off') 



# --------------------------------------------------------- 
def main(player = None, reward = None, model = None):
    global is_human
    if not is_human:
        player = player
        reward = reward

    customize_plot()

    # call the animator	 
    anim = animation.FuncAnimation(fig, animate, init_func=world_init, 
                                frames=int(FRAMES), interval=FRAME_INTERVAL, blit=True, repeat=False) 
    plt.show()

if __name__ == '__main__':
    main()
else:
    is_human = False




    