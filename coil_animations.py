
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
plt.style.use('dark_background')

fig = plt.figure() 
ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50)) 
line, = ax.plot([], [], lw=2) 

# initialization function 
def init(): 
	# creating an empty plot/frame 
	line.set_data([], []) 
	return line, 

# lists to store x and y axis points 
xdata, ydata = [], [] 

sign = 1
theta = .1
adjuster1 = 1
adjuster2 = 1
# animation function 
def animate(i): 
	global sign
	global theta
	global adjuster1
	global adjuster2
	theta = theta + sign*.1
	# t is a parameter 
	t = 0.1*i 
	if i % adjuster2 == 0:
		sign *= -1
		temp = adjuster1
		adjuster1 = adjuster2
		adjuster2 += temp
	t = t*sign
	# x, y values to be plotted 
	x = t*np.sin(theta)*sign
	y = t*np.cos(theta)*sign
	
	# appending new points to x, y axes points list 
	xdata.append(x) 
	ydata.append(y) 
	line.set_data(xdata, ydata) 
	return line, 

# setting a title for the plot 
plt.title('Alternating Coil') 
# hiding the axis details 
plt.axis('off') 

# call the animator	 
anim = animation.FuncAnimation(fig, animate, init_func=init, 
							frames=500, interval=20, blit=True) 

plt.show()

# save the animation as mp4 video file 
anim.save('coil2.gif') 