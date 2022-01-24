# Dot-Finding-Agent
This project aims to use machine learning, specifically an Actor-Critic method, to have an agent learn how to play a minigame.

### The minigame
This is a simple game where the player controls one dot in a 2d plane and is trying to move their dot to the randomly placed 'target' dot. The player uses the arrow keys to apply forces to their dot, and this (hopefully) pushes their dot towards the target dot. Once the target dot is collided with, it gets randomly moved to a new location and the game ends after once time expires. There is a bit of momentum and a slight drag force on the player dot to make the game less trivial. 

### Agent
The Agent that is learning to play the game is using an Actor-Critic method, which uses two neural networks. One for choosing actions and one for evaluating how those actions will perform. I am trying to migrate my algorithm to be closer to a Proximal Policy Optimization, which from my understanding is just an actor critic method with many added 'tricks' that help the actor learn better. 

### Success
Success for the agent is ultimately determined by how fast it scores points, or in otherwords, how fast it can collide with the target dot. While this might not always be the reward function for the agent, it is ultimately how I view the agent as successful.

While there is more work to be done, my algorithm so far has produced an agent that can play the game somewhat well. It will actively seek out the target dot and will collide with it an average of over 3 times over a 5 second game. <insert media here>

### Purpose
I want to pratice implementing ML algorithms and gain a better understanding of how they work. I felt the best way to help me understand would be through creation. 

### File Breakdown
Will update in the future.

### Other Notes
- I implemented this game using matplotlib because I was familiar with Matlab. Another, more standard way to display the game might have been through using something like turtle, but matplotlib is working perfectly fine and the way I choose to graphically display the game does not really matter.
- There are better algorithms to use for this type of game (especially off-policy algorithms), but I specifically wanted to practice implementing an actor-critic and this is the game I chose to use. 
- This is a work in progress and I am definitely new to ML so some of the code might be messier/less standard than the stuff I normally create and organize.  
