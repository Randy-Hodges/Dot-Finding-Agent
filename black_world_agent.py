
from typing import Any, List, Tuple
from numpy.core.numeric import Inf
# Machine learning
import tensorboard
import numpy as np
import tensorflow as tf
# helpful
import statistics
import collections
import tqdm
import datetime
import pickle
# animation
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
# custom
import dot_environment
from ActorCritic import ActorCritic
from Interactive_Objects import Player, Reward
from bw_configs import FRAMES, FRAME_INTERVAL, DOT_SIZE, REWARD_SIZE, BORDER_DELTA, xlower_bound, \
    xupper_bound, ylower_bound, yupper_bound
print('----------------------------')

# Create Environment
env = dot_environment.dot_environment()
player = Player()
reward_obj = Reward()

# Model inits
num_actions = env.action_space.n
num_hidden_units = 128
optimizer = tf.keras.optimizers.Adam(learning_rate=.1, clipvalue=2.0)
model = ActorCritic(num_actions, num_hidden_units)
gamma = .95

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_log_dir = 'logs/gradient_tape/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Custom inits
print_action = False
print_loss = False
actions_str = ''


# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

# region Algorithm
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  global actions_str
  if print_action:
    actions_str += str(action)
  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])


def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int
    ):
    '''Runs a single episode to collect data'''
    global print_action

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # convert state into a batched tensor (batch size = 1), flattens the state?*
        state = tf.expand_dims(state, 0)

        # run model with state to get action_probs and critic value
        action_logits_t, value = model(state)

        # sample next action from the distribution
        action = tf.random.categorical(action_logits_t, 1)[0,0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value)) # idk why squeeze

        # Store action probs
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # apply action
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(
  rewards: tf.Tensor,
  gamma: float,
  standardize: bool = True
  ):
  '''Compute expected rewards per timestep'''
  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(tf.float32, size=n)

  # Start from the end of rewards and accumulate reward sums backwards into returns
  rewards = tf.cast(rewards[::-1], dtype=tf.float32) # Flip rewards
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + discounted_sum*gamma
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = (returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + eps)

  return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
  action_probs: tf.Tensor,
  values: tf.Tensor,
  returns:tf.Tensor
  ):
  '''Compute combined actor and critic loss'''
  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs*advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


@tf.function
def training_step(
  initial_state: tf.Tensor,
  model: tf.keras.Model,
  gamma: float,
  max_steps: int,
  optimizer: tf.keras.optimizers.Optimizer
  ) -> tf.Tensor:
  '''Runs a model training step'''

  with tf.GradientTape() as tape:

    # run the model for one training step to collect data
    action_probs, values, rewards = run_episode(initial_state, model, max_steps)

    # calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # compute loss
    loss = compute_loss(action_probs, values, returns)
   
   
  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward, loss
# endregion


def render_episode(env, model: tf.keras.Model, max_steps: int): 

  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))
    tf.print(action, end='')

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    env.render()

    if done:
      break


def render(env, model, print_states = False):
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

  initial_state = tf.constant(env.reset(), dtype=tf.float32)

  def animate(framecount):

    # convert state into a batched tensor (batch size = 1), flattens the state?*
    state = tf.expand_dims(env.get_state(), 0)

    # run model with state to get action_probs and critic value
    action_logits_t, value = model(state)

    # sample next action from the distribution
    action = tf.random.categorical(action_logits_t, 1)[0,0]

    # apply action
    state, reward, done, _ = tf_env_step(action)
    state = tf.constant(state, dtype=tf.float32)
    
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



# ------------- Run Training Loop -------------
# region
print('=============================')
model.compile(optimizer=optimizer)
min_episodes = 100 # also determines length of running reward/loss
max_episodes = 30000
view_rate = 100 # rate at which episode data is displayed (every x episodes)
render_rate = 2000 # rate at which episodes are rendered (every x episodes)
max_steps = FRAMES
# Reward tracking
reward_threshold = 300
running_reward = 0
running_loss = 0


def run_training_loop(model: tf.keras.Model):
  '''
  Trains the agent on the environment
  
  :returns: True if success criteria met, False if not
  '''
  global actions_str
  global print_action

  # improvement tracking
  highest_running_reward = -Inf
  non_improvement_count = 0 # increases by one every min episodes
  non_improvement_cap = 5 # resets

  # store best model weights
  model_weights = None
  # don't reset for the same weights too many times
  same_reset_count = 0
  # min episodes for best-weights reset is same_reset_thresh*view_rate
  same_reset_thresh = 5 # resets


  # keep a log of rewards
  episodes_rewards: collections.deque = collections.deque(maxlen=min_episodes)
  episodes_loss: collections.deque = collections.deque(maxlen=min_episodes)

  # Run training loop
  with tqdm.trange(max_episodes) as t:
    for i in t:
      # get episode reward
      initial_state = env.reset()
      episode_reward, loss = map(int, list(training_step(initial_state, model, gamma, FRAMES, optimizer)))
      with train_summary_writer.as_default():
        tf.summary.scalar('Loss', loss, step=i)
        tf.summary.scalar('Episode Reward', episode_reward, step=i)

      # log reward
      episodes_rewards.append(episode_reward)
      running_reward = statistics.mean(episodes_rewards)
      # log loss
      episodes_loss.append(loss)
      running_loss = statistics.mean(episodes_loss)

      # describe tqdm stats
      t.set_description(f'Episode {i}')
      t.set_postfix(episode_reward = episode_reward, running_reward = running_reward)      

      # Show average episode reward every view_rate episodes and update highest_running_reward
      if i % view_rate == 0:
        print(f'\nEpisode {i}: running reward: {running_reward} Running Loss: {running_loss} Highest_running_reward: {highest_running_reward} Same Reset Count: {same_reset_count}')
        # Update highest running reward, discounted by 5% so progress can be made
        if running_reward > highest_running_reward*.95 and i > min_episodes:
          highest_running_reward = running_reward
          print(f'highest_running_reward: {highest_running_reward}')
          # update saved weights
          model_weights = model.get_weights()
          same_reset_count = 0
        else:
          non_improvement_count += 1
          if non_improvement_count >= non_improvement_cap and i > min_episodes:
            # reset weights
            print('----- no improvement, resetting -----')
            model.set_weights(model_weights)
            non_improvement_count = 0
            same_reset_count += 1
            if same_reset_count >= same_reset_thresh:
              print('********** Resetting highest_running_reward ************')
              same_reset_count = 0
              highest_running_reward = -Inf

        print_action = True
        print_loss = True

      # only print action once
      elif print_action == True:
        t.write(actions_str)
        print_action = False
        actions_str = ''
        print_loss = False
      
      # Break if success criteria are met
      if running_reward >= reward_threshold and i >= min_episodes:
        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
        model.save('./models/best_model1')
        model_weights = model.get_weights()
        render(env, model)
        return True, model_weights

      # Render episode every render_rate episodes
      if i%render_rate == 0 and i != 0:
        print_action = False
        actions_str = ''
        render(env, model, print_states=False)

  return False, model_weights


success, model_weights = run_training_loop(model)
# store model weights
if success:
  model.save_weights('good_weights')

# training_cap = 3
# for i in range(training_cap):
#   success = run_training_loop(model)
#   if not success:
#     print('model failed')
#   else:
#     break

for _ in range(20):
  render(env, model)
# endregion
exit()

# ----------------- TESTING CODE ---------------------
num_actions = env.action_space.n
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)
gamma = .98

print('=============================')

initial_state = env.reset(player, reward_obj)
action_probs, values, rewards = run_episode(tf.convert_to_tensor(initial_state), model, FRAMES)

returns = get_expected_return(rewards, gamma)
bool_mask = tf.cast(rewards, tf.bool)
no_zeros = tf.boolean_mask(rewards, bool_mask)
tf.print()
tf.print(no_zeros)

loss = compute_loss(action_probs, values, returns)
tf.print('loss:', loss)


optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
episode_rewards = int(training_step(initial_state, model, gamma, FRAMES, optimizer))
tf.print('\n rewards:', episode_rewards)





