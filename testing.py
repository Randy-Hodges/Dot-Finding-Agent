from tensorflow.python.keras.saving.save import load_model
from ActorCritic import ActorCritic
import pickle


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# num_actions = 9
# num_hidden_units = 128
# model = ActorCritic(num_actions, num_hidden_units)
# model_weights = model.get_weights()
# model.save_weights('test_weights')
# #model.save('test_model')
# model = model.load_weights('test_weights.index')
# #model2 = load_model('test_model')
# print('yaaay')
exit()
import tensorflow as tf
import dot_environment
from Interactive_Objects import Player, Reward
from black_world_agent import ActorCritic

env = dot_environment.dot_environment()
player = Player()
reward = Reward()
model = ActorCritic(4, 128)

state = env.reset(player, reward)
state = tf.expand_dims(state, 0)

# run model with state to get action_probs and critic value
action_logits_t, value = model(state)

# sample next action from the distribution
action = tf.random.categorical(action_logits_t, 1)[0,0]
print('-----------------------')
tf.print(action)