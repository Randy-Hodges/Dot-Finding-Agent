from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers



class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):

    super().__init__()
    self.common = layers.Dense(num_hidden_units, activation="tanh")
    self.common2 = layers.Dense(num_hidden_units, activation="tanh")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    x2 = self.common2(x)
    return self.actor(x2), self.critic(x2)
