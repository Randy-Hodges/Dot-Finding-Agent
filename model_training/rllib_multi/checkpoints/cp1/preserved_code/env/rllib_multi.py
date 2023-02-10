# 
# TODO: 
import argparse
import os
import random  

import tensorflow as tf
import re

import ray
# from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.examples.models.shared_weights_model import (
#     SharedWeightsModel1,
#     SharedWeightsModel2,
# )
# from ray.rllib.env.env_context import EnvContext
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
import warnings
warnings.filterwarnings('ignore')

from multi_dot_environment import render_multi
from multi_dot_environment import MultiDotEnvironment
from bw_configs import *
# -------------------------------------------------------------------------------------------
# region Trying to turn off GPU. Doesn't turn it off, but code is faster so I'm leaving it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
# endregion


def gen_policy(i):
    """Generates a policy for an agent"""
    gammas = [0.90, 0.95, 0.99]
    config = {
        "gamma": gammas[i%3],
    }
    return PolicySpec(config=config)


def policy_mapping_fn(agent_id, episode = None, worker = None, **kwargs):
    """Given an Agent Id, this function returns a corresponding policy ID """
    pol_id = policy_ids[int(agent_id)] # kinda jank/feels wrong atm, but works
    return pol_id


def find_checkpoint_path():
    entries = os.listdir(r"model_training\rllib_multi\checkpoints\cp1")
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    entries = sorted(entries, key=numericalSort)
    # for entry in entries:
    #     print(entry)
    return entries[-2]


# region Parser with arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=80000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=500, help="Reward at which we stop training."
)
parser.add_argument(
    "--perform", type=bool, default=False, help="Whether or not to only display the results of the load path"
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--checkpoint-path", type=str, default=r"model_training\rllib_multi\checkpoints\cp1", help="Path where checkpoint is saved."
)
parser.add_argument(
    "--load-path", type=str, default=r"model_training\rllib_multi\checkpoints\cp1" + "\\" + find_checkpoint_path(), help="Path where checkpoint is saved."
)
parser.add_argument("--num-agents", type=int, default=NUM_STARTING_AGENTS)
parser.add_argument("--num-policies", type=int, default=NUM_STARTING_AGENTS)
# endregion


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=args.local_mode)

    # Policies
    policies = {str(i): gen_policy(i) for i in range(args.num_policies)}
    policy_ids = list(policies.keys()) 

    # Configs
    config = (
        PPOConfig()
        # .environment(MultiDotEnvironment) # was having trouble with this one
        .framework(args.framework)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=0)
    )
    config = config.to_dict()
    config["env"] = MultiDotEnvironment
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # Algorithm/Training
    args.load = True
    args.stop_timesteps = 8000
    args.save = True
    # args.tune = True
    # args.perform = True
    rllib_trainer = "my model doesn't exist yet"
    trainer = ppo.PPO(config=config, env=MultiDotEnvironment)
    if args.load:
        trainer.restore(args.load_path)
    # run manual training loop and print results after each iteration
    for i in range(args.stop_iters):
        if args.perform:
            break
        result = trainer.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
            result["timesteps_total"] >= args.stop_timesteps
            # or result["episode_reward_mean"] >= args.stop_reward
        ):
            rllib_trainer = trainer
            if args.save:
                path_to_checkpoint = trainer.save(args.checkpoint_path)
                print(
                    "An Algorithm checkpoint has been created inside directory: "
                    f"'{path_to_checkpoint}'."
                )
            break
         
    ray.shutdown()

    # Visualization
    env = MultiDotEnvironment()
    for _ in range(2):
        render_multi(env, rllib_trainer, rllib=True)