import argparse
import os
from turtle import dot

import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
import warnings
warnings.filterwarnings('ignore')

from dot_environment import render
from  dot_environment import dot_environment
# -------------------------------------------------------------------------------------------

# region Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf2",
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
    "--stop-timesteps", type=int, default=6000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=300, help="Reward at which we stop training."
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
    "--checkpoint-path", type=str, default=r"model_training\rllib_single\checkpoints\cp1", help="Path where checkpoint is saved."
)
# parser.add_argument(
#     "--checkpoint-path", type=str, default=r"C:\Users\Randy Hodges\Documents\GitHub\Dot-Finding-Agent\checkpoints\rllib_single\checkpoint1", help="Path where checkpoint is saved."
# )
# endregion

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    args.no_tune = True
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)
    
    config = {
        "env": dot_environment, 
        "env_config": {
            "corridor_length": 5,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0,
        # "num_workers": 1,  # parallelism
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    rllib_trainer = "my model doesn't exist yet"
    if args.no_tune:
        print("no tune")
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPO(config=ppo_config, env=dot_environment)
        trainer.restore(args.checkpoint_path)
        exit()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                rllib_trainer = trainer
                path_to_checkpoint = trainer.save(args.checkpoint_path)
                print(
                    "An Algorithm checkpoint has been created inside directory: "
                    f"'{path_to_checkpoint}'."
                )
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run, param_space=config, run_config=air.RunConfig(stop=stop)
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

    print(rllib_trainer)
    env = dot_environment()
    for _ in range(2):
        render(env, rllib_trainer, rllib=True)