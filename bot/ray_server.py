from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from JAS.Helper.rl_helper import RLHelper
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.utils.annotations import override


import os
import sys
import getopt


import ray
# from ray.rllib.agents.a3c import A3CAgent
from ray.rllib.agents.dqn.dqn import DQNTrainer #DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.agents.dqn.apex import ApexTrainer #, DQN_CONFIG
# from ray.rllib.utils import merge_dicts
from ray.rllib.env.external_env import ExternalEnv
from utils.policy_server import PolicyServer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint.out"

ALGORITHM = "DQN" # APEX


class CartpoleServing(ExternalEnv):
    # class var

    def __init__(self):
        ExternalEnv.__init__(
            self, RLHelper.action_space(),
            RLHelper.observation_space())
        self.port = SERVER_PORT

    def run(self):
        while True:
            try:
                server = PolicyServer(self, SERVER_ADDRESS, self.port)
                print("Policy server listening at {}:{}".format(
                    SERVER_ADDRESS, self.port))
                server.serve_forever()

                break
            except OSError:
                self.port += 1


class MyModelClass(Model):

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):

        # hyperparameters
        inputs = input_dict["obs"]
        hiddens = options.get("fcnet_hiddens")
        activation = tf.nn.relu

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    activation_fn=activation,
                    scope=label)
                i += 1
            label = "fc_out"
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None,
                scope=label)
            return output, last_layer


def main(argv):
    ModelCatalog.register_custom_model("my_model", MyModelClass)

    model = {
        # cusom model options
        "custom_model": "my_model",
        "custom_preprocessor": None,
        # Extra options to pass to the custom classes
        "custom_options": {},

        # built in options
        # Number of hidden layers for fully connected net
        "fcnet_hiddens": [256,256,256,256],
    }

    num_workers = 2

    # read out command line arguments
    try:
        opts, args = getopt.getopt(argv, "hn:", ["number-worker="])
    except getopt.GetoptError:
        print('ray_server.py -n <number-worker>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ray_server.py -n <number-worker>')
            print('-n --number-worker  - number of worker to start')
            sys.exit()
        elif opt in ("-n", "--number-worker"):
            num_workers = int(arg)

    ray.init()
    print("[RAY] Initialized")
    register_env("srv", lambda _: CartpoleServing())

    if ALGORITHM == "APEX":
        dqn = ApexTrainer(
            env="srv",
            config={
                #Ape-X for gpu...
                "optimizer_class": "AsyncReplayOptimizer",
                "schedule_max_timesteps": 100000,
                # "n_step": 3, # warum haben wir den mist hier angegegeben?!!?!?
                "target_network_update_freq": 500000,
                "timesteps_per_iteration": 25000,
                "per_worker_exploration": True,
                "worker_side_prioritization": True,
                "min_iter_time_s": 30,
                # Learning rate - defaults to 5e-4
                "lr": 0.0001,
                # Use a single process to avoid needing to set up a load balancer
                "num_workers": num_workers,
                # mehrere Threads fuer worker! fuer debugging auf false setzen
                # "sample_async": True,
                #"grad_clip": 0.5,
                "model": model,
                "gamma": 0.99,
                "noisy": False,
                "num_gpus": 1,
                # Size of rollout batch
                # Default sample batch size (unroll length). Batches of this size are
                # collected from workers until train_batch_size is met. When using
                # multiple envs per worker, this is multiplied by num_envs_per_worker.
                "sample_batch_size": 512,
                # Training batch size, if applicable. Should be >= sample_batch_size.
                # Samples batches will be concatenated together to this size for training.
                "train_batch_size": 1024,
                # Size of the replay buffer. Note that if async_updates is set, then
                # each worker will have a replay buffer of this size. default 50000
                "buffer_size": 2000000,
                "learning_starts": 50000,
            })
    else:
        dqn = DQNTrainer(
            env="srv",
            config={
                # model
                # mehrere Threads fuer worker! fuer debugging auf false setzen
                # "sample_async": True,
                # "grad_clip": 0.5,
                "model": model,
                "gamma": 0.99,
                "noisy": False,
                "num_gpus": 1,

                # evaluation
                # everything default, see dqn.py

                # exploration
                "target_network_update_freq": 500000,
                # rest: everything default, see dqn.py

                # replay buffer
                # Size of the replay buffer. Note that if async_updates is set, then
                # each worker will have a replay buffer of this size. default 50000
                "buffer_size": 2000000,
                # If True prioritized replay buffer will be used.
                "prioritized_replay": False,
                # here are many parameters, untouched from me (see dqn.py)

                # Optimization
                # Learning rate - defaults to 5e-4
                "lr": 0.0001,
                # Update the replay buffer with this many samples at once. Note that
                # this setting applies per-worker if num_workers > 1. (agent history length)
                "sample_batch_size": 1024,
                # How many steps of the model to sample before learning starts
                "learning_starts": 50000,
                # Size of a batched sampled from replay buffer for training. Note that
                # if async_updates is set, then each worker returns gradients for a
                # batch of this size. (Minibatch size) hould be >= sample_batch_size
                # Samples batches will be concatenated together to this size for training.
                "train_batch_size": 2048,

                # parallelism
                # Number of workers for collecting samples with. This only makes sense
                # to increase if your environment is particularly slow to sample, or if
                # you"re using the Async or Ape-X optimizers.
                "num_workers": num_workers,
                # distribute epsilon over workers
                "per_worker_exploration": True,
                # compute worker side prioritazation (False, because in DQN this was not ipmlemented)
                "worker_side_prioritization": False,
            })

    # write policy graph to tensorboard (for debugging purposes)
    policy_graph = dqn.local_evaluator.policy_map["default_policy"].sess.graph
    writer = tf.summary.FileWriter(dqn._result_logger.logdir, policy_graph)
    writer.close()

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        dqn.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(dqn.train()))
        checkpoint_path = dqn.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
