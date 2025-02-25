# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import sys

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent.parent
sys.path.extend([UTIL_DIR, str(CUR_DIR)])
import tuner
import util
from ray import tune


class RLGamesJobCfg(tuner.JobCfg):
    """In order to be compatible with :meth: invoke_tuning_run, and
    :class:IsaacLabTuneTrainable , configurations should
    be in a similar format to this class. This class can vary env count/horizon length,
    CNN structure, and MLP structure. Broad possible ranges are set, the specific values
    that work can be found via tuning. Tuning results can inform better ranges for a second tuning run.
    These ranges were selected for demonstration purposes. Best ranges are run/task specific."""

    @staticmethod
    def _get_batch_size_divisors(batch_size: int, min_size: int = 128) -> list[int]:
        """Get valid batch divisors to combine with num_envs and horizon length"""
        divisors = [i for i in range(min_size, batch_size + 1) if batch_size % i == 0]
        return divisors if divisors else [min_size]

    def __init__(self, cfg={}, vary_env_count: bool = False, vary_mlp: bool = False,
                max_num_mlp_layers: int = 10, max_neurons_per_mlp_layer: int = 2048):
        cfg = util.populate_isaac_ray_cfg_args(cfg)

        # Basic configuration
        cfg["runner_args"]["headless_singleton"] = "--headless"
        cfg["hydra_args"]["agent.params.config.max_epochs"] = 1500

        if vary_env_count:  # Vary the env count, and horizon length, and select a compatible mini-batch size
            # Check from 512 to 8196 envs in powers of 2
            # check horizon lengths of 8 to 256
            # More envs should be better, but different batch sizes can improve gradient estimation
            env_counts = [2**x for x in range(9, 13)]
            horizon_lengths = [2**x for x in range(3, 8)]

            selected_env_count = tune.choice(env_counts)
            selected_horizon = tune.choice(horizon_lengths)

            cfg["runner_args"]["--num_envs"] = selected_env_count
            cfg["hydra_args"]["agent.params.config.horizon_length"] = selected_horizon

            def get_valid_batch_size(config):
                num_envs = config["runner_args"]["--num_envs"]
                horizon_length = config["hydra_args"]["agent.params.config.horizon_length"]
                total_batch = horizon_length * num_envs
                divisors = self._get_batch_size_divisors(total_batch)
                return divisors[0]

            cfg["hydra_args"]["agent.params.config.minibatch_size"] = tune.sample_from(get_valid_batch_size)

        if vary_mlp:  # Vary the MLP structure; neurons (units) per layer, number of layers,

            max_num_layers = max_num_mlp_layers
            max_neurons_per_layer = 128
            if "env.observations.policy.image.params.model_name" in cfg["hydra_args"]:
                # By decreasing MLP size when using pretrained helps prevent out of memory on L4
                max_num_layers = 3
                max_neurons_per_layer = 32
            if "agent.params.network.cnn.convs" in cfg["hydra_args"]:
                # decrease MLP size to prevent running out of memory on L4
                max_num_layers = 2
                max_neurons_per_layer = 32

            num_layers = tune.randint(1, max_num_layers)

            def get_mlp_layers(_):
                return [tune.randint(4, max_neurons_per_layer).sample() for _ in range(num_layers.sample())]

            cfg["hydra_args"]["agent.params.network.mlp.units"] = tune.sample_from(get_mlp_layers)
            cfg["hydra_args"]["agent.params.network.mlp.initializer.name"] = tune.choice(["default"]).sample()
            cfg["hydra_args"]["agent.params.network.mlp.activation"] = tune.choice(
                ["relu", "tanh", "sigmoid", "elu"]
            ).sample()

        super().__init__(cfg)