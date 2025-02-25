# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])
import blind_cfg
import util
from ray import tune


class AnymalBlindJobCfg(blind_cfg.BlindJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Velocity-Flat-Anymal-C-Direct-v0"])
        super().__init__(cfg, vary_env_count=False, vary_mlp=True, vary_networks_type=False, vary_algorithm_param=False)
