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
import util
from rl_games_cfg import RLGamesJobCfg
from ray import tune


class HandCfg(RLGamesJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Repose-Cube-Shadow-Direct-v0"])
        super().__init__(cfg, vary_env_count=False, vary_mlp=True)