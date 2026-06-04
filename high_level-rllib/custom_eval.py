from typing import Tuple
import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.typing import ResultDict

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.evaluation import Episode
from ray.rllib.env import BaseEnv


class LogCustomMetrics(DefaultCallbacks):
    def on_episode_created(self, *, episode: Episode, **kwargs):
        # episode.user_data["obj_list"] = []
        pass

    def on_episode_step(self, *, episode: Episode, base_env: BaseEnv, **kwargs):
        # info = episode.last_info_for()
        # sum_obj = -info['obj_contrib']
        # episode.user_data["obj_list"].append(sum_obj)
        pass

    def on_episode_end(self, *, episode: Episode, base_env: BaseEnv, **kwargs):
        # sum_obj_list = episode.user_data["obj_list"]

        # SGI = base_env.get_sub_environments()[0].unwrapped.episode_return

        info = episode.last_info_for()
        SGI = -info['episode_return']
        episode.custom_metrics['SGI'] = SGI
