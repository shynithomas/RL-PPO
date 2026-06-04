# pylint: disable-all
# modified cleanRL PPO code for multi-agent surveillance
import warnings
warnings.filterwarnings("ignore")
import os
import ray
import random
import time
# from dataclasses import dataclass
# import gymnasium as gym
import numpy as np
import torch
from torch.optim import AdamW
# import torch.nn as nn
# import torch.optim as optim
# import tyro
# import pickle as pkl
# from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# import multiprocessing as mp
from rllib_model import MultiAgentModel
from utilities import utils
from utilities.data import Data
from multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
# from actor import MultiAgentActor
# from critic import MultiAgentCritic
# from gymnasium.envs.registration import register

from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from ray import tune
from ray.train import RunConfig, CheckpointConfig

from custom_eval import LogCustomMetrics

import json
from datetime import datetime
import argparse
import pickle


parser = argparse.ArgumentParser()
# parser.add_argument('seed_idx', type = int, help ='seed idx(1-5) to pick from a set of random seeds')
parser.add_argument('num_nodes', type=int)
parser.add_argument('num_agents', type=int, nargs='?', default=None)
parser.add_argument('--graph', type=str, default=None)
parser.add_argument('--graph_type', type=str, default="GRID-1")
parser.add_argument('--station_flag', type=bool, default=False)
parser.add_argument('--sparse_flag', type=bool, default=False)
parser.add_argument('--active_agents', default=None)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()


if args.station_flag:
    source_dir = "Station_graphs"
    log_dir = "tuner_logs_NH"
else:
    source_dir = "NoStation_graphs"
    log_dir = "tuner_logs_NS"


if args.graph is None:
    training_graphs = None
else:
    training_graphs = [args.graph]
    # training_graphs = ["25N-3A-15T_G1_C15"]

if training_graphs is None:
    if args.num_agents is None:
        matching_string = f"graph_{args.num_nodes}"
        log_dir = f"{log_dir}/{args.num_nodes}N_general_policy"
    else:
        matching_string = f"graph_{args.num_nodes}N-{args.num_agents}A"
        log_dir = f"{log_dir}/{args.num_nodes}N_{args.num_agents}A"
    training_graphs = []
    for root, subdirs, files in os.walk(source_dir):
        for filename in files:
            if (matching_string in filename) and (args.graph_type in filename):
                graph = filename.replace("graph_", "")
                training_graphs.append(graph)
else:
    assert len(training_graphs)==1
    log_dir = f"{log_dir}/{training_graphs[0]}"

print("Training graphs: ", training_graphs)
# exit()

os.makedirs(log_dir, exist_ok=True)

if args.seed is None:
    training_seed = np.random.randint(100, 1000)
else:
    training_seed = args.seed
print("Training seed: ", training_seed)

global data
data = Data([f"./{source_dir}", 1, 1, 'ppo', None,
                               None, None])
(model_config, _, _, sol_array_dict) = utils.create_configs(data, None)
# data.sol_array_dict = sol_array_dict
data.sol_array_dict = {key: sol_array_dict[key] for key in training_graphs}

max_agents_dict = {15:3, 25:4, 50:6, 100:10}
max_stn_dict = {15:2, 25:3}
if args.num_agents is None:
    max_num_agents = max_agents_dict[args.num_nodes]
else:
    max_num_agents = args.num_agents

num_edges = [sol_array_dict[key][0][0].nx_graph.number_of_edges() for key in data.sol_array_dict.keys()]
max_num_edges = max(num_edges)
print("Max number of edges in training graphs: ", max_num_edges)
data.sim_data['num_closest_stn_in_node_feat'] = 0 if not args.station_flag else max_stn_dict[args.num_nodes]
data.sim_data['num_closest_agents_in_node_feat'] = max_num_agents
data.sim_data['max_num_nodes'] = args.num_nodes
data.sim_data['max_num_agents'] = max_num_agents
data.sim_data['max_num_edges'] = max_num_edges

data.sim_data['sparse_flag'] = args.sparse_flag
data.sim_data['active_agents'] = args.active_agents


class CustomAdamWPPOPolicy(PPOTorchPolicy):
    def optimizer(self):
        # Replace default Adam with AdamW
        return AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=0.01)

def env_creator(env_config):
    return MultiAgentEnv(data)

with open("ppo_config.json", 'r') as f:
    ppo_config = json.load(f)
    training_config = ppo_config['training']
    model_config = ppo_config['model']
    resource_config = ppo_config['resource']

register_env("MultiAgentEnv-v1", env_creator)
ModelCatalog.register_custom_model("MultiAgentModel-v1", MultiAgentModel)
ray.init(num_cpus=1+resource_config['num_env_runners'])
from ray.rllib.models.preprocessors import get_preprocessor
env = env_creator(data)
prep = get_preprocessor(env.observation_space)(env.observation_space)

config = (
    PPOConfig()
    .environment(env="MultiAgentEnv-v1",
        )

    .framework("torch")
    .training(model={
        "custom_model": "MultiAgentModel-v1",
        "custom_model_config": {"obs_space_dummy": env.observation_space, "enc_hidden_size": model_config['enc_hidden_size']},
        })
    .env_runners(num_env_runners=resource_config['num_env_runners'])
    # .learners(num_learners=1, num_cpus_per_learner=1, num_gpus_per_learner=0)
    .resources(num_gpus=resource_config['num_gpus'])
    .debugging(log_level='ERROR',
               seed=training_seed)# INFO, DEBUG, ERROR, WARN
    .callbacks(LogCustomMetrics)
)
config["preprocessor_pref"] = None
config["_disable_preprocessor_api"] = True
config.policies = {
    "default_policy": (CustomAdamWPPOPolicy, None, None, {})
}

config.update_from_dict(training_config)

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=RunConfig(
        storage_path=os.path.abspath(log_dir),
        name=f"TuneExp_seed-{training_seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        stop={"training_iteration": training_config['training_iter']},
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=training_config['checkpoint_freq'],
        ),
    ),
)

results = tuner.fit()

# These custom logs are only saved at the end of training. Make sure to run the training till the end to not lose this info.
# TODO: Try to save these at the start of training. (how to retrive the experiment path??) 
for result in results:
    trial_dir = result.path
    os.makedirs(f"{trial_dir}/custom_logs")
    with open(f"{trial_dir}/custom_logs/ppo_config.json", 'w') as f:
        json.dump(ppo_config, f, indent=4)
    with open(f"{trial_dir}/custom_logs/settings.json", 'w') as f:
        json.dump({"sim_data": data.sim_data}, f, indent=4)
        json.dump({"training_graphs": training_graphs}, f, indent=4)

ray.shutdown()
exit()
