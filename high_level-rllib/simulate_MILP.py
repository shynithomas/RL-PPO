import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import numpy as np
import time
import csv
import copy
import gymnasium as gym
import argparse
import os

from multi_agent_env import MultiAgentEnv
from utilities import utils
from utilities.data import Data
from rllib_model import MultiAgentModel
from ray.rllib.algorithms.ppo import PPOConfig


parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int)
parser.add_argument('num_agents', type=int)
parser.add_argument('episode_length', type=int)
parser.add_argument('--dyn_p_flag', type=bool, default=False)

args = parser.parse_args()

def env_creator(env_confg):
    return MultiAgentEnv(data)

global data
data = Data([f"./Dynamic_graphs", 1, 1, 'ppo', None,
                               None, None])
(model_config, _, _, sol_array_dict) = utils.create_configs(data, None)
data.sol_array_dict = sol_array_dict

# # register_env("MultiAgentEnv-v1", env_creator)
# ModelCatalog.register_custom_model("MultiAgentModel-v1", MultiAgentModel)

# ray.init(num_cpus=5)

eval_graphs = ["40N-5A-150T-RP-RD-IRR-2"]

log_dir = f"./SGOpt_dyn_results/{eval_graphs[0]}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/summary", exist_ok=True)
os.makedirs(f"{log_dir}/detailed", exist_ok=True)
results_path = f"{log_dir}/summary/SGOpt.csv"
detail_path = f"{log_dir}/detailed/SGOpt.csv"
sol_dir = "./SGOpt_dyn"

method = f"SGOpt"
with open(results_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])

with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])

# Evaluation Loop
def make_env():
    env = env_creator(data)
    return env

num_edges = [sol_array_dict[key][0][0].nx_graph.number_of_edges() for key in eval_graphs]
max_num_edges = max(num_edges)
print("Max number of edges in training graphs: ", max_num_edges)
data.sol_array_dict = {key: sol_array_dict[key] for key in eval_graphs}
data.sim_data['num_closest_agents_in_node_feat'] = args.num_agents
data.sim_data['max_num_nodes'] = args.num_nodes
data.sim_data['max_num_agents'] = args.num_agents
data.sim_data['max_num_edges'] = max_num_edges

# register_env("MultiAgentEnv-v1", env_creator)

class Opt_Agent():
    def __init__(self, graph_path, instance):
        super().__init__()

        sol_path = graph_path + f"/test_instances/opt_sol_instance_{instance}.csv"

        self.policy = {}
        with open(sol_path, 'r') as f:
            reader = csv.reader(f)

            start = False
            for i, row in enumerate(reader):
                if 'obj' in row:
                    self.obj = float(row[1])
                if 'sol_time' in row:
                    self.sol_time = float(row[1])
                if 'agent' in row:
                    start = True
                    continue
                if start:
                    if row[0] not in self.policy.keys():
                        self.policy[row[0]] = {}
                    self.policy[row[0]][row[1]] = row[2]

        print(f"Policy length: {len(self.policy)} timesteps")
        print(f"Solution Time: {self.sol_time}s")


    def get_action(self, x, timer):

        batch_size = len(x['num_active_agents'])
        assert batch_size==1

        action = [-1 for agent_id in range(x['num_active_agents'][0])]

        for agent_id in range(x['num_active_agents'][0]):
            options = x['valid_neigh'][int(agent_id)]
            # print(f"Agent {agent_id} options: {options}")
            next_node = int(self.policy[str(timer+1)][str(agent_id)])
            action[agent_id] = np.where(options == next_node)[0]
            #backup in case there is no action (in case of irregular graphs)
            print("Og action: ", action[agent_id])
            if len(action[agent_id])==0:
                action[agent_id] = -1

        return action

eval_env = env_creator(data)

overall_SGI = []
overall_opts = []
overall_times = []
for graph_id in eval_graphs:
    sol_array = sol_array_dict[graph_id]
    all_SGI = []
    all_opts = []
    all_times = []
    for i, sol in enumerate(sol_array):
        
        graph_path = f"{sol_dir}/{graph_id}_SGOpt_rolling"
        agent = Opt_Agent(graph_path=graph_path, instance=i+1)

        print("Evaluating instance no. ", i+1)
        # init_time = time.time()
        eval_graph = sol[0]
        options={'graph': eval_graph, 'episode_length': args.episode_length, 'rolling_window': 15}
        if args.dyn_p_flag:
            print(args.dyn_p_flag)
            options['dyn_p'] = sol[3]
        obs, info = eval_env.reset(options=options)

        done = False
        ep_reward = 0
        ep_return = 0
        inter_SGI = []
        timer = 0
        # while not done:
        #     # print("Timer: ", timer)
        #     action = agent.get_action(obs, timer)
        #     print("Action: ", action)
        #     obs, reward, terminated, done, info = eval_env.step(action)
        #     ep_reward += reward
        #     ep_return += -info['obj_contrib']
        #     inter_SGI.append(float(-info['episode_return']))
        #     timer += 1

        # ep_time = time.time() - init_time
        ep_time = agent.sol_time

        # SGI = -info['episode_return']
        SGI = agent.obj
        # opt_gap = 100*(round(SGI)-round(sol[1]))/round(sol[1])
        opt_gap = np.nan

        # print("Episode Reward: ", ep_reward)
        # print(f"Opt Gap: {opt_gap:.2f}%")

        all_SGI.append(SGI)
        all_opts.append(opt_gap)
        overall_SGI.append(SGI)
        overall_opts.append(opt_gap)

        all_times.append(ep_time)
        overall_times.append(ep_time)

        with open(f"{detail_path}", 'a', newline='') as f:
            writer = csv.writer(f)
            data_row = [method, graph_id, i+1, SGI, opt_gap, ep_time, inter_SGI]
            writer.writerow(data_row)

    print("\nAvg opt gap:", np.mean(all_opts), np.std(all_opts))
    print("Avg SGI:", np.mean(all_SGI), np.std(all_SGI))
    print("Avg Time:", np.mean(all_times), np.std(all_times))

    data_row = [method, graph_id, np.mean(all_SGI), np.std(all_SGI), np.mean(all_opts), np.std(all_opts), np.mean(all_times), np.std(all_times)]
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)


print("\n Overall opt gap: ", np.mean(overall_opts), np.std(overall_opts))
print(" Overall SGI: ", np.mean(overall_SGI), np.std(overall_SGI))
print(" Overall Time: ", np.mean(overall_times), np.std(overall_times))