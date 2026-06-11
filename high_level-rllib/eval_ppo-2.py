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
from torch.optim import AdamW

from multi_agent_env import MultiAgentEnv
from utilities import utils
from utilities.data import Data
from rllib_model import MultiAgentModel
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from custom_eval import LogCustomMetrics


def run_eval(env_creator, eval_graphs, sol_array_dict, trained_algo, results_dir, args):
    summary_path = f"{results_dir}/summary/{args.method}.csv"
    detail_path = f"{results_dir}/detailed/{args.method}.csv"
    node_seq_path = f"{results_dir}/node_seq/{args.method}.csv"
    os.makedirs(f"{results_dir}/summary", exist_ok=True)
    os.makedirs(f"{results_dir}/detailed", exist_ok=True)
    os.makedirs(f"{results_dir}/node_seq", exist_ok=True)
    with open(summary_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])
    with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])
    with open(f"{node_seq_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'Node Sequence'])

    eval_env = env_creator()

    overall_SGI = []
    overall_opts = []
    overall_times = []
    for graph_id in eval_graphs:
        sol_array = sol_array_dict[graph_id]
        all_SGI = []
        all_opts = []
        all_times = []
        for i, sol in enumerate(sol_array):
            print("Evaluating instance no. ", i+1)
            init_time = time.time()
            init_graph = sol[0]
            config_options = {'graph': init_graph, 
                              'episode_length': args.episode_length, 
                              'rolling_window': 15}
            if args.dpf:
                config_options['dyn_p'] = sol[3]
            else:
                pass

            obs, info = eval_env.reset(options=config_options)

            ep_reward = 0
            ep_return = 0
            inter_SGI = []
            done = False

            # Extracting node id sequences from current position and further actions
            NUM_AGENTS = int(obs['num_active_agents'][0])
            init_nodes = [int(obs['Agent'][agent_id][1])+1 for agent_id in range(NUM_AGENTS)]
            #mapping [0,N-1] range to [1,N] for N number of nodes
            node_seq = [init_nodes]
            while not done:
                action = trained_algo.compute_single_action(obs, explore=False)
                next_nodes = [int(obs['valid_neigh'][agent_id][action[agent_id]])+1 for agent_id in range(NUM_AGENTS)]
                node_seq.append(next_nodes)
                # print("Action: ", action)
                obs, reward, terminated, done, info = eval_env.step(action)
                ep_reward += reward
                ep_return += -info['obj_contrib']
                inter_SGI.append(float(-info['episode_return']))

            ep_time = time.time() - init_time

            SGI = -info['episode_return']
            opt_gap = 100*(SGI-sol[1])/sol[1]

            all_SGI.append(SGI)
            all_opts.append(opt_gap)
            overall_SGI.append(SGI)
            overall_opts.append(opt_gap)

            all_times.append(ep_time)
            overall_times.append(ep_time)

            with open(f"{detail_path}", 'a', newline='') as f:
                writer = csv.writer(f)
                data_row = [args.method, graph_id, i+1, SGI, opt_gap, ep_time, inter_SGI]
                writer.writerow(data_row)

            with open(f"{node_seq_path}", 'a', newline='') as f:
                writer = csv.writer(f)

                agent_traj = {i+1: [] for i in range(NUM_AGENTS)}
                for actions in node_seq:
                    for id in range(NUM_AGENTS):
                        agent_traj[id+1].append(actions[id])

                data_row = [args.method, graph_id, i+1, agent_traj]
                writer.writerow(data_row)

        print("\nAvg opt gap:", np.mean(all_opts), np.std(all_opts))
        print("Avg SGI:", np.mean(all_SGI), np.std(all_SGI))
        print("Avg Time:", np.mean(all_times), np.std(all_times))

        data_row = [args.method, graph_id, np.mean(all_SGI), np.std(all_SGI), np.mean(all_opts), np.std(all_opts), np.mean(all_times), np.std(all_times)]
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

    # print("\n Overall opt gap: ", np.mean(overall_opts), np.std(overall_opts))
    # print(" Overall SGI: ", np.mean(overall_SGI), np.std(overall_SGI))
    # print(" Overall Time: ", np.mean(overall_times), np.std(overall_times))

def Best_of_MC_eval(trained_policy, temp_envs, env_state, eval_graph, timer, args):
    N = args.num_samples
    obs, info = temp_envs.reset(options={'graph': eval_graph, 'state': env_state, 'timer': timer, 'episode_length': args.episode_length, 'rolling_window': 15, 'eval_length': 15})
    dones = [False]*N
    temp_rewards = [0]*N
    temp_act_seq = []
    env_returns = [0]*N
    env_act_seq = []
    env_dones = np.zeros(N)
    while True:
        actions, states, action_info = trained_policy.compute_actions_from_input_dict({'obs': obs}, explore=True)
        temp_act_seq.append(actions)
        obs, rewards, terminated, dones, info = temp_envs.step(actions)
        temp_rewards += rewards
        if dones.any():
            for i in range(N):
                if (not env_dones[i]) and dones[i]:
                    env_returns[i] = temp_rewards[i]
                    env_dones[i] = 1
        if env_dones.all():
            break
    temp_act_seq = np.array(temp_act_seq)
    best_act_seq = temp_act_seq[:, np.argmax(temp_rewards)]
    return best_act_seq

def run_eval_MC(env_creator, eval_graphs, sol_array_dict, trained_algo, results_dir, args):
    summary_path = f"{results_dir}/summary/{args.method}.csv"
    detail_path = f"{results_dir}/detailed/{args.method}.csv"
    node_seq_path = f"{results_dir}/node_seq/{args.method}.csv"
    os.makedirs(f"{results_dir}/summary", exist_ok=True)
    os.makedirs(f"{results_dir}/detailed", exist_ok=True)
    os.makedirs(f"{results_dir}/node_seq", exist_ok=True)

    with open(summary_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])
    with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])
    with open(f"{node_seq_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'Node Sequence'])

    eval_env = env_creator()
    temp_envs = gym.vector.SyncVectorEnv([env_creator for i in range(args.num_samples)])
    trained_policy = trained_algo.get_policy()

    overall_SGI = []
    overall_opts = []
    overall_times = []
    for graph_id in eval_graphs:
        sol_array = sol_array_dict[graph_id]
        all_SGI = []
        all_opts = []
        all_times = []
        for i, sol in enumerate(sol_array):
            print("Evaluating instance no. ", i+1)
            init_time = time.time()
            init_graph = sol[0]
            config_options = {'graph': init_graph, 
                              'episode_length': args.episode_length, 
                              'rolling_window': 15}
            if args.dpf:
                config_options['dyn_p'] = sol[3]
            else:
                pass

            obs, info = eval_env.reset(options=config_options)

            ep_reward = 0
            ep_return = 0
            inter_SGI = []
            done = False

            # Extracting node id sequences from current position and further actions
            NUM_AGENTS = int(obs['num_active_agents'][0])
            init_nodes = [int(obs['Agent'][agent_id][1])+1 for agent_id in range(NUM_AGENTS)]
            #mapping [0,N-1] range to [1,N] for N number of nodes
            node_seq = [init_nodes]
            while not done:
                action_seq = Best_of_MC_eval(trained_policy, temp_envs, obs, init_graph, eval_env.timer, args)
                # print(f"Found the best action sequence of length {len(action_seq)}")
                assert len(action_seq)==15
                for action in action_seq:
                    next_nodes = [int(obs['valid_neigh'][agent_id][action[agent_id]])+1 for agent_id in range(NUM_AGENTS)]
                    node_seq.append(next_nodes)
                    
                    obs, reward, terminated, done, info = eval_env.step(action)
                    ep_reward += reward
                    ep_return += -info['obj_contrib']
                    inter_SGI.append(float(-info['episode_return']))

            ep_time = time.time() - init_time

            SGI = -info['episode_return']
            opt_gap = 100*(SGI-sol[1])/sol[1]

            all_SGI.append(SGI)
            all_opts.append(opt_gap)
            overall_SGI.append(SGI)
            overall_opts.append(opt_gap)

            all_times.append(ep_time)
            overall_times.append(ep_time)

            with open(f"{detail_path}", 'a', newline='') as f:
                writer = csv.writer(f)
                data_row = [args.method, graph_id, i+1, SGI, opt_gap, ep_time, inter_SGI]
                writer.writerow(data_row)

            with open(f"{node_seq_path}", 'a', newline='') as f:
                writer = csv.writer(f)

                agent_traj = {i+1: [] for i in range(NUM_AGENTS)}
                for actions in node_seq:
                    for id in range(NUM_AGENTS):
                        agent_traj[id+1].append(actions[id])

                data_row = [args.method, graph_id, i+1, agent_traj]
                writer.writerow(data_row)

        print("\nAvg opt gap:", np.mean(all_opts), np.std(all_opts))
        print("Avg SGI:", np.mean(all_SGI), np.std(all_SGI))
        print("Avg Time:", np.mean(all_times), np.std(all_times))

        data_row = [args.method, graph_id, np.mean(all_SGI), np.std(all_SGI), np.mean(all_opts), np.std(all_opts), np.mean(all_times), np.std(all_times)]
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

    # print("\n Overall opt gap: ", np.mean(overall_opts), np.std(overall_opts))
    # print(" Overall SGI: ", np.mean(overall_SGI), np.std(overall_SGI))
    # print(" Overall Time: ", np.mean(overall_times), np.std(overall_times))

def MCTS_eval(trained_policy, temp_envs, env_state, eval_graph, timer, args):
    N = args.num_samples
    depth = args.depth
    obs, info = temp_envs.reset(options={'graph': eval_graph, 
                                         'state': env_state, 
                                         'timer': timer, 
                                         'episode_length': args.episode_length, 
                                         'rolling_window': 15})
    dones = False
    temp_rewards = [0]*N
    first_actions = None
    for i in range(depth):
        actions, states, action_info = trained_policy.compute_actions_from_input_dict({'obs': obs}, explore=True)
        if i==0:
            first_actions = actions
        obs, rewards, terminated, dones, info = temp_envs.step(actions)
        temp_rewards += rewards
        if dones.any():
            break
        
    _, _, action_info =  trained_policy.compute_actions_from_input_dict({'obs': obs}, explore=True)
    value_estimates = action_info['vf_preds']
    temp_rewards += value_estimates
    best_action = first_actions[np.argmax(temp_rewards)]
    return best_action

def run_eval_MCTS(env_creator, eval_graphs, sol_array_dict, trained_algo, results_dir, args):
    summary_path = f"{results_dir}/summary/{args.method}.csv"
    detail_path = f"{results_dir}/detailed/{args.method}.csv"
    node_seq_path = f"{results_dir}/node_seq/{args.method}.csv"
    os.makedirs(f"{results_dir}/summary", exist_ok=True)
    os.makedirs(f"{results_dir}/detailed", exist_ok=True)
    os.makedirs(f"{results_dir}/node_seq", exist_ok=True)

    with open(summary_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])
    with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])
    with open(f"{node_seq_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'Node Sequence'])

    eval_env = env_creator()
    temp_envs = gym.vector.SyncVectorEnv([env_creator for i in range(args.num_samples)])
    trained_policy = trained_algo.get_policy()

    overall_SGI = []
    overall_opts = []
    overall_times = []
    for graph_id in eval_graphs:
        sol_array = sol_array_dict[graph_id]
        all_SGI = []
        all_opts = []
        all_times = []
        for i, sol in enumerate(sol_array):
            print("Evaluating instance no. ", i+1)
            init_time = time.time()
            init_graph = sol[0]
            config_options = {'graph': init_graph, 
                              'episode_length': args.episode_length, 
                              'rolling_window': 15}
            if args.dpf:
                config_options['dyn_p'] = sol[3]
            else:
                pass

            obs, info = eval_env.reset(options=config_options)

            ep_reward = 0
            ep_return = 0
            inter_SGI = []
            done = False

            # Extracting node id sequences from current position and further actions
            NUM_AGENTS = int(obs['num_active_agents'][0])
            init_nodes = [int(obs['Agent'][agent_id][1])+1 for agent_id in range(NUM_AGENTS)]
            #mapping [0,N-1] range to [1,N] for N number of nodes
            node_seq = [init_nodes]
            while not done:
                action = MCTS_eval(trained_policy, temp_envs, obs, init_graph, eval_env.timer, args)

                next_nodes = [int(obs['valid_neigh'][agent_id][action[agent_id]])+1 for agent_id in range(NUM_AGENTS)]
                node_seq.append(next_nodes)

                obs, reward, terminated, done, info = eval_env.step(action)
                ep_reward += reward
                ep_return += -info['obj_contrib']
                inter_SGI.append(float(-info['episode_return']))
            
            ep_time = time.time() - init_time

            SGI = -info['episode_return']
            opt_gap = 100*(SGI-sol[1])/sol[1]

            all_SGI.append(SGI)
            all_opts.append(opt_gap)
            overall_SGI.append(SGI)
            overall_opts.append(opt_gap)

            all_times.append(ep_time)
            overall_times.append(ep_time)

            with open(f"{detail_path}", 'a', newline='') as f:
                writer = csv.writer(f)
                data_row = [args.method, graph_id, i+1, SGI, opt_gap, ep_time, inter_SGI]
                writer.writerow(data_row)

            with open(f"{node_seq_path}", 'a', newline='') as f:
                writer = csv.writer(f)

                agent_traj = {i+1: [] for i in range(NUM_AGENTS)}
                for actions in node_seq:
                    for id in range(NUM_AGENTS):
                        agent_traj[id+1].append(actions[id])

                data_row = [args.method, graph_id, i+1, agent_traj]
                writer.writerow(data_row)

        print("\nAvg opt gap:", np.mean(all_opts), np.std(all_opts))
        print("Avg SGI:", np.mean(all_SGI), np.std(all_SGI))
        print("Avg Time:", np.mean(all_times), np.std(all_times))

        data_row = [args.method, graph_id, np.mean(all_SGI), np.std(all_SGI), np.mean(all_opts), np.std(all_opts), np.mean(all_times), np.std(all_times)]
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

    # print("\n Overall opt gap: ", np.mean(overall_opts), np.std(overall_opts))
    # print(" Overall SGI: ", np.mean(overall_SGI), np.std(overall_SGI))
    # print(" Overall Time: ", np.mean(overall_times), np.std(overall_times))


if __name__ == "__main__":
    '''
    Evaluation script for regular as well as rolling horizon implementation depending on the episode length.
    - Recommended to only use for evaluation on a single graph at a time.
    '''

    #ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str) #RL, RL-MC or RL-MCTS
    # parser.add_argument('num_nodes', type=int)
    # parser.add_argument('num_agents', type=int)
    parser.add_argument('episode_length', type=int)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--graph', type=str)
    parser.add_argument('--max_agents', type=int, default=None)
    parser.add_argument('--graph_list', type=list)
    parser.add_argument('--station_flag', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--dpf', type=bool, default=False, help="Dynamic priority flag")
    args = parser.parse_args()

    if args.station_flag:
        source_dir = "Station_graphs"
        eval_dir = "eval_logs_NH"
    else:
        source_dir = "NoStation_graphs"
        eval_dir = "eval_logs_NS"

    if args.graph is not None:
        eval_graphs = [args.graph]
        results_dir = f"{eval_dir}/results-{args.episode_length}steps/{eval_graphs[0]}"
    elif args.graph_list is not None:
        eval_graphs = args.graph_list
        results_dir = f"{eval_dir}/results-{args.episode_length}steps/Misc"
    else:
        raise Exception("An evaluation graph must be provided!")
    
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_path = args.checkpoint

    #Prepare the environment config
    global data
    data = Data([f"./{source_dir}/", 1, 1, 'ppo', None,
                                None, None])
    (model_config, _, _, sol_array_dict) = utils.create_configs(data, None)

    # data.sol_array_dict = sol_array_dict
    data.sol_array_dict = {key: sol_array_dict[key] for key in eval_graphs}

    # max_agents_dict = {15:3, 25:4, 50:6, 100:10}
    max_stn_dict = {15:2, 25:3} 

    num_nodes = [sol_array_dict[key][0][0].nx_graph.number_of_nodes() for key in eval_graphs]
    num_edges = [sol_array_dict[key][0][0].nx_graph.number_of_edges() for key in eval_graphs]
    num_agents = [sol_array_dict[key][0][0].max_num_agents for key in eval_graphs]
    data.sim_data['max_num_nodes'] = max(num_nodes)
    data.sim_data['max_num_edges'] = max(num_edges)
    data.sim_data['max_num_agents'] = max(num_agents) if args.max_agents is None else args.max_agents
    data.sim_data['num_closest_agents_in_node_feat'] = max(num_agents) if args.max_agents is None else args.max_agents
    data.sim_data['num_closest_stn_in_node_feat'] = 0 if not args.station_flag else max_stn_dict[args.num_nodes]

    #Register env and model
    def env_creator(env_config=None):
        #env_config is a dummy arg required for rllib
        return MultiAgentEnv(data)

    register_env("MultiAgentEnv-v1", env_creator)
    ModelCatalog.register_custom_model("MultiAgentModel-v1", MultiAgentModel)

    #Define config for algo initialization (Should be compatible with saved algorithm state)
    ray.init(num_cpus=1)
    class CustomAdamWPPOPolicy(PPOTorchPolicy):
        def optimizer(self):
            # Replace default Adam with AdamW
            return AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=0.01)
    env = env_creator()
    config = (
        PPOConfig()
        .environment(env="MultiAgentEnv-v1",
            )

        .framework("torch")
        .training(model={
            "custom_model": "MultiAgentModel-v1",
            "custom_model_config": {"obs_space_dummy": env.observation_space, "enc_hidden_size": args.hidden_size},
            })
        .env_runners(num_env_runners=1)
        # .learners(num_learners=0, num_cpus_per_learner=1, num_gpus_per_learner=0)
        # .resources(num_gpus=1)
        .debugging(log_level='ERROR')
        .callbacks(LogCustomMetrics)
    )
    #AdamW used while training (Optional for evaluation)
    config.policies = {
        "default_policy": (CustomAdamWPPOPolicy, None, None, {})
    }
    # config["preprocessor_pref"] = None
    config["_disable_preprocessor_api"] = True

    #Restore trained state from saved checkpoint
    algo = config.build()
    algo.restore(checkpoint_path)   #restore_from_path method does not work as intended

    #Run the evaluation function
    if args.method == 'RL':
        run_eval(env_creator, eval_graphs, sol_array_dict, algo, results_dir, args)
    elif args.method == 'RL-MC':
        vars(args)['method'] = f'RL-MC({args.num_samples})'
        run_eval_MC(env_creator, eval_graphs, sol_array_dict, algo, results_dir, args)
    elif args.method == 'RL-MCTS':
        vars(args)['method'] = f'RL-MCTS({args.num_samples},{args.depth})'
        run_eval_MCTS(env_creator, eval_graphs, sol_array_dict, algo, results_dir, args)
    else:
        raise Exception("Invalid method provided! Must be one of (RL, RL-MC, RL-MCTS)")

    algo.stop()