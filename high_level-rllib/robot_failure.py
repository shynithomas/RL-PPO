""" Evaluation script to simulate robot failure scenarios. """

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


def gen_events(eval_seed=42, episode_length=150, num_agents=4):
    ''' function to generate dynamic agent events '''

    np.random.seed(eval_seed)

    event_times = []
    for i in range(episode_length):
        if len(event_times)==0 or i in event_times:
            next_event = i + np.random.randint(30, 50)
            if next_event < episode_length:
                event_times.append(next_event)

    active = [a for a in range(1, num_agents+1)]
    failed = []
    event_details = {}
    for event in event_times:
        p1 = len(active)/(len(active)+len(failed))
        p2 = len(failed)/(len(active)+len(failed))
        if len(active)==1:
            assert len(failed)>0
            p1 = 0
            p2 = 1
        
        event_type = np.random.choice(['fail', 'recover'], p=[p1, p2])
        if event_type=='fail':
            #guid stands for global unique identifier
            agent_guid = np.random.choice(active)
            active.remove(agent_guid)
            failed.append(agent_guid)
        else:
            agent_guid = np.random.choice(failed)
            failed.remove(agent_guid)
            active.append(agent_guid)
        
        event_details[event] = {'type': event_type, 'agent_guid': agent_guid}

    return event_details

def run_eval(env_creator, eval_graphs, sol_array_dict, trained_algo, results_dir, args):
    summary_path = f"{results_dir}/summary/{args.method}.csv"
    detail_path = f"{results_dir}/detailed/{args.method}.csv"
    os.makedirs(f"{results_dir}/summary", exist_ok=True)
    os.makedirs(f"{results_dir}/detailed", exist_ok=True)
    with open(summary_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])
    with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])

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
            dyn_agent_events = gen_events(eval_seed=i, episode_length=args.episode_length, num_agents=init_graph.max_num_agents)
            config_options = {'graph': init_graph, 
                              'episode_length': args.episode_length, 
                              'rolling_window': 15,
                              'dyn_agent_events': dyn_agent_events
                              }
            if args.dpf:
                config_options['dyn_p'] = sol[3]
            else:
                pass

            obs, info = eval_env.reset(options=config_options)

            ep_reward = 0
            ep_return = 0
            inter_SGI = []
            done = False
            while not done:
                action = trained_algo.compute_single_action(obs, explore=False)
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
    os.makedirs(f"{results_dir}/summary", exist_ok=True)
    os.makedirs(f"{results_dir}/detailed", exist_ok=True)
    with open(summary_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Avg SGI', 'Std SGI', 'Avg Opt-Gap', 'Std Opt-Gap', 'Avg Ep-Time', 'Std Ep-Time'])
    with open(f"{detail_path}", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Graph', 'Instance', 'SGI', 'Opt-Gap', 'Ep-Time', 'SGI_log'])

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
            dyn_agent_events = gen_events(eval_seed=i, episode_length=args.episode_length, num_agents=init_graph.max_num_agents)
            config_options = {'graph': init_graph, 
                              'episode_length': args.episode_length, 
                              'rolling_window': 15,
                              'dyn_agent_events': dyn_agent_events
                              }
            if args.dpf:
                config_options['dyn_p'] = sol[3]
            else:
                pass

            obs, info = eval_env.reset(options=config_options)

            ep_reward = 0
            ep_return = 0
            inter_SGI = []
            done = False
            while not done:
                action = MCTS_eval(trained_policy, temp_envs, obs, init_graph, eval_env.timer, args)
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
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--dpf', type=bool, default=False, help="Dynamic priority flag")
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--active_agents', default=None)
    args = parser.parse_args()

    eval_graphs = ["25N-4A-15T-RP-RD-GRID-1"]
    assert len(eval_graphs)==1
    # log_dir = "tuner_logs_NS/25N_general_policy/TuneExp_32/PPO_MultiAgentEnv-v1_dfa8c_00000_0_2025-11-25_13-23-47/checkpoint_000004"
    log_dir = "tuner_logs_NS/25N-4A-15T-RP-RD-GRID-1/TuneExp_32/PPO_MultiAgentEnv-v1_f65b9_00000_0_2025-11-25_12-05-41/checkpoint_000004"
    checkpoint_path = log_dir

    #Path to saved checkpoint
    # checkpoint = args.checkpoint
    # if checkpoint=='last':
    #     checkpoint_path = log_dir + "/last_checkpoint"
    # else: 
    #     checkpoint_path = log_dir + f"/checkpoint-{checkpoint}"

    #Path to where the results get saved
    # results_dir = f"{log_dir}/results/checkpoint-{checkpoint}"
    results_dir = f"results_dyn_agents-try/{eval_graphs[0]}"
    os.makedirs(results_dir, exist_ok=True)
    # results_path = f"{results_dir}/{method}.csv"

    #Prepare the environment config
    global data
    data = Data([f"./NoStation_graphs/", 1, 1, 'ppo', None,
                                None, None])
    (model_config, _, _, sol_array_dict) = utils.create_configs(data, None)

    # data.sol_array_dict = sol_array_dict
    data.sol_array_dict = {key: sol_array_dict[key] for key in eval_graphs}

    num_nodes = [sol_array_dict[key][0][0].nx_graph.number_of_nodes() for key in eval_graphs]
    num_edges = [sol_array_dict[key][0][0].nx_graph.number_of_edges() for key in eval_graphs]
    num_agents = [sol_array_dict[key][0][0].max_num_agents for key in eval_graphs]
    data.sim_data['max_num_nodes'] = max(num_nodes)
    data.sim_data['max_num_edges'] = max(num_edges)
    data.sim_data['max_num_agents'] = 4
    data.sim_data['num_closest_agents_in_node_feat'] = 4

    data.sim_data['active_agents'] = args.active_agents

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
    # elif args.method == 'RL-MC':
    #     vars(args)['method'] = f'RL-MC({args.num_samples})'
    #     run_eval_MC(env_creator, eval_graphs, sol_array_dict, algo, results_dir, args)
    elif args.method == 'RL-MCTS':
        vars(args)['method'] = f'RL-MCTS({args.num_samples},{args.depth})'
        run_eval_MCTS(env_creator, eval_graphs, sol_array_dict, algo, results_dir, args)
    else:
        raise Exception("Invalid method provided! Must be one of (RL, RL-MC, RL-MCTS)")

    algo.stop()