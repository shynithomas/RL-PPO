import os
import sys
sys.path.append('CBLS/')

import datetime as dt
import pytz
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# from Env.env import obsMap
# from dummy import obsMap

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
# from Env.env import Env
from dummy import Env
from CBLS_policy import ConcurrentBayesianLearning
from CBLS_policy import get_key
# from Env.env import CONST
# from dummy import CONST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def main(data, sol_array, graph_id, args):

    #initializing environment and other dependent variables
    env = Env(data)
    # CONST = env.CONST
    # obsMap = env.obsMap

    #### Added to store data to df ####
    results_path = "CBLS/results-dyn_agents"
    df = pd.DataFrame()
    # df_ob = str(type(obsMap)).split('.')[-2]
    df_ob = "grid"

    ctime = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = f'{ctime.year}_{ctime.month}_{ctime.day}_{ctime.hour}' +\
        f'_{ctime.minute}_{ctime.second}_{ctime.microsecond}'
    df_id = df_ob+".csv"
    results_path = (results_path + "/" + f"{graph_id}" + "/" + f"ep_length-{args.episode_length}" + "/CBLS_" + current_time + "/")
    df_path = results_path + df_id

    os.makedirs(results_path, exist_ok=True)
    # config_path = results_path+"config.txt"
    # with open(config_path, 'a+') as f:
    #     print("OBSTACLE:"+df_ob+"\n", file=f)
    #     print(CONST, file=f)

    #### Added to store data to df ####

    CBLS = ConcurrentBayesianLearning(env)
    idleness = []
    idleness_greedy = []
    idleness_rand = []
    idleness_bayes = []
    batch_step = 0
    Global_idle = []
    Global_idle_greedy = []
    GLobal_idle_bayes = []
    Global_ent = []
    Global_ent_greedy = []
    Global_ent_bayes = []
    all_ep_r = []

    Sum_idle = []
    Opt_gap_list = []

    for episode in range(len(sol_array)):

        if args.render:
            if episode == args.episode - 1:
                pass 
            else:
                continue

        if args.dyn_flag:
            dyn_p = sol_array[episode][3]
        else:
            dyn_p=None
        
        dyn_agent_events = gen_events(eval_seed=episode, episode_length=args.episode_length, num_agents=sol_array[episode][0].max_num_agents) if args.dyn_agents else None
        state, nodes_last_visit, last_vertex, neighbor_nodes, flag, dest_nodes = env.reset(options={'graph':sol_array[episode][0], 'dyn_p':dyn_p, 'episode_length':args.episode_length, 
        'dyn_agent_events': dyn_agent_events})
        # state, nodes_last_visit, last_vertex, neighbor_nodes, flag = env.reset(initial_demand=0)
        initial_demands = [int(x[-1]) for x in state]
        # print("Init Weighted Demands: ", initial_demands)

        CONST = env.CONST
        obsMap = env.obsMap

        CBLS.reset(env)
        SDI = []
        SDI_greedy = []
        SDI_bayes = []
        nodes_last_goal = [dest_nodes[i] for i in range(CONST.NUM_AGENTS)]
        # print("nodes last goal: ", nodes_last_goal)
        episode_reward = 0
        t0 = time.time()

        change_reward = 0
        change_step = 0
        decision_index = -torch.ones((1, CONST.NUM_AGENTS)).to(device)

        step_count = 0
        while True:

            # if (episode + 1) % 25 == 0:
            #     env.render()

            decision_flag = torch.ones((1, CONST.NUM_AGENTS), dtype=bool).to(device)
            greedynodes = []

            # state (batch_size=1,graph_size,3)
            # nodes_last_list : ( batch_size=1,num_agent)
            # decision_index  : (batch_size=1, num_agent)
            # decision_flag  : (batch_size=1,  num_agent)
            # choose_node(self, state, nodes_last_list,decision_index,decision_flag)


            for i in range(CONST.NUM_AGENTS):
                decision_flag[0][i] = flag[i]
            
            # print("Decision Flags: ", decision_flag)

            if step_count == 0:
                nodes, log_prob = CBLS.choose(state, nodes_last_visit, env.visits, decision_flag, nodes_last_goal)

            elif any(decision_flag[0]):

                nodes, log_prob = CBLS.choose(state, nodes_last_visit, env.visits, decision_flag, nodes_last_goal)
                change_reward = 0

            # choose node: list: num_agent,2 log_prob_choose: list: num_agent
            # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)

            else:
                nodes = []
                for i in range(CONST.NUM_AGENTS):
                    nodes.append(obsMap.nodes[int(decision_index[0][i])])

            step_count +=1

            a = time.time()
            # next_state

            # for i in range(CONST.NUM_AGENTS):
            #     if nodes[i] == nodes_last_visit[i]:
            #         print(f"{i} {nodes[i]} {nodes_last_goal[i]}")

            # update decision_index:

            decision_index = [get_key(nodes[i], obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
            decision_index = np.array(decision_index)
            decision_index = torch.tensor(decision_index, device=device).unsqueeze(0)  # (1,num_agents)
            next_state, agent_pos_list, current_map_state, nb_nodes, local_heatmap_list, mini_map, \
            shared_reward, flag, last_vertex_list, n_last_visit, visit_flag, done, info = env.step(
                nodes, nodes_last_visit, last_vertex, state)

            if episode == 0:
                idleness.append(env.avg_i)
                SDI.append(env.interval)
            elif episode == 1:
                idleness_greedy.append(env.avg_i)
                SDI_greedy.append(env.interval)
            else:
                idleness_bayes.append(env.avg_i)
                SDI_bayes.append(env.interval)
            last_vertex = [last_vertex_list[i] for i in range(CONST.NUM_AGENTS)]
            nodes_last_goal = [nodes[i] for i in range(CONST.NUM_AGENTS)]
            change_reward += shared_reward  # change_reward
            if any(decision_flag[0]):
                change_step += 1

            state = next_state
            nodes_last_visit = n_last_visit
            episode_reward += shared_reward

            # graph_demands.append(np.sum(state[:, 2]).squeeze())
            # avg_graph_demands.append(env.sum_idleness/(t+1))
            # rolling_window = 15
            # divider = (t+1) if t<rolling_window else rolling_window
            # rolling_avg.append(np.sum(graph_demands[-rolling_window:])/divider)

            if done:
                break
        
        # graph_demands = info['ep_sum_demands']
        # rolling_window = 15
        # avg_graph_demands = []
        # rolling_avg = []
        # for i, gd in enumerate(graph_demands):
        #     avg_graph_demands.append(sum(graph_demands[:i+1])/i+1)
        #     divider = i+1 if i<rolling_window else rolling_window
        #     rolling_avg.append(sum(graph_demands[:i+1])/divider)

        if episode == 0:
            all_ep_r.append(episode_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)

        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | Idleness : {:.4f} | Global Idleness : {:.4f} | Sum Idleness: {:.4f}'.format(
                episode, 50, episode_reward,
                time.time() - t0, env.avg_idleness, env.avg_i_sum / CONST.LEN_EPISODE, env.sum_idleness
            )
        )
        # print("Node visits: ", env.visits)

        if sol_array[episode][2]==CONST.LEN_EPISODE:
            opt_sol = sol_array[episode][1]
            gap = env.sum_idleness - opt_sol
            opt_gap = 100*gap/opt_sol
        else:
            opt_gap = -1

        new_row = {"Episode": episode,
                   "Episode Length": CONST.LEN_EPISODE,
                   "Episode Reward": episode_reward,
                   "Run Time": time.time() - t0,
                   "Avg Idleness": env.avg_idleness,
                   "Global Idleness": env.avg_i_sum / CONST.LEN_EPISODE,
                   "Max Idleness": env.max_idleness,
                   "Sum Idleness": env.sum_idleness,
                   "Opt Gap": opt_gap}

        Sum_idle.append(env.sum_idleness)
        Opt_gap_list.append(opt_gap)
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        plt.cla()

        if (episode + 1) % 10 == 0 or (episode+1)==len(sol_array):
            # print(all_ep_r)
            df.to_csv(df_path)

        if args.render:
            os.makedirs(f"{results_path}/vids", exist_ok=True)
            animate_agents_on_graph(info['ep_info'], save_path=f"{results_dir}/vids/{args.method}_instance_{episode+1}.mp4")

    print(f"Avg Sum Idleness: {np.mean(Sum_idle)} +- {np.std(Sum_idle)}")
    print(f"Avg Optimal %Gap: {round(np.mean(Opt_gap_list), 2)} +- {round(np.std(Opt_gap_list), 2)}")

    # print(len(graph_demands))

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    # ax1.plot(graph_demands)
    # ax1.set_title("Graph Demands at each timestep")

    # ax2.plot(rolling_avg)
    # ax2.set_title(f"Rolling Avg of Graph Demands ({rolling_window})")

    # ax3.plot(avg_graph_demands)
    # ax3.set_title("Runnning Avg of Graph Demands")

    # fig.tight_layout()
    # fig.savefig(f'{results_path}/demands.png')
    # plt.show()