from CBLS.main import main
import argparse
import datetime as dt
import pytz
from utilities.data import Data
from utilities import utils
import numpy as np

import random
# random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('eval_graphs_pth', nargs='?', default=f"./NoStation_graphs")
parser.add_argument('graph_num', nargs='?' , default='25N-3A-15T-RP-RD-GRID-1')
parser.add_argument('learner_num', nargs='?', default=1000)
parser.add_argument('algorithm', nargs='?', default='ppo')
parser.add_argument('saved_results_path', nargs='?', default=None)
parser.add_argument('checkpoint_num', nargs='?', default=None)
# for convenient saving of results when running multiple tests (instances of main())
parser.add_argument('--test_name', nargs='?', default=None)
parser.add_argument('--test_instance', nargs='?', default=None)

#additional args
parser.add_argument('--num_agents', type=int, default=3)
parser.add_argument('--episode_length', type=int, default=150)
parser.add_argument('--dyn_p', type=bool, default=False)
parser.add_argument('--dyn_agents', type=bool, default=False)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--episode', type=int, default=1, help="Episode number for rendering")

args = parser.parse_args()

if __name__ == '__main__':
    
    eval_graphs_pth = args.eval_graphs_pth
    graph_num = args.graph_num
    learner_num = args.learner_num
    algo = args.algorithm
    saved_results_pth = args.saved_results_path
    checkpoint_num = args.checkpoint_num
    # for convenient saving of results when running multiple tests (instances of main())
    test_name = args.test_name
    t_instance = args.test_instance

    data = Data([eval_graphs_pth, graph_num,
                    learner_num, algo, saved_results_pth,
                    checkpoint_num, test_name])
    (model_config, env_config, _, sol_array_dict) = utils.create_configs(data, t_instance)
    if t_instance != None:
        sol_array = [sol_array_dict[t_instance]]

    data.sol_array_dict = {graph_num: sol_array_dict[graph_num]}
    sol_array = sol_array_dict[graph_num]

    ### --------------------------------------------------------------------------------------------------------
    # print(sol_array)

    # env_config['steps_per_episode'] = args.episode_length
    # env_config['graph'].num_agents = args.num_agents

    # for node in env_config['graph'].nx_graph.nodes():
    #     env_config['graph'].nx_graph.nodes[node]['station'] = [0]
    # for graph in sol_array:
    #     # graph[0].num_agents = args.num_agents
    #     for node in graph[0].nx_graph.nodes():
    #         graph[0].nx_graph.nodes[node]['station'] = [0]
    #         graph[0].nx_graph.nodes[node]['stn_dem'] = [0]*graph[0].num_agents
    #         graph[0].nx_graph.nodes[node]['occupied'] = graph[0].nx_graph.nodes[node]['occupied'][:graph[0].num_agents]

    env_config['dyn_flag'] = args.dyn_flag
    main(data, sol_array, graph_num, args)
