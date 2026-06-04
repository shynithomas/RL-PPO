# pylint: disable-all

'''Module for all helper functions'''
from utilities.data import Data
import pickle
import pprint
from prettytable import PrettyTable
# from opt_solve import Solve_Main
import networkx as nx
import numpy as np
import copy
import os

import time

from matplotlib import pyplot as plt


def time_it(f):
    '''To time functions'''
    def inner(*args, **kwargs):
        t_start = time.time()
        res = f(*args, **kwargs)
        t_end = time.time()

        print(f'FUNCTION TIMER: {f.__name__}: {(t_end-t_start)}s')
        return res

    return inner


def create_configs(data: Data, instance_num):
    '''Function to create the configs for the NN models'''
    # generate random graph
    num_nodes = data.sim_data.get('num_nodes')
    steps_per_episode = data.sim_data.get('steps_per_episode')
    eval_graphs_path = data.paths.get('eval_graphs_path')
    num_closest_stn_in_node_feat = data.sim_data.get('num_closest_stn_in_node_feat')
    num_closest_agents_in_node_feat = data.sim_data.get('num_closest_agents_in_node_feat')
    max_num_nodes = data.sim_data.get('max_num_nodes')
    max_num_agents = data.sim_data.get('max_num_agents')
    max_num_neigh = data.sim_data.get('max_num_neigh')

    graph = nx_graph_opt = None
    sol_array_dict = {}

    for filename in os.listdir(f"{eval_graphs_path}/"):
        if filename.startswith('graph_'):
            with open(f"{eval_graphs_path}/{filename}", 'rb') as f:
                key = filename.split('graph_')[-1]
                sol_array_dict[key] = pickle.load(f)

    # for key in sol_array_dict.keys():
    #     for sol_graph in sol_array_dict[key]:
    #         coordinates = {}
            
    #         for node in sol_graph[0].nx_graph.nodes():
    #             if len(list(sol_graph[0].nx_graph.nodes())) == 15:
    #                 coordinates[f'{node}'] = [int(node%5), int(node//5)]
    #                 sol_graph[0].nx_graph.nodes[node]['coord_x'] = [coordinates[f'{node}'][0]]
    #                 sol_graph[0].nx_graph.nodes[node]['coord_y'] = [coordinates[f'{node}'][1]]
    
    # create the environment config
    env_config = {
        'steps_per_episode': data.sim_data.get('steps_per_episode'),
        'station_dyn_flag': data.sim_data.get('station_dyn_flag'),
        'energy_flag': data.sim_data.get('energy_flag'),
        'objective_type': data.sim_data.get('objective_type'),
        'p-norm': data.sim_data.get('p-norm'),
        'num_closest_stn_in_node_feat': data.sim_data.get('num_closest_stn_in_node_feat'),
        'num_closest_agents_in_node_feat': data.sim_data.get('num_closest_agents_in_node_feat'),
    }

    # create the model config
    model_config = {
        'feature_fields': {
            'nodes': ['coord_x', 'coord_y', 'priority', 'demand', 'time_to_go',
                      *['station' for _ in range(num_closest_stn_in_node_feat)], 
                      *['station' for _ in range(num_closest_agents_in_node_feat)],],  # 'occupied'],
            'agents': ['stn_dem', 'time_to_dest', 'curr_batt', 'max_energy',],
        },
        'node_enc_hidden_size': data.sim_data['node_enc_hidden_size']\
        if 'node_enc_hidden_size' in data.sim_data else 128,
        'agent_enc_hidden_size': data.sim_data['agent_enc_hidden_size']\
        if 'agent_enc_hidden_size' in data.sim_data else 128,
        'cuda': data.cuda_status,
        'max_num_nodes': data.sim_data['max_num_nodes'],
        'max_num_neigh': data.sim_data['max_num_neigh'],
        'max_num_agents': data.sim_data['max_num_agents'],
        'max_num_edges': data.sim_data['max_num_edges'],
    }
    optimal_obj_val = -1
    if eval_graphs_path is None:
        opt_sol = get_optimal_solution(
            graph_opt=nx_graph_opt,
            steps_per_episode=steps_per_episode,
            num_agents=1,
        )
        optimal_obj_val = opt_sol.model.getVal(opt_sol.obj)
        print(f'Optimal Solution obtained. Obj val={optimal_obj_val}')

    return model_config, env_config, optimal_obj_val, sol_array_dict


def get_optimal_solution(graph_opt, steps_per_episode, num_agents):
    '''Function to get the optimal solution'''
    return (graph_opt, steps_per_episode, num_agents)


def write_model_summary(model, config_path):
    '''Function to write the model summary'''
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    with open(config_path, 'a', encoding='utf-8') as f:
        f.write(str(model) + '\n\n')
        f.write(table.get_string() + '\n')
        f.write(f'Total Trainable Params: {total_params}\n\n')

    print(table)
    print(f'Total Trainable Params: {total_params}')


def get_travel_time_along_path(nx_graph, path):
    '''Return the travel time along the given path'''
    assert isinstance(nx_graph, type(nx.DiGraph())), \
        f'Invalid Graph type: {type(nx_graph)}'
    total_travel_time = 0.0
    for i in range(len(path)-1):
        s = path[i]
        t = path[i+1]
        total_travel_time += nx_graph.edges[(s, t)]['travel_time'][0]
    return total_travel_time


def get_weighted_adjacency_matrix(nx_graph: nx.classes.digraph.DiGraph):
    num_nodes = nx_graph.number_of_nodes()
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in nx_graph.edges:
        if edge[0] == edge[1]:
            adj_matrix[edge[0], edge[1]] = -1
        else:
            adj_matrix[edge[0], edge[1]
                       ] = nx_graph.edges[edge]['travel_time'][0]

    return adj_matrix
