# pylint: disable-all

'''Module for the gym environment for multiple agents'''

import time
import torch
import math
import random
import numpy as np
import pickle as pkl
import networkx as nx
import gymnasium as gym
from itertools import count
from utilities import utils
from utilities.data import Data
from iteration_utilities import deepflatten
from gymnasium.spaces import Box, Discrete, Sequence, MultiDiscrete
# from torch_geometric.utils.convert import from_networkx


class MultiAgentEnv(gym.Env):
    '''Class for the multi-agent environment'''

    def __init__(self, config=None):

        self.graph_config = None

        self.num_nearest_stations_to_track = config.sim_data.get('num_closest_stn_in_node_feat')
        self.num_nearest_agents_to_track = config.sim_data.get('num_closest_agents_in_node_feat')

        self.max_num_nodes = config.sim_data.get('max_num_nodes')
        self.max_num_neigh = config.sim_data.get('max_num_neigh')
        self.max_num_agents = config.sim_data.get('max_num_agents')
        self.max_num_edges = config.sim_data.get('max_num_edges')
        self.sol_graph_dict = config.sol_array_dict

        self.max_distance_bw_nodes = -100

        self.n_obs_idx = {
                                    'coord_x': 0,
                                    'coord_y': 1,
                                    'priority': 2,
                                    'demand': 3,
                                    'time_to_go': 4
                                }

        self.fixed_node_feat = ['coord_x', 'coord_y', 'time_to_go']

        self.num_high_p = {10: 3, 11: 3, 15: 3, 20: 3, 25: 5, 40: 8, 50: 10, 100: 20}

        # self.num_low_p = {20: 17, 25: 20, 40: 32, 50: 40, 100: 80}

        self.node_feat_range = {'high_p': 
                                        {'demand': 
                                                {'low': 10,
                                                 'high': 19
                                                 },
                                         'priority': 
                                                {'low': 5,
                                                 'high': 7
                                                 }
                                        },
                                'low_p':
                                        {'demand': 
                                                {'low': 1,
                                                 'high': 4
                                                 },
                                         'priority': 
                                                {'low': 1,
                                                 'high': 2
                                                 }
                                        }
                                }
        
        if 'sparse_flag' in config.sim_data:
            if config.sim_data.get('sparse_flag')==True:
                print("Sparse flag is enabled!")
                self.node_feat_range['low_p']['priority']['low']=0
                self.node_feat_range['low_p']['priority']['high']=0
                self.node_feat_range['low_p']['demand']['low']=0
                self.node_feat_range['low_p']['demand']['high']=0

        self.active_agents = None
        if 'active_agents' in config.sim_data:
            if config.sim_data.get('active_agents') is not None:
                print(f"Manually setting active number of agents as {config.sim_data.get('active_agents')} for training!")
                self.active_agents = config.sim_data.get('active_agents')

        self.total_node_feat = len(list(self.n_obs_idx.keys())) + self.num_nearest_stations_to_track + self.num_nearest_agents_to_track

        self.a_obs_idx = {
                                    'max_batt': 0,
                                    'dest_node': 1,
                                    'time_to_dest': 2,
                                    't_since_last_stn_visit': 3,
                                    'curr_batt': 4,
                                    
                                 }

        self.agent_key_alias = {
                                    'max_batt': 'max_energy',
                                    'dest_node': 'occ_node',
                                    'time_to_dest': 'time_to_dest',
                                    't_since_last_stn_visit': 'stn_dem',
                                    'curr_batt': 'curr_batt',
                                 }

        self.fixed_agent_feat = ['max_batt']

        self.total_agent_feat = len(list(self.a_obs_idx.keys())) + self.total_node_feat

        self.observation_space = gym.spaces.Dict(
                                    {
                                        'node_feat': Box(low=-np.inf, 
                                                         high=np.inf,
                                                shape=(self.max_num_nodes, self.total_node_feat)
                                                        ),

                                        'node_feat_state': Box(low=-np.inf,
                                                               high=np.inf,\
                                                shape=(self.max_num_nodes, self.total_node_feat+self.total_agent_feat-1,)
                                                        ),

                                        'edge_indices': Box(low=-1, high=\
                                                            self.max_num_nodes,
                                                shape=(2, self.max_num_edges)
                                                        ),

                                        'edge_mask': Box(low=0, high=1,
                                                shape=(self.max_num_edges,)
                                                        ),

                                        'edge_weight': Box(low=0, high=np.inf,
                                                shape=(self.max_num_edges,)
                                                        ),

                                        'TimeLeft': Box(low=0, high=np.inf,
                                                        shape=(1,)),

                                        'num_active_agents': Box(low=0, high=\
                                                            self.max_num_agents,
                                                            shape=(1,)
                                                        ),

                                        'valid_neigh': Box(low=0,
                                                high=self.max_num_nodes, 
                                                    shape=(self.max_num_agents,
                                                        self.max_num_neigh,)),

                                        'num_valid_neigh': Box(low=0,
                                                high=self.max_num_neigh, 
                                                    shape=(self.max_num_agents,)),

                                        'Agent': Box(low=-np.inf, high=np.inf, 
                                                shape=(self.max_num_agents,
                                                       self.total_agent_feat,)),

                                        'agent_feat': Box(low=-np.inf,
                                                          high=np.inf, 
                                                          shape=(\
                                                            self.max_num_agents,
                                                    self.total_node_feat + self.total_agent_feat-1,)),

                                        'num_nodes': Box(low=0, high=\
                                                            self.max_num_nodes,
                                                            shape=(1,)
                                                        ),

                                        'actor_mask': Box(\
                                                    low=0, high=1, 
                                                    shape=(self.max_num_agents, (self.max_num_nodes)+self.max_num_agents,)
                                                        ),

                                        'critic_mask': Box(\
                                                    low=0, high=1, 
                                                    shape=((self.max_num_nodes)+self.max_num_agents, (self.max_num_nodes)+self.max_num_agents,)
                                                        ),

                                        'valid_act_mask': Box(\
                                                    low=0, high=1, 
                                                    shape=(self.max_num_agents, 
                                                           self.max_num_neigh)
                                                        ),

                                        # 'valid_agent_mask': Box(\
                                        #             low=0, high=1, 
                                        #             shape=(self.max_num_agents,)
                                        #                 )
                                     }
                                    )

        self.action_space = MultiDiscrete([self.max_num_neigh \
                                            for _ in range(self.max_num_agents)]
                                            ) 

        # self.action_space = gym.spaces.Tuple((Discrete(self.max_num_neigh)
        #                                     for _ in range(self.max_num_agents))
        #                                     ) 


        # print(f"\n\n\n\n\n\nENV OBS SPACE: {self.observation_space}\n\n\n\n\n\n")
        # num_var = 0
        # for key, val in self.observation_space.items():
        #     num_var += math.prod(val.shape)
        # print(f"================================")
        # print(f"[ENV]num_var: {num_var}")
        # print(f"================================")

    def _ret_weight(self, n1, n2, edges):
        """Function to aid shortes time path copmputation"""
        return float(edges['travel_time'][0])

    def get_station_nodes(self):
        """returns the list of station nodes in the graph"""
        self.station_nodes = []
        for node in self.graph.nx_graph.nodes:
            if self.graph.nx_graph.nodes[node]['station'][0]:
                self.station_nodes.append(node)


    def reset(self, seed=None, config={'a': 'b'}, options={}):

        self.timer = 0

        # create a random observation or load the state
        # print(f"\n\n\n\n\n\nRESET_CALLED\n\n\n\n\n\n")

        # config = self.graph_config
        if options is not None:
            if len(options.keys())>0:
                config=options

        if config != {'a': 'b'}:
            if 'graph' in list(config.keys()):
                self.graph = config['graph']
                self.sample_graph = False

            else:
                self.sol_graph_dict = config['sol_graph_dict']
                self.sample_graph = True

                if 'sol_graph_key' in  list(config.keys()):
                    self.key = config['sol_graph_key']
                else:
                    self.key = random.choice(list(self.sol_graph_dict.keys()))

                self.graph = self.sol_graph_dict[self.key][0][0]

        else:
            self.sample_graph = True
            self.key = random.choice(list(self.sol_graph_dict.keys()))
            self.graph = self.sol_graph_dict[self.key][0][0]
        
        #options for adding episode length and rolling horizon
        if 'episode_length' in list(config.keys()):
            self.episode_length = config['episode_length']
        else:
            self.episode_length = None
        
        if 'rolling_window' in list(config.keys()):
            self.rolling_window = config['rolling_window']
        else:
            self.rolling_window = None
        if 'eval_length' in list(config.keys()):
            self.eval_length = config['eval_length']
        else:
            self.eval_length = None

        #option for adding robotic failure events
        if 'dyn_agent_events' in list(config.keys()):
            self.dyn_agent_events = config['dyn_agent_events']
        else:
            self.dyn_agent_events = None

        #adding station flag
        if 'station' in self.graph.nx_graph.nodes[0]:
            self.station_flag = True
        else:
            self.station_flag = False

        if self.station_flag:
            self.get_station_nodes()
        else:
            self.station_nodes = []


        #Temporary fix for model's dependency on TimeLeft
        #Caution!!! - will need to change this value if the training horizon or rolling horizon changes!
        for node in self.graph.nx_graph.nodes():
            # print(self.graph.nx_graph.nodes[node]['time_to_go'])
            self.graph.nx_graph.nodes[node]['time_to_go'] = [15]

        for node in self.graph.nx_graph.nodes():
            if sum(self.graph.nx_graph.nodes[node]['occupied']) > 0:
                self.graph.nx_graph.nodes[node]['demand'] = [0]

        for agent_id in self.graph.agent_features.keys():
            if self.graph.agent_features[agent_id]['occ_node'] in self.station_nodes:
                    self.graph.agent_features[agent_id]['stn_dem'] = [0]
                    self.graph.agent_features[agent_id]['curr_batt'] = [self.graph.agent_features[agent_id]['max_energy']]
            if not self.station_flag:
                    self.graph.agent_features[agent_id]['stn_dem'] = [0]
                    self.graph.agent_features[str(agent_id)]['curr_batt'] = self.graph.nx_graph.nodes[0]['time_to_go'][0]

        self.shortest_path_lengths = dict(nx.all_pairs_bellman_ford_path_length(self.graph.nx_graph, weight=self._ret_weight))

        for node in self.graph.nx_graph.nodes():
            self.shortest_path_lengths[node][node] = 0

        self.distance_to_all_stn_nodes = {}
        self.min_time_to_stn = {}

        for node in self.graph.nx_graph.nodes():
            if self.station_flag:
                self.distance_to_all_stn_nodes[node] = []
                self.min_time_to_stn[node] = None
                for stn_node in self.station_nodes:
                    if stn_node == node:
                        self.distance_to_all_stn_nodes[node].append(0.0)
                        self.min_time_to_stn[node] = 0
                    else:
                        self.distance_to_all_stn_nodes[node].append(\
                                        self.shortest_path_lengths[node][stn_node])

                self.distance_to_all_stn_nodes[node].sort()

                if node not in self.station_nodes:
                    self.min_time_to_stn[node] = float(self.distance_to_all_stn_nodes[node][0])

                len_temp_list = len(self.distance_to_all_stn_nodes[node])
                
                if len_temp_list < self.num_nearest_stations_to_track:
                    self.distance_to_all_stn_nodes[node] = [*self.distance_to_all_stn_nodes[node], *[self.max_distance_bw_nodes for _ in range(self.num_nearest_stations_to_track - len_temp_list)]]
            else:
                self.distance_to_all_stn_nodes[node] = [0]*self.num_nearest_stations_to_track
                self.min_time_to_stn[node] = 0


        if 'state' not in config.keys():

            self.state = {}

            #defining episode_length and rolling_horizon
            if self.episode_length is None:
                self.episode_length = self.graph.nx_graph.nodes[0]['time_to_go'][0]
                self.state['TimeLeft'] = self.graph.nx_graph.nodes[0]['time_to_go'][0]
            else:
                self.state['TimeLeft'] = self.episode_length
                pass
                
            if self.rolling_window is None:
                self.rolling_window = self.episode_length
            else:
                self.state['TimeLeft'] = self.rolling_window
            
            if self.eval_length is None:
                self.eval_length = self.episode_length

            if self.active_agents is not None:
                if self.active_agents == "random":
                    self.state['num_active_agents'] = np.random.randint(1, self.max_num_agents+1)
                elif self.active_agents == "max":
                    self.state['num_active_agents'] = self.max_num_agents
                else:
                    try:
                        assert int(self.active_agents) <= self.max_num_agents
                        self.state['num_active_agents'] = int(self.active_agents)
                    except:
                        raise Exception("active_agents config value not recognized")
            else:
                self.state['num_active_agents'] = self.graph.num_agents

            # print("Active agents: ", self.state['num_active_agents'])

            if self.sample_graph:
                all_nodes_feat = []
                if self.station_flag:
                    temp_list = []
                    for node in self.graph.nx_graph.nodes:
                        if not self.graph.nx_graph.nodes[node]['station'][0]:
                            temp_list.append(node)
                else:
                    temp_list = [node for node in self.graph.nx_graph.nodes]

                random.shuffle(temp_list)
                high_p_nodes = temp_list[:self.num_high_p[len(list(self.graph.nx_graph.nodes()))]]


                for node in self.graph.nx_graph.nodes():
                    temp_node_feat = [None for _ in self.n_obs_idx.keys()]

                    for feat, index in self.n_obs_idx.items():
                        if (feat in self.fixed_node_feat) or (node not in temp_list):
                            temp_node_feat[index] = self.graph.nx_graph.nodes[node][feat][0]
                        else:
                            if node in high_p_nodes:
                                temp_node_feat[index] = np.random.randint(\
                                            low=self.node_feat_range['high_p'][feat]['low'],
                                            high=self.node_feat_range['high_p'][feat]['high']+1)
                            else:
                                temp_node_feat[index] = np.random.randint(\
                                            low=self.node_feat_range['low_p'][feat]['low'],
                                            high=self.node_feat_range['low_p'][feat]['high']+1)

                    all_nodes_feat.append(list(deepflatten(temp_node_feat)))

                some_agent_at_node = False
                all_agents_feat = []
                self.agent_dests = {}
                self.agent_time_to_dest = {}

                for agent_id in range(self.state['num_active_agents']):
                    temp_feat_array = [None for _ in self.a_obs_idx.keys()]

                    for feat, alias in self.agent_key_alias.items():
                        if feat in self.fixed_agent_feat:
                            temp_feat_array[self.a_obs_idx[feat]] = self.graph.agent_features[str(agent_id)][alias]

                        else:
                            if feat == 'dest_node':
                                edge_ind_samp = np.random.randint(0, \
                                          high=len(list(self.graph.nx_graph.edges)))
                                edge = list(self.graph.nx_graph.edges)[edge_ind_samp]
                                # edge = (n_1, n_2)
                                dest_node_samp = edge[1]
                                trav_time_on_edge = self.graph.nx_graph.edges[\
                                                            edge]['travel_time'][0]-1
                                if trav_time_on_edge:
                                    time_to_dest_samp = np.random.randint(0, \
                                                           high=trav_time_on_edge)
                                else:
                                    time_to_dest_samp = 0
                                some_agent_at_node = \
                                       some_agent_at_node or (not time_to_dest_samp)

                                if (agent_id == self.state['num_active_agents'] - 1)\
                                    and (not some_agent_at_node):
                                    time_to_dest_samp = 0

                                temp_feat_array[self.a_obs_idx['dest_node']] = \
                                                                    dest_node_samp
                                self.agent_dests[int(agent_id)] = temp_feat_array[\
                                                        self.a_obs_idx['dest_node']]
                                temp_feat_array[self.a_obs_idx['time_to_dest']] = \
                                                                time_to_dest_samp
                                self.agent_time_to_dest[int(agent_id)] = \
                                     temp_feat_array[self.a_obs_idx['time_to_dest']]

                            elif feat == 'time_to_dest':
                                pass

                            elif feat == 't_since_last_stn_visit':
                                if self.station_flag:
                                    t_since_stn_samp = np.random.randint(\
                                                        low=self.min_time_to_stn[\
                                                                temp_feat_array[\
                                                self.a_obs_idx['dest_node']]] + \
                                                    temp_feat_array[self.a_obs_idx[\
                                                                'time_to_dest']], \
                                                    high=3*self.state['TimeLeft']
                                                        )
                                else:
                                    t_since_stn_samp = 0

                                temp_feat_array[self.a_obs_idx[\
                                   't_since_last_stn_visit']] = t_since_stn_samp

                            elif feat == 'curr_batt':
                                if self.station_flag:
                                    low = self.min_time_to_stn[temp_feat_array[\
                                                self.a_obs_idx['dest_node']]] + \
                                                    temp_feat_array[self.a_obs_idx[\
                                                                    'time_to_dest']]
                                    high = temp_feat_array[self.a_obs_idx[\
                                                                        'max_batt']]

                                    if low == 0 or (low == high):
                                        curr_batt_samp = high

                                    else:
                                        curr_batt_samp = np.random.randint(\
                                                            low=max(1., 
                                                            self.min_time_to_stn[\
                                                                temp_feat_array[\
                                                self.a_obs_idx['dest_node']]] + \
                                                    temp_feat_array[self.a_obs_idx[\
                                                                'time_to_dest']]), \
                                            high=temp_feat_array[self.a_obs_idx[\
                                                                    'max_batt']]\
                                                            )
                                else:
                                    curr_batt_samp = self.state['TimeLeft']
                                temp_feat_array[self.a_obs_idx[\
                                                  'curr_batt']] = curr_batt_samp

                    all_agents_feat.append(temp_feat_array)

            else:
                all_nodes_feat = []

                for node in self.graph.nx_graph.nodes():
                    temp_node_feat = [None for _ in self.n_obs_idx.keys()]

                    for feat, index in self.n_obs_idx.items():
                        temp_node_feat[index] = self.graph.nx_graph.nodes[node][feat][0]
                    all_nodes_feat.append(list(deepflatten(temp_node_feat)))

                all_agents_feat = []
                for agent_id in range(self.state['num_active_agents']):
                    temp_feat_array = [None for _ in self.a_obs_idx.keys()]
                    if self.graph.agent_features[str(agent_id)]['occ_node'] in self.station_nodes:
                        self.graph.agent_features[str(agent_id)]['stn_dem'] = [0]
                        self.graph.agent_features[str(agent_id)]['curr_batt'] = [self.graph.agent_features[str(agent_id)]['max_energy']]
                    for feat, alias in self.agent_key_alias.items():
                        temp_feat_array[self.a_obs_idx[feat]] = self.graph.agent_features[str(agent_id)][alias]
                    all_agents_feat.append(np.array(list(deepflatten(temp_feat_array))))
                    if all_agents_feat[int(agent_id)][self.a_obs_idx['dest_node']] == -1:
                        all_agents_feat[int(agent_id)][self.a_obs_idx['dest_node']] =\
                                    self.graph.agent_features[str(agent_id)]['edge'][-1]

            for ind, node in enumerate(self.graph.nx_graph.nodes()):
                all_nodes_feat[node] = [*all_nodes_feat[node], *self.distance_to_all_stn_nodes[node][:self.num_nearest_stations_to_track]]

                time_to_agents = [self.shortest_path_lengths[node][all_agents_feat[int(agent_id)][self.a_obs_idx['dest_node']]] + all_agents_feat[int(agent_id)][self.a_obs_idx['time_to_dest']] for agent_id in range(self.state['num_active_agents'])]
                time_to_agents.sort()
                len_temp_list = len(time_to_agents)
                if len_temp_list < self.num_nearest_agents_to_track:
                    time_to_agents = [*time_to_agents, *[\
                                                    self.max_distance_bw_nodes \
                                      for _ in range(\
                                            self.num_nearest_agents_to_track - \
                                                                len_temp_list)]]

                all_nodes_feat[node] = [*all_nodes_feat[node], *time_to_agents[\
                                            :self.num_nearest_agents_to_track]]

            edge_node_feat_list = []
            edge_to_node_map = []
            edge_from_node_map = []
            self.edge_node_relation_index_dict = {}

            for node in self.graph.nx_graph.nodes():
                node_feat = all_nodes_feat[node]
                successors = list(self.graph.nx_graph.successors(node))
                successors.sort()
                for neigh in successors:
                    neigh_feat = all_nodes_feat[neigh]
                    edge_feat = [self.graph.nx_graph[node][neigh][\
                                           'travel_time'][0] for _ in node_feat]
                    edge_node_feat_list.append([*node_feat, *edge_feat,
                                                *neigh_feat])
                    edge_from_node_map.append(node)
                    edge_to_node_map.append(neigh)
                    self.edge_node_relation_index_dict[(node, neigh)] = \
                              pkl.loads(pkl.dumps(len(edge_node_feat_list) - 1))

            edge_node_feat = np.array(edge_node_feat_list)
            edge_node_feat_padding = np.zeros_like(edge_node_feat[0,:].reshape(\
                                                                         1, -1))
            edge_node_feat_padding = edge_node_feat_padding.repeat(\
                    self.max_num_edges - len(list(self.graph.nx_graph.edges())),
                                                                   axis=0)
            edge_node_feat_mask = np.ones((1, 1, \
                                        len(list(self.graph.nx_graph.edges()))))
            edge_node_feat_mask_padding = np.zeros((1, 1, \
                   self.max_num_edges - len(list(self.graph.nx_graph.edges()))))
            edge_node_feat_mask = np.concatenate((edge_node_feat_mask,
                                                  edge_node_feat_mask_padding),
                                                 axis=-1)
            edge_to_node_map = np.array(edge_to_node_map)
            edge_node_map_padding = -np.ones(self.max_num_edges - \
                                             len(edge_to_node_map))
            all_nodes_feat = np.array(all_nodes_feat, dtype=np.float32)

            self.state['num_nodes'] = all_nodes_feat.shape[0]

            node_padding = np.zeros((self.max_num_nodes-self.state['num_nodes'],
                                     all_nodes_feat.shape[-1]))
            node_mask_ = np.ones(self.state['num_nodes'])
            node_mask_padding = np.zeros(node_padding.shape[0])
            node_mask = np.concatenate((node_mask_, node_mask_padding), axis=-1)

            all_nodes_feat = np.concatenate((all_nodes_feat, node_padding), axis=0)

            for agent_id in range(self.state['num_active_agents']):
                all_agents_feat[int(agent_id)] = [*all_agents_feat[\
                                                               int(agent_id)], \
                                        *all_nodes_feat[int(all_agents_feat[\
                                                 int(agent_id)][self.a_obs_idx[\
                                                                'dest_node']])]]

            num_nodes = len(self.graph.nx_graph.nodes())
            num_edges = len(self.graph.nx_graph.edges())
            edge_links = self.np_random.integers(
                    low=0, high=num_nodes, size=(num_edges, 2), dtype=np.int32
                )

            self.state['edge_indices'] = np.zeros(shape=(2, self.max_num_edges))
            self.state['edge_mask'] = np.zeros(shape=(self.max_num_edges,), dtype=np.bool)#.bool()
            self.state['edge_weight'] = np.zeros(shape=(self.max_num_edges,))

            for ind, edge in enumerate(self.graph.nx_graph.edges()):
                self.state['edge_indices'][0][ind] = edge[0]
                self.state['edge_indices'][1][ind] = edge[1]
                self.state['edge_weight'][ind] = self.graph.nx_graph.edges[edge]['travel_time'][0]
                self.state['edge_mask'][ind] = True

            node_seq_pad = np.zeros(shape=(self.max_num_nodes-self.state['num_nodes'], self.total_node_feat))

            self.state['node_feat'] = all_nodes_feat

            all_agents_feat = np.array(all_agents_feat)

            agent_mask_ = np.ones(self.state['num_active_agents'])
            agent_mask_padding = np.zeros(self.max_num_agents - self.state['num_active_agents'])
            agent_mask = np.concatenate((agent_mask_, agent_mask_padding), axis=-1)

            mask = np.concatenate((node_mask, agent_mask), axis=-1).reshape(1, 1, -1)
            mask_rep = mask.repeat(self.state['num_active_agents'], axis=1)
            mask_pad = np.zeros_like(mask).repeat(self.max_num_agents-self.state['num_active_agents'], axis=1)

            # self.state['actor_mask'] = np.squeeze(np.concatenate((mask_rep, mask_pad), axis=1))
            self.state['actor_mask'] = np.squeeze(np.concatenate((mask_rep, mask_pad), axis=1), axis=0)

            critic_mask = mask.repeat(num_nodes, axis=1)
            critic_mask_pad = np.zeros_like(mask).repeat(self.max_num_nodes-num_nodes, axis=1)
            critic_mask = np.concatenate((critic_mask, critic_mask_pad), axis=1)
            critic_mask_agent = mask.repeat(self.state['num_active_agents'], axis=1)
            critic_mask = np.concatenate((critic_mask, critic_mask_agent), axis=1)
            critic_mask_agent_pad = np.zeros_like(mask).repeat(self.max_num_agents-self.state['num_active_agents'], axis=1)

            self.state['critic_mask'] = np.squeeze(np.concatenate((critic_mask, critic_mask_agent_pad), axis=1))

            agent_padding = np.zeros((self.max_num_agents-self.state['num_active_agents'], all_agents_feat.shape[-1]))

            self.state['Agent'] = np.concatenate((all_agents_feat, agent_padding), axis=0)

            self.update_node_state_feat_and_agent_feat()
            self.update_valid_neigh()

        else:
            if self.rolling_window is None:
                self.rolling_window = self.episode_length
            if self.eval_length is None:
                self.eval_length = self.episode_length

            self.state = pkl.loads(pkl.dumps(config['state']))

            self.state['TimeLeft'] = int(self.state['TimeLeft'][0])
            self.state['num_nodes'] = int(self.state['num_nodes'][0])
            self.state['num_active_agents'] = int(self.state['num_active_agents'][0])

        #initialize agent tracking - used to  track agents ids and features in case of dynamic agents.
        #reset() should only be called once for an episode to initialize all the agents present in the graph. 
        #reset() will not be called again in case of agent removal or addition, that will be handled internally in the step funcition.
        self.init_agents()

        if 'timer' in config.keys():
            #to be provided along with 'state' option
            self.timer = config['timer']

        self.dyn_p_flag = False
        if 'dyn_p' in config.keys():
            self.dyn_p = config['dyn_p']
            if self.dyn_p is not None:
                self.dyn_p_flag = True
                self.define_dyn_p()
            


        self.edge_travel_time_dict = {}
        for ind, edge in enumerate(self.graph.nx_graph.edges()):
            self.edge_travel_time_dict[tuple(edge)] = self.graph.nx_graph.edges[edge]['travel_time'][0]

        self.update_action_masks()

        info = {}
        self.episode_return = 0

        assert min([
                    self.state['Agent'][int(agent_id)][\
                        self.a_obs_idx['time_to_dest']] \
                        for agent_id in range(self.state['num_active_agents'])
                    ]
                ) == 0

        # print(f"self.state: {self.state}")

        # print("\nPrinting shapes self.state")
        for key, val in self.state.items():
            if isinstance(val, torch.Tensor):
                self.state[key] = val.float().numpy()
            if isinstance(val, int):
                self.state[key] = np.asarray(val).reshape(1,)

            # print(key, self.state[key].shape, type(self.state[key]))

        return self.state, info


    def update_action_masks(self):

        # self.state['valid_agent_mask'] = torch.zeros(self.max_num_agents, 
        #                                              dtype=torch.bool)

        self.state['valid_act_mask'] = torch.zeros((self.max_num_agents,
                                                    self.max_num_neigh),
                                                   dtype=torch.bool)

        # self.state['valid_agent_mask'][list(range(self.state['num_active_agents']))] = True  # Only set valid agents

        for agent_id in range(self.state['num_active_agents']):
            self.state['valid_act_mask'][agent_id, list(range(int(self.state['num_valid_neigh'][agent_id])))] = True
            if int(self.state['num_valid_neigh'][agent_id])==0:
                self.state['valid_act_mask'][agent_id, 0] = True

        for agent_id in range(self.state['num_active_agents'], \
                                                        self.max_num_agents):
            self.state['valid_act_mask'][agent_id, 0] = True



    def update_node_state_feat_and_agent_feat(self):

        """update node and agent features as needed for state"""

        temp_nx_graph = pkl.loads(pkl.dumps(self.graph.nx_graph))
        nodes_feat_t_pyG = self.state['node_feat']
        node_zero_pad = torch.zeros((nodes_feat_t_pyG.shape[0], 
                                     self.total_agent_feat-1))
        agent_zero_pad = torch.zeros((self.state['Agent'].shape[0],
                                      self.total_node_feat))
        all_node_feat_temp = torch.cat((node_zero_pad,
                                        torch.from_numpy(nodes_feat_t_pyG)),
                                       dim=-1)
        all_agents_feat_temp = torch.cat((torch.from_numpy(self.state['Agent'][\
                          :, :1]), torch.from_numpy(self.state['Agent'][:, 2:]),
                                          agent_zero_pad), dim=-1)

        self.state['node_feat_state'] = all_node_feat_temp
        self.state['agent_feat'] = all_agents_feat_temp


    def update_time_to_agents_and_dest_node_feat_in_obs(self):

        """ update time to each agents for each node and the 
            destination node features for each agent"""

        for node in self.graph.nx_graph.nodes():
            time_to_agents = [self.shortest_path_lengths[node][\
                                        int(self.state['Agent'][int(agent_id)][\
                                              self.a_obs_idx['dest_node']])] + \
                                            self.state['Agent'][int(agent_id)][\
                                               self.a_obs_idx['time_to_dest']] \
                        for agent_id in range(self.state['num_active_agents'])]
            time_to_agents.sort()

            len_temp_list = len(time_to_agents)
            if len_temp_list < self.num_nearest_agents_to_track:
                time_to_agents = [*time_to_agents, *[self.max_distance_bw_nodes\
              for _ in range(self.num_nearest_agents_to_track - len_temp_list)]]

            self.state['node_feat'][node][\
                                        -self.num_nearest_agents_to_track:] = \
                torch.tensor(time_to_agents[:self.num_nearest_agents_to_track])

        for agent_id in range(self.state['num_active_agents']):
            self.state['Agent'][int(agent_id)][-int(self.total_node_feat):] = \
                            self.state['node_feat'][int(self.state['Agent'][\
                                   int(agent_id)][self.a_obs_idx['dest_node']])]
        

    def action_wrapper(self, act):

        """ convert the action tensor to action dictionary with agents as keys 
            and its next node as value """

        action = {}
        for agent_id in range(self.state['num_active_agents']):
            if (int(act[int(agent_id)]) == -1) or \
                            (not self.state['num_valid_neigh'][int(agent_id)]):
                action[int(agent_id)] = -1
            else:
                assert int(act[agent_id]) < self.state['num_valid_neigh'][\
                                                                  int(agent_id)]
                action[int(agent_id)] = self.state['valid_neigh'][\
                                              int(agent_id)][int(act[agent_id])]

        return action

    def define_dyn_p(self):
        print("Dynamic priority enabled!")
        #recording the list of timestamps when priority changing events occur
        self.event_times = [int(k) for k in self.dyn_p.keys()]

        #accessing a dummy dict to capture list of survey nodes
        dummy_dict = list(self.dyn_p.values())[0]
        self.survey_nodes = [int(n) for n in dummy_dict.keys()]

    def reset_priorities(self, event_time):
        #reset demands to zero for zero-priority nodes, before updating priorities
        #this makes sure that the pseudo demands are neglected
        for node in self.survey_nodes:
            if self.state['node_feat'][node][self.n_obs_idx['priority']] == 0:
                self.state['node_feat'][node][self.n_obs_idx['demand']] = 0
        # print(f"\nUpdating priorities at timestep {event_time}")
        for node in self.survey_nodes:
            self.state['node_feat'][node][self.n_obs_idx['priority']] = self.dyn_p[str(event_time)][str(node)]

    def init_agents(self):
        self.failed_agent_dict = {} #adding global unique id (guid) for each agent (1,2,...,n)
        #(CAREFUL: the agent_id variable used in this file represent agent index instead of the unique id)
        self.guid_id_map = {} #mapping guid to id
        self.id_guid_map = {} #mapping id to guid
        for agent_id in range(self.state['num_active_agents']):
            # self.agent_dict[agent_id+1] = self.state['Agent'][int(agent_id)]
            #agent indices represent unique ids at the start of an episode
            self.guid_id_map[agent_id+1] = agent_id
            self.id_guid_map[agent_id] = agent_id+1

    def agent_deletion(self, failed_agent_guid):
        """ function to remove a failed agent from the agent features """

        # print(self.state['num_active_agents'])
        failed_agent_id = self.guid_id_map[failed_agent_guid]
        assert failed_agent_id < self.state['num_active_agents']
        self.failed_agent_dict[failed_agent_guid] = pkl.loads(pkl.dumps(self.state['Agent'][int(failed_agent_id)]))

        for agent_id in range(failed_agent_id, self.state['num_active_agents']-1):
            self.id_guid_map[agent_id] = int(self.id_guid_map[agent_id+1])
            self.guid_id_map[self.id_guid_map[agent_id]] = agent_id

            self.state['Agent'][int(agent_id)] = pkl.loads(pkl.dumps(\
                                        self.state['Agent'][int(agent_id+1)]))
        
        self.id_guid_map[self.state['num_active_agents']-1] = None
        self.guid_id_map[failed_agent_guid] = None
        # print(failed_agent_guid, self.guid_id_map[failed_agent_guid])
        self.state['Agent'][self.state['num_active_agents']-1] = np.zeros(\
                                        shape=(self.state['Agent'].shape[1],))
        
        self.state['num_active_agents'] -= 1

        self.state['critic_mask'][self.max_num_nodes + self.state['num_active_agents']:, :] = 0
        self.state['critic_mask'][:, self.max_num_nodes + self.state['num_active_agents']:] = 0

        self.update_time_to_agents_and_dest_node_feat_in_obs()
        self.update_valid_neigh()
        self.update_node_state_feat_and_agent_feat()

    def agent_addition(self, agent_guid):
        """ function to add back a failed agent """

        # print("agent_id: ", self.guid_id_map[agent_guid])
        assert self.guid_id_map[agent_guid] is None
        assert self.max_num_agents>self.state['num_active_agents']

        self.state['Agent'][int(self.state['num_active_agents'])] = self.failed_agent_dict[agent_guid]
        self.id_guid_map[self.state['num_active_agents']] = agent_guid
        self.guid_id_map[agent_guid] = self.state['num_active_agents']

        self.state['num_active_agents'] += 1

        #update critic mask
        # ---------- WRITE CODE HERE ------------
        self.state['critic_mask'][self.max_num_nodes+self.state['num_active_agents']-1, :self.max_num_nodes+self.state['num_active_agents']] = 1
        self.state['critic_mask'][:self.max_num_nodes+self.state['num_active_agents'], self.max_num_nodes+self.state['num_active_agents']-1] = 1

        self.update_time_to_agents_and_dest_node_feat_in_obs()
        self.update_valid_neigh()
        self.update_node_state_feat_and_agent_feat()


    def step(self, action_):
        # print(f"********************************action_: {action_}")

        """ step function for environment """

        self.state['TimeLeft'] = int(self.state['TimeLeft'][0])
        self.state['num_nodes'] = int(self.state['num_nodes'][0])
        self.state['num_active_agents'] = int(self.state['num_active_agents'][0])

        if self.state['TimeLeft'] == 0:
            return self.state, 0, True, True, {}

        if not isinstance(action_, dict):
            action = self.action_wrapper(action_)
        else:
            print(f"action_: {action_}")
            action = action_

        info = {}
        assert min([
                    self.state['Agent'][int(agent_id)][\
                        self.a_obs_idx['time_to_dest']] \
                        for agent_id in range(self.state['num_active_agents'])
                    ]
                ) == 0
        self.reward = 0
        self.obj_contrib = 0

        self.update_agent_dest_timetodest_feat_from_action(action)
        
        for _ in count():

            self.timer += 1

            if self.episode_length - self.timer < self.rolling_window:
                self.state['TimeLeft'] -= 1

            reaching_nodes = self.update_agent_feat_return_reaching_nodes()
            self.update_node_features(reaching_nodes)
            self.update_time_to_agents_and_dest_node_feat_in_obs()
            self.update_valid_neigh()
            self.update_node_state_feat_and_agent_feat()
            self.update_reward()

            #updating priorities if they are dynamically changing through the episode
            if self.dyn_p_flag:
                if self.timer in self.event_times:
                    self.reset_priorities(self.timer)
                    self.update_node_state_feat_and_agent_feat()

            if self.dyn_agent_events is not None:
                if self.timer in self.dyn_agent_events.keys():
                    print(f"Agent {self.dyn_agent_events[self.timer]['agent_guid']} {self.dyn_agent_events[self.timer]['type']}ed at time {self.timer}")
                    if self.dyn_agent_events[self.timer]['type']=='fail':
                        self.agent_deletion(self.dyn_agent_events[self.timer]['agent_guid'])
                    elif self.dyn_agent_events[self.timer]['type']=='recover':
                        self.agent_addition(self.dyn_agent_events[self.timer]['agent_guid'])
                
            # print("Timer: ", self.timer)
            # for node in sorted(self.survey_nodes):
            #     print(int(self.state['node_feat'][node][self.n_obs_idx['priority']]))

            if (len(reaching_nodes) > 0) or (self.state['TimeLeft'] == 0) or (self.timer%self.eval_length == 0):
                break

        self.update_action_masks()

        done = terminated = (self.state['TimeLeft'] == 0) or (self.timer%self.eval_length==0)
        self.episode_return += self.obj_contrib
        self.final_ret = self.episode_return
        self.last_state = pkl.loads(pkl.dumps(self.state))

        for key, val in self.state.items():
            if isinstance(val, torch.Tensor):
                self.state[key] = val.numpy()

        info['final_observation'] = {}
        for key, val in self.state.items():
            info['final_observation'][key] = tuple([val])

        info['episode_return'] = self.episode_return
        info['obj_contrib'] = self.obj_contrib

        # print(f"self.state: {self.state}")

        for key, val in self.state.items():
            if isinstance(val, torch.Tensor):
                self.state[key] = val.float().numpy()
            if isinstance(val, int):
                self.state[key] = np.asarray(val).reshape(1,)

        return self.state, self.reward, terminated, done, info

    def update_valid_neigh(self):

        """ update the set of valid neighbouring nodes for each agent """

        self.state['valid_neigh'] = -np.ones(shape=(self.max_num_agents,
                                                    self.max_num_neigh,))
        self.state['num_valid_neigh'] = np.zeros(shape=(self.max_num_agents,))

        for agent_id in range(self.state['num_active_agents']):
            agent_time_to_dest = self.state['Agent'][int(agent_id)][\
                                                 self.a_obs_idx['time_to_dest']]
            count_ = 0

            if int(agent_time_to_dest) == 0:
                agent_loc = int(self.state['Agent'][int(agent_id)][\
                                                   self.a_obs_idx['dest_node']])
                agent_bat = self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                                                   'curr_batt']]
                successors = list(self.graph.nx_graph.successors(agent_loc))
                successors.sort()
                for node in successors:
                    if agent_bat >= self.graph.nx_graph.edges[(agent_loc, \
                        node)]['travel_time'][0] + self.min_time_to_stn[node]:
                        self.state['valid_neigh'][int(agent_id)][count_] = node
                        count_ += 1

                self.state['num_valid_neigh'][int(agent_id)] = count_
                     

    def update_reward(self):

        """ update the reward and the objective function with corresponding
            contribution from the current time step """

        temp_rew = sum(
                        [
                         self.state['node_feat'][ind][\
                                                self.n_obs_idx['priority']]*\
                         self.state['node_feat'][ind][\
                                                    self.n_obs_idx['demand']]
                         for ind in range(self.state['num_nodes'])
                        ]
                        ) + \
                   sum(
                        [
                         self.state['Agent'][int(agent_id)][\
                                     self.a_obs_idx['t_since_last_stn_visit']]\
                         for agent_id in range(self.state['num_active_agents'])
                        ]
                        )
        
        if not self.station_flag:
            assert temp_rew == sum(
                        [
                         self.state['node_feat'][ind][\
                                                self.n_obs_idx['priority']]*\
                         self.state['node_feat'][ind][\
                                                    self.n_obs_idx['demand']]
                         for ind in range(self.state['num_nodes'])
                        ]
                        )

        self.reward -= temp_rew/(500)

        self.obj_contrib -= temp_rew

    def update_agent_dest_timetodest_feat_from_action(self, act):

        """ update time to destination and destination node for each active
            agent as per the current action """

        for agent_id in range(self.state['num_active_agents']):

            if act[int(agent_id)] != -1:

                assert int(self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                                                        'time_to_dest']]) == 0

                edge = (int(self.state['Agent'][int(agent_id)][\
                                self.a_obs_idx['dest_node']]), int(act[int(agent_id)]))

                self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                'time_to_dest']] = \
                                        self.edge_travel_time_dict[tuple(edge)]

                
                self.state['Agent'][int(agent_id)][self.a_obs_idx['dest_node']] = \
                                                                int(act[int(agent_id)])

            else:
                if int(self.state['Agent'][int(agent_id)][self.a_obs_idx['time_to_dest']]) <= 0:
                    print(f"act[{agent_id}]: {act[int(agent_id)]}, self.state['Agent'][int(agent_id)][self.a_obs_idx['time_to_dest']]: {self.state['Agent'][int(agent_id)][self.a_obs_idx['time_to_dest']]}")

                assert int(self.state['Agent'][int(agent_id)][self.a_obs_idx['time_to_dest']]) > 0


    def update_node_features(self, reaching_nodes):

        """ update node demands depending on agent presence at the current time
            step """

        for ind in range(self.state['num_nodes']):
            self.state['node_feat'][ind][self.n_obs_idx['time_to_go']] = \
                                                        self.state['TimeLeft']
            
            if (ind in reaching_nodes) or (ind in self.station_nodes):
                self.state['node_feat'][ind][self.n_obs_idx['demand']] = 0
            else:
                self.state['node_feat'][ind][self.n_obs_idx['demand']] += 1


    def update_agent_feat_return_reaching_nodes(self):

        """ update agent features and return the nodes that are being reached by any agent in the current time step """

        list_of_reaching_nodes = []

        for agent_id in range(self.state['num_active_agents']):
            self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                                           'time_to_dest']] -= 1
            if self.station_flag:
                self.state['Agent'][int(agent_id)][self.a_obs_idx['curr_batt']] -= 1
                self.state['Agent'][int(agent_id)][\
                                    self.a_obs_idx['t_since_last_stn_visit']] += 1

            if (int(self.state['Agent'][int(agent_id)][\
                                    self.a_obs_idx['time_to_dest']]) == 0
                ):

                list_of_reaching_nodes.append(int(self.state['Agent'][\
                                   int(agent_id)][self.a_obs_idx['dest_node']]))

                if list_of_reaching_nodes[-1] in self.station_nodes:
                    self.state['Agent'][int(agent_id)][\
                                self.a_obs_idx['t_since_last_stn_visit']] = 0

                    self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                                            'curr_batt']] = \
                            self.state['Agent'][int(agent_id)][self.a_obs_idx[\
                                                                    'max_batt']]

        return list_of_reaching_nodes
