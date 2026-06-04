import gymnasium as gym
import numpy as np
from multi_agent_env import MultiAgentEnv
import random
import pickle as pkl
import networkx as nx

class obsMap():
    #dummy class of obsMap
    def __init__(self, env):
        
        self.G = env.graph.nx_graph

        self.patrol_nodes = list(self.G.nodes())
        self.nodes = {}
        for node in self.patrol_nodes:
            self.nodes[node] = [self.G.nodes[node]['coord_x'][0], self.G.nodes[node]['coord_y'][0]]
        # self.nb_list = [[nb for nb in self.G[n]] for n in self.G.nodes()]

        self.patrol_points = [self.nodes[self.patrol_nodes[i]][0] for i in range(len(self.patrol_nodes))], \
                                [self.nodes[self.patrol_nodes[i]][1] for i in range(len(self.patrol_nodes))]
        
        G = pkl.loads(pkl.dumps(self.G))  #deep copy
        for u,v,d in G.edges(data=True):
            #minor fix needed due to edge weights not being scalar
            d['travel_time'] = d['travel_time'][0]
        self.adj = nx.to_numpy_array(G, nodelist=self.patrol_nodes, weight='travel_time')

    def get_key(self, val, my_dict):
        for key, value in my_dict.items():
            if val[0] == value[0] and val[1] == value[1]:
                return key
            
    def get_neighbor_nodes_number(self, node):
        #return a list of neighbors given node index
        nb_list = [nb for nb in self.G[node]]
        return nb_list

class CONST():
    def __init__(self, env):

        self.NUM_AGENTS = env.state['num_active_agents'][0]
        self.LEN_EPISODE = env.episode_length

class Env():
    #dummy env wrapper over multi_agent_env
    def __init__(self, env_config):
        self.env = MultiAgentEnv(env_config)
        # self.obsMap = obsMap(self.env)
        # self.adj = self.env.adj
        # self.patrol_points = self.obsMap.patrol_points
        # self.CONST = CONST(self.env)
        # self.interval = np.zeros(len(self.patrol_points[0]))
        # # self.visits = self.env.visits
        # self.visits = np.zeros(len(self.patrol_points[0]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # else:
        #     obs, info = self.env.reset(initial_demand=kwargs['initial_demand'])
        self.obsMap = obsMap(self.env)
        self.adj = self.obsMap.adj
        self.patrol_points = self.obsMap.patrol_points
        self.CONST = CONST(self.env)
        self.interval = np.zeros(len(self.patrol_points[0]))
        # self.visits = self.env.visits
        self.visits = np.zeros(len(self.patrol_points[0]))

        priorities = []

        state = []
        #assuming all nodes are being patrolled
        for node in self.obsMap.patrol_nodes:
            state.append([obs['node_feat'][node][self.env.n_obs_idx['coord_x']],
                            obs['node_feat'][node][self.env.n_obs_idx['coord_y']],
                            obs['node_feat'][node][self.env.n_obs_idx['demand']]*obs['node_feat'][node][self.env.n_obs_idx['priority']]
            ])
            priorities.append(obs['node_feat'][node][self.env.n_obs_idx['priority']])
        
        # print(priorities)
        # print(info['loc'], [obs.agent_features[str(i)]['edge'] for i in range(self.CONST.NUM_AGENTS)])

        last_vertex = [None for i in range(self.CONST.NUM_AGENTS)]   #not being used by CBLS policy

        # nodes_last_visit = info['loc']  #index of each node an agent occupies (what happens when agent is on edge???)
        nodes_last_visit = [[-1,-1] for i in range(self.CONST.NUM_AGENTS)]  # x-y coordinates for the occupied nodes

        dest_nodes = [[-1,-1] for i in range(self.CONST.NUM_AGENTS)]  #different than last visit nodes if agents start on edge

        neighbor_nodes = {}
        flag = [1 for i in range(self.CONST.NUM_AGENTS)]
        for i in range(self.CONST.NUM_AGENTS):
            if obs['Agent'][i][self.env.a_obs_idx['time_to_dest']]==0:
                flag[i]=1
                prev_node = obs['Agent'][i][self.env.a_obs_idx['dest_node']]
                nodes_last_visit[i] = self.obsMap.nodes[prev_node]
                dest_nodes[i] = self.obsMap.nodes[prev_node]
            else:
                #Agent on edge
                flag[i]=0
                prev_node = self.env.graph.agent_features[str(i)]['edge'][0]    #only works for directed graphs
                nodes_last_visit[i] = self.obsMap.nodes[prev_node]

                next_node = self.env.graph.agent_features[str(i)]['edge'][1]
                dest_nodes[i] = self.obsMap.nodes[next_node]

            neighbor_nodes[i] = self.obsMap.get_neighbor_nodes_number(prev_node)

        # print("nodes last visit: ", nodes_last_visit)

        self.avg_i = 0
        self.avg_i_sum = 0
        self.avg_i_last = 0
        self.avg_idleness50 = 0
        self.max_idleness = 0
        self.sum_idleness = 0

        return np.array(state), nodes_last_visit, last_vertex, neighbor_nodes, flag, dest_nodes

    def step(self, nodes, nodes_last_visit, last_vertex, state):
        action = {}
        for a in range(self.CONST.NUM_AGENTS):
            action[a] = self.obsMap.get_key(nodes[a], self.obsMap.nodes)

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        self.interval = np.zeros(len(self.patrol_points[0]))

        for a in range(self.CONST.NUM_AGENTS):
            if obs['Agent'][a][self.env.a_obs_idx['time_to_dest']]==0:
                node_visited = int(obs['Agent'][a][self.env.a_obs_idx['dest_node']])
                self.visits[node_visited] += 1
                self.interval[node_visited] = state[node_visited][2]    #return demand values for nodes just visited, zero for all other nodes

        for i, node in enumerate(self.obsMap.patrol_nodes):
            # state[i][2] = obs.nx_graph.nodes[node]['demand'][0]
            state[i][2] = obs['node_feat'][node][self.env.n_obs_idx['demand']]*obs['node_feat'][node][self.env.n_obs_idx['priority']]
        state = np.array(state).squeeze()

        self.avg_idleness = np.sum(self.env.timer / (self.visits + 1))

        # self.avg_i = np.mean(state[:, 2]).squeeze()
        # self.max_idleness = max(self.max_idleness, np.max(state[:, 2]).squeeze())
        # self.sum_idleness += np.sum(state[:, 2]).squeeze()
        # shared_reward = -self.avg_i / 100 + 1
        # self.avg_i_last = self.avg_i
        # self.avg_i_sum += self.avg_i

        self.max_idleness = max(self.max_idleness, np.max(state[:, 2]).squeeze())
        self.sum_idleness = -self.env.episode_return
        shared_reward = -(self.sum_idleness/len(self.patrol_points[0]) - self.avg_i_sum)/100 + 1
        self.avg_i_sum = self.sum_idleness/len(self.patrol_points[0])

        flag = [1 for i in range(self.CONST.NUM_AGENTS)]
        n_last_visit = [[-1,-1] for i in range(self.CONST.NUM_AGENTS)]
        for i in range(self.CONST.NUM_AGENTS):
            if obs['Agent'][i][self.env.a_obs_idx['time_to_dest']]==0:
                flag[i]=1
                prev_node = obs['Agent'][i][self.env.a_obs_idx['dest_node']]
                n_last_visit[i] = self.obsMap.nodes[prev_node]
            else:
                #Agent on edge
                flag[i]=0
                n_last_visit[i] = nodes_last_visit[i]
        
        # print("n last visit: ", n_last_visit)

        last_vertex_list = [None for i in range(self.CONST.NUM_AGENTS)]  #not being used by CBLS policy

        return state, None, None, None, None, None, \
                shared_reward, flag, last_vertex_list, n_last_visit, None, done, info

    def render(self):
        #not implemented
        pass
