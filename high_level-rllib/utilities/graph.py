'''Module for defining a custom graph and a graph gym space'''
import gymnasium as gym
import networkx as nx

class Graph:
    '''Class for custom graph for multi agent system'''
    def __init__(self, max_num_agents, graph=None):
        if graph is None:
            self.nx_graph = nx.digraph.DiGraph()
        else:
            self.nx_graph = graph
        self.max_num_agents = max_num_agents
        self.num_agents = max_num_agents
        self.agent_features = {
                str(_): {
                        'active': False,
                        'on_edge': False,
                        'edge': (),
                        'time_to_dest': 0,
                        'occ_node': -1,
                        'stn_dem': -1,
                        'curr_batt': -1,
                        'max_energy': -1,
                    } for _ in range(self.num_agents)
                }
        if graph is None:
            self.max_num_neighs = 0
        else:
            #self.max_num_neighs = int(max(
            #    (d/2 for _, d in list(self.nx_graph.degree()))
            #    ))
            self.max_num_neighs = int(max(
                (len(list(self.nx_graph.successors(_)))\
                        for _ in self.nx_graph.nodes)
                ))
            print(f"\n\n\n*****self.max_num_neighs: {self.max_num_neighs}\n\n\n")
        
    def update_agent_features(self, agent_id, **kwargs):
        for feature, value in kwargs.items():
            self.agent_features[str(agent_id)][feature] = value

    def get_active_agents(self):
        active_agent_ids = []
        for agent_id in self.agent_features.keys():
            if self.agent_features[agent_id]['active']:
                active_agent_ids.append(agent_id)
        return active_agent_ids

    def update_num_agents(self):
        self.num_agents = len(self.agent_features)

    def get_num_agents(self):
        self.update_num_agents()
        return self.num_agents

    def get_max_num_neighs(self):
        return self.max_num_neighs

class GraphSpace(gym.Space):
    '''Class for custom gym space for the graph'''
    def __init__(self, graph:Graph):
        gym.Space.__init__(self)
        self.graph = graph
