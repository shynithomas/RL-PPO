#pylint: disable-all

'''Module for the policy network'''

import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from utilities.nn_utils import BatchedAFTFullConv, AFTFullMultiHead, MultiheadCrossAttention

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.annotations import override


class MultiAgentModel(TorchModelV2, nn.Module):
    '''PyTorch implementation of Policy network for PPO'''

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.max_num_nodes = obs_space['node_feat_state'].shape[0]
        self.max_num_agents = obs_space['agent_feat'].shape[0]
        self.max_num_neighs = obs_space['valid_act_mask'].shape[-1]
        self.enc_hidden_size = model_config['custom_model_config']['enc_hidden_size']
        self.node_enc_input_size = obs_space['node_feat_state'].shape[-1]

        self.gnn_1 = BatchedAFTFullConv(in_channels=self.node_enc_input_size,
                                      hidden_channels=self.enc_hidden_size,
                                      out_channels=self.node_enc_input_size)

        # self.gnn_2 = BatchedAFTFullConv(in_channels=self.node_enc_input_size,
        #                               hidden_channels=self.enc_hidden_size,
        #                               out_channels=self.node_enc_input_size)

        self.attention_view = AFTFullMultiHead(max_seqlen=\
                                        self.max_num_nodes+self.max_num_agents,
                                        q_dim=2*self.node_enc_input_size,
                                        k_dim=self.node_enc_input_size,
                                        hidden_dim=self.enc_hidden_size)

        self.critic_encoder = AFTFullMultiHead(max_seqlen=\
                                        self.max_num_nodes+self.max_num_agents,
                                        q_dim=None,
                                        k_dim=self.node_enc_input_size,
                                        hidden_dim=self.enc_hidden_size)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        if len(input_dict['obs']['node_feat_state'].shape) > 3:
            for key, obs in input_dict['obs'].items():
                input_dict['obs'][key] = obs.squeeze(0)
                
        rnn_input = input_dict['obs']
        batch_size = rnn_input['valid_neigh'].shape[0] 
        valid_neigh = rnn_input['valid_neigh'].long()
        valid_act_mask = rnn_input['valid_act_mask'].bool()
        if valid_act_mask.sum() == 0:
            valid_act_mask[:, : , 0] = torch.ones_like(valid_act_mask[:, : , 0]).bool()

        valid_neigh_mask = (valid_neigh >= 0).bool()
        num_valid_actions = rnn_input['num_valid_neigh'].long()
        agent_rnn_input = rnn_input['agent_feat'].float()
        node_mask = rnn_input['critic_mask'][:, :self.max_num_nodes,
                                             :self.max_num_nodes].bool()
        if node_mask.sum() == 0:
            node_mask[:, : , 0] = torch.ones_like(node_mask[:, : , 0]).bool()
        mask = rnn_input['critic_mask'][:, 0, :].unsqueeze(1).bool()
        if mask.sum() == 0:
            mask[:, : , 0] = torch.ones_like(mask[:, : , 0]).bool()

        assert (batch_size == agent_rnn_input.shape[0])
        if batch_size == 0:
            return torch.zeros(batch_size,
                               self.max_num_neighs*self.max_num_agents)

        nodes_feat = rnn_input['node_feat_state'].float()
        edge_indices = rnn_input['edge_indices'].long()
        edge_weight = rnn_input['edge_weight'].long()
        edge_mask = rnn_input['edge_mask'].bool()
        if edge_mask.sum() == 0:
            edge_mask[: , 0] = torch.ones_like(edge_mask[:, 0]).bool()
        # nodes_feat_critic = self.gnn_2(x=nodes_feat, edge_index=edge_indices,
        #                       edge_weight=edge_weight, edge_mask=edge_mask,
        #                       node_mask=node_mask[:, 0, :])

        nodes_feat = self.gnn_1(x=nodes_feat, edge_index=edge_indices,
                              edge_weight=edge_weight, edge_mask=edge_mask,
                              node_mask=node_mask[:, 0, :])
        nodes_feat_critic = nodes_feat
        nodes_feat = nodes_feat.reshape(batch_size, self.max_num_nodes, -1)
        node_out_seq = nodes_feat.reshape(batch_size, -1, nodes_feat.shape[-1])
        agent_out_seq = agent_rnn_input
        out_seq = torch.concat((node_out_seq, agent_out_seq), dim=1)

        nodes_feat_critic = nodes_feat_critic.reshape(batch_size, self.max_num_nodes, -1)
        node_out_seq_critic = nodes_feat_critic.reshape(batch_size, -1, nodes_feat_critic.shape[-1])
        out_seq_critic = torch.concat((node_out_seq_critic, agent_out_seq), dim=1)
        
        indices_expanded = valid_neigh.unsqueeze(-1).expand(-1, -1, -1, node_out_seq.shape[-1])
        indices_expanded = indices_expanded.masked_fill(indices_expanded==-1, 0)
        # Gather
        next_possible_nodes = torch.gather(node_out_seq.unsqueeze(1).repeat_interleave(self.max_num_agents, dim=1), dim=2, index=indices_expanded).reshape(batch_size, -1, node_out_seq.shape[-1])
        agent_out_seq_rep = agent_out_seq.repeat_interleave(self.max_num_neighs,
                                                            dim=1)
        act_embed = torch.cat((agent_out_seq_rep, next_possible_nodes),
                                  dim=-1)
        scores = self.attention_view(q=act_embed, k=out_seq,
                                     mask=mask).reshape(batch_size,
                                                        self.max_num_agents,
                                                        self.max_num_neighs)
        # print(f"[RLLIB_MODEL.PY][FORWARD][BEFORE MASK] scores: {scores}")

        scores[~valid_act_mask] += -torch.inf * torch.ones_like(scores)[\
                                                                ~valid_act_mask]

        # print(f"[RLLIB_MODEL.PY][FORWARD][AFTER MASK] scores: {scores}")


        self._value_out = self.critic_encoder(k=out_seq_critic, mask=mask).reshape(-1)
        # print(f"[RLLIB_MODEL.PY][FORWARD] self._value_out: {self._value_out}")
        return scores.reshape(batch_size, -1), [] #state

    @override(ModelV2)
    def value_function(self):
        # print(f"[RLLIB_MODEL.PY][VALUE_FUNCTION] self._value_out: {self._value_out}")

        return self._value_out#.reshape(-1)
