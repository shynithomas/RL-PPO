'''Data class to store and manage all data'''
import os
import datetime as dt
import pytz
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings
import sys


class Data:
    '''Data class to store and manage all data'''

    def __init__(self, args, make_dirs=True):
        self.paths = {}
        self.sim_data = {}
        (self.eval_graphs_path_init, self.graph_num, self.learner_num, self.algo,
         self.saved_results_path, self.checkpoint_num, self.test_name) = args

        if make_dirs:
            self.make_dirs()
        self.init_sim_data()
        self.rewards, self.mean_rewards, self.opt_gaps = [], [], []
        self.loss = {'actor_loss': [], 'critic_loss': []}
        self.start_time = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
        self.cuda_status = False

    def init_sim_data(self):
        # Algorithm hyperparameters
        if self.algo == 'ppo':
            self.sim_data = {
                'algo': self.algo,
                'actor_lr': 3e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'num_minibatches': 4,
                'num_nodes': 25,
                'steps_per_episode': 15,
                'grad_clip_value': 100,
                'eval_freq': 100,
                'updates_per_iteration': 10,
                'clip': 0.2,
                'update_every': 128,
                'saved_results_path': self.saved_results_path,
                'CUDA': torch.cuda.is_available(),
                'normalize': False,
                'use_gae': True,
                'gae_lambda': 0.95,
                'eps_start': 1e-3,
                'eps_end': 1e-5,
                'eps_decay': 100,
                'vf_coeff': 1.0,
                'station_flag': True,
                'energy_flag': True,
                'objective_type': 'sum',  # max, sum
                'station_dyn_flag': True,
                'num_closest_agents_in_node_feat': 4,
                'num_closest_stn_in_node_feat': 3,
                'node_enc_hidden_size': 128, 
                'agent_enc_hidden_size': 128, # ignored by rnn_comb
                'beam_search': False,
                'beam_width': 10,
                'max_num_nodes': 25,
                'max_num_neigh': 5,
                'max_num_agents': 4,
                'max_num_edges': 105,
            }
        else:
            raise ValueError(f'Invalid Algorithm:{self.algo}')

    def make_dirs(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        time = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
        current_time = f'{time.year}_{time.month}_{time.day}_{time.hour}' +\
            f'_{time.minute}_{time.second}_{time.microsecond}'
        results_path = os.path.join(file_path, f'results_{current_time}')

        if self.eval_graphs_path_init is not None:
            self.eval_graphs_path = os.path.join(f'{self.eval_graphs_path_init}',
                                                 f'graph_{self.graph_num}')

        if self.graph_num is not None and self.learner_num is not None\
                and self.eval_graphs_path is not None:
            results_path = os.path.join(os.path.dirname(self.eval_graphs_path),
                                        'eval', f'graph_{self.graph_num}',
                                        f'learner_{self.learner_num}',
                                        f'results_{current_time}')

        eval_models_path = os.path.join(results_path, 'eval_models')
        
        self.paths['config_path'] = os.path.join(results_path, 'config.txt')
        self.paths['graph_path'] = os.path.join(results_path, 'graph.png')
        self.paths['demand_graph_path'] = os.path.join(results_path,
                                                       'demand.png')
        self.paths['eval_graphs_path'] = self.eval_graphs_path_init
        self.paths['loss_graph_path'] = os.path.join(results_path, 'loss.png')
        self.paths['best_model_path'] = os.path.join(
            results_path, 'best_model')
        self.paths['opt_gap_path'] = os.path.join(results_path, 'opt_gap.png')
        self.paths['eval_models'] = eval_models_path
        self.paths['opt_data_csv'] = os.path.join(results_path, 'opt_data.csv')

        # for convenient saving of results when running multiple tests (instances of main())
        if self.test_name is not None:
            main_path = os.path.abspath(str(sys.modules['__main__'].__file__))
            main_path = os.path.split(main_path)[0]
            test_path = os.path.abspath(os.path.join(main_path, '..', self.test_name))
            self.paths['test_name'] = test_path
        else:
            self.paths['test_name'] = None

    def plot_data(self, **kwargs):
        eval_freq = self.sim_data['eval_freq']
        ########### Demands ###########
        plt.figure()
        plt.clf()
        plt.title('Training demands')
        plt.xlabel(f'Episode/{eval_freq}')
        plt.ylabel('Demand')
        plt.plot(self.rewards, label='Demand')
        # Take 100 episode averages and plot them too
        if len(self.rewards) >= 10:
            self.mean_rewards = self.running_mean(self.rewards, 10)
        else:
            self.mean_rewards.append(0)
        plt.plot(self.mean_rewards, label='10 episode avg of demand')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.paths['demand_graph_path'], dpi=300)

        ########### Losses ###########
        plt.figure()
        plt.clf()
        if 'logger' in kwargs:
            logger: dict = kwargs['logger']
            num_losses = len(list(logger.keys()))
            i = 1
            for loss_name, loss_list in logger.items():
                if loss_name == 'entropy':
                    continue
                plt.subplot(num_losses, 1, i)
                plt.title(loss_name)
                plt.xlabel('')
                plt.ylabel('Loss')
                plt.plot(loss_list)
                i += 1
        else:
            plt.subplot(2, 1, 1)
            plt.title('Actor loss')
            plt.xlabel('')
            plt.ylabel('Loss')
            plt.plot(self.loss['actor_loss'])

            plt.subplot(2, 1, 2)
            plt.title('Critic Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.plot(self.loss['critic_loss'])
        plt.tight_layout()

        plt.savefig(self.paths['loss_graph_path'], dpi=400)

        if 'logger' in kwargs and 'entropy' in kwargs['logger']:
            logger: dict = kwargs['logger']
            plt.figure()
            plt.clf()
            plt.plot(logger['entropy'])
            plt.title('Entropy')
            plt.xlabel('')
            plt.ylabel('Entropy')

            plt.tight_layout()
            plt.savefig(self.paths['loss_graph_path']+str('_entropy.png'),
                        dpi=300)

        ########### Optimality Gaps ###########
        plt.figure()
        plt.clf()
        plt.title('Percentage Optimal Gap')
        plt.xlabel(f'Episode/{eval_freq}')
        plt.ylabel('% optimal gap')
        plt.plot(self.opt_gaps)
        plt.tight_layout()

        plt.savefig(self.paths['opt_gap_path'], dpi=300)

        plt.close('all')

    def running_mean(self, x, n):
        return np.convolve(x, np.ones(n)/n, mode='valid')
