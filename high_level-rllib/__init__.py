'''__init__ to register the gym env'''
from env.multi_agent_env import MultiAgentEnv
from gymnasium.envs.registration import register

register(
     id='MultiAgentEnv-v1',
     entry_point='env.multi_agent_env:MultiAgentEnv',
)
