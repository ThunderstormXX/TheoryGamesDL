"""
Агенты для теоретико-игровых задач
"""

from .qlearning import calc_q, sd_qlearning
from .neural_agent import DQNAgent, A2CAgent, train_dqn_agents, train_a2c_agents, evaluate_neural_agents