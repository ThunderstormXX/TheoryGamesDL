import numpy as np
from scipy.linalg import eig
from simple_agent import SimpleQLearningAgent

class AlphaRankAgent(SimpleQLearningAgent):
    def __init__(self, agent_id, state_size=4, action_size=2, lr=0.001):
        super().__init__(agent_id, action_size, lr)
        self.agent_id = agent_id

def compute_alpharank(payoff_matrix, alpha=0.01):
    """
    Вычисляет AlphaRank для матрицы выплат
    
    Args:
        payoff_matrix: матрица выплат размера (n_agents, n_strategies, n_strategies)
        alpha: параметр регуляризации
    
    Returns:
        ranking: стационарное распределение стратегий
    """
    n_agents, n_strategies, _ = payoff_matrix.shape
    
    # Упрощенная версия AlphaRank - используем средние выплаты
    agent_scores = np.zeros(n_agents)
    
    for i in range(n_agents):
        # Средняя выплата агента по всем стратегиям
        agent_scores[i] = payoff_matrix[i].mean()
    
    # Нормализуем в распределение вероятностей
    if agent_scores.sum() > 0:
        agent_scores = agent_scores / agent_scores.sum()
    else:
        agent_scores = np.ones(n_agents) / n_agents
    
    # Преобразуем в форму (n_agents, n_strategies)
    ranking = np.zeros((n_agents, n_strategies))
    for i in range(n_agents):
        ranking[i, 0] = agent_scores[i] * 0.6  # Больший вес кооперации
        ranking[i, 1] = agent_scores[i] * 0.4  # Меньший вес дефекта
    
    return ranking

def compute_payoff_matrix(agents, game_payoffs):
    """
    Вычисляет матрицу выплат на основе стратегий агентов
    
    Args:
        agents: список агентов
        game_payoffs: базовые выплаты игры [CC, DD, DC, CD]
    
    Returns:
        payoff_matrix: матрица выплат
    """
    n_agents = len(agents)
    n_strategies = 2  # Cooperate, Defect
    
    payoff_matrix = np.zeros((n_agents, n_strategies, n_strategies))
    
    for i, agent in enumerate(agents):
        strategy = agent.get_strategy()
        
        for my_action in range(n_strategies):
            for opp_action in range(n_strategies):
                # Индекс в game_payoffs: [CC, DD, DC, CD]
                if my_action == 0 and opp_action == 0:  # CC
                    payoff = game_payoffs[0]
                elif my_action == 1 and opp_action == 1:  # DD
                    payoff = game_payoffs[1]
                elif my_action == 1 and opp_action == 0:  # DC
                    payoff = game_payoffs[2]
                else:  # CD
                    payoff = game_payoffs[3]
                
                payoff_matrix[i, my_action, opp_action] = payoff
    
    return payoff_matrix