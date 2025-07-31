import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random

# Отключаем предупреждения о градиентах
torch.autograd.set_detect_anomaly(False)

class AuctionNetwork(nn.Module):
    """Нейросеть для агента аукциона"""
    
    def __init__(self, input_size=1, hidden_size=64, n_actions=11):
        super(AuctionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class AuctionAgent:
    """Агент для двустороннего аукциона"""
    
    def __init__(self, agent_id, agent_type, value_or_cost, n_actions=11, lr=0.001):
        self.agent_id = agent_id
        self.agent_type = agent_type  # 'buyer' или 'seller'
        self.value_or_cost = value_or_cost  # value для покупателя, cost для продавца
        self.n_actions = n_actions  # 0, 1, 2, ..., N
        
        # Нейросеть
        self.network = AuctionNetwork(input_size=1, n_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # История для обучения
        self.memory = deque(maxlen=10000)
        self.strategy_history = []
        
    def get_input(self):
        """Получает вход для нейросети (константа)"""
        return torch.FloatTensor([1.0]).unsqueeze(0)
    
    def get_strategy(self):
        """Возвращает текущую стратегию как вероятности действий"""
        with torch.no_grad():
            input_tensor = self.get_input()
            probs = self.network(input_tensor)
            return probs.cpu().numpy().flatten()
    
    def sample_action(self):
        """Сэмплирует действие и возвращает его с log_prob"""
        input_tensor = self.get_input()
        probs = self.network(input_tensor)
        
        # Создаем категориальное распределение
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def act(self, deterministic=False):
        """Выбирает действие (ставку)"""
        if deterministic:
            # Детерминистический выбор (максимальная вероятность)
            strategy = self.get_strategy()
            return np.argmax(strategy)
        else:
            # Стохастический выбор
            action, _ = self.sample_action()
            return action
    
    def remember(self, state, action, reward, next_state, done, log_prob=None):
        """Сохраняет опыт (только для статистики)"""
        self.memory.append((state, action, reward, next_state, done, None))  # Не сохраняем log_prob
    
    def learn_from_action(self, log_prob, reward):
        """Немедленное обучение от одного действия"""
        if log_prob is not None and reward != 0:
            # REINFORCE: loss = -log_prob * reward
            loss = -log_prob * reward
            
            # Обратное распространение
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def replay(self, batch_size=32):
        """Пустой метод для совместимости"""
        pass
    
    def update_strategy_history(self):
        """Обновляет историю стратегий"""
        strategy = self.get_strategy()
        self.strategy_history.append(strategy)
    
    def __str__(self):
        return f"{self.agent_type.capitalize()} {self.agent_id} ({self.agent_type[0]}{self.value_or_cost})"

def compute_auction_payoffs(buyer_bid, seller_ask, buyer_value, seller_cost):
    """
    Вычисляет выплаты в двустороннем аукционе
    
    Args:
        buyer_bid: ставка покупателя
        seller_ask: ставка продавца  
        buyer_value: ценность для покупателя
        seller_cost: стоимость для продавца
    
    Returns:
        buyer_reward, seller_reward
    """
    if buyer_bid >= seller_ask:
        # Сделка состоялась
        price = (buyer_bid + seller_ask) / 2.0
        buyer_reward = buyer_value - price
        seller_reward = price - seller_cost
    else:
        # Сделка не состоялась
        buyer_reward = 0.0
        seller_reward = 0.0
    
    return buyer_reward, seller_reward

def create_auction_agents(n_buyers=3, n_sellers=3, n_actions=11):
    """Создает агентов для аукциона"""
    agents = []
    
    # Создаем покупателей с разными ценностями
    buyer_values = np.linspace(5, 10, n_buyers)
    for i, value in enumerate(buyer_values):
        agent = AuctionAgent(i, 'buyer', value, n_actions)
        agents.append(agent)
    
    # Создаем продавцов с разными стоимостями
    seller_costs = np.linspace(1, 6, n_sellers)
    for i, cost in enumerate(seller_costs):
        agent = AuctionAgent(i + n_buyers, 'seller', cost, n_actions)
        agents.append(agent)
    
    return agents