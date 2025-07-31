import numpy as np
import random
from collections import deque

class SimpleQLearningAgent:
    """Упрощенный Q-learning агент без нейросетей"""
    
    def __init__(self, agent_id, action_size=2, lr=0.1, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-таблица для простых состояний
        self.q_table = {}
        self.memory = deque(maxlen=2000)
        self.strategy_history = []
        
    def get_state_key(self, state):
        """Преобразует состояние в ключ для Q-таблицы"""
        return tuple(np.round(state, 2))
    
    def get_strategy(self):
        """Возвращает текущую стратегию как вероятности действий"""
        if not self.q_table:
            return np.ones(self.action_size) / self.action_size
        
        # Усредняем Q-значения по всем состояниям
        q_values = np.zeros(self.action_size)
        for state_key, q_vals in self.q_table.items():
            q_values += np.array(q_vals)
        
        if len(self.q_table) > 0:
            q_values /= len(self.q_table)
        
        # Применяем softmax для получения вероятностей
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        return probs
    
    def act(self, state, epsilon=None):
        """Выбирает действие"""
        if epsilon is None:
            epsilon = self.epsilon
            
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        
        # Epsilon-greedy стратегия
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Обучение на основе сохраненного опыта"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * self.action_size
            
            # Q-learning обновление
            target = reward
            if not done:
                target += self.gamma * max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
    
    def update_strategy_history(self):
        """Обновляет историю стратегий"""
        strategy = self.get_strategy()
        self.strategy_history.append(strategy)