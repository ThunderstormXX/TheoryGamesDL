"""
Нейросетевые агенты для теоретико-игровых задач (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


class DQNModel(nn.Module):
    """
    Нейронная сеть для DQN агента
    """
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Агент глубокого Q-обучения (Deep Q-Network)
    """
    def __init__(self, state_size, action_size, memory_size=2000, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        """
        Инициализация агента DQN
        
        Args:
            state_size (int): Размерность состояния
            action_size (int): Количество возможных действий
            memory_size (int): Размер памяти для experience replay
            gamma (float): Коэффициент дисконтирования
            epsilon (float): Начальное значение эпсилон для epsilon-greedy политики
            epsilon_min (float): Минимальное значение эпсилон
            epsilon_decay (float): Коэффициент затухания эпсилон
            learning_rate (float): Скорость обучения
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Сохранение опыта в памяти
        
        Args:
            state: Текущее состояние
            action: Выбранное действие
            reward: Полученное вознаграждение
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Выбор действия с использованием epsilon-greedy политики
        
        Args:
            state: Текущее состояние
            
        Returns:
            int: Выбранное действие
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        """
        Обучение на мини-батче из памяти (experience replay)
        
        Args:
            batch_size (int): Размер мини-батча
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            target_f = self.model(state_tensor)
            target_f_clone = target_f.clone()
            target_f_clone[0, action] = target
            
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, target_f_clone)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """
        Загрузка весов модели
        
        Args:
            name (str): Путь к файлу с весами
        """
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        """
        Сохранение весов модели
        
        Args:
            name (str): Путь для сохранения весов
        """
        torch.save(self.model.state_dict(), name)


class ActorModel(nn.Module):
    """
    Модель актора для A2C агента
    """
    def __init__(self, state_size, action_size):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class CriticModel(nn.Module):
    """
    Модель критика для A2C агента
    """
    def __init__(self, state_size):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class A2CAgent:
    """
    Агент Advantage Actor-Critic (A2C)
    """
    def __init__(self, state_size, action_size, gamma=0.95, actor_lr=0.001, critic_lr=0.001):
        """
        Инициализация агента A2C
        
        Args:
            state_size (int): Размерность состояния
            action_size (int): Количество возможных действий
            gamma (float): Коэффициент дисконтирования
            actor_lr (float): Скорость обучения для актора
            critic_lr (float): Скорость обучения для критика
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Создание моделей актора и критика
        self.actor = ActorModel(state_size, action_size).to(self.device)
        self.critic = CriticModel(state_size).to(self.device)
        
        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def act(self, state):
        """
        Выбор действия на основе вероятностей от актора
        
        Args:
            state: Текущее состояние
            
        Returns:
            int: Выбранное действие
            float: Вероятность выбранного действия
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy = self.actor(state_tensor).cpu().numpy()[0]
        action = np.random.choice(self.action_size, p=policy)
        return action, policy[action]
    
    def train(self, state, action, reward, next_state, done):
        """
        Обучение моделей актора и критика
        
        Args:
            state: Текущее состояние
            action: Выбранное действие
            reward: Полученное вознаграждение
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        # Обучение критика
        target = reward
        if not done:
            with torch.no_grad():
                target = reward + self.gamma * self.critic(next_state_tensor).item()
        
        value = self.critic(state_tensor)
        
        target_tensor = torch.full_like(value, float(target))  # тот же dtype и device, что у value
        critic_loss = F.mse_loss(value, target_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Вычисление advantage
        advantage = target - value.item()
        
        # Обучение актора
        policy = self.actor(state_tensor)
        # Преобразуем action в тензор типа long
        action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
        # Получаем вероятность выбранного действия
        selected_action_prob = policy.gather(1, action_tensor)
        # Вычисляем логарифм вероятности
        log_prob = torch.log(selected_action_prob)
        # Вычисляем функцию потерь
        actor_loss = -log_prob * torch.FloatTensor([[advantage]]).to(self.device)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def load(self, actor_name, critic_name):
        """
        Загрузка весов моделей
        
        Args:
            actor_name (str): Путь к файлу с весами актора
            critic_name (str): Путь к файлу с весами критика
        """
        self.actor.load_state_dict(torch.load(actor_name))
        self.critic.load_state_dict(torch.load(critic_name))
    
    def save(self, actor_name, critic_name):
        """
        Сохранение весов моделей
        
        Args:
            actor_name (str): Путь для сохранения весов актора
            critic_name (str): Путь для сохранения весов критика
        """
        torch.save(self.actor.state_dict(), actor_name)
        torch.save(self.critic.state_dict(), critic_name)


def train_dqn_agents(game, episodes=1000, batch_size=32, state_size=2, action_size=2):
    """
    Обучение двух DQN агентов для игры
    
    Args:
        game: Объект игры
        episodes (int): Количество эпизодов обучения
        batch_size (int): Размер мини-батча для обучения
        state_size (int): Размерность состояния
        action_size (int): Количество возможных действий
        
    Returns:
        tuple: (агент1, агент2, история действий, история вознаграждений)
    """
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    
    history = []
    rewards_history = []
    
    for e in range(episodes):
        game.reset()
        state1 = np.zeros((1, state_size))
        state2 = np.zeros((1, state_size))
        
        # Добавляем информацию о текущем состоянии игры
        state1[0][0] = game.curr_state
        state2[0][0] = game.curr_state
        
        # Добавляем информацию о предыдущих действиях (изначально 0)
        state1[0][1] = 0
        state2[0][1] = 0
        
        total_reward1 = 0
        total_reward2 = 0
        
        done = False
        
        while not done:
            # Агент 1 выбирает действие
            action1 = agent1.act(state1)
            
            # Агент 2 выбирает действие
            action2 = agent2.act(state2)
            
            # Выполняем действия в игре
            game.action()
            
            # Получаем вознаграждения
            reward1, err1 = game.reward_func(action1, action2, 0)
            reward2, err2 = game.reward_func(action2, action1, 1)
            
            # Формируем новые состояния
            next_state1 = np.zeros((1, state_size))
            next_state2 = np.zeros((1, state_size))
            
            next_state1[0][0] = game.curr_state
            next_state2[0][0] = game.curr_state
            
            next_state1[0][1] = action2  # Действие оппонента
            next_state2[0][1] = action1  # Действие оппонента
            
            # Проверяем, завершен ли эпизод
            done = game.curr_state >= game.steps_number - 1
            
            # Сохраняем опыт в памяти
            agent1.remember(state1, action1, reward1, next_state1, done)
            agent2.remember(state2, action2, reward2, next_state2, done)
            
            # Обновляем состояния
            state1 = next_state1
            state2 = next_state2
            
            # Накапливаем вознаграждения
            total_reward1 += reward1
            total_reward2 += reward2
            
            # Сохраняем историю действий и вознаграждений
            history.append([action1, action2])
            rewards_history.append([reward1, reward2])
            
            # Обучаем агентов
            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size)
            
            if len(agent2.memory) > batch_size:
                agent2.replay(batch_size)
        
        # Уменьшаем epsilon для обоих агентов
        if agent1.epsilon > agent1.epsilon_min:
            agent1.epsilon *= agent1.epsilon_decay
        
        if agent2.epsilon > agent2.epsilon_min:
            agent2.epsilon *= agent2.epsilon_decay
    
    return agent1, agent2, history, rewards_history


def train_a2c_agents(game, episodes=1000, state_size=2, action_size=2):
    """
    Обучение двух A2C агентов для игры
    
    Args:
        game: Объект игры
        episodes (int): Количество эпизодов обучения
        state_size (int): Размерность состояния
        action_size (int): Количество возможных действий
        
    Returns:
        tuple: (агент1, агент2, история действий, история вознаграждений)
    """
    agent1 = A2CAgent(state_size, action_size)
    agent2 = A2CAgent(state_size, action_size)
    
    history = []
    rewards_history = []
    
    for e in range(episodes):
        game.reset()
        state1 = np.zeros((1, state_size))
        state2 = np.zeros((1, state_size))
        
        # Добавляем информацию о текущем состоянии игры
        state1[0][0] = game.curr_state
        state2[0][0] = game.curr_state
        
        # Добавляем информацию о предыдущих действиях (изначально 0)
        state1[0][1] = 0
        state2[0][1] = 0
        
        total_reward1 = 0
        total_reward2 = 0
        
        done = False
        
        while not done:
            # Агент 1 выбирает действие
            action1, _ = agent1.act(state1)
            
            # Агент 2 выбирает действие
            action2, _ = agent2.act(state2)
            
            # Выполняем действия в игре
            game.action()
            
            # Получаем вознаграждения
            reward1, err1 = game.reward_func(action1, action2, 0)
            reward2, err2 = game.reward_func(action2, action1, 1)
            
            # Формируем новые состояния
            next_state1 = np.zeros((1, state_size))
            next_state2 = np.zeros((1, state_size))
            
            next_state1[0][0] = game.curr_state
            next_state2[0][0] = game.curr_state
            
            next_state1[0][1] = action2  # Действие оппонента
            next_state2[0][1] = action1  # Действие оппонента
            
            # Проверяем, завершен ли эпизод
            done = game.curr_state >= game.steps_number - 1
            
            # Обучаем агентов
            agent1.train(state1, action1, reward1, next_state1, done)
            agent2.train(state2, action2, reward2, next_state2, done)
            
            # Обновляем состояния
            state1 = next_state1
            state2 = next_state2
            
            # Накапливаем вознаграждения
            total_reward1 += reward1
            total_reward2 += reward2
            
            # Сохраняем историю действий и вознаграждений
            history.append([action1, action2])
            rewards_history.append([reward1, reward2])
    
    return agent1, agent2, history, rewards_history


def evaluate_neural_agents(game, agent1, agent2, episodes=100, state_size=2):
    """
    Оценка обученных нейросетевых агентов
    
    Args:
        game: Объект игры
        agent1: Первый агент
        agent2: Второй агент
        episodes (int): Количество эпизодов для оценки
        state_size (int): Размерность состояния
        
    Returns:
        tuple: (история действий, история вознаграждений, вероятности сотрудничества)
    """
    history = []
    rewards_history = []
    cooperation_probs1 = []
    cooperation_probs2 = []
    
    for e in range(episodes):
        game.reset()
        state1 = np.zeros((1, state_size))
        state2 = np.zeros((1, state_size))
        
        # Добавляем информацию о текущем состоянии игры
        state1[0][0] = game.curr_state
        state2[0][0] = game.curr_state
        
        # Добавляем информацию о предыдущих действиях (изначально 0)
        state1[0][1] = 0
        state2[0][1] = 0
        
        done = False
        
        while not done:
            # Для DQN агента
            if isinstance(agent1, DQNAgent):
                state_tensor1 = torch.FloatTensor(state1).to(agent1.device)
                with torch.no_grad():
                    act_values1 = agent1.model(state_tensor1)
                action1 = torch.argmax(act_values1).item()
                # Вероятность сотрудничества (действие 1)
                act_values1_np = act_values1.cpu().numpy()[0]
                cooperation_prob1 = act_values1_np[1] / np.sum(act_values1_np)
            # Для A2C агента
            else:
                state_tensor1 = torch.FloatTensor(state1).to(agent1.device)
                with torch.no_grad():
                    policy1 = agent1.actor(state_tensor1).cpu().numpy()[0]
                action1 = np.argmax(policy1)
                cooperation_prob1 = policy1[1]
            
            # Для DQN агента
            if isinstance(agent2, DQNAgent):
                state_tensor2 = torch.FloatTensor(state2).to(agent2.device)
                with torch.no_grad():
                    act_values2 = agent2.model(state_tensor2)
                action2 = torch.argmax(act_values2).item()
                # Вероятность сотрудничества (действие 1)
                act_values2_np = act_values2.cpu().numpy()[0]
                cooperation_prob2 = act_values2_np[1] / np.sum(act_values2_np)
            # Для A2C агента
            else:
                state_tensor2 = torch.FloatTensor(state2).to(agent2.device)
                with torch.no_grad():
                    policy2 = agent2.actor(state_tensor2).cpu().numpy()[0]
                action2 = np.argmax(policy2)
                cooperation_prob2 = policy2[1]
            
            # Выполняем действия в игре
            game.action()
            
            # Получаем вознаграждения
            reward1, _ = game.reward_func(action1, action2, 0)
            reward2, _ = game.reward_func(action2, action1, 1)
            
            # Формируем новые состояния
            next_state1 = np.zeros((1, state_size))
            next_state2 = np.zeros((1, state_size))
            
            next_state1[0][0] = game.curr_state
            next_state2[0][0] = game.curr_state
            
            next_state1[0][1] = action2  # Действие оппонента
            next_state2[0][1] = action1  # Действие оппонента
            
            # Обновляем состояния
            state1 = next_state1
            state2 = next_state2
            
            # Сохраняем историю действий и вознаграждений
            history.append([action1, action2])
            rewards_history.append([reward1, reward2])
            cooperation_probs1.append(cooperation_prob1)
            cooperation_probs2.append(cooperation_prob2)
            
            # Проверяем, завершен ли эпизод
            done = game.curr_state >= game.steps_number - 1
    
    return history, rewards_history, cooperation_probs1, cooperation_probs2