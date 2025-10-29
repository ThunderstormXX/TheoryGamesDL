import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from typing import List, Tuple

class TensorGame:
    """N-player game with tensor payoffs"""
    
    def __init__(self, payoff_tensor: np.ndarray):
        self.payoff_tensor = payoff_tensor
        self.n_players = len(payoff_tensor.shape) - 1
        self.n_actions = payoff_tensor.shape[:-1]
        
    def get_payoffs(self, actions: List[int]) -> List[float]:
        """Get payoffs for pure action profile"""
        return [float(self.payoff_tensor[tuple(actions)][i]) for i in range(self.n_players)]


class ProbabilisticTensorGame:
    """N-player game where payoffs are sampled from stored distributions"""
    
    def __init__(self, mean_tensor: np.ndarray, std_tensor: np.ndarray):
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor
        self.n_players = len(mean_tensor.shape) - 1
        self.n_actions = mean_tensor.shape[:-1]
        
    def get_payoffs(self, actions: List[int]) -> List[float]:
        idx = tuple(actions)
        return [
            float(np.random.normal(loc=self.mean_tensor[idx][i], scale=self.std_tensor[idx][i]))
            for i in range(self.n_players)
        ]


def create_probabilistic_game_with_lower_offdiag(
    n_players: int,
    n_actions: int,
    base_values: list,
    diag_variance=4.0,
    off_diag_variance=1.0,
    off_diag_base_upper_bound=None  # верхняя граница для uniform вне диагонали
) -> ProbabilisticTensorGame:
    shape = [n_actions] * n_players + [n_players]
    mean_tensor = np.zeros(shape)
    std_tensor = np.zeros(shape)

    # Определим верхнюю границу для вне диагонали, если не передана
    if off_diag_base_upper_bound is None:
        off_diag_base_upper_bound = min(base_values) * 1  # например 80% от минимального диагонального выигрыша

    # Заполняем диагональ
    for i in range(n_actions):
        diag_idx = tuple([i] * n_players)
        for player in range(n_players):
            mean_tensor[diag_idx][player] = base_values[i] if i < len(base_values) else 5
            std_tensor[diag_idx][player] = diag_variance

    # Заполняем вне диагонали случайным средним из uniform [0, off_diag_base_upper_bound]
    for idx in np.ndindex(*shape[:-1]):
        if len(set(idx)) > 1:
            for player in range(n_players):
                base_val = np.random.uniform(0, off_diag_base_upper_bound)
                mean_tensor[idx][player] = base_val
                std_tensor[idx][player] = off_diag_variance

    return ProbabilisticTensorGame(mean_tensor, std_tensor)


def collect_statistics(game: ProbabilisticTensorGame, n_samples: int = 1000):
    records = []
    n_players = game.n_players
    n_actions = game.n_actions[0]

    for state in np.ndindex((n_actions,) * n_players):
        for _ in range(n_samples):
            payoffs = game.get_payoffs(list(state))
            for agent_id, payoff in enumerate(payoffs):
                records.append({
                    'state': state,
                    'agent': f'Agent {agent_id+1}',
                    'payoff': payoff
                })

    df = pd.DataFrame(records)
    stats = df.groupby(['state', 'agent'])['payoff'].agg(['mean', 'std']).reset_index()
    return stats



def plot_stats_barplot(stats_df):
    plt.figure(figsize=(18, 7))

    # Средние выигрыши
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=stats_df,
        x=stats_df['state'].astype(str),
        y='mean',
        hue='agent',
        errorbar=None
    )
    plt.title("Средний выигрыш агентов по состояниям")
    plt.xlabel("Состояния (действия)")
    plt.ylabel("Средний выигрыш")
    plt.xticks(rotation=90)

    # Стандартное отклонение
    plt.subplot(1, 2, 2)
    sns.barplot(
        data=stats_df,
        x=stats_df['state'].astype(str),
        y='std',
        hue='agent',
        errorbar=None
    )
    plt.title("Стандартное отклонение выигрыша агентов")
    plt.xlabel("Состояния (действия)")
    plt.ylabel("Стандартное отклонение")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()



class MultiAgent:
    """Base class for multi-agent RL"""
    
    def __init__(self, n_actions: int, lr: float = 0.01, epsilon: float = 0.1, optimizer_type: str = 'Adam'):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def get_action_probs(self) -> np.ndarray:
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1)
            logits = self.model(dummy_input)
            probs = F.softmax(logits, dim=-1)
            return probs.squeeze().numpy()
    
    def sample_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            probs = self.get_action_probs()
            return np.random.choice(self.n_actions, p=probs)
    
    def update(self, action: int, reward: float):
        dummy_input = torch.zeros(1, 1)
        logits = self.model(dummy_input)
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = -log_probs[0, action] * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class AdversarialAgent(MultiAgent):
    """Agent that attacks other agents' loss functions"""
    
    def __init__(self, n_actions: int, lr: float = 0.01, epsilon: float = 0.0, optimizer_type: str = 'Adam'):
        super().__init__(n_actions, lr, epsilon, optimizer_type)
        self.victim_models = []  # Store references to other agents' models
        self.attack_strength = 0.1
    
    def set_victims(self, victim_agents: List['MultiAgent']):
        """Set victim agents to attack"""
        self.victim_models = [agent.model for agent in victim_agents]
    
    def adversarial_update(self, action: int, reward: float, victim_actions: List[int], victim_rewards: List[float]):
        """Update with adversarial loss to manipulate victims"""
        dummy_input = torch.zeros(1, 1)
        
        # Own reward maximization
        logits = self.model(dummy_input)
        log_probs = F.log_softmax(logits, dim=-1)
        own_loss = -log_probs[0, action] * reward
        
        # Adversarial component: minimize victims' rewards
        adversarial_loss = 0
        for i, victim_model in enumerate(self.victim_models):
            victim_logits = victim_model(dummy_input)
            victim_log_probs = F.log_softmax(victim_logits, dim=-1)
            # Encourage victims to take suboptimal actions
            adversarial_loss += victim_log_probs[0, victim_actions[i]] * victim_rewards[i] * self.attack_strength
        
        total_loss = own_loss + adversarial_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

def create_diagonal_tensor_game(n_players: int, n_actions: int) -> TensorGame:
    """Create tensor game with equilibria on diagonal"""
    shape = [n_actions] * n_players + [n_players]
    payoff_tensor = np.zeros(shape)
    
    # Fill diagonal with increasing payoffs
    for i in range(n_actions):
        diagonal_idx = tuple([i] * n_players)
        for player in range(n_players):
            payoff_tensor[diagonal_idx][player] = (i + 1) * 5
    
    # Add some noise to off-diagonal elements
    np.random.seed(42)
    for idx in np.ndindex(tuple(shape[:-1])):
        if len(set(idx)) > 1:  # Off-diagonal
            for player in range(n_players):
                payoff_tensor[idx][player] = np.random.uniform(0, 3)
    
    return TensorGame(payoff_tensor)

def train_multi_agents(game: TensorGame, agents: List[MultiAgent], n_episodes: int = 5000) -> List[List[float]]:
    """Train multiple agents on tensor game"""
    payoff_history = []
    
    # Set up adversarial agent if present
    adversarial_agent = None
    victim_agents = []
    for i, agent in enumerate(agents):
        if isinstance(agent, AdversarialAgent):
            adversarial_agent = agent
            victim_agents = [a for j, a in enumerate(agents) if j != i]
            agent.set_victims(victim_agents)
            break
    
    for episode in range(n_episodes):
        # Sample actions from all agents
        actions = [agent.sample_action() for agent in agents]
        
        # Get rewards
        rewards = game.get_payoffs(actions)
        
        # Update agents
        for i, (agent, action, reward) in enumerate(zip(agents, actions, rewards)):
            if isinstance(agent, AdversarialAgent):
                # Adversarial update
                victim_actions = [actions[j] for j in range(len(actions)) if j != i]
                victim_rewards = [rewards[j] for j in range(len(rewards)) if j != i]
                agent.adversarial_update(action, reward, victim_actions, victim_rewards)
            else:
                # Normal update
                agent.update(action, reward)
        
        payoff_history.append(rewards)
        
        if episode % 1000 == 0:
            avg_rewards = np.mean(payoff_history[-100:], axis=0)
            print(f"Episode {episode}: Avg rewards {[f'{r:.3f}' for r in avg_rewards]}")
    
    return payoff_history

def plot_training_history(payoff_history: List[List[float]], window_size: int = 100):
    df = pd.DataFrame(payoff_history, columns=[f'Agent {i+1}' for i in range(len(payoff_history[0]))])
    df_smooth = df.rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    for agent in df.columns:
        plt.plot(df_smooth[agent], label=agent)
    
    plt.title(f"Среднее вознаграждение агентов (скользящее окно {window_size})")
    plt.xlabel("Эпизод")
    plt.ylabel("Сглаженное вознаграждение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
