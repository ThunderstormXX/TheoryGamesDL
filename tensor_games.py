import numpy as np
import torch
import torch.nn as nn
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