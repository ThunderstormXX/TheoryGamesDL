import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from game_theory import TwoPlayerGame, MatrixGame

class MLP(nn.Module):
    """Multi-layer perceptron for learning game strategies"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GamePlayer:
    """Neural network player for game theory experiments"""
    
    def __init__(self, n_actions: int, hidden_dims: List[int] = [64, 32]):
        self.n_actions = n_actions
        self.model = MLP(n_actions, hidden_dims, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
    def get_strategy(self, opponent_strategy: np.ndarray) -> np.ndarray:
        """Get strategy given opponent's strategy"""
        with torch.no_grad():
            x = torch.FloatTensor(opponent_strategy).unsqueeze(0)
            strategy = self.model(x).squeeze().numpy()
        return strategy
    
    def update(self, opponent_strategy: np.ndarray, reward: float):
        """Update model based on received reward"""
        x = torch.FloatTensor(opponent_strategy).unsqueeze(0)
        strategy = self.model(x)
        
        # Simple policy gradient update
        loss = -torch.log(strategy.max()) * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_players(game: TwoPlayerGame, n_episodes: int = 1000) -> Tuple[GamePlayer, GamePlayer]:
    """Train two neural network players against each other"""
    n_actions = game.payoff_p1.shape[0]
    
    player1 = GamePlayer(n_actions)
    player2 = GamePlayer(n_actions)
    
    # Initialize with random strategies
    s1 = np.random.dirichlet(np.ones(n_actions))
    s2 = np.random.dirichlet(np.ones(n_actions))
    
    for episode in range(n_episodes):
        # Get strategies
        s1 = player1.get_strategy(s2)
        s2 = player2.get_strategy(s1)
        
        # Calculate payoffs
        payoff1, payoff2 = game.get_payoffs((s1, s2))
        
        # Update players
        player1.update(s2, payoff1)
        player2.update(s1, payoff2)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: P1={payoff1:.3f}, P2={payoff2:.3f}")
    
    return player1, player2

class RLAgent:
    """RL agent for matrix games"""
    
    def __init__(self, n_actions: int, lr: float = 1e-3, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
    def get_action_probs(self) -> np.ndarray:
        """Get action probability distribution"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1)
            logits = self.model(dummy_input)
            probs = F.softmax(logits, dim=-1)
            return probs.squeeze().numpy()
    
    def sample_action(self) -> int:
        """Sample action with epsilon-greedy exploration"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            probs = self.get_action_probs()
            return np.random.choice(self.n_actions, p=probs)
    
    def update(self, action: int, reward: float):
        """Update policy using REINFORCE"""
        dummy_input = torch.zeros(1, 1)
        logits = self.model(dummy_input)
        log_probs = F.log_softmax(logits, dim=-1)
        
        loss = -log_probs[0, action] * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_rl_agents(game: MatrixGame, n_episodes: int = 5000, epsilon1: float = 0.1, epsilon2: float = 0.1) -> Tuple[RLAgent, RLAgent]:
    """Train two RL agents on matrix game"""
    agent1 = RLAgent(game.n_actions_p1, epsilon=epsilon1)
    agent2 = RLAgent(game.n_actions_p2, epsilon=epsilon2)
    
    payoff_history = []
    
    for episode in range(n_episodes):
        # Sample actions
        action1 = agent1.sample_action()
        action2 = agent2.sample_action()
        
        # Get rewards
        reward1, reward2 = game.get_payoffs(action1, action2)
        
        # Update agents
        agent1.update(action1, reward1)
        agent2.update(action2, reward2)
        
        payoff_history.append((reward1, reward2))
        
        if episode % 1000 == 0:
            avg_r1 = np.mean([r[0] for r in payoff_history[-100:]])
            avg_r2 = np.mean([r[1] for r in payoff_history[-100:]])
            print(f"Episode {episode}: Avg rewards P1={avg_r1:.3f}, P2={avg_r2:.3f}")
    
    return agent1, agent2, payoff_history

def evaluate_equilibrium(game: TwoPlayerGame, player1: GamePlayer, player2: GamePlayer) -> dict:
    """Evaluate if learned strategies form Nash equilibrium"""
    # Get final strategies
    s2_init = np.random.dirichlet(np.ones(game.payoff_p1.shape[0]))
    s1_init = np.random.dirichlet(np.ones(game.payoff_p1.shape[0]))
    
    s1 = player1.get_strategy(s2_init)
    s2 = player2.get_strategy(s1_init)
    
    # Refine strategies through iteration
    for _ in range(10):
        s1_new = player1.get_strategy(s2)
        s2_new = player2.get_strategy(s1)
        s1, s2 = s1_new, s2_new
    
    payoff1, payoff2 = game.get_payoffs((s1, s2))
    
    from game_theory import is_nash_equilibrium
    is_equilibrium = is_nash_equilibrium(game, (s1, s2))
    
    return {
        'strategy_p1': s1,
        'strategy_p2': s2,
        'payoff_p1': payoff1,
        'payoff_p2': payoff2,
        'is_nash': is_equilibrium
    }