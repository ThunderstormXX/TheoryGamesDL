"""GPU-Optimized Learner Module using PyTorch"""
import torch
import numpy as np
from gpu_utils import gpu_config

class GPULearner:
    """Base GPU-optimized learner with Q-table stored as tensor"""
    def __init__(self, action_space_size=2, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=0.1, strategy='epsilon_greedy', temperature=1.0,
                 max_states=10):
        """
        Args:
            max_states: Maximum number of possible states (for tensor pre-allocation)
        """
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.strategy = strategy
        self.temp = temperature
        self.device = gpu_config.device
        
        # Pre-allocate Q-table as tensor (state_id, action)
        self.q_table = torch.zeros(max_states, action_space_size, device=self.device)
        self.max_states = max_states
        self.state_counter = 0
        self.state_mapping = {}  # Map state values to indices
        
    def _get_state_idx(self, state):
        """Get or create index for state"""
        state_key = str(int(state))
        if state_key not in self.state_mapping:
            if self.state_counter >= self.max_states:
                # Dynamically grow table if needed
                new_size = self.max_states * 2
                new_table = torch.zeros(new_size, self.action_space_size, device=self.device)
                new_table[:self.max_states] = self.q_table
                self.q_table = new_table
                self.max_states = new_size
            
            self.state_mapping[state_key] = self.state_counter
            self.state_counter += 1
        
        return self.state_mapping[state_key]
    
    def get_q_values(self, state):
        """Get Q-values for a state as tensor"""
        state_idx = self._get_state_idx(state)
        return self.q_table[state_idx]
    
    def get_q_numpy(self, state):
        """Get Q-values as numpy array"""
        return self.get_q_values(state).cpu().numpy()
    
    def get_probs(self, state):
        """Compute action probabilities"""
        q_values = self.get_q_values(state)
        
        if self.strategy == 'epsilon_greedy':
            probs = torch.ones(self.action_space_size, device=self.device) * (self.epsilon / self.action_space_size)
            best_action = torch.argmax(q_values)
            probs[best_action] += (1.0 - self.epsilon)
            
        elif self.strategy == 'boltzmann':
            # Numerical stability
            q_stable = q_values - torch.max(q_values)
            exp_q = torch.exp(q_stable / self.temp)
            probs = exp_q / torch.sum(exp_q)
        else:
            probs = torch.ones(self.action_space_size, device=self.device) / self.action_space_size
        
        return probs.cpu().numpy()
    
    def choose_action(self, state):
        """Choose action based on policy"""
        probs = self.get_probs(state)
        return np.random.choice(len(probs), p=probs)
    
    def step(self, state, action, reward, next_state, done=False):
        """Update Q-values (to be overridden in subclasses)"""
        pass


class GPUQLearner(GPULearner):
    """GPU-optimized Q-Learning"""
    def step(self, state, action, reward, next_state, done=False):
        """Q-Learning update rule"""
        state_idx = self._get_state_idx(state)
        next_state_idx = self._get_state_idx(next_state)
        
        current_q = self.q_table[state_idx, action]
        next_q_max = torch.max(self.q_table[next_state_idx]) if not done else torch.tensor(0.0, device=self.device)
        
        td_target = reward + self.gamma * next_q_max
        td_error = td_target - current_q
        
        self.q_table[state_idx, action] += self.lr * td_error


class GPUSARSALearner(GPULearner):
    """GPU-optimized SARSA Learning"""
    def step(self, state, action, reward, next_state, next_action, done=False):
        """SARSA update rule"""
        state_idx = self._get_state_idx(state)
        next_state_idx = self._get_state_idx(next_state)
        
        current_q = self.q_table[state_idx, action]
        next_q_val = self.q_table[next_state_idx, next_action] if not done else torch.tensor(0.0, device=self.device)
        
        td_target = reward + self.gamma * next_q_val
        td_error = td_target - current_q
        
        self.q_table[state_idx, action] += self.lr * td_error
