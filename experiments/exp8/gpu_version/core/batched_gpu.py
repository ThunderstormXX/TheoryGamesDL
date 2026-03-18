import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config
from experiments.exp8.gpu_version.core.graph_structure import SmallWorldGraph, StarGraph, WheelGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager

class BatchedGPUQLearner:
    """
    Batched Q-Learner for multiple simulations running in parallel.
    Manages Q-tables for (batch_size * n_agents) agents.
    """
    def __init__(self, batch_size, n_agents, action_space_size=2, 
                 learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=0.1, strategy='epsilon_greedy', 
                 temperature=1.0, max_states=1001):
        
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.total_agents = batch_size * n_agents
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.strategy = strategy
        self.temp = temperature
        self.device = gpu_config.device
        self.max_states = max_states
        
        # Q-table shape: (batch_size, n_agents, max_states, action_space_size)
        # We assume state is an integer index from 0 to max_states-1.
        self.q_table = torch.zeros(
            (batch_size, n_agents, max_states, action_space_size), 
            device=self.device, dtype=torch.float32
        )
        
    def get_actions(self, states):
        """
        states: (batch_size, n_agents) tensor of state indices
        Returns: (batch_size, n_agents) tensor of actions
        """
        # Flatten for easier processing
        flat_states = states.view(-1)
        
        # Create indices for batch and agent: 0, 1, ..., total_agents-1
        batch_agent_linear_idx = torch.arange(self.total_agents, device=self.device)
        
        # Select Q-values for current states: (B*N, A)
        # q_table is (B, N, S, A) -> view (B*N, S, A)
        # For each (batch*agent)_i, we want q_table[i, flat_states[i], :]
        flat_q = self.q_table.view(-1, self.max_states, self.action_space_size)
        
        # Advanced indexing:
        # flat_q[i, state[i]] gives (A) vector for each i
        gathered_q = flat_q[batch_agent_linear_idx, flat_states] # (B*N, A)
        
        if self.strategy == 'epsilon_greedy':
            # Create a uniform distribution mask for epsilon-greedy
            random_probs = torch.rand(self.total_agents, device=self.device)
            random_actions = torch.randint(0, self.action_space_size, (self.total_agents,), device=self.device)
            
            # Select max Q actions
            greedy_actions = torch.argmax(gathered_q, dim=1)
            
            # Apply epsilon: pick random action with probability epsilon
            actions = torch.where(random_probs < self.epsilon, random_actions, greedy_actions)
            
        elif self.strategy == 'softmax' or self.strategy == 'boltzmann':
            # Softmax on Q/temp
            # Using higher temperature makes actions more random
            # Using lower temperature makes it more greedy
            probabilities = torch.softmax(gathered_q / self.temp, dim=1)
            actions = torch.multinomial(probabilities, 1).squeeze()
        else:
            actions = torch.argmax(gathered_q, dim=1)
            
        return actions.view(self.batch_size, self.n_agents)

    def update(self, states, actions, rewards, next_states, mask=None):
        """
        Batch update Q-values.
        """
        flat_states = states.view(-1)
        flat_actions = actions.view(-1)
        
        current_lr = self.lr
            
        flat_rewards = rewards.view(-1)
        flat_next_states = next_states.view(-1)
        
        # Linear indices
        batch_agent_idx = torch.arange(self.total_agents, device=self.device)
        
        # Current Q(S, A)
        flat_q = self.q_table.view(-1, self.max_states, self.action_space_size)
        
        # Max Next Q(S', a')
        next_q_values = flat_q[batch_agent_idx, flat_next_states] # (B*N, A)
        max_next_q, _ = torch.max(next_q_values, dim=1)
        
        # Target: Q_target = R + gamma * max(Q(S', a'))
        target = flat_rewards + self.gamma * max_next_q
        
        # Update current Q value with scatter_
        flat_q_data = self.q_table.view(-1)
        indices = batch_agent_idx * (self.max_states * self.action_space_size) + \
                  flat_states * self.action_space_size + flat_actions
        
        # Current values
        current_q = flat_q_data[indices]
        
        # Incremental update: Q = (1-lr)*Q + lr*target
        new_q_val = (1.0 - current_lr) * current_q + current_lr * target

        if mask is not None:
             flat_mask = mask.view(-1)
             indices = indices[flat_mask]
             new_q_val = new_q_val[flat_mask]

        flat_q_data.scatter_(0, indices, new_q_val)
        
    def get_q_table(self):
        return self.q_table


class BatchedGPUMonteKarloPairGame:
    def __init__(self, batch_size, n_agents_per_sim, graph_params, learner_params, reward_params, graph_type='small_world', reward_type='pf', n_anchors=None):
        self.batch_size = batch_size
        self.n_agents = n_agents_per_sim
        self.graph_params = graph_params
        self.reward_params = reward_params
        self.n_anchors = n_anchors
        self.device = gpu_config.device
        
        # Reward Manager
        self.b = reward_params.get('b', 1.5)
        self.c = reward_params.get('c', 1.0)
        self.reward_manager = RewardManager(reward_type, self.b, self.c)
        
        # Graph
        if graph_type == 'small_world':
            self.graph = SmallWorldGraph(n_agents_per_sim, **graph_params, device=self.device)
        elif graph_type == 'star':
            self.graph = StarGraph(n_agents_per_sim, device=self.device)
        elif graph_type == 'wheel':
            self.graph = WheelGraph(n_agents_per_sim, device=self.device)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
            
        self.adjacency_matrix = self.graph.get_batched_adjacency(batch_size).to(self.device)
        # Degree per agent: (B, N)
        self.degrees = self.adjacency_matrix.sum(dim=2)
        
        # Learner
        self.learner = BatchedGPUQLearner(
            batch_size, n_agents_per_sim, 
            **learner_params
        )
        
        # Initial State
        # Random initial actions: 0 or 1
        self.current_actions = torch.randint(0, 2, (batch_size, n_agents_per_sim), device=self.device)
        self.current_states = self._get_states(self.current_actions)
        
        # History
        self.history_actions = []
        self.history_rewards = []
        # Or explicit matrix: [[R, S], [T, P]]
        # R = b-c, S = -c, T = b, P = 0
        
    def _get_states(self, actions):
        """
        Calculate state for each agent based on neighbors' actions.
        State = number of cooperating neighbors (action 0).
        """
        # actions: (B, N). 0=Coop, 1=Defect.
        # We want number of cooperators. Cooperators are where action is 0.
        # Let's invert action to count cooperators: (1 - action)
        cooperators = 1 - actions # (B, N)
        
        # Neighbors sum
        # Adjacency (B, N, N) * Cooperators (B, N, 1) -> (B, N, 1)
        # Use bmm (batch matrix multiply)
        # cooperators needs to be (B, N, 1) to multiply with (B, N, N)
        coop_expanded = cooperators.unsqueeze(2) # (B, N, 1)
        
        # We want sum over neighbors j for each i: sum_j A[i,j] * C[j]
        # BMM: (B, N, N) x (B, N, 1) -> (B, N, 1)
        # MPS requires float tensors for bmm usually
        neighbor_coops = torch.bmm(self.adjacency_matrix, coop_expanded.float()).squeeze(2) # (B, N)
        
        # State is just this count (rounded to int just in case, though adj is 1.0/0.0)
        states = neighbor_coops.long()
        
        # Ensure states < max_states
        states = torch.clamp(states, 0, self.learner.max_states - 1)
        
        return states

    def round(self):
        """
        Execute one round of the game.
        Supports both full-update (n_anchors=None) and k-anchors update.
        """
        # Define update mask: (B, N)
        if self.n_anchors is not None:
            # 1. Select anchors: (B, n_anchors)
            anchors_idx = torch.randint(0, self.n_agents, (self.batch_size, self.n_anchors), device=self.device)
            
            # 2. Update mask: Anchors AND their neighbors
            # Start with anchors
            mask = torch.zeros((self.batch_size, self.n_agents), device=self.device, dtype=torch.bool)
            mask.scatter_(1, anchors_idx, True) # Set anchors to True
            
            # Neighbors of anchors: (B, N, N) * (B, N, 1) -> (B, N, 1)
            # A[i,j] is 1 if j is neighbor of i.
            # We want all j such that A[anchor, j] == 1.
            anchor_one_hot = torch.zeros((self.batch_size, self.n_agents, 1), device=self.device)
            anchor_one_hot.scatter_(1, anchors_idx.unsqueeze(2), 1.0)
            
            # (B, N, N)^T * (B, N, 1) -> (B, N, 1)
            # Actually, since adj is symmetric for these graphs:
            neigh_signals = torch.bmm(self.adjacency_matrix, anchor_one_hot).squeeze(2)
            mask = (mask | (neigh_signals > 0)) # Union of anchors and neighbors
        else:
            mask = None # Full update for all agents

        # 1. Get actions from learner for all agents (required for state calculation)
        new_actions = self.learner.get_actions(self.current_states)
        
        # 2. Calculate rewards for all agents
        cooperators = (1 - new_actions).float()
        rewards = self.reward_manager.calculate_rewards(
            cooperators, 
            self.adjacency_matrix, 
            self.degrees
        )

        # 3. Determine next states
        coop_expanded = cooperators.unsqueeze(2)
        neighbor_coops = torch.bmm(self.adjacency_matrix, coop_expanded).squeeze(2)
        next_states = neighbor_coops.long()
        next_states = torch.clamp(next_states, 0, self.learner.max_states - 1)
        
        # 4. Update Learner (only for agents in mask if mask is provided)
        self.learner.update(self.current_states, new_actions, rewards, next_states, mask=mask)
        
        # 5. Internal state update
        if mask is not None:
             # If using k-anchors, only 'active' agents update their current state/action?
             # Standard Monte Carlo Pair usually updates only active ones.
             # However, actions for others are still what they were.
             self.current_states = torch.where(mask, next_states, self.current_states)
             self.current_actions = torch.where(mask, new_actions, self.current_actions)
        else:
             self.current_states = next_states
             self.current_actions = new_actions
        
        # Log: (batch_size) vector of cooperación rates
        batch_coop_rates = cooperators.mean(dim=1) # (B)
        
        return {
            "mean_cooperation": batch_coop_rates.mean().item(),
            "batch_coop_rates": batch_coop_rates,
            "mean_reward": rewards.mean().item()
        }

    def get_history(self):
        # Implementation depends on how we store history
        return self.history_actions
        
    def plot_q_function(self, batch_idx, agent_idx):
        """
        Plots the Q-function for a specific agent in a specific batch.
        """
        q_table = self.learner.q_table[batch_idx, agent_idx] # (max_states, action_space)
        q_np = q_table.cpu().detach().numpy()
        
        # Filter out states that are never visited? 
        # Or just plot the whole thing up to max possible neighbors
        # Max neighbors = k.
        k = self.graph_params.get('k', 4)
        valid_states = k + 1
        q_subset = q_np[:valid_states, :]
        
        plt.figure(figsize=(10, 6))
        plt.plot(q_subset[:, 0], label='Action 0 (Cooperate)')
        plt.plot(q_subset[:, 1], label='Action 1 (Defect)')
        plt.xlabel('State (Number of Cooperating Neighbors)')
        plt.ylabel('Q-Value')
        plt.title(f'Q-Function for Agent {agent_idx} in Batch {batch_idx}')
        plt.legend()
        plt.grid(True)
        return plt

