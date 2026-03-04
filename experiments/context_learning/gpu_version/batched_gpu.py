import torch
import numpy as np
from gpu_utils import gpu_config
from graph_structure import SmallWorldGraph

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
            # Random actions
            random_actions = torch.randint(0, self.action_space_size, (self.total_agents,), device=self.device)
            # Greedy actions
            greedy_actions = torch.argmax(gathered_q, dim=1)
            
            # Mask for exploration
            mask = torch.rand((self.total_agents,), device=self.device) < self.epsilon
            actions = torch.where(mask, random_actions, greedy_actions)
            
        elif self.strategy == 'boltzmann':
            # Numerical stability
            q_max, _ = torch.max(gathered_q, dim=1, keepdim=True)
            q_stable = gathered_q - q_max
            exp_q = torch.exp(q_stable / self.temp)
            probs = exp_q / torch.sum(exp_q, dim=1, keepdim=True)
            
            # Sampling
            actions = torch.multinomial(probs, 1).squeeze()
        else:
            actions = torch.randint(0, self.action_space_size, (self.total_agents,), device=self.device)
            
        return actions.view(self.batch_size, self.n_agents)

    def update(self, states, actions, rewards, next_states, mask=None):
        """
        Batch update Q-values.
        states, actions, rewards, next_states: (batch_size, n_agents)
        mask: (batch_size, n_agents) boolean tensor, true if update should happen
        """
        # Flatten everything
        flat_q = self.q_table.view(-1, self.max_states, self.action_space_size)
        flat_states = states.view(-1)
        flat_actions = actions.view(-1)
        flat_rewards = rewards.view(-1)
        flat_next_states = next_states.view(-1)
        
        if mask is not None:
            flat_mask = mask.view(-1)
            # Filter indices where mask is True
            valid_indices = torch.nonzero(flat_mask).squeeze()
            if valid_indices.numel() == 0:
                return
                
            batch_agent_indices = valid_indices
            states_subset = flat_states[valid_indices]
            actions_subset = flat_actions[valid_indices]
            rewards_subset = flat_rewards[valid_indices]
            next_states_subset = flat_next_states[valid_indices]
        else:
            batch_agent_indices = torch.arange(self.total_agents, device=self.device)
            states_subset = flat_states
            actions_subset = flat_actions
            rewards_subset = flat_rewards
            next_states_subset = flat_next_states

        # Current Q(s, a)
        current_q_vals = flat_q[batch_agent_indices, states_subset, actions_subset]
        
        # Max Q(s', a')
        next_q_all = flat_q[batch_agent_indices, next_states_subset] # (K, A) where K is num valid
        next_q_max, _ = torch.max(next_q_all, dim=1)
        
        # TD Target
        td_target = rewards_subset + self.gamma * next_q_max
        td_error = td_target - current_q_vals
        
        # Update in-place
        flat_q[batch_agent_indices, states_subset, actions_subset] += self.lr * td_error


class BatchedGPUMonteKarloPairGame:
    def __init__(self, batch_size, n_agents_per_sim, graph_params, learner_params, reward_params, k_anchors=1):
        self.batch_size = batch_size
        self.n_agents = n_agents_per_sim
        self.device = gpu_config.device
        self.k_anchors = k_anchors
        
        # Initialize one learner managing all agents
        # max_states needs to be n_agents + 1
        learner_params['max_states'] = self.n_agents + 1
        self.learner = BatchedGPUQLearner(batch_size, self.n_agents, **learner_params)
        
        # Generate graphs
        # adjs needs to be (B, N, N)
        # We can create one and repeat, or B random ones.
        # Repeating one is faster and uses less memory if we use expand, BUT we need mutable separate copies?
        # Actually adj is static, so one copy expanded is fine. 
        # BUT for true experiment we need different graphs per sim.
        
        # For now, let's create 1 graph and repeat to save startup time, as creating 100 graphs is slow.
        # Or better: create 'batch_size' graphs.
        
        # Optimization: Create ONE graph if 'graph_params' indicates so, or many.
        # User wants "100 experiments", implying statistical validity over graph realizations.
        # So we should generate B graphs.
        
        print(f"Generating {batch_size} graphs (CPU)...")
        # Pre-allocate tensor
        self.adj_tensor = torch.zeros((batch_size, self.n_agents, self.n_agents), device=self.device, dtype=torch.float32)
        
        # Generate sequentially
        # This part might be slow for large B.
        for i in range(batch_size):
            g = SmallWorldGraph(**graph_params)
            adj = torch.from_numpy(g.get_adj_matrix()).float().to(self.device)
            self.adj_tensor[i] = adj
            
        self.degrees = torch.sum(self.adj_tensor, dim=2) # (B, N)
        
        # Strategies: (B, N) random init
        self.strategies = torch.randint(0, 2, (batch_size, self.n_agents), device=self.device).float()
        
        self.b = reward_params['b']
        self.c = reward_params.get('c', 1.0)
        
        self.history_coop = []

    def _get_states(self, strategies):
        """
        Compute states for all agents in all batches.
        State = number of cooperating neighbors.
        Returns: (B, N) LongTensor
        """
        # (B, N, N) bmm (B, N, 1) -> (B, N, 1)
        # strategies must be float for matmul
        neighbor_coops = torch.bmm(self.adj_tensor, strategies.unsqueeze(2)).squeeze(2)
        return neighbor_coops.long()

    def round(self):
        # 1. Get current states for all agents
        states = self._get_states(self.strategies)
        
        # 2. Learners choose actions (proposed next strategies)
        # We get actions for everyone, but will only use some
        actions = self.learner.get_actions(states)
        
        # 3. Determine Active Nodes (Mask)
        if self.k_anchors < self.n_agents:
            # Select k anchors PER BATCH randomly efficiently
            # We use topk on random noise
            rand_vals = torch.rand((self.batch_size, self.n_agents), device=self.device)
            _, anchors = torch.topk(rand_vals, self.k_anchors, dim=1) # (B, k)
            
            # Create mask (B, N)
            active_mask = torch.zeros((self.batch_size, self.n_agents), device=self.device, dtype=torch.bool)
            # scatter_ requires index to have same dim as src, or broadcast?
            # scatter_(dim, index, src)
            # anchors is (B, k). active_mask is (B, N).
            # We want active_mask[b, anchors[b, k]] = True.
            active_mask.scatter_(1, anchors, True)
            
            # Expand to neighbors
            # (B, N, N) @ (B, N, 1) -> (B, N, 1) > 0
            # neighbor_mask is true if any neighbor is anchor
            # Since adj is symmetric, "my neighbor is anchor" <=> "I am neighbor of anchor"
            has_anchor_neighbor = torch.bmm(self.adj_tensor, active_mask.float().unsqueeze(2)).squeeze(2) > 0
            
            final_mask = active_mask | has_anchor_neighbor
        else:
            final_mask = torch.ones((self.batch_size, self.n_agents), device=self.device, dtype=torch.bool)
            
        # 4. Update strategies
        # Only active nodes update their strategy to the chosen action
        next_strategies = torch.where(final_mask, actions.float(), self.strategies)
        
        # 5. Compute rewards 
        # Based on NEW strategies
        n_coop_neighbors = torch.bmm(self.adj_tensor, next_strategies.unsqueeze(2)).squeeze(2) # (B, N)
        
        # Payoff (GPUPPReward): b * sum(xj) - c * xi * ki
        rewards = self.b * n_coop_neighbors - self.c * next_strategies * self.degrees
        
        # 6. Learn
        # We need next_state (state at t+1) based on strategies at t+1
        next_states = self._get_states(next_strategies)
        
        # Update Q-values for active agents
        # (s, a, r, s') where:
        # s = state before action
        # a = action taken
        # r = reward received after action
        # s' = state resulting from everyone's actions
        
        # Optimization: pass mask only if needed
        self.learner.update(states, actions, rewards, next_states, mask=final_mask)
        
        # 7. Advance state
        self.strategies = next_strategies
        
        # Record history mean coop
        mean_coop = self.strategies.mean(dim=1).cpu().numpy() # (B,)
        self.history_coop.append(mean_coop)
        
    def get_history(self):
        """
        Returns: (episodes, batch_size) -> (batch_size, episodes)??
        history_coop is list of (B,) arrays.
        Vertical stack -> (episodes, B)
        Transpose -> (B, episodes)
        """
        return np.vstack(self.history_coop).T

