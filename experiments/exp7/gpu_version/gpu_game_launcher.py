"""GPU-Optimized Game Launchers"""
import torch
import numpy as np
from gpu_utils import gpu_config

class GPUGame:
    """Base GPU-optimized game class"""
    def __init__(self, graph, learners, reward_model):
        self.graph = graph
        self.learners = learners
        self.reward_model = reward_model
        
        # Strategies as tensor
        n_agents = len(learners)
        self.strategies = torch.from_numpy(
            np.random.randint(0, 2, size=n_agents)
        ).float().to(gpu_config.device)
        
        self.history = []
        self.adj_tensor = None
        self.degrees_tensor = None
        self.sum_adj = None
        self.device = gpu_config.device
        self._cache_graph_data()
    
    def _cache_graph_data(self):
        """Pre-compute and cache graph data as tensors"""
        adj = self.graph.get_adj_matrix()
        self.adj_tensor = torch.from_numpy(adj).float().to(self.device)
        
        degrees = np.array(self.graph.get_degree())
        self.degrees_tensor = torch.from_numpy(degrees).float().to(self.device)
        
        self.sum_adj = torch.sum(self.adj_tensor).item()
    
    def round(self):
        """Execute one game round"""
        pass
    
    def result(self):
        """Get game history"""
        return self.history
    
    def get_pairwise_cooperation(self):
        """Compute pairwise cooperation metric"""
        result = (self.strategies @ self.adj_tensor @ self.strategies) / self.sum_adj
        return result.item()
    
    def _get_states_batch(self, agent_indices, adj, strategies):
        """Efficiently compute states for multiple agents"""
        # For each agent in indices, count cooperating neighbors
        states = []
        for agent_id in agent_indices:
            neighbors = torch.nonzero(adj[agent_id]).squeeze()
            if neighbors.dim() == 0:
                neighbor_state = 0
            else:
                neighbor_state = int(torch.sum(strategies[neighbors]))
            states.append(neighbor_state)
        return states


class GPUMonteKarloPairGame(GPUGame):
    """GPU-optimized Monte Carlo Pair Game"""
    def __init__(self, graph, learners, reward_model, k_anchors=1):
        super().__init__(graph, learners, reward_model)
        self.k = k_anchors

    def round(self):
        """Execute one round of Monte Carlo Pair Game"""
        adj = self.adj_tensor
        n_agents = len(self.learners)
        
        # Select anchors
        if self.k == n_agents:
            active_nodes = list(range(n_agents))
        else:
            anchors = np.random.choice(n_agents, self.k, replace=False)
            active_nodes = set(anchors)
            for a in anchors:
                neighbors = torch.nonzero(adj[a]).squeeze()
                if neighbors.dim() == 0:
                    if neighbors.item() != a:
                        active_nodes.add(neighbors.item())
                else:
                    active_nodes.update(neighbors.tolist())
            active_nodes = list(active_nodes)
        
        # Create induced subgraph mask
        mask = torch.zeros_like(adj)
        for i, j in [(i, j) for i in active_nodes for j in active_nodes]:
            mask[i, j] = 1
        sub_adj = adj * mask
        
        # 1. Choose actions
        transitions = {}
        for i in active_nodes:
            state = self._get_state(i, sub_adj, self.strategies)
            action = self.learners[i].choose_action(state)
            transitions[i] = (state, action)
        
        # 2. Update strategies
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
        
        # 3. Compute rewards
        sub_degrees = torch.sum(sub_adj, dim=1)
        rewards = self.reward_model.get_all_rewards(self.strategies, sub_adj, sub_degrees)
        
        # 4. Learn
        for i in active_nodes:
            s, a = transitions[i]
            r = rewards[i].item() if isinstance(rewards[i], torch.Tensor) else float(rewards[i])
            next_state = self._get_state(i, sub_adj, self.strategies)
            next_action = self.learners[i].choose_action(next_state)
            
            if self.learners[i].__class__.__name__ == 'GPUSARSALearner':
                self.learners[i].step(s, a, r, next_state, next_action)
            else:
                self.learners[i].step(s, a, r, next_state)
        
        # Record history
        self.history.append({
            'active_nodes': active_nodes,
            'strategies': self.strategies.cpu().numpy().copy(),
            'rewards': rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards.copy()
        })
    
    def _get_state(self, agent_id, adj, strategies):
        """Get state for single agent"""
        neighbors = torch.nonzero(adj[agent_id]).squeeze()
        if neighbors.dim() == 0:
            return 0
        return int(torch.sum(strategies[neighbors]))


class GPUPairGame(GPUGame):
    """GPU-optimized Pair Game (all nodes each round)"""
    def round(self):
        """Execute one round of Pair Game"""
        adj = self.adj_tensor
        
        # 1. Choose actions
        transitions = {}
        for i in range(len(self.learners)):
            state = self._get_state(i, adj, self.strategies)
            action = self.learners[i].choose_action(state)
            transitions[i] = (state, action)
        
        # 2. Update strategies
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
        
        # 3. Compute rewards
        rewards = self.reward_model.get_all_rewards(self.strategies, adj, self.degrees_tensor)
        
        # 4. Learn
        for i in range(len(self.learners)):
            s, a = transitions[i]
            r = rewards[i].item() if isinstance(rewards[i], torch.Tensor) else float(rewards[i])
            next_state = self._get_state(i, adj, self.strategies)
            next_action = self.learners[i].choose_action(next_state)
            
            if self.learners[i].__class__.__name__ == 'GPUSARSALearner':
                self.learners[i].step(s, a, r, next_state, next_action)
            else:
                self.learners[i].step(s, a, r, next_state)
        
        self.history.append({
            'strategies': self.strategies.cpu().numpy().copy(),
            'rewards': rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards.copy()
        })
    
    def _get_state(self, agent_id, adj, strategies):
        """Get state for single agent"""
        neighbors = torch.nonzero(adj[agent_id]).squeeze()
        if neighbors.dim() == 0:
            return 0
        return int(torch.sum(strategies[neighbors]))


class GPUMonteKarloNotPairGame(GPUGame):
    """GPU-optimized Monte Carlo Not Pair Game (clique topology)"""
    def __init__(self, graph, learners, reward_model, k_anchors=1):
        super().__init__(graph, learners, reward_model)
        self.k = k_anchors

    def round(self):
        """Execute one round"""
        adj = self.adj_tensor
        n_agents = len(self.learners)
        
        # Select anchors and their neighbors
        anchors = np.random.choice(n_agents, self.k, replace=False)
        active_nodes = set(anchors)
        for a in anchors:
            neighbors = torch.nonzero(adj[a]).squeeze()
            if neighbors.dim() == 0:
                if neighbors.item() != a:
                    active_nodes.add(neighbors.item())
            else:
                active_nodes.update(neighbors.tolist())
        active_nodes = list(active_nodes)
        
        # Create clique (complete graph on active nodes)
        clique_adj = torch.zeros_like(adj)
        for i, j in [(i, j) for i in active_nodes for j in active_nodes if i != j]:
            clique_adj[i, j] = 1
        
        # 1. Choose actions
        transitions = {}
        for i in active_nodes:
            state = self._get_state(i, clique_adj, self.strategies)
            action = self.learners[i].choose_action(state)
            transitions[i] = (state, action)
        
        # 2. Update strategies
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
        
        # 3. Compute rewards
        clique_degrees = torch.sum(clique_adj, dim=1)
        rewards = self.reward_model.get_all_rewards(self.strategies, clique_adj, clique_degrees)
        
        # 4. Learn
        for i in active_nodes:
            s, a = transitions[i]
            r = rewards[i].item() if isinstance(rewards[i], torch.Tensor) else float(rewards[i])
            next_state = self._get_state(i, clique_adj, self.strategies)
            next_action = self.learners[i].choose_action(next_state)
            
            if self.learners[i].__class__.__name__ == 'GPUSARSALearner':
                self.learners[i].step(s, a, r, next_state, next_action)
            else:
                self.learners[i].step(s, a, r, next_state)
        
        self.history.append({
            'active_nodes': active_nodes,
            'strategies': self.strategies.cpu().numpy().copy()
        })
    
    def _get_state(self, agent_id, adj, strategies):
        """Get state for single agent"""
        neighbors = torch.nonzero(adj[agent_id]).squeeze()
        if neighbors.dim() == 0:
            return 0
        return int(torch.sum(strategies[neighbors]))
