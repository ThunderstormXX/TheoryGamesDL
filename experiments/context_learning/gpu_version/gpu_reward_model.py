"""GPU-Optimized Reward Models using PyTorch"""
import torch
import numpy as np
from gpu_utils import gpu_config

class GPUReward:
    """Base class for GPU-optimized reward models"""
    def __init__(self, b, c):
        self.b = torch.tensor(b, dtype=torch.float32, device=gpu_config.device)
        self.c = torch.tensor(c, dtype=torch.float32, device=gpu_config.device)
        self.device = gpu_config.device
    
    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        """Single reward calculation - override in subclass"""
        raise NotImplementedError
    
    def get_all_rewards(self, strategies, adj_matrix, degrees):
        """
        Vectorized reward calculation for all agents
        
        Args:
            strategies: 1D tensor/array of 0/1
            adj_matrix: 2D tensor/array adjacency matrix
            degrees: 1D tensor/array of degrees
            
        Returns:
            1D tensor of rewards for all agents
        """
        # Convert to tensors if needed
        if isinstance(strategies, np.ndarray):
            strategies = torch.from_numpy(strategies).float().to(self.device)
        elif not isinstance(strategies, torch.Tensor):
            strategies = torch.tensor(strategies, dtype=torch.float32, device=self.device)
        
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float().to(self.device)
        elif not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=self.device)
        
        if isinstance(degrees, np.ndarray):
            degrees = torch.from_numpy(degrees).float().to(self.device)
        elif not isinstance(degrees, torch.Tensor):
            degrees = torch.tensor(degrees, dtype=torch.float32, device=self.device)
        
        return self._vectorized_rewards(strategies, adj_matrix, degrees)
    
    def _vectorized_rewards(self, strategies, adj_matrix, degrees):
        """Override in subclass with vectorized computation"""
        raise NotImplementedError


class GPUPPReward(GPUReward):
    """GPU-optimized PP Reward: g(xi) = b * sum(wij * xj) - c * xi * ki"""
    def _vectorized_rewards(self, strategies, adj_matrix, degrees):
        # Sum of neighbor strategies: adj @ strategies
        neighbor_sum = adj_matrix @ strategies
        xi = strategies
        ki = degrees
        
        rewards = self.b * neighbor_sum - self.c * xi * ki
        return rewards


class GPUPFReward(GPUReward):
    """GPU-optimized PF Reward: g(xi) = b * sum(wij * xj) - c * xi"""
    def _vectorized_rewards(self, strategies, adj_matrix, degrees):
        neighbor_sum = adj_matrix @ strategies
        xi = strategies
        
        rewards = self.b * neighbor_sum - self.c * xi
        return rewards


class GPUFPReward(GPUReward):
    """GPU-optimized FP Reward: g(xi) = b * sum(wij * xj / kj) - c * xi * ki"""
    def _vectorized_rewards(self, strategies, adj_matrix, degrees):
        # Avoid division by zero
        degrees_safe = torch.clamp(degrees, min=1e-8)
        weighted_strategies = strategies / degrees_safe
        
        neighbor_weighted_sum = adj_matrix @ weighted_strategies
        xi = strategies
        ki = degrees
        
        rewards = self.b * neighbor_weighted_sum - self.c * xi * ki
        return rewards


class GPUFFReward(GPUReward):
    """GPU-optimized FF Reward: g(xi) = b * sum(wij * xj / kj) - c * xi"""
    def _vectorized_rewards(self, strategies, adj_matrix, degrees):
        degrees_safe = torch.clamp(degrees, min=1e-8)
        weighted_strategies = strategies / degrees_safe
        
        neighbor_weighted_sum = adj_matrix @ weighted_strategies
        xi = strategies
        
        rewards = self.b * neighbor_weighted_sum - self.c * xi
        return rewards


# Factory function for easy creation
def create_gpu_reward(reward_type, b, c):
    """Create GPU reward model by type"""
    reward_classes = {
        'pp': GPUPPReward,
        'pf': GPUPFReward,
        'fp': GPUFPReward,
        'ff': GPUFFReward,
    }
    
    if reward_type.lower() not in reward_classes:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward_classes[reward_type.lower()](b, c)
