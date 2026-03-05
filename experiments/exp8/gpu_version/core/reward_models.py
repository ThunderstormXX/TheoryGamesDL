import torch

class RewardManager:
    """
    Manages different reward calculation models for the game.
    All calculations are performed on tensors to maintain GPU efficiency.
    """
    def __init__(self, reward_type='pf', b=1.5, c=1.0):
        self.reward_type = reward_type
        self.b = b
        self.c = c

    def calculate_rewards(self, cooperators, adjacency_matrix, degrees):
        """
        Calculates rewards based on the specified model.
        
        Args:
            cooperators: (B, N) tensor where 1 is Cooperate, 0 is Defect.
            adjacency_matrix: (B, N, N) tensor.
            degrees: (B, N) tensor of node degrees (k_i).
            
        Returns:
            (B, N) tensor of rewards.
        """
        coop_expanded = cooperators.unsqueeze(2)
        k = degrees
        
        if self.reward_type == 'pp':
            # g(i) = b * sum(A_ij * x_j) - c * x_i * k_i
            pool = torch.bmm(adjacency_matrix, coop_expanded).squeeze(2)
            return self.b * pool - self.c * cooperators * k
            
        elif self.reward_type == 'pf':
            # g(i) = b * sum(A_ij * x_j) - c * x_i
            pool = torch.bmm(adjacency_matrix, coop_expanded).squeeze(2)
            return self.b * pool - self.c * cooperators
            
        elif self.reward_type == 'ff':
            # g(i) = b * sum(A_ij * x_j / k_j) - c * x_i
            # x_j / k_j: benefit scaled by neighbor's degree
            scaled_coop = (cooperators / torch.clamp(k, min=1)).unsqueeze(2)
            pool = torch.bmm(adjacency_matrix, scaled_coop).squeeze(2)
            return self.b * pool - self.c * cooperators
            
        elif self.reward_type == 'fp':
            # g(i) = b * sum(A_ij * x_j / k_j) - c * x_i * k_i
            scaled_coop = (cooperators / torch.clamp(k, min=1)).unsqueeze(2)
            pool = torch.bmm(adjacency_matrix, scaled_coop).squeeze(2)
            return self.b * pool - self.c * cooperators * k
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
