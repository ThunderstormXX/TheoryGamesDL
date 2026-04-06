import torch
from .reward_models import RewardManager

class BonusRewardManager(RewardManager):
    """
    Modified RewardManager that adds +1.0 bonus to every agent in every outcome.
    """
    def __init__(self, reward_type='pp', b=3.0, c=1.0, bonus=1.0):
        super().__init__(reward_type=reward_type, b=b, c=c)
        self.bonus = bonus

    def calculate_rewards(self, cooperators, adjacency_matrix, degrees):
        rewards = super().calculate_rewards(cooperators, adjacency_matrix, degrees)
        return rewards + self.bonus
