import numpy as np

class Reward():
    def __init__(self, b, c):
        self.b = b
        self.c = c

    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        raise NotImplementedError

    def get_all_rewards(self, strategies, adj_matrix, degrees):
        # strategies: 1D array of 0/1
        # adj_matrix: 2D adjacency
        # degrees: 1D array of degrees
        rewards = []
        for i in range(len(strategies)):
            rewards.append(self.get_reward(i, strategies, adj_matrix, degrees))
        return np.array(rewards)

class PPReward(Reward):
    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        # pp: g(xi) = b * sum(wij * xj) - c * xi * ki
        neighbors = np.nonzero(adj_matrix[agent_id])[0]
        sum_neighbor_strat = np.sum(strategies[neighbors])
        xi = strategies[agent_id]
        ki = degrees[agent_id]
        return self.b * sum_neighbor_strat - self.c * xi * ki

class PFReward(Reward):
    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        # pf: g(xi) = b * sum(wij * xj) - c * xi
        neighbors = np.nonzero(adj_matrix[agent_id])[0]
        sum_neighbor_strat = np.sum(strategies[neighbors])
        xi = strategies[agent_id]
        return self.b * sum_neighbor_strat - self.c * xi

class FPReward(Reward):
    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        # fp: g(xi) = b * sum(wij * xj / kj) - c * xi * ki
        neighbors = np.nonzero(adj_matrix[agent_id])[0]
        term1 = 0.0
        for j in neighbors:
            kj = degrees[j]
            if kj > 0:
                term1 += strategies[j] / kj
        xi = strategies[agent_id]
        ki = degrees[agent_id]
        return self.b * term1 - self.c * xi * ki

class FFReward(Reward):
    def get_reward(self, agent_id, strategies, adj_matrix, degrees):
        # ff: g(xi) = b * sum(wij * xj / kj) - c * xi
        neighbors = np.nonzero(adj_matrix[agent_id])[0]
        term1 = 0.0
        for j in neighbors:
            kj = degrees[j]
            if kj > 0:
                term1 += strategies[j] / kj
        xi = strategies[agent_id]
        return self.b * term1 - self.c * xi
