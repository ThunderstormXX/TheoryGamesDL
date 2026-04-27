import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
DEVICE = torch.device('cpu')
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph

class CustomSixNodeGraph(BaseGraph):
    def __init__(self, device=None):
        super().__init__(num_nodes=6, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((6, 6), device=self.device, dtype=torch.float32)
        edges = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)]
        for u, v in edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        return adj

def quick_run(graph, gamma, beta, iters=50000):
    TEMP = 1.0 / beta
    np.random.seed(42)
    torch.manual_seed(42)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    num_nodes = graph.num_nodes
    
    REPS = 32
    states = torch.zeros((REPS, num_nodes), dtype=torch.long, device=DEVICE)
    
    learner = BatchedGPUQLearner(
        batch_size=REPS, n_agents=num_nodes, action_space_size=2,
        learning_rate=0.01, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=TEMP, max_states=1,
    )
    reward_manager = BonusRewardManager(reward_type='pp', b=2.0, c=1.0, bonus=1.0)
    
    adj_batched = adj_t.unsqueeze(0).expand(REPS, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(REPS, -1)
    
    with torch.no_grad():
        for t in range(1, iters + 1):
            actions = learner.get_actions(states) 
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
    q_now = learner.q_table[:, :, 0, :].cpu()
    probs = torch.softmax(q_now / TEMP, dim=-1).numpy()
    final_p = probs[..., 1]
    return final_p.mean(axis=0)

if __name__ == '__main__':
    print("--- Testing Custom 6-Node Graph ---")
    graph = CustomSixNodeGraph(DEVICE)
    for gamma in [0.0, 0.5, 0.9]:
        p = quick_run(graph, gamma, beta=1.0, iters=50000)
        print(f"Gamma={gamma:.1f} | Node A (neighbor deg 2): {p[0]:.3f} | Node D (neighbor deg 4): {p[3]:.3f}")

    print("\\n--- Testing Phase Transition (Complete Graph 4 nodes) ---")
    from experiments.exp8.gpu_version.core.graph_structure import CompleteGraph
    graph_comp = CompleteGraph(4, DEVICE)
    
    gammas = [0.0, 0.3, 0.6, 0.8, 0.95]
    betas = [0.5, 1.0, 2.0]
    
    print("      " + " ".join([f"B={b:.1f}" for b in betas]))
    for g in gammas:
        row = []
        for b in betas:
            p = quick_run(graph_comp, g, b, iters=20000)
            row.append(f"{p.mean():.3f}")
        print(f"G={g:.2f} " + " ".join(row))
