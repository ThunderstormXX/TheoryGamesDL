
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config

def test_convergence():
    b, c, t = 3.0, 1.0, 1.0
    k = 4
    n_batches = 10
    n_agents = 200
    
    # Setup similar to reogranized_experiment.py
    graph_params = {'k': k, 'p': 0.1}
    
    # Stateless
    max_states = 1
    
    learner_params = {
        'learning_rate': 0.05,
        'discount_factor': 0.9,
        'exploration_rate': 0.0, # Pure softmax (code ignores epsilon for softmax anyway)
        'strategy': 'softmax',
        'temperature': t,
        'max_states': max_states
    }
    
    reward_params = {'b': b, 'c': c}
    
    game = BatchedGPUMonteKarloPairGame(
        n_batches, n_agents, 
        graph_params, learner_params, reward_params,
        reward_type='pf'
    )
    
    # Override for stateless
    def stateless_get_states(actions):
        return torch.zeros((n_batches, n_agents), device=game.device, dtype=torch.long)
    game._get_states = stateless_get_states
    game.current_states = game._get_states(game.current_actions)
    
    print(f"Theory value (P(C)): {1/(1+np.exp(c/t)):.4f}")
    
    for i in range(1001):
        metrics = game.round()
        if i % 200 == 0:
            print(f"Round {i}: Mean Coop = {metrics['mean_cooperation']:.4f}")
            
            # Check Q-values average difference
            q_table = game.learner.q_table # (B, N, S, A)
            q0 = q_table[:, :, 0, 0].mean().item() # Coop
            q1 = q_table[:, :, 0, 1].mean().item() # Defect
            print(f"  Avg Q(C)={q0:.4f}, Avg Q(D)={q1:.4f}, Diff={q0-q1:.4f}")
            
    # Final check
    q_table = game.learner.q_table
    q_diff = q_table[:, :, 0, 0] - q_table[:, :, 0, 1]
    # Calculate prob from q_diff
    probs = torch.sigmoid(q_diff / t)
    print(f"Avg Prob from Q: {probs.mean().item():.4f}")
    print(f"Theory Prob: {1/(1+np.exp(c/t)):.4f}")
    
    # Calculate var
    print(f"Var(Q_diff): {q_diff.var().item():.4f}")
    
    # Check if Jensen's approx matches
    # mean_prob approx sigmoid(mean_diff) + 0.5 * sigmoid'' * var
    
if __name__ == "__main__":
    test_convergence()
