import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.visualization.plotting import plot_cooperation_with_std

def run_reward_verification(reward_type, b_value, n_rounds=2000):
    print(f"\n--- Verifying Stateless {reward_type} with b={b_value} ---")
    
    batch_size = 100
    n_agents = 50
    k_neighbors = 4
    graph_params = {'k': k_neighbors, 'p': 0.1}
    
    # Stateless: max_states = 1
    learner_params = {
        'learning_rate': 0.05,
        'discount_factor': 0.9,
        'exploration_rate': 0.1,
        'strategy': 'softmax',
        'temperature': 0.5,
        'max_states': 1
    }
    
    # We use c=1.0 for easier verification of the formula
    reward_params = {'b': b_value, 'c': 1.0}
    
    game = BatchedGPUMonteKarloPairGame(
        batch_size, n_agents, 
        graph_params, learner_params, reward_params,
        reward_type=reward_type
    )
    
    # Override to stateless
    def stateless_get_states(actions):
        return torch.zeros((batch_size, n_agents), device=game.device, dtype=torch.long)
    game._get_states = stateless_get_states
    game.current_states = game._get_states(game.current_actions)

    history_tensor = torch.zeros(n_rounds, batch_size)
    for i in range(n_rounds):
        metrics = game.round()
        history_tensor[i] = metrics['batch_coop_rates'].cpu()
        
    final_avg = torch.mean(history_tensor[-200:]).item()
    print(f"Final Average Coop: {final_avg:.4f}")
    
    # Theoretical Expected Cooperation for stateless softmax:
    # Action 0 (Coop) Reward: R_c
    # Action 1 (Defect) Reward: R_d = 0 (since x_i=0)
    # Prob(Coop) = exp(R_c/T) / (exp(R_c/T) + exp(0/T)) = 1 / (1 + exp(-R_c/T))
    # For pf/ff: R_c = -c (since neighboring x_j averages out or doesn't matter for convergence if b doesn't affect it)
    # Wait, the user formula says sigma(exp(c/T)). 
    # Usually it's softmax: P(C) = exp(Q_c/T) / (exp(Q_c/T) + exp(Q_d/T))
    
    return history_tensor.numpy(), final_avg

if __name__ == "__main__":
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results_plots"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Test for PF with different B values
    h1, avg1 = run_reward_verification('pf', b_value=1.5)
    h2, avg2 = run_reward_verification('pf', b_value=10.0)
    
    print(f"\nPF Verification: b=1.5 -> {avg1:.4f}, b=10.0 -> {avg2:.4f}")
    print(f"Difference: {abs(avg1 - avg2):.4f} (Should be near 0 if independent of b)")

    # Test for PP with different B values
    h3, avg3 = run_reward_verification('pp', b_value=1.5)
    h4, avg4 = run_reward_verification('pp', b_value=10.0)
    
    print(f"PP Verification: b=1.5 -> {avg3:.4f}, b=10.0 -> {avg4:.4f}")
    print(f"Difference: {abs(avg3 - avg4):.4f} (Should be near 0 if independent of b)")

    plot_cooperation_with_std(
        [h1, h2, h3, h4], 
        ["PF (b=1.5)", "PF (b=10.0)", "PP (b=1.5)", "PP (b=10.0)"],
        title="Stateless Independence of B Verification",
        save_path=os.path.join(results_dir, "stateless_verification.png")
    )
    
    print(f"\nVerification plots saved to: {results_dir}")
