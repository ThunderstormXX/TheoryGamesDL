import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Refactored imports
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.visualization.plotting import plot_cooperation_with_std, plot_q_table

def run_experiment_with_data(name, state_type, b, c, t, k, reward_type='pf', n_batches=100, n_rounds=1000):
    print(f"\nRunning Experiment: {name} (Reward: {reward_type})")
    
    batch_size = n_batches
    n_agents = 200
    k_neighbors = k
    
    graph_params = {'k': k_neighbors, 'p': 0.1}
    
    max_states = (k_neighbors + 1) if state_type == 'neighbor_coop' else 1
    
    learner_params = {
        'learning_rate': 0.05,
        'discount_factor': 0.9,
        'exploration_rate': 0.1,
        'strategy': 'softmax',
        'temperature': t,
        'max_states': max_states
    }
    
    reward_params = {'b': b, 'c': c}
    
    game = BatchedGPUMonteKarloPairGame(
        batch_size, n_agents, 
        graph_params, learner_params, reward_params,
        reward_type=reward_type
    )
    
    # Stateless simulation override
    if state_type == 'stateless':
        def stateless_get_states(actions):
            return torch.zeros((batch_size, n_agents), device=game.device, dtype=torch.long)
        game._get_states = stateless_get_states
        game.current_states = game._get_states(game.current_actions)

    # Historical data: (n_rounds, batch_size)
    history_tensor = torch.zeros(n_rounds, batch_size)
    
    for i in range(n_rounds):
        metrics = game.round()
        # Save batch-wise rates for calculating STD
        history_tensor[i] = metrics['batch_coop_rates'].cpu()
        if i % 200 == 0:
            print(f"Round {i}: Mean Coop = {metrics['mean_cooperation']:.4f}")
            
    final_avg = metrics['mean_cooperation']
    print(f"Final Average Cooperation ({name}): {final_avg:.4f}")
    
    # Also plot one Q-table
    plot_q_table(game.learner.q_table, k_neighbors, save_path=f"../results/q_table_{state_type}.png")
    
    return history_tensor.numpy()

if __name__ == "__main__":
    b, c, t = 3, 1, 1
    k = 4
    n_rounds = 10000 
    n_batches = 200
    
    reward_types = ['pf']
    
    for r_type in reward_types:
        experiment_info = {
            "b": b,
            "c": c,
            "T": t,
            "N": 200,
            "K": k,
            "Batches": n_batches,
            "Reward Type": r_type
        }
        
        # pf/ff limit: 1 / (1 + exp(c/T))
        # pp/fp limit: 1 / (1 + exp(c*k/T))
        if r_type in ['pf', 'ff']:
            theory_val = 1.0 / (1.0 + np.exp(c / t))
        else: # pp/fp
            theory_val = 1.0 / (1.0 + np.exp((c * k) / t))
            
        history = run_experiment_with_data(
            f"Stateless ({r_type.upper()})", 
            "stateless", 
            b, c, t, k, 
            reward_type=r_type, 
            n_batches=n_batches, 
            n_rounds=n_rounds
        )
        
        final_save_path = os.path.join(os.path.dirname(__file__), f"../results/stateless_{r_type}_experiment.png")
        
        plot_cooperation_with_std(
            [history], 
            [f"Stateless {r_type.upper()}"],
            title=f"Stateless Convergence: {r_type.upper()} Model",
            save_path=final_save_path,
            theory_values={f"Theory {r_type.upper()}": theory_val},
            experiment_info=experiment_info
        )
    
    print(f"\nAll 4 experiments completed. Results are in experiments/exp8/results/")
