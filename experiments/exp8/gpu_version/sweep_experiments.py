import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Refactored imports
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.visualization.plotting import plot_cooperation_with_std

def run_experiment_with_data(name, state_type, b, c, t, k, reward_type='pf', n_batches=100, n_rounds=1000, n_anchors=None, gamma=0.9):
    print(f"  > {name}: b={b}, gamma={gamma}, rounds={n_rounds}")
    
    batch_size = n_batches
    n_agents = 200
    k_neighbors = k
    
    graph_params = {'k': k_neighbors, 'p': 0.1}
    max_states = (k_neighbors + 1) if state_type == 'neighbor_coop' else 1
    
    learner_params = {
        'learning_rate': 0.05,
        'discount_factor': gamma,
        'exploration_rate': 0.1,
        'strategy': 'softmax',
        'temperature': t,
        'max_states': max_states
    }
    
    reward_params = {'b': b, 'c': c}
    
    game = BatchedGPUMonteKarloPairGame(
        batch_size, n_agents, 
        graph_params, learner_params, reward_params,
        reward_type=reward_type,
        n_anchors=n_anchors
    )
    
    # Calculate average degree from the graph
    avg_degree = game.degrees.mean().item()

    # Stateless simulation override
    if state_type == 'stateless':
        def stateless_get_states(actions):
            return torch.zeros((batch_size, n_agents), device=game.device, dtype=torch.long)
        game._get_states = stateless_get_states
        game.current_states = game._get_states(game.current_actions)

    history_tensor = torch.zeros(n_rounds, batch_size)
    
    for i in range(n_rounds):
        metrics = game.round()
        history_tensor[i] = metrics['batch_coop_rates'].cpu()
            
    return history_tensor.numpy(), game.degrees.cpu().numpy()

if __name__ == "__main__":
    c, t = 1, 1
    k = 4
    n_batches = 100
    n_anchors = None
    
    # Setup results directory
    base_res_dir = os.path.join(os.path.dirname(__file__), "../results/sweeps/")
    os.makedirs(base_res_dir, exist_ok=True)

    # --- Experiment 1: PF reward with different gamma values grouped by b ---
    b_values = [1.5, 2.5, 3.5]
    gamma_values = [0.5, 0.9, 0.99]
    n_rounds_pf = 2000 
    
    
    
    for reward_model in ['pf', 'ff', 'pp', 'fp']:
        pf_dir = os.path.join(base_res_dir, f"{reward_model}_gamma_sweep")
        os.makedirs(pf_dir, exist_ok=True)
        print(f"\nRunning {reward_model.upper()} experiments (grouped by b)...")
        for b in b_values:
            histories = []
            labels = []
            
            for gamma in gamma_values:
                name = f"gamma={gamma}"
                history, degrees = run_experiment_with_data(
                    f"{reward_model.upper()}_b{b}_g{gamma}", "stateless", 
                    b, c, t, k, 
                    reward_type=reward_model, 
                    n_batches=n_batches, 
                    n_rounds=n_rounds_pf,
                    gamma=gamma
                )
                histories.append(history)
                labels.append(name)
            save_path = os.path.join(pf_dir, f"{reward_model}_b{b}_gamma_comparison.png")
            
            if reward_model in ['pf', 'ff']:
                theory_val = 1.0 / (1.0 + np.exp(c / t))
            else:
                # Sum over all nodes in all batches for the exact theoretical average
                # theory_i = 1 / (1 + exp(c * k_i / T))
                # degrees: (n_batches, n_agents)
                theory_per_node = 1.0 / (1.0 + np.exp((c * degrees) / t))
                theory_val = np.mean(theory_per_node)

            plot_cooperation_with_std(
                histories, 
                labels,
                title=f"{reward_model.upper()} Reward: Gamma sweep for b={b}",
                save_path=save_path,
                theory_values={f"Theory {reward_model.upper()}": theory_val},
                experiment_info={"b": b, "c": c, "T": t, "K": k, "Rounds": n_rounds_pf}
            )

    # --- Experiment 2: Other reward types with 1000 rounds ---
    other_reward_types = ['pp', 'ff', 'fp']
    n_rounds_others = 1000
    b_fixed = 3.0
    gamma_fixed = 0.9
    
    others_dir = os.path.join(base_res_dir, "others")
    os.makedirs(others_dir, exist_ok=True)

    print("\nRunning experiments for other reward types (1000 rounds)...")
    for r_type in other_reward_types:
        history, degrees = run_experiment_with_data(
            f"{r_type}_fixed", "stateless", 
            b_fixed, c, t, k, 
            reward_type=r_type, 
            n_batches=n_batches, 
            n_rounds=n_rounds_others,
            gamma=gamma_fixed
        )
        
        if r_type in ['pf', 'ff']:
            theory_val = 1.0 / (1.0 + np.exp(c / t))
        else:
            # Exact sum over all nodes
            theory_per_node = 1.0 / (1.0 + np.exp((c * degrees) / t))
            theory_val = np.mean(theory_per_node)
            
        save_path = os.path.join(others_dir, f"stateless_{r_type}_1000.png")
        
        plot_cooperation_with_std(
            [history], 
            [f"Stateless {r_type.upper()}"],
            title=f"Stateless {r_type.upper()}: {n_rounds_others} rounds",
            save_path=save_path,
            theory_values={f"Theory {r_type.upper()}": theory_val},
            experiment_info={"b": b_fixed, "gamma": gamma_fixed, "rounds": n_rounds_others}
        )
    
    print(f"\nAll experiments completed. Results saved to {base_res_dir}")
