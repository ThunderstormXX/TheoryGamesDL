import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config


def calculate_theoretical_q_diff(reward_type, c, degrees):
    """
    Calculates the theoretical difference Q(C) - Q(D) for a stateless agent in PD.
    Diff = R(C) - R(D) = - Cost_of_cooperation
    """
    if reward_type == 'pp':
        # Cost is c * k_i
        cost = c * degrees
    elif reward_type == 'pf':
        # Cost is c
        cost = torch.full_like(degrees, c)
    elif reward_type == 'ff':
        # Cost is c
        cost = torch.full_like(degrees, c)
    elif reward_type == 'fp':
        # Cost is c * k_i
        cost = c * degrees
        
    # Diff = - cost
    return -cost

def run_stateless_convergence_experiment(graph_type = 'star_graph'):
    output_dir = "experiments/exp8/results/stateless_convergence_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    batch_size = 1
    num_iterations = 5000
    gamma = 0.0 # ZERO gamma for stateless/myopic agent
    lr = 0.1
    c = 1.0
    epsilon = 0.1 # Constant epsilon to ensure visitation
    
    # Use a Star Graph for simple analysis
    # Node 0 is center (Hub), Nodes 1-4 are leaves.
    print("-"*50 + f"\n{graph_type} (Stateless Gamma=0)")

    match graph_type:
        case 'star_graph':
            num_nodes = 5
            graph = StarGraph(num_nodes=num_nodes, device=gpu_config.device)
        case 'wheel_graph':
            num_nodes = 5
            graph = WheelGraph(num_nodes=num_nodes, device=gpu_config.device)
        case 'small_world_graph':
            num_nodes = 15
            graph = SmallWorldGraph(num_nodes=num_nodes, device=gpu_config.device)
    adj_matrix = graph.generate_adjacency_matrix()
    degrees = torch.sum(adj_matrix, dim=1)
    
    print(f"Graph: {graph_type} with {num_nodes} nodes.")
    print(f"Degrees: {degrees}")
    

    # Reward types to test
    reward_types = ['pp', 'pf', 'ff', 'fp']
    b_values = [1.5, 3.0, 5.0, 10.0]
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for r_type in reward_types:
        print(f"Running experiment for reward type: {r_type}")
        
        # Storage for all B values
        all_b_results = {} # {b_val: delta_q_history}
        
        # Determine Theoretical Delta Q (Independent of b)
        th_delta_q = calculate_theoretical_q_diff(r_type, c, degrees)
        print(f"Theoretical Delta Q (Q(C)-Q(D)): {th_delta_q}")
        
        for b_idx, b_val in enumerate(b_values):
            print(f"  Running for b={b_val}")
            
            # Setup Agent
            # We use max_states=1 for pure stateless Q-learning
            # And discount_factor=0.0
            player_lr = lr

            learner = BatchedGPUQLearner(
                batch_size=batch_size, 
                n_agents=num_nodes,
                action_space_size=2,
                learning_rate=player_lr,
                discount_factor=gamma, # 0.0
                exploration_rate=epsilon,
                max_states=1 
            )
            
            reward_manager = RewardManager(reward_type=r_type, b=b_val, c=c)
            
            # Tracking
            # We track Delta Q = Q(C) - Q(D)
            delta_q_history = {i: [] for i in range(num_nodes)}
            
            # Initial State (all 0)
            states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=gpu_config.device)
            
            # Run Simulation
            for t in range(num_iterations):
                # 1. Get Actions
                actions = learner.get_actions(states)
                
                # 2. Calculate Rewards
                rewards = reward_manager.calculate_rewards(actions.float(), adj_matrix.unsqueeze(0), degrees.unsqueeze(0))
                
                # 3. Update
                next_states = torch.zeros_like(states)
                learner.update(states, actions, rewards, next_states)
                
                # Track Q(C) - Q(D) for State 0
                for n in range(num_nodes):
                    q_c = learner.q_table[0, n, 0, 1].item() # Action 1 (Cooperate)
                    q_d = learner.q_table[0, n, 0, 0].item() # Action 0 (Defect)
                    delta_q = q_c - q_d
                    delta_q_history[n].append(delta_q)
            
            all_b_results[b_val] = delta_q_history

        # Plotting for this reward type
        nodes_to_plot = min(num_nodes, 5)
        fig, axes = plt.subplots(nodes_to_plot, 1, figsize=(10, 3*nodes_to_plot), sharex=True)
        if nodes_to_plot == 1: axes = [axes]
        
        fig.suptitle(f"Delta Q Convergence (Stateless, Gamma=0) - {r_type.upper()}\n Graph type: {graph_type}", fontsize=16)
        
        for n in range(nodes_to_plot):
            ax = axes[n]
            
            # Plot actual Delta Q for each b
            for b_idx, b_val in enumerate(b_values):
                hist = all_b_results[b_val][n]
                ax.plot(hist, label=f'b={b_val}', color=colors[b_idx % len(colors)], alpha=0.6, linewidth=1.5)
            
            # Plot Theoretical Delta Q
            th_val = th_delta_q[n].item()
            ax.axhline(y=th_val, color='r', linestyle='--', linewidth=2.0, label=f'Theoretical: {th_val:.2f}')
            
            ax.set_ylabel(f"Agent {n} (k={int(degrees[n].item())})\nDelta Q")
            if n == 0:
                ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        plt.xlabel("Iterations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Ensure subdirectory exists
        param_dir = os.path.join(output_dir, graph_type)
        os.makedirs(param_dir, exist_ok=True)
        
        output_path = os.path.join(param_dir, f"stateless_delta_q_{r_type}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    run_stateless_convergence_experiment('star_graph')
    run_stateless_convergence_experiment('wheel_graph')
    run_stateless_convergence_experiment('small_world_graph')
