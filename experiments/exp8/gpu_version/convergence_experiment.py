
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


def calculate_theoretical_q_diff(reward_type, c, gamma, degrees):
    """
    Calculates the theoretical difference Q(C) - Q(D) for a stateless agent in PD.
    Diff = R(C) - R(D) = - Cost_of_cooperation
    Note: The future value (gamma * max(Q)) term cancels out because 
    the next state (and max Q) is independent of the current action in a stateless game.
    """
    bs = degrees.shape[0]
    
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

def run_convergence_experiment(graph_type = 'star_graph'):
    output_dir = "experiments/exp8/results/convergence_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    batch_size = 1
    num_iterations = 10000
    gamma = 0.9
    lr = 0.1
    b = 1.5
    c = 1.0
    epsilon = 0.1 # Constant epsilon to ensure visitation
    
    # Use a Star Graph for simple analysis
    # Node 0 is center (Hub), Nodes 1-4 are leaves.
    print("-"*50 + f"\n{graph_type}")

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
    
    print(f"Graph: Star Graph with {num_nodes} nodes.")
    print(f"Degrees: {degrees}")
    
    # Reward types to test
    reward_types = ['pp', 'pf', 'ff', 'fp']
    
    for r_type in reward_types:
        print(f"Running experiment for reward type: {r_type}")
        
        # Setup Agent
        # We use max_states=1 for pure stateless Q-learning
        learner = BatchedGPUQLearner(
            batch_size=batch_size, 
            n_agents=num_nodes,
            action_space_size=2,
            learning_rate=lr,
            discount_factor=gamma,
            exploration_rate=epsilon,
            max_states=1 
        )
        
        reward_manager = RewardManager(reward_type=r_type, b=b, c=c)
        
        # Tracking
        # We track Delta Q = Q(C) - Q(D)
        delta_q_history = {i: [] for i in range(num_nodes)}
        
        # Determine Theoretical Delta Q
        th_delta_q = calculate_theoretical_q_diff(r_type, c, gamma, degrees)
        print(f"Theoretical Delta Q (Q(C)-Q(D)): {th_delta_q}")
        
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

        # Plotting for this reward type
        # We create one plot per reward type, with subplots for agents? 
        # Or one plot with lines for all agents?
        # User asked: "graphics for each agent".
        # Let's do subplots for clarity.
        
        fig, axes = plt.subplots(min(num_nodes, 5), 1, figsize=(10, 3*min(num_nodes, 5)), sharex=True)
        if num_nodes == 1: axes = [axes]
        
        fig.suptitle(f"Delta Q (Q(C) - Q(D)) Convergence - {r_type.upper()}\n Graph type: {graph_type}", fontsize=16)
        
        for n in range(min(num_nodes, 5)):
            ax = axes[n]
            # Plot actual Delta Q
            ax.plot(delta_q_history[n], label=f'Agent {n} (deg={int(degrees[n].item())}) Simulated')
            
            # Plot Theoretical Delta Q
            th_val = th_delta_q[n].item()
            ax.axhline(y=th_val, color='r', linestyle='--', label=f'Theoretical: {th_val:.2f}')
            
            ax.set_ylabel("Delta Q")
            ax.legend()
            ax.grid(True)
            
        plt.xlabel("Iterations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(output_dir, f"{graph_type}/delta_q_{r_type}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    # run_convergence_experiment('star_graph')
    # run_convergence_experiment('wheel_graph')
    run_convergence_experiment('small_world_graph')
    
