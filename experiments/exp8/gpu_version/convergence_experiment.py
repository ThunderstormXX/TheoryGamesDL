
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
    num_iterations = 50000
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
    b_values = [1.5, 3.0, 5.0, 10.0]
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for r_type in reward_types:
        print(f"Running experiment for reward type: {r_type}")
        
        # Storage for all B values
        all_b_results = {} # {b_val: delta_q_history}
        
        # Determine Theoretical Delta Q (Independent of b)
        th_delta_q = calculate_theoretical_q_diff(r_type, c, gamma, degrees)
        print(f"Theoretical Delta Q (Q(C)-Q(D)): {th_delta_q}")
        
        for b_idx, b_val in enumerate(b_values):
            print(f"  Running for b={b_val}")
            
            # Setup Agent
            # State space size depends on max degree (number of neighbors + 1)
            max_degree = int(degrees.max().item())
            state_space_size = max_degree + 1
            
            player_lr = lr
            if r_type == 'pf' or r_type == 'ff':
               # For these linear rewards, Q-values can get large if b is large, but relative diff is constant.
               pass

            learner = BatchedGPUQLearner(
                batch_size=batch_size, 
                n_agents=num_nodes,
                action_space_size=2,
                learning_rate=player_lr,
                discount_factor=gamma,
                exploration_rate=epsilon,
                max_states=state_space_size 
            )
            
            reward_manager = RewardManager(reward_type=r_type, b=b_val, c=c)
            
            # Tracking
            # We track Delta Q = Q(C) - Q(D) for ALL states over time
            # shape: (num_iterations, num_nodes, state_space_size)
            history_delta_q = np.zeros((num_iterations, num_nodes, state_space_size))
            
            # Initial State (all 0)
            states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=gpu_config.device)
            
            # Run Simulation
            for t in range(num_iterations):
                # 1. Get Actions
                actions = learner.get_actions(states)
                
                # 2. Calculate Rewards
                rewards = reward_manager.calculate_rewards(actions.float(), adj_matrix.unsqueeze(0), degrees.unsqueeze(0))
                
                # 3. Update
                # Calculate next states: number of cooperating neighbors
                next_states = torch.matmul(actions.float(), adj_matrix).long()
                
                learner.update(states, actions, rewards, next_states)
                
                # Record the full Delta Q-Table at this timestep
                # q_table is (B, N, S, A)
                current_q = learner.q_table[0].detach() # (N, S, A)
                # Compute Q(C) - Q(D) across all states
                diff = current_q[:, :, 1] - current_q[:, :, 0] # (N, S)
                
                # Store in history. 
                # Note: learner.q_table may cover up to max_states=max_degree+1.
                # history_delta_q is initialized with max_states size.
                history_delta_q[t] = diff.cpu().numpy()

                states = next_states
            
            
            # --- PLOTTING FOR THIS B_VAL AND AGENT ---
            # We plot the Time Series of Delta Q for EACH STATE for EACH AGENT.
            
            nodes_to_plot_indices = range(min(num_nodes, 5)) 
            
            # Create a Figure for THIS B_VAL
            # Subplots: One per agent
            fig, axes = plt.subplots(len(nodes_to_plot_indices), 1, figsize=(12, 5*len(nodes_to_plot_indices)), sharex=True)
            if len(nodes_to_plot_indices) == 1: axes = [axes] # Ensure iterable if only 1 node
            
            fig.suptitle(f"Delta Q Convergence by State - {r_type.upper()} b={b_val}\n{graph_type}", fontsize=16)

            for idx, n in enumerate(nodes_to_plot_indices):
                if isinstance(axes, np.ndarray):
                    ax = axes[idx]
                else:
                    ax = axes[idx]
                
                agent_degree = int(degrees[n].item())
                valid_states = range(agent_degree + 1) # 0 to k neighbors
                
                # Plot a line for each state
                for s in valid_states:
                    series = history_delta_q[:, n, s]
                    # Check if this state was ever visited/updated? 
                    # If it's always 0 (initial), it might clutter. 
                    # But usually epsilon ensures visitation.
                    # We plot all valid states.
                    ax.plot(series, label=f'State {s} (k={s})', linewidth=1.5, alpha=0.8)
                
                ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
                
                # Add theoretical horizontal line
                th_val = th_delta_q[n].item()
                ax.axhline(th_val, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Theoretical (-Cost)')

                ax.set_title(f"Agent {n} (Degree {agent_degree})")
                ax.set_ylabel("Delta Q (Q(C) - Q(D))")
                ax.grid(True, alpha=0.3)
                
                # Place legend outside or smartly
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Num Coops")

            plt.xlabel("Iterations")
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) # Make room for legend on the right
            
            save_dir = os.path.join(output_dir, "states_convergence_plots", graph_type, r_type)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"b_{b_val}_convergence_by_state.png"
            output_path = os.path.join(save_dir, filename)
            plt.savefig(output_path)
            plt.close()
            print(f"    Saved convergence plot to {output_path}")

        # (Original loop plotting code removed/commented out as requested - actually I will replace it)


if __name__ == "__main__":
    run_convergence_experiment('star_graph')
    run_convergence_experiment('wheel_graph')
    # run_convergence_experiment('small_world_graph')
    
