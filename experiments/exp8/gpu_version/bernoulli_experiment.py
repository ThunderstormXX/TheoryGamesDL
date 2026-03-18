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

def bootstrap_ci(data, n_boot=1000, ci=95):
    """
    Computes the bootstrapped confidence interval for the mean of the data.
    Returns (mean, low_ci, high_ci)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    # Convert manually to numpy if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    means = np.array(means)
    low_p = (100 - ci) / 2
    high_p = 100 - low_p
    
    return np.mean(data), np.percentile(means, low_p), np.percentile(means, high_p)

def run_bernoulli_experiment():
    output_dir = "experiments/exp8/results/bernoulli_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Experiment Parameters
    num_iterations = 10000
    warmup_period = 5000 # Discard first N steps
    batch_size = 1 # Single simulation per config
    
    # Agent Parameters
    gamma = 0.9      # For stateful
    gamma_stateless = 0.0 # For stateless
    lr = 0.1
    c = 1.0
    epsilon = 0.05   # Small epsilon to allow some noise but mostly exploit
    
    b_values = [1.2, 1.5, 2.0, 3.0, 5.0]
    graph_types = ['star_graph', 'wheel_graph'] #, 'small_world_graph']
    modes = ['state', 'stateless']
    r_type = 'pp' # Standard PD
    
    results_summary = []

    print(f"Starting Bernoulli Cooperation Experiment")
    print(f"Iterations: {num_iterations}, Warmup: {warmup_period}")
    
    for graph_type in graph_types:
        print(f"\nGraph: {graph_type}")
        
        # Setup Graph
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
        
        for mode in modes:
            print(f"  Mode: {mode}")
            
            for b_val in b_values:
                print(f"    b = {b_val}")
                
                # Determine Agent Config
                if mode == 'state':
                    max_degree = int(degrees.max().item())
                    max_states = max_degree + 1
                    current_gamma = gamma
                else: # stateless
                    max_states = 1
                    current_gamma = gamma_stateless
                
                # Setup Learner
                learner = BatchedGPUQLearner(
                    batch_size=batch_size, 
                    n_agents=num_nodes,
                    action_space_size=2,
                    learning_rate=lr,
                    discount_factor=current_gamma,
                    exploration_rate=epsilon,
                    max_states=max_states 
                )
                
                reward_manager = RewardManager(reward_type=r_type, b=b_val, c=c)
                
                # Initial State
                if mode == 'state':
                    # Initially 0 neighbors correspond to state 0 (or some assumption)
                    # We start with all Defect actions effectively, so state is 0
                    states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=gpu_config.device)
                else:
                    states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=gpu_config.device)
                    
                # Data Collection
                # We want to store actions after warmup
                # Shape: (num_samples, num_agents)
                collected_actions = []
                
                for t in range(num_iterations):
                    # 1. Get Actions
                    actions = learner.get_actions(states) # (B, N)
                    
                    # 2. Rewards
                    rewards = reward_manager.calculate_rewards(actions.float(), adj_matrix.unsqueeze(0), degrees.unsqueeze(0))
                    
                    # 3. Next State
                    if mode == 'state':
                        # Count cooperating neighbors
                        next_states = torch.matmul(actions.float(), adj_matrix).long()
                    else:
                        # Always 0
                        next_states = torch.zeros_like(states)
                    
                    learner.update(states, actions, rewards, next_states)
                    states = next_states
                    
                    # Collect data if past warmup
                    if t >= warmup_period:
                        # actions is (1, N)
                        collected_actions.append(actions.cpu().numpy()[0])
                
                # Process Results for this Config
                print(f"      Processing {len(collected_actions)} samples...")
                
                collected_actions = np.array(collected_actions) # (Samples, N)
                
                # For each agent, calculate p and CI
                for agent_idx in range(num_nodes):
                    agent_history = collected_actions[:, agent_idx]
                    mean_p, ci_low, ci_high = bootstrap_ci(agent_history)
                    
                    # Agent role description
                    if graph_type == 'star_graph':
                        role = "Hub" if agent_idx == 0 else "Leaf"
                    elif graph_type == 'wheel_graph':
                        role = "Hub" if agent_idx == 0 else "Rim"
                    else:
                        role = f"Node_{agent_idx}"
                        
                    result_entry = {
                        "graph": graph_type,
                        "mode": mode,
                        "b": b_val,
                        "agent_id": agent_idx,
                        "agent_role": role,
                        "degree": degrees[agent_idx].item(),
                        "p_mean": mean_p,
                        "ci_lower": ci_low,
                        "ci_upper": ci_high
                    }
                    results_summary.append(result_entry)
                    # print(f"        Agent {agent_idx} ({role}): p={mean_p:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # Visualization of Results
    plot_results(results_summary, b_values, output_dir)
    save_results_csv(results_summary, output_dir)
    print(f"Experiment Complete. Results saved to {output_dir}")

def save_results_csv(results, output_dir):
    import csv
    if not results:
        return
    
    keys = results[0].keys()
    with open(os.path.join(output_dir, "bernoulli_results.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results, b_values, output_dir):
    # We want to plot p with error bars for each agent type vs b, for each graph and mode
    
    # Organize data manually
    # structure: data[graph][mode][role][b] -> list of {p_mean, ci_lower, ci_upper}
    
    data_struct = {}
    
    for row in results:
        g = row['graph']
        m = row['mode']
        r = row['agent_role']
        b = row['b']
        
        if g not in data_struct: data_struct[g] = {}
        if m not in data_struct[g]: data_struct[g][m] = {}
        if r not in data_struct[g][m]: data_struct[g][m][r] = {}
        if b not in data_struct[g][m][r]: data_struct[g][m][r][b] = []
        
        data_struct[g][m][r][b].append(row)
        
    # Plotting
    for g in data_struct:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f"Cooperation Rate (p) with 95% CI - {g}", fontsize=16)
        
        modes = ['state', 'stateless']
        for idx, m in enumerate(modes):
            ax = axes[idx]
            
            if m not in data_struct[g]:
                continue
                
            roles = list(data_struct[g][m].keys())
            
            for role in roles:
                role_data_by_b = data_struct[g][m][role]
                
                # Sort b values
                bs = sorted(role_data_by_b.keys())
                
                means = []
                lower_errs = []
                upper_errs = []
                
                for b_val in bs:
                    # Average over agents of same role if multiple
                    entries = role_data_by_b[b_val]
                    # We average the means and CIs (approximate visualization)
                    p_mean = np.mean([e['p_mean'] for e in entries])
                    ci_low = np.mean([e['ci_lower'] for e in entries])
                    ci_high = np.mean([e['ci_upper'] for e in entries])
                    
                    means.append(p_mean)
                    lower_errs.append(p_mean - ci_low)
                    upper_errs.append(ci_high - p_mean)
                
                # Plot
                ax.errorbar(bs, means, yerr=[lower_errs, upper_errs], label=role, capsize=5, marker='o')
            
            # Theoretical baseline for Epsilon-Greedy with Epsilon=0.05
            theoretical_p = 0.05 / 2.0
            ax.axhline(y=theoretical_p, color='red', linestyle='--', alpha=0.7, label=f'Theoretical ({theoretical_p})')
            
            ax.set_title(f"Mode: {m}")
            ax.set_xlabel("Benefit (b)")
            ax.set_ylabel("Cooperation Probability p")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"p_values_{g}.png"))
        plt.close()

def load_results_csv(file_path):
    import csv
    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric types
            row['b'] = float(row['b'])
            row['degree'] = float(row['degree'])
            row['p_mean'] = float(row['p_mean'])
            row['ci_lower'] = float(row['ci_lower'])
            row['ci_upper'] = float(row['ci_upper'])
            results.append(row)
    return results

if __name__ == "__main__":
    # If standard run is needed:
    # run_bernoulli_experiment()
    
    # For plotting existing results without re-running simulation:
    output_dir = "experiments/exp8/results/bernoulli_test"
    csv_path = os.path.join(output_dir, "bernoulli_results.csv")
    
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}...")
        results = load_results_csv(csv_path)
        plot_results(results, None, output_dir)
        print(f"Plots updated in {output_dir}")
    else:
        print("No existing results found. Please run the experiment first.")
        # run_bernoulli_experiment()