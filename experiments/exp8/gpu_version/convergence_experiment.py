import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config

N_WORKERS = max(1, cpu_count() - 2) if cpu_count() > 4 else 4
print(f"Using {N_WORKERS} parallel workers")


def calculate_theoretical_q_diff(reward_type, c, gamma, degrees):
    if reward_type == 'pp':
        cost = c * degrees
    elif reward_type == 'pf':
        cost = torch.full_like(degrees, c)
    elif reward_type == 'ff':
        cost = torch.full_like(degrees, c)
    elif reward_type == 'fp':
        cost = c * degrees
    return -cost


def run_single_simulation(args):
    """
    Worker function. All parameters passed via args tuple.
    """
    (b_val, seed, graph_type, r_type, num_nodes, adj_matrix_np, degrees_np,
     num_iterations, warmup_start, gamma, lr, c, epsilon) = args
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    adj_matrix = torch.from_numpy(adj_matrix_np).float().to(gpu_config.device)
    degrees = torch.from_numpy(degrees_np).float().to(gpu_config.device)
    
    max_degree = int(degrees.max().item())
    state_space_size = max_degree + 1

    learner = BatchedGPUQLearner(
        batch_size=1,
        n_agents=num_nodes,
        action_space_size=2,
        learning_rate=lr,
        discount_factor=gamma,
        exploration_rate=epsilon,
        max_states=state_space_size
    )
    
    reward_manager = RewardManager(reward_type=r_type, b=b_val, c=c)
    
    history_delta_q = np.zeros((num_iterations, num_nodes, state_space_size))
    cooperation_history = np.zeros((num_iterations, num_nodes), dtype=bool)
    
    states = torch.zeros((1, num_nodes), dtype=torch.long, device=gpu_config.device)
    
    for t in range(num_iterations):
        actions = learner.get_actions(states)
        cooperation_history[t] = (actions.cpu().numpy()[0] == 1)
        
        rewards = reward_manager.calculate_rewards(
            actions.float(), 
            adj_matrix.unsqueeze(0), 
            degrees.unsqueeze(0)
        )
        
        next_states = torch.matmul(actions.float(), adj_matrix).long()
        learner.update(states, actions, rewards, next_states)
        
        current_q = learner.q_table[0].detach()
        diff = current_q[:, :, 1] - current_q[:, :, 0]
        history_delta_q[t] = diff.cpu().numpy()
        states = next_states
    
    post_warmup_coop = cooperation_history[warmup_start:]
    coop_rate = np.mean(post_warmup_coop, axis=0)
    
    return {
        'b_val': b_val,
        'seed': seed,
        'history_delta_q': history_delta_q,
        'coop_rate': coop_rate,
    }


def run_convergence_experiment(graph_type='star_graph'):
    output_dir = "experiments/exp8/results/convergence_test"
    os.makedirs(output_dir, exist_ok=True)
    
    num_iterations = 50000
    warmup_start = int(num_iterations * 0.8)
    gamma = 0.9
    lr = 0.1
    c = 1.0
    epsilon = 0.1
    n_replications = 5
    
    print("-" * 50 + f"\n{graph_type}")

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
    
    adj_matrix_np = adj_matrix.cpu().numpy()
    degrees_np = degrees.cpu().numpy()
    
    print(f"Graph: {graph_type} with {num_nodes} nodes.")
    print(f"Degrees: {degrees}")

    reward_types = ['pp', 'pf', 'ff', 'fp']
    b_values = [1.5, 3.0, 5.0, 10.0]
    
    for r_type in reward_types:
        print(f"\nReward type: {r_type}")
        
        th_delta_q = calculate_theoretical_q_diff(r_type, c, gamma, degrees)
        print(f"Theoretical Delta Q: {th_delta_q}")
        
        # Build task list
        all_tasks = []
        for b_val in b_values:
            for seed in range(n_replications):
                task = (
                    b_val, seed, graph_type, r_type, num_nodes, 
                    adj_matrix_np, degrees_np,
                    num_iterations, warmup_start, gamma, lr, c, epsilon
                )
                all_tasks.append(task)
        
        print(f"  Running {len(all_tasks)} parallel simulations...")
        
        # Execute with progress bar
        with Pool(processes=N_WORKERS) as pool:
            # Use imap_unordered with tqdm for progress bar
            results = list(tqdm(
                pool.imap_unordered(run_single_simulation, all_tasks),
                total=len(all_tasks),
                desc=f"{r_type:2s}",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ))
        
        # Aggregate by b_val
        aggregated = {b: [] for b in b_values}
        for res in results:
            aggregated[res['b_val']].append(res)
        
        cooperation_rates = {}
        avg_delta_q_history = {}
        
        for b_val in b_values:
            replicates = aggregated[b_val]
            coop_rates = np.array([r['coop_rate'] for r in replicates])
            cooperation_rates[b_val] = np.mean(coop_rates, axis=0)
            avg_delta_q_history[b_val] = replicates[0]['history_delta_q']
            print(f"  b={b_val}: coop_rates = {cooperation_rates[b_val].round(3)}")
        
        # Plotting with progress bar
        print("  Plotting...")
        for b_val in tqdm(b_values, desc="Plots", ncols=80, leave=False):
            plot_convergence_by_state(
                avg_delta_q_history[b_val], 
                cooperation_rates[b_val],
                b_val, r_type, graph_type, gamma, 
                degrees, th_delta_q, output_dir
            )
        
        plot_cooperation_rate(cooperation_rates, b_values, degrees, r_type, graph_type, output_dir, epsilon)
        print(f"  ✓ {r_type} complete")


def plot_convergence_by_state(history_delta_q, coop_rates, b_val, r_type, graph_type, 
                              gamma, degrees, th_delta_q, output_dir):
    num_nodes = len(degrees)
    nodes_to_plot_indices = range(min(num_nodes, 5))
    
    fig, axes = plt.subplots(len(nodes_to_plot_indices), 1, 
                            figsize=(12, 5 * len(nodes_to_plot_indices)), 
                            sharex=True)
    if len(nodes_to_plot_indices) == 1:
        axes = [axes]
    
    fig.suptitle(f"Delta Q Convergence by State - {r_type.upper()} b={b_val}\n{graph_type}, γ={gamma}", 
                 fontsize=16)

    for idx, n in enumerate(nodes_to_plot_indices):
        ax = axes[idx]
        agent_degree = int(degrees[n].item())
        valid_states = range(agent_degree + 1)
        
        for s in valid_states:
            series = history_delta_q[:, n, s]
            ax.plot(series, label=f'State {s} (k={s})', linewidth=1.5, alpha=0.8)
        
        ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
        
        th_val = th_delta_q[n].item()
        ax.axhline(th_val, color='red', linestyle='--', alpha=0.8, 
                   linewidth=1.5, label=f'Theoretical (-Cost)')

        ax.set_title(f"Agent {n} (Degree {agent_degree}, p={coop_rates[n]:.3f})")
        ax.set_ylabel("Delta Q (Q(C) - Q(D))")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize='small', title="Num Coops")

    plt.xlabel("Iterations")
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    
    save_dir = os.path.join(output_dir, "states_convergence_plots", graph_type, r_type)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"b_{b_val}_convergence_by_state.png"
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cooperation_rate(cooperation_rates, b_values, degrees, r_type, graph_type, 
                          output_dir, epsilon):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_nodes = len(degrees)
    bs = sorted(cooperation_rates.keys())
    
    coop_by_agent = {i: [] for i in range(num_nodes)}
    for b in bs:
        rates = cooperation_rates[b]
        for i in range(num_nodes):
            coop_by_agent[i].append(rates[i])
    
    colors = {'Hub': '#d62728', 'Leaf': '#2ca02c', 'Rim': '#ff7f0e'}
    
    for i in range(num_nodes):
        agent_degree = int(degrees[i].item())
        
        if graph_type == 'star_graph':
            role = "Hub" if i == 0 else "Leaf"
        elif graph_type == 'wheel_graph':
            role = "Hub" if i == 0 else "Rim"
        else:
            role = f"Node_{i}"
        
        color = colors.get(role, plt.cm.tab10(i))
        
        ax.plot(bs, coop_by_agent[i], marker='o', linewidth=2, markersize=8,
                label=f'{role} (deg={agent_degree})', color=color, alpha=0.8)
    
    theoretical_p = epsilon * 0.5
    ax.axhline(y=theoretical_p, color='black', linestyle='--', alpha=0.5,
               label=f'Random baseline ({theoretical_p:.3f})')
    
    ax.set_xlabel('Benefit (b)', fontsize=12)
    ax.set_ylabel('Cooperation Rate p', fontsize=12)
    ax.set_title(f'Cooperation Rate vs Benefit\n{graph_type} | {r_type.upper()} | γ={0.9}', 
                 fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "cooperation_rate_plots", graph_type)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{r_type}_cooperation_rate.png"
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    import time
    start_time = time.time()
    
    run_convergence_experiment('star_graph')
    run_convergence_experiment('wheel_graph')
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Workers used: {N_WORKERS}")