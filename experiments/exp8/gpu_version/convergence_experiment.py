import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config

# A100 Optimization: Use all resources
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Use TF32 when beneficial

# Detect A100 and optimize
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"✓ GPU: {GPU_NAME}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # A100 has 80GB or 40GB - use large batch processing
    MAX_BATCH_SIZE = 32 if "A100" in GPU_NAME else 8
else:
    DEVICE = torch.device("cpu")
    MAX_BATCH_SIZE = 1
    print("⚠ No GPU detected")

# Override gpu_config
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager

# CPU workers for data processing
N_WORKERS = min(32, cpu_count())  # A100 systems usually have many CPU cores

# AGGRESSIVE PARAMETERS FOR MAXIMUM RESULTS
N_REPLICATIONS = 100       # Statistical power
NUM_ITERATIONS = 100000    # Long convergence
WARMUP_PERIOD = 80000      # 80% warmup
B_VALUES = [1.2, 1.3, 1.5, 1.8, 2.0, 3.0, 5.0, 7.0, 10.0]  # Extended range
GRAPH_TYPES = ['star_graph', 'wheel_graph', 'small_world_graph']
REWARD_TYPES = ['pp', 'pf', 'ff', 'fp']


def calculate_theoretical_q_diff(reward_type, c, degrees):
    if reward_type in ['pp', 'fp']:
        cost = c * degrees
    else:
        cost = torch.full_like(degrees, c)
    return -cost


class BatchedA100Experiment:
    """
    Batched experiment runner for A100 - processes multiple replications 
    simultaneously on GPU for maximum throughput.
    """
    def __init__(self, batch_size, num_nodes, max_states, lr, gamma, epsilon, r_type, b_val, c, adj_matrix, degrees):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        
        self.learner = BatchedGPUQLearner(
            batch_size=batch_size,
            n_agents=num_nodes,
            action_space_size=2,
            learning_rate=lr,
            discount_factor=gamma,
            exploration_rate=epsilon,
            max_states=max_states
        )
        
        self.reward_manager = RewardManager(reward_type=r_type, b=b_val, c=c)
        self.adj_matrix = adj_matrix
        self.degrees = degrees
        
        # Pre-allocate buffers on GPU
        self.states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=DEVICE)
        self.action_buffer = torch.zeros((NUM_ITERATIONS, batch_size, num_nodes), dtype=torch.bool, device=DEVICE)
        self.delta_q_buffer = torch.zeros((NUM_ITERATIONS, batch_size, num_nodes, max_states), device=DEVICE)
        
    def run(self):
        for t in range(NUM_ITERATIONS):
            actions = self.learner.get_actions(self.states)
            self.action_buffer[t] = (actions == 1)
            
            rewards = self.reward_manager.calculate_rewards(
                actions.float(),
                self.adj_matrix.unsqueeze(0).expand(self.batch_size, -1, -1),
                self.degrees.unsqueeze(0).expand(self.batch_size, -1)
            )
            
            next_states = torch.matmul(actions.float(), self.adj_matrix).long()
            self.learner.update(self.states, actions, rewards, next_states)
            
            # Record Delta Q for all states
            current_q = self.learner.q_table  # (B, N, S, A)
            delta_q = current_q[:, :, :, 1] - current_q[:, :, :, 0]  # (B, N, S)
            self.delta_q_buffer[t] = delta_q
            
            self.states = next_states
        
        # Calculate cooperation rates (after warmup)
        post_warmup = self.action_buffer[WARMUP_PERIOD:]
        coop_rates = post_warmup.float().mean(dim=0).cpu().numpy()  # (B, N)
        
        # Move Delta Q history to CPU
        delta_q_history = self.delta_q_buffer.cpu().numpy()  # (T, B, N, S)
        
        return coop_rates, delta_q_history


def run_single_simulation_optimized(args):
    """
    Optimized worker for A100 - uses batched GPU execution.
    """
    (rep_batch, b_val, graph_type, r_type, num_nodes, adj_matrix_np, 
     degrees_np, mode, gamma) = args
    
    # Each worker handles a batch of replications
    batch_size = len(rep_batch)
    
    torch.cuda.set_device(0)  # A100 usually single GPU per process
    torch.manual_seed(rep_batch[0])
    
    adj_matrix = torch.from_numpy(adj_matrix_np).float().to(DEVICE)
    degrees = torch.from_numpy(degrees_np).float().to(DEVICE)
    
    max_degree = int(degrees.max().item())
    max_states = max_degree + 1 if mode == 'state' else 1
    actual_gamma = gamma if mode == 'state' else 0.0
    
    # Run batched experiment
    experiment = BatchedA100Experiment(
        batch_size=batch_size,
        num_nodes=num_nodes,
        max_states=max_states,
        lr=0.1,
        gamma=actual_gamma,
        epsilon=0.05,
        r_type=r_type,
        b_val=b_val,
        c=1.0,
        adj_matrix=adj_matrix,
        degrees=degrees
    )
    
    coop_rates, delta_q_history = experiment.run()
    
    # Return individual results
    results = []
    for i, rep in enumerate(rep_batch):
        results.append({
            'rep': rep,
            'b_val': b_val,
            'coop_rate': coop_rates[i],
            'delta_q_history': delta_q_history[:, i, :, :],  # (T, N, S)
        })
    
    return results


def run_convergence_experiment_a100(graph_type='star_graph'):
    output_dir = "experiments/exp8/results/a100_maximum"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"MAXIMUM A100 EXPERIMENT: {graph_type}")
    print(f"{'='*70}")
    print(f"Replications: {N_REPLICATIONS}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"B values: {B_VALUES}")
    print(f"Reward types: {REWARD_TYPES}")
    
    match graph_type:
        case 'star_graph':
            num_nodes = 5
            graph = StarGraph(num_nodes=num_nodes, device=DEVICE)
        case 'wheel_graph':
            num_nodes = 5
            graph = WheelGraph(num_nodes=num_nodes, device=DEVICE)
        case 'small_world_graph':
            num_nodes = 50  # Larger for small-world
            graph = SmallWorldGraph(num_nodes=num_nodes, device=DEVICE)
    
    adj_matrix = graph.generate_adjacency_matrix()
    degrees = torch.sum(adj_matrix, dim=1)
    
    adj_matrix_np = adj_matrix.cpu().numpy()
    degrees_np = degrees.cpu().numpy()
    
    print(f"Nodes: {num_nodes}")
    print(f"Degrees: {degrees.cpu().numpy()}")
    
    modes = ['state', 'stateless']
    
    for r_type in REWARD_TYPES:
        for mode in modes:
            print(f"\n{'-'*50}")
            print(f"Reward: {r_type}, Mode: {mode}")
            
            th_delta_q = calculate_theoretical_q_diff(r_type, 1.0, degrees)
            
            # Process all B values
            all_results = {b: [] for b in B_VALUES}
            
            for b_val in B_VALUES:
                # Create batches for GPU efficiency
                # Each batch processes multiple replications simultaneously
                batch_size = min(MAX_BATCH_SIZE, N_REPLICATIONS)
                n_batches = (N_REPLICATIONS + batch_size - 1) // batch_size
                
                print(f"  b={b_val}: {n_batches} batches of size ≤{batch_size}")
                
                # Prepare batch tasks
                batch_tasks = []
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, N_REPLICATIONS)
                    rep_batch = list(range(start_idx, end_idx))
                    
                    batch_tasks.append((
                        rep_batch, b_val, graph_type, r_type, num_nodes,
                        adj_matrix_np, degrees_np, mode, 0.9
                    ))
                
                # Execute batches with progress bar
                with tqdm(total=len(batch_tasks), desc=f"b={b_val}", ncols=80) as pbar:
                    for task in batch_tasks:
                        batch_results = run_single_simulation_optimized(task)
                        for res in batch_results:
                            all_results[b_val].append(res)
                        pbar.update(1)
                        # Clear GPU cache between batches
                        torch.cuda.empty_cache()
            
            # Aggregate and plot
            cooperation_rates = {}
            all_delta_q_histories = {}
            
            for b_val in B_VALUES:
                replicates = all_results[b_val]
                coop_rates = np.array([r['coop_rate'] for r in replicates])
                cooperation_rates[b_val] = {
                    'mean': np.mean(coop_rates, axis=0),
                    'std': np.std(coop_rates, axis=0),
                    'all': coop_rates  # (n_reps, n_nodes)
                }
                all_delta_q_histories[b_val] = np.array([r['delta_q_history'] for r in replicates])
            
            # Generate all plots
            print("  Generating plots...")
            
            # 1. Standard convergence plots
            for b_val in tqdm(B_VALUES, desc="Convergence", leave=False):
                plot_convergence_a100(
                    all_delta_q_histories[b_val],
                    cooperation_rates[b_val],
                    b_val, r_type, mode, graph_type, 
                    degrees.cpu().numpy(), th_delta_q.cpu().numpy(), output_dir
                )
            
            # 2. NEW: Delta Q distribution over time
            for b_val in tqdm(B_VALUES, desc="Distributions", leave=False):
                plot_delta_q_distribution_a100(
                    all_delta_q_histories[b_val],
                    b_val, r_type, mode, graph_type,
                    degrees.cpu().numpy(), output_dir
                )
            
            # 3. Cooperation rate summary
            plot_cooperation_rate_a100(
                cooperation_rates, B_VALUES, degrees.cpu().numpy(),
                r_type, mode, graph_type, output_dir
            )
            
            # 4. Statistical summary plot
            plot_statistical_summary(
                cooperation_rates, all_delta_q_histories, B_VALUES,
                r_type, mode, graph_type, output_dir
            )


def plot_convergence_a100(delta_q_histories, coop_stats, b_val, r_type, mode, 
                          graph_type, degrees, th_delta_q, output_dir):
    """
    Plot mean Delta Q convergence with confidence bands.
    delta_q_histories: (n_reps, T, N, S)
    """
    n_reps, T, N, S = delta_q_histories.shape
    
    # Average over replications
    mean_delta_q = np.mean(delta_q_histories, axis=0)  # (T, N, S)
    std_delta_q = np.std(delta_q_histories, axis=0)
    
    nodes_to_plot = min(N, 5)
    states_to_plot = min(S, 3)  # Limit states for clarity
    
    fig, axes = plt.subplots(nodes_to_plot, 1, figsize=(12, 4 * nodes_to_plot), sharex=True)
    if nodes_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(f"ΔQ Convergence - {graph_type} | {r_type} | {mode} | b={b_val}\n"
                 f"Mean ± std over {n_reps} replications", fontsize=14)
    
    for n in range(nodes_to_plot):
        ax = axes[n]
        agent_degree = int(degrees[n])
        
        for s in range(min(agent_degree + 1, states_to_plot)):
            mean_traj = mean_delta_q[:, n, s]
            std_traj = std_delta_q[:, n, s]
            
            ax.plot(mean_traj, linewidth=1.5, alpha=0.8, label=f'State {s}')
            ax.fill_between(range(T), mean_traj - std_traj, mean_traj + std_traj,
                           alpha=0.2)
        
        # Theoretical line
        ax.axhline(th_delta_q[n], color='r', linestyle='--', alpha=0.8,
                  label=f'Theory: {th_delta_q[n]:.2f}')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        mean_coop = coop_stats['mean'][n]
        std_coop = coop_stats['std'][n]
        ax.set_title(f"Agent {n} (k={agent_degree}, p={mean_coop:.3f}±{std_coop:.3f})")
        ax.set_ylabel("ΔQ")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.xlabel("Iterations")
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "convergence", graph_type, r_type, mode)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"b_{b_val}.png"), dpi=150)
    plt.close()


def plot_delta_q_distribution_a100(delta_q_histories, b_val, r_type, mode, 
                                   graph_type, degrees, output_dir):
    """
    Plot distribution of Delta Q over time - A100 version with high resolution.
    """
    n_reps, T, N, S = delta_q_histories.shape
    
    # Time points: logarithmic progression for early dynamics + linear for late
    time_points = np.unique(np.concatenate([
        np.logspace(0, np.log10(T//10), 10).astype(int),  # Early dynamics
        np.linspace(T//10, T-1, 10).astype(int)  # Late convergence
    ]))
    time_points = np.clip(time_points, 0, T-1)
    time_labels = [f"t={t}" for t in time_points]
    
    nodes_to_plot = min(N, 5)
    
    # Create figure with subplots
    n_cols = len(time_points)
    n_rows = nodes_to_plot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows),
                            sharex='row', sharey='row')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'ΔQ Distribution Evolution - {graph_type} | {r_type} | {mode} | b={b_val}\n'
                 f'{n_reps} replications', fontsize=14)
    
    # Global range for consistent bins
    all_vals = delta_q_histories[:, :, :nodes_to_plot, :].flatten()
    vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
    
    for row, n in enumerate(range(nodes_to_plot)):
        for col, (t, label) in enumerate(zip(time_points, time_labels)):
            ax = axes[row, col]
            
            # Collect all states for this agent at time t
            agent_vals = delta_q_histories[:, t, n, :].flatten()
            
            # Kernel density estimate + histogram
            ax.hist(agent_vals, bins=50, density=True, alpha=0.6,
                   color='steelblue', range=(vmin, vmax))
            
            # Add statistics
            mean_val = np.mean(agent_vals)
            median_val = np.median(agent_vals)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'μ={mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5,
                      label=f'med={median_val:.2f}')
            
            if row == 0:
                ax.set_title(label, fontsize=9)
            
            if col == 0:
                agent_degree = int(degrees[n])
                ax.set_ylabel(f'Agent {n} (k={agent_degree})\nDensity', fontsize=8)
            
            if row == n_rows - 1:
                ax.set_xlabel('ΔQ', fontsize=8)
            
            ax.set_xlim(vmin, vmax)
            ax.tick_params(axis='both', labelsize=6)
            ax.legend(loc='upper right', fontsize=5)
            ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "distributions", graph_type, r_type, mode)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"b_{b_val}_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_cooperation_rate_a100(cooperation_stats, b_values, degrees, r_type, mode, 
                                graph_type, output_dir):
    """
    Plot cooperation rate with error bars from replications.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    N = len(degrees)
    bs = sorted(b_values)
    
    colors = {'Hub': '#d62728', 'Leaf': '#2ca02c', 'Rim': '#ff7f0e'}
    
    for i in range(N):
        agent_degree = int(degrees[i])
        
        means = [cooperation_stats[b]['mean'][i] for b in bs]
        stds = [cooperation_stats[b]['std'][i] for b in bs]
        
        if graph_type == 'star_graph':
            role = "Hub" if i == 0 else "Leaf"
        elif graph_type == 'wheel_graph':
            role = "Hub" if i == 0 else "Rim"
        else:
            role = f"Node_{i}"
        
        color = colors.get(role, plt.cm.tab10(i))
        
        ax.errorbar(bs, means, yerr=stds, marker='o', linewidth=2, markersize=8,
                   capsize=4, label=f'{role} (deg={agent_degree})', 
                   color=color, alpha=0.8)
    
    ax.axhline(0.025, color='black', linestyle='--', alpha=0.5, label='Random (0.025)')
    
    ax.set_xlabel('Benefit (b)', fontsize=12)
    ax.set_ylabel('Cooperation Rate p', fontsize=12)
    ax.set_title(f'Cooperation Rate vs Benefit\n{graph_type} | {r_type} | {mode} | '
                 f'{N_REPLICATIONS} replications', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "cooperation_rates", graph_type)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{r_type}_{mode}.png"), dpi=150)
    plt.close()


def plot_statistical_summary(cooperation_stats, delta_q_histories, b_values,
                            r_type, mode, graph_type, output_dir):
    """
    Advanced statistical summary: variance decomposition, convergence speed, etc.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Statistical Summary - {graph_type} | {r_type} | {mode}', fontsize=14)
    
    # 1. Variance of cooperation across replications vs b
    ax1 = axes[0, 0]
    bs = sorted(b_values)
    for i in range(len(cooperation_stats[bs[0]]['mean'])):
        variances = [cooperation_stats[b]['std'][i]**2 for b in bs]
        ax1.plot(bs, variances, marker='o', label=f'Agent {i}')
    ax1.set_xlabel('b')
    ax1.set_ylabel('Var(p)')
    ax1.set_title('Variance of Cooperation Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence speed: time to reach 90% of final ΔQ
    ax2 = axes[0, 1]
    # Calculate for each replication and average
    convergence_times = {b: [] for b in bs}
    for b in bs:
        histories = delta_q_histories[b]  # (n_reps, T, N, S)
        final_vals = histories[:, -100:, :, :].mean(axis=1)  # (n_reps, N, S)
        for rep in range(histories.shape[0]):
            for n in range(histories.shape[2]):
                for s in range(min(3, histories.shape[3])):
                    traj = histories[rep, :, n, s]
                    target = 0.9 * final_vals[rep, n, s]
                    # Find first time within 10% of final
                    close_indices = np.where(np.abs(traj - final_vals[rep, n, s]) < 0.1 * np.abs(final_vals[rep, n, s]))[0]
                    if len(close_indices) > 0:
                        convergence_times[b].append(close_indices[0])
    mean_conv = [np.mean(convergence_times[b]) if convergence_times[b] else NUM_ITERATIONS 
                 for b in bs]
    ax2.plot(bs, mean_conv, marker='o', linewidth=2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('Iterations to convergence')
    ax2.set_title('Convergence Speed')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of final ΔQ across replications
    ax3 = axes[1, 0]
    final_deltas = {b: [] for b in bs}
    for b in bs:
        histories = delta_q_histories[b]
        final_vals = histories[:, -1000:, :, :].mean(axis=(1, 2, 3))  # (n_reps,)
        final_deltas[b] = final_vals
    bp_data = [final_deltas[b] for b in bs]
    ax3.boxplot(bp_data, labels=[str(b) for b in bs])
    ax3.set_xlabel('b')
    ax3.set_ylabel('Final ΔQ (mean over agents and states)')
    ax3.set_title('Distribution of Final ΔQ')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation between agents
    ax4 = axes[1, 1]
    correlations = []
    for b in bs:
        coop_matrix = cooperation_stats[b]['all']  # (n_reps, n_nodes)
        if coop_matrix.shape[1] > 1:
            # Average pairwise correlation
            corrs = []
            for i in range(coop_matrix.shape[1]):
                for j in range(i+1, coop_matrix.shape[1]):
                    corrs.append(np.corrcoef(coop_matrix[:, i], coop_matrix[:, j])[0, 1])
            correlations.append(np.mean(corrs))
        else:
            correlations.append(0)
    ax4.plot(bs, correlations, marker='o', linewidth=2)
    ax4.set_xlabel('b')
    ax4.set_ylabel('Mean pairwise correlation')
    ax4.set_title('Inter-Agent Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "statistics", graph_type)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{r_type}_{mode}_summary.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    start_time = __import__('time').time()
    
    # Run all graph types
    for graph_type in GRAPH_TYPES:
        run_convergence_experiment_a100(graph_type)
    
    elapsed = __import__('time').time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE! Total time: {elapsed/3600:.2f} hours")
    print(f"Output: experiments/exp8/results/a100_maximum/")