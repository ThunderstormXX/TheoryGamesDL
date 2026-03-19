import os
import sys
import warnings
import gc

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# A100 Maximum Optimization - Single Process, Full GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-optimize conv algorithms
torch.set_float32_matmul_precision('high')

# Force single thread CPU
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")

# Override gpu_config
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager

# MAXIMUM PARAMETERS FOR A100
N_REPLICATIONS = 100      # Total replications
BATCH_SIZE = 64           # Process 64 at once on GPU (A100 can handle more)
NUM_ITERATIONS = 100000   # Long runs
WARMUP_PERIOD = 80000
B_VALUES = [1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
GRAPH_TYPES = ['star_graph', 'wheel_graph', 'small_world_graph']
REWARD_TYPES = ['pp', 'pf', 'ff', 'fp']


def calculate_theoretical_q_diff(reward_type, c, degrees):
    """Vectorized theoretical calculation."""
    if reward_type in ['pp', 'fp']:
        return -c * degrees
    else:
        return -torch.full_like(degrees, c)


def run_batched_simulations(batch_reps, b_val, r_type, mode, num_nodes, 
                            adj_matrix, degrees, gamma):
    """
    Run BATCH_SIZE replications simultaneously on GPU.
    """
    actual_batch = len(batch_reps)
    max_degree = int(degrees.max().item())
    max_states = max_degree + 1 if mode == 'state' else 1
    actual_gamma = gamma if mode == 'state' else 0.0
    
    # УБРАТЬ .to(DEVICE) - learner сам использует gpu_config.device
    learner = BatchedGPUQLearner(
        batch_size=actual_batch,
        n_agents=num_nodes,
        action_space_size=2,
        learning_rate=0.1,
        discount_factor=actual_gamma,
        exploration_rate=0.05,
        max_states=max_states
    )
    # .to(DEVICE) УДАЛЕНО!
    
    reward_manager = RewardManager(reward_type=r_type, b=b_val, c=1.0)
    
    # Pre-allocate ALL tensors on GPU (no CPU-GPU transfers in loop)
    states = torch.zeros((actual_batch, num_nodes), dtype=torch.long, device=DEVICE)
    
    # Storage: keep on GPU until end, then move to CPU
    # Shape: (iterations, batch, nodes, states)
    delta_q_buffer = torch.zeros((NUM_ITERATIONS, actual_batch, num_nodes, max_states), 
                                  device=DEVICE, dtype=torch.float32)
    action_buffer = torch.zeros((NUM_ITERATIONS, actual_batch, num_nodes), 
                                 device=DEVICE, dtype=torch.bool)
    
    # Main simulation loop - ALL on GPU
    for t in range(NUM_ITERATIONS):
        actions = learner.get_actions(states)
        action_buffer[t] = (actions == 1)
        
        # Batch rewards calculation
        rewards = reward_manager.calculate_rewards(
            actions.float(),
            adj_matrix.unsqueeze(0).expand(actual_batch, -1, -1),
            degrees.unsqueeze(0).expand(actual_batch, -1)
        )
        
        # Next states
        if mode == 'state':
            next_states = torch.matmul(actions.float(), adj_matrix).long()
        else:
            next_states = torch.zeros_like(states)
        
        learner.update(states, actions, rewards, next_states)
        
        # Record Delta Q (all on GPU)
        q_table = learner.q_table  # (B, N, S, A)
        delta_q = q_table[:, :, :, 1] - q_table[:, :, :, 0]  # (B, N, S)
        delta_q_buffer[t] = delta_q
        
        states = next_states
    
    # Calculate cooperation rates (GPU)
    post_warmup = action_buffer[WARMUP_PERIOD:]
    coop_rates = post_warmup.float().mean(dim=0)  # (batch, nodes)
    
    # Move to CPU ONCE at the end
    coop_rates_cpu = coop_rates.cpu().numpy()
    delta_q_cpu = delta_q_buffer.cpu().numpy()
    
    # Clear GPU cache
    del learner, states, delta_q_buffer, action_buffer
    torch.cuda.empty_cache()
    gc.collect()
    
    return coop_rates_cpu, delta_q_cpu


def run_experiment_single_process(graph_type='star_graph'):
    """
    Single process, maximum GPU utilization.
    No multiprocessing - pure GPU batching.
    """
    output_dir = "experiments/exp8/results/a100_single_process"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SINGLE PROCESS A100 MAXIMUM: {graph_type}")
    print(f"{'='*70}")
    print(f"Total replications: {N_REPLICATIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per config: {(N_REPLICATIONS + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Setup graph
    match graph_type:
        case 'star_graph':
            num_nodes = 5
            graph = StarGraph(num_nodes=num_nodes, device=DEVICE)
        case 'wheel_graph':
            num_nodes = 5
            graph = WheelGraph(num_nodes=num_nodes, device=DEVICE)
        case 'small_world_graph':
            num_nodes = 50
            graph = SmallWorldGraph(num_nodes=num_nodes, device=DEVICE)
    
    adj_matrix = graph.generate_adjacency_matrix()
    degrees = torch.sum(adj_matrix, dim=1)
    
    # Move to GPU once
    adj_matrix = adj_matrix.to(DEVICE)
    degrees = degrees.to(DEVICE)
    
    print(f"Nodes: {num_nodes}")
    print(f"Degrees: {degrees.cpu().numpy()}")
    
    modes = ['state', 'stateless']
    
    for r_type in REWARD_TYPES:
        for mode in modes:
            print(f"\n{'-'*50}")
            print(f"Reward: {r_type}, Mode: {mode}")
            
            th_delta_q = calculate_theoretical_q_diff(r_type, 1.0, degrees)
            
            # Process all B values
            all_results = {b: {'coop': [], 'delta_q': []} for b in B_VALUES}
            
            for b_val in B_VALUES:
                print(f"  b={b_val}")
                
                # Process replications in GPU batches
                n_batches = (N_REPLICATIONS + BATCH_SIZE - 1) // BATCH_SIZE
                
                for batch_idx in tqdm(range(n_batches), desc=f"b={b_val}", ncols=70):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min((batch_idx + 1) * BATCH_SIZE, N_REPLICATIONS)
                    batch_reps = list(range(start_idx, end_idx))
                    actual_batch = len(batch_reps)
                    
                    # SINGLE GPU CALL processes entire batch
                    coop_rates, delta_q_hist = run_batched_simulations(
                        batch_reps, b_val, r_type, mode, num_nodes,
                        adj_matrix, degrees, gamma=0.9
                    )
                    
                    # Store results
                    for i in range(actual_batch):
                        all_results[b_val]['coop'].append(coop_rates[i])
                        all_results[b_val]['delta_q'].append(delta_q_hist[:, i, :, :])
                
                # Stack all replications for this b_val
                all_results[b_val]['coop'] = np.array(all_results[b_val]['coop'])
                all_results[b_val]['delta_q'] = np.array(all_results[b_val]['delta_q'])
                
                print(f"    Shape: {all_results[b_val]['delta_q'].shape}")
            
            # Generate plots
            print("  Generating plots...")
            plot_all_results(all_results, B_VALUES, r_type, mode, graph_type, 
                           degrees.cpu().numpy(), th_delta_q.cpu().numpy(), output_dir)


def plot_all_results(results, b_values, r_type, mode, graph_type, degrees, 
                     th_delta_q, output_dir):
    """Generate all plot types."""
    
    # 1. Convergence plots (mean ± std)
    for b_val in tqdm(b_values, desc="Convergence", leave=False):
        plot_convergence(results[b_val], b_val, r_type, mode, graph_type, 
                        degrees, th_delta_q, output_dir)
    
    # 2. Distribution evolution (NEW - main request)
    for b_val in tqdm(b_values, desc="Distributions", leave=False):
        plot_delta_q_distribution(results[b_val], b_val, r_type, mode, 
                                  graph_type, degrees, output_dir)
    
    # 3. Cooperation rates
    plot_cooperation_rates(results, b_values, degrees, r_type, mode, 
                          graph_type, output_dir)
    
    # 4. Statistical summary
    plot_statistics(results, b_values, r_type, mode, graph_type, output_dir)


def plot_convergence(result_data, b_val, r_type, mode, graph_type, 
                     degrees, th_delta_q, output_dir):
    """Plot mean Delta Q convergence with confidence bands."""
    delta_q_hist = result_data['delta_q']  # (n_reps, T, N, S)
    n_reps, T, N, S = delta_q_hist.shape
    
    mean_dq = delta_q_hist.mean(axis=0)
    std_dq = delta_q_hist.std(axis=0)
    
    nodes_to_plot = min(N, 5)
    states_to_plot = min(S, 3)
    
    fig, axes = plt.subplots(nodes_to_plot, 1, figsize=(12, 3.5 * nodes_to_plot), 
                            sharex=True)
    if nodes_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(f"ΔQ Convergence - {graph_type} | {r_type} | {mode} | b={b_val}\n"
                 f"Mean ± std, n={n_reps}", fontsize=12)
    
    for n in range(nodes_to_plot):
        ax = axes[n]
        for s in range(min(states_to_plot, int(degrees[n]) + 1)):
            mean_traj = mean_dq[:, n, s]
            std_traj = std_dq[:, n, s]
            
            ax.plot(mean_traj, linewidth=1.5, label=f'State {s}', alpha=0.8)
            ax.fill_between(range(T), mean_traj - std_traj, mean_traj + std_traj,
                           alpha=0.2)
        
        ax.axhline(th_delta_q[n], color='r', linestyle='--', alpha=0.7,
                  label=f'Theory: {th_delta_q[n]:.2f}')
        ax.axhline(0, color='k', linestyle='-', alpha=0.2)
        
        coop_mean = result_data['coop'][:, n].mean()
        coop_std = result_data['coop'][:, n].std()
        ax.set_title(f"Agent {n} (k={int(degrees[n])}, p={coop_mean:.3f}±{coop_std:.3f})")
        ax.set_ylabel("ΔQ")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.xlabel("Iterations")
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "convergence", graph_type, r_type, mode)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"b_{b_val}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_delta_q_distribution(result_data, b_val, r_type, mode, 
                              graph_type, degrees, output_dir):
    """
    Plot distribution of Delta Q over time - evolution from initial to equilibrium.
    """
    delta_q_hist = result_data['delta_q']  # (n_reps, T, N, S)
    n_reps, T, N, S = delta_q_hist.shape
    
    # Log-spaced early times + linear late times
    early_times = np.unique(np.logspace(0, np.log10(max(T//20, 10)), 8).astype(int))
    late_times = np.linspace(T//5, T-1, 7).astype(int)
    time_points = np.unique(np.concatenate([early_times, late_times]))
    time_points = np.clip(time_points, 0, T-1)
    
    nodes_to_plot = min(N, 5)
    n_times = len(time_points)
    
    fig, axes = plt.subplots(nodes_to_plot, n_times, 
                            figsize=(2.2 * n_times, 2 * nodes_to_plot),
                            sharex='row', sharey='row')
    
    if nodes_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'ΔQ Distribution Evolution - {graph_type} | {r_type} | {mode} | b={b_val}\n'
                 f'{n_reps} replications, {N_REPLICATIONS} total', fontsize=13)
    
    # Global range
    all_vals = delta_q_hist[:, :, :nodes_to_plot, :].flatten()
    vmin, vmax = np.percentile(all_vals, [1, 99])
    
    for row, n in enumerate(range(nodes_to_plot)):
        for col, t in enumerate(time_points):
            ax = axes[row, col]
            
            # All states for this agent at time t
            vals = delta_q_hist[:, t, n, :].flatten()
            
            # Histogram with KDE-like smoothing via many bins
            ax.hist(vals, bins=40, density=True, alpha=0.65,
                   color='steelblue', range=(vmin, vmax), edgecolor='white', linewidth=0.3)
            
            # Statistics
            mean_v = np.mean(vals)
            median_v = np.median(vals)
            std_v = np.std(vals)
            
            ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f'μ={mean_v:.2f}')
            ax.axvline(median_v, color='green', linestyle=':', linewidth=1.2, label=f'med={median_v:.2f}')
            ax.axvline(th_delta_q[n], color='orange', linestyle='-', linewidth=1, alpha=0.7,
                      label=f'th={th_delta_q[n]:.2f}')
            
            if row == 0:
                ax.set_title(f't={t}', fontsize=9)
            
            if col == 0:
                ax.set_ylabel(f'A{n} (k={int(degrees[n])})\nDensity', fontsize=8)
            
            if row == nodes_to_plot - 1:
                ax.set_xlabel('ΔQ', fontsize=8)
            
            ax.set_xlim(vmin, vmax)
            ax.tick_params(labelsize=6)
            ax.legend(loc='upper right', fontsize=5, handlelength=0.8)
            ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "distributions", graph_type, r_type, mode)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"b_{b_val}_dist.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_cooperation_rates(results, b_values, degrees, r_type, mode, 
                           graph_type, output_dir):
    """Plot cooperation rates vs b with error bars."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    N = len(degrees)
    bs = sorted(b_values)
    colors = {'Hub': '#d62728', 'Leaf': '#2ca02c', 'Rim': '#ff7f0e'}
    
    for i in range(N):
        means = [results[b]['coop'][:, i].mean() for b in bs]
        stds = [results[b]['coop'][:, i].std() for b in bs]
        
        if graph_type == 'star_graph':
            role = "Hub" if i == 0 else "Leaf"
        elif graph_type == 'wheel_graph':
            role = "Hub" if i == 0 else "Rim"
        else:
            role = f"Node_{i}"
        
        color = colors.get(role, plt.cm.tab10(i))
        
        ax.errorbar(bs, means, yerr=stds, marker='o', markersize=7, linewidth=2,
                   capsize=3, label=f'{role} (deg={int(degrees[i])})', color=color)
    
    ax.axhline(0.025, color='black', linestyle='--', alpha=0.4, label='Random')
    
    ax.set_xlabel('Benefit (b)', fontsize=12)
    ax.set_ylabel('Cooperation Rate p', fontsize=12)
    ax.set_title(f'Cooperation vs Benefit - {graph_type} | {r_type} | {mode}\n'
                 f'n={N_REPLICATIONS} replications', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "cooperation", graph_type)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{r_type}_{mode}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_statistics(results, b_values, r_type, mode, graph_type, output_dir):
    """Advanced statistical analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Statistical Analysis - {graph_type} | {r_type} | {mode}', fontsize=13)
    
    bs = sorted(b_values)
    N = len(results[bs[0]]['coop'][0])
    
    # 1. Variance of cooperation across replications
    ax1 = axes[0, 0]
    for i in range(min(N, 5)):
        variances = [results[b]['coop'][:, i].var() for b in bs]
        ax1.plot(bs, variances, marker='o', label=f'Agent {i}')
    ax1.set_xlabel('b')
    ax1.set_ylabel('Var(p)')
    ax1.set_title('Variance of Cooperation')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence time (iterations to within 5% of final)
    ax2 = axes[0, 1]
    conv_times = {b: [] for b in bs}
    for b in bs:
        hist = results[b]['delta_q']  # (n_reps, T, N, S)
        final_vals = hist[:, -1000:, :, :].mean(axis=1)  # (n_reps, N, S)
        for rep in range(hist.shape[0]):
            for n in range(min(N, 5)):
                for s in range(min(3, hist.shape[3])):
                    traj = hist[rep, :, n, s]
                    target = final_vals[rep, n, s]
                    close_idx = np.where(np.abs(traj - target) < 0.05 * abs(target))[0]
                    if len(close_idx) > 0:
                        conv_times[b].append(close_idx[0])
    mean_conv = [np.mean(conv_times[b]) if conv_times[b] else NUM_ITERATIONS for b in bs]
    ax2.plot(bs, mean_conv, marker='o', linewidth=2, color='purple')
    ax2.set_xlabel('b')
    ax2.set_ylabel('Iterations to convergence')
    ax2.set_title('Convergence Speed')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of final Delta Q
    ax3 = axes[1, 0]
    final_data = []
    for b in bs:
        hist = results[b]['delta_q']
        finals = hist[:, -1000:, :, :].mean(axis=(1, 2, 3))  # (n_reps,)
        final_data.append(finals)
    bp = ax3.boxplot(final_data, labels=[str(b) for b in bs], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_xlabel('b')
    ax3.set_ylabel('Final ΔQ (mean over agents/states)')
    ax3.set_title('Distribution of Final ΔQ')
    ax3.grid(True, alpha=0.3)
    
    # 4. Inter-agent correlation
    ax4 = axes[1, 1]
    correlations = []
    for b in bs:
        coop = results[b]['coop']  # (n_reps, N)
        if N > 1:
            corrs = []
            for i in range(N):
                for j in range(i+1, N):
                    corrs.append(np.corrcoef(coop[:, i], coop[:, j])[0, 1])
            correlations.append(np.mean(corrs))
        else:
            correlations.append(0)
    ax4.plot(bs, correlations, marker='o', linewidth=2, color='coral')
    ax4.set_xlabel('b')
    ax4.set_ylabel('Mean pairwise correlation')
    ax4.set_title('Inter-Agent Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir = os.path.join(output_dir, "statistics", graph_type)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{r_type}_{mode}_stats.png"), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import time
    start = time.time()
    
    for graph_type in GRAPH_TYPES:
        run_experiment_single_process(graph_type)
    
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {elapsed/3600:.2f} hours")
    print(f"Output: experiments/exp8/results/a100_single_process/")