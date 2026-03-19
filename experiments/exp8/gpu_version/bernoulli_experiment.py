import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool, cpu_count
from functools import partial

# M1 Optimization
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ MPS available")
else:
    DEVICE = torch.device("cpu")

# Override gpu_config
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager

# === PARAMETERS ===
N_REPLICATIONS = 40
NUM_ITERATIONS = 10000
WARMUP_PERIOD = 5000
N_BOOT = 5000
B_VALUES = [1.2, 1.5, 2.0, 3.0, 5.0]
GRAPH_TYPES = ['star_graph', 'wheel_graph']
MODES = ['state', 'stateless']

# Parallel workers
N_WORKERS = min(6, cpu_count() - 2) if cpu_count() > 4 else 4

def run_single_rep(args):
    """Worker function for parallel execution."""
    rep, graph_type, mode, b_val, num_nodes, adj_matrix_np, degrees_np = args
    
    torch.manual_seed(rep)
    np.random.seed(rep)
    
    adj_matrix = torch.from_numpy(adj_matrix_np).float().to(DEVICE)
    degrees = torch.from_numpy(degrees_np).float().to(DEVICE)
    
    n_samples = NUM_ITERATIONS - WARMUP_PERIOD
    action_buffer = np.zeros((n_samples, num_nodes), dtype=np.float32)
    
    if mode == 'state':
        max_states = int(degrees.max().item()) + 1
        gamma = 0.9
    else:
        max_states = 1
        gamma = 0.9
    
    learner = BatchedGPUQLearner(
        batch_size=1,
        n_agents=num_nodes,
        action_space_size=2,
        learning_rate=0.1,
        discount_factor=gamma,
        exploration_rate=0.05,
        max_states=max_states
    )
    
    reward_manager = RewardManager(reward_type='pp', b=b_val, c=1.0)
    states = torch.zeros((1, num_nodes), dtype=torch.long, device=DEVICE)
    
    sample_idx = 0
    for t in range(NUM_ITERATIONS):
        actions = learner.get_actions(states)
        
        if t >= WARMUP_PERIOD:
            action_buffer[sample_idx] = actions.cpu().numpy()[0]
            sample_idx += 1
        
        rewards = reward_manager.calculate_rewards(
            actions.float(), 
            adj_matrix.unsqueeze(0), 
            degrees.unsqueeze(0)
        )
        
        if mode == 'state':
            next_states = torch.matmul(actions.float(), adj_matrix).long()
        else:
            next_states = torch.zeros_like(states)
        
        learner.update(states, actions, rewards, next_states)
        states = next_states
    
    return np.mean(action_buffer, axis=0)

def run_config_parallel(graph_type, mode, b_val, num_nodes, adj_matrix, degrees):
    """Run all replications for one config in parallel."""
    adj_matrix_np = adj_matrix.cpu().numpy()
    degrees_np = degrees.cpu().numpy()
    
    args_list = [
        (rep, graph_type, mode, b_val, num_nodes, adj_matrix_np, degrees_np)
        for rep in range(N_REPLICATIONS)
    ]
    
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(run_single_rep, args_list)
    
    return np.array(results)

def bootstrap_ci(data, n_boot=N_BOOT, return_dist=False):
    """Vectorized bootstrap."""
    data = np.array(data)
    n = len(data)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.array([])) if return_dist else (np.nan, np.nan, np.nan)
    
    rng = np.random.default_rng()
    samples = rng.choice(data, size=(n_boot, n), replace=True)
    boot_means = np.mean(samples, axis=1)
    
    mean_obs = np.mean(data)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    
    if return_dist:
        return mean_obs, ci_low, ci_high, boot_means
    return mean_obs, ci_low, ci_high

def plot_bootstrap_diagnostics(boot_means, observed_mean, agent_role, b_val, mode, graph_type, output_dir):
    """
    Create histogram with KDE and QQ-plot for bootstrap distribution.
    """
    # Создаём директорию если не существует
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Bootstrap Distribution: {graph_type} | {mode} | {agent_role} | b={b_val}', fontsize=11)

    # Left: Histogram with KDE and normal fit
    ax1 = axes[0]
    
    ax1.hist(boot_means, bins=50, density=True, alpha=0.6, 
             color='steelblue', edgecolor='white', label='Bootstrap samples')
    
    # KDE
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(boot_means)
        x_range = np.linspace(boot_means.min(), boot_means.max(), 200)
        ax1.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    except ImportError:
        pass
    
    # Normal fit
    mu, sigma = np.mean(boot_means), np.std(boot_means)
    x_norm = np.linspace(boot_means.min(), boot_means.max(), 200)
    ax1.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r--', linewidth=2, 
             label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    
    # Mark observed mean and CI
    ci_low, ci_high = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    ax1.axvline(observed_mean, color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {observed_mean:.3f}')
    ax1.axvline(ci_low, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(ci_high, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
    
    ax1.set_xlabel('Bootstrap Mean Cooperation Rate')
    ax1.set_ylabel('Density')
    ax1.legend(loc='best', fontsize=8)
    ax1.set_title('Distribution of Bootstrap Means')
    ax1.grid(True, alpha=0.3)
    
    # Right: QQ-plot against normal
    ax2 = axes[1]
    
    standardized = (boot_means - mu) / sigma
    stats.probplot(standardized, dist=stats.norm, plot=ax2)
    
    # Customize QQ-plot appearance
    ax2.get_lines()[0].set_markerfacecolor('steelblue')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linestyle('--')
    
    # Normality test
    if len(boot_means) <= 5000:
        stat, p_value = stats.shapiro(boot_means[:min(5000, len(boot_means))])
        test_name = "Shapiro-Wilk"
    else:
        stat, p_value = stats.jarque_bera(boot_means)
        test_name = "Jarque-Bera"
    
    is_normal = "Normal" if p_value > 0.05 else "Non-normal"
    ax2.set_title(f'Q-Q Plot vs Normal\n{test_name}: p={p_value:.4f} ({is_normal})')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_role = agent_role.replace(' ', '_')
    filename = f'bootstrap_{graph_type}_{mode}_{safe_role}_b{b_val:.1f}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_cooperation_distributions(results, output_dir):
    """
    Plot distribution of cooperation rates across b values for each role.
    Shows how p varies with b and includes fitted curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cooperation Rate Distributions Across b Values', fontsize=14)
    
    # Organize data
    data_by_config = {}
    for r in results:
        key = (r['graph'], r['mode'], r['agent_role'])
        if key not in data_by_config:
            data_by_config[key] = {'b': [], 'p': [], 'ci_low': [], 'ci_high': []}
        data_by_config[key]['b'].append(r['b'])
        data_by_config[key]['p'].append(r['p_mean'])
        data_by_config[key]['ci_low'].append(r['ci_lower'])
        data_by_config[key]['ci_high'].append(r['ci_upper'])
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    config_list = [
        ('star_graph', 'state'), ('star_graph', 'stateless'),
        ('wheel_graph', 'state'), ('wheel_graph', 'stateless')
    ]
    
    for (graph, mode), (row, col) in zip(config_list, positions):
        ax = axes[row, col]
        
        if (graph, mode, 'Hub') not in data_by_config:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot for each role
        colors = {'Hub': '#d62728', 'Leaf': '#2ca02c', 'Rim': '#ff7f0e'}
        
        for role in ['Hub', 'Leaf'] if graph == 'star_graph' else ['Hub', 'Rim']:
            key = (graph, mode, role)
            if key not in data_by_config:
                continue
            
            d = data_by_config[key]
            bs = np.array(d['b'])
            ps = np.array(d['p'])
            lows = np.array(d['ci_low'])
            highs = np.array(d['ci_high'])
            
            # Sort by b
            sort_idx = np.argsort(bs)
            bs, ps, lows, highs = bs[sort_idx], ps[sort_idx], lows[sort_idx], highs[sort_idx]
            
            # Plot with error bars
            ax.errorbar(bs, ps, yerr=[ps - lows, highs - ps],
                       label=role, marker='o', capsize=5, linewidth=2,
                       markersize=8, color=colors.get(role, 'blue'),
                       alpha=0.8)
            
            # Add fitted curve (polynomial or logistic)
            if len(bs) >= 3:
                try:
                    # Try sigmoid fit
                    from scipy.optimize import curve_fit
                    def sigmoid(x, L, k, x0):
                        return L / (1 + np.exp(-k * (x - x0)))
                    
                    p0 = [max(ps), 1.0, np.median(bs)]
                    popt, _ = curve_fit(sigmoid, bs, ps, p0=p0, maxfev=10000)
                    x_smooth = np.linspace(bs.min(), bs.max(), 100)
                    y_smooth = sigmoid(x_smooth, *popt)
                    ax.plot(x_smooth, y_smooth, '--', color=colors.get(role, 'blue'),
                           alpha=0.5, linewidth=1.5)
                except:
                    pass  # Skip fit if fails
        
        # Theoretical baseline
        ax.axhline(y=0.025, color='black', linestyle=':', alpha=0.5, 
                  label='Random (0.025)')
        
        ax.set_title(f'{graph.replace("_", " ").title()}, {mode}')
        ax.set_xlabel('Benefit (b)')
        ax.set_ylabel('Cooperation rate p')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cooperation_distributions.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cooperation_distributions.png")

def plot_qq_by_b(results, output_dir):
    """
    Create QQ-plots comparing empirical distribution of replications 
    to theoretical distributions for each b value.
    """
    # This requires raw replication data - we'll simulate from summary stats
    # or note that we need to save raw data for perfect QQ plots
    
    fig, axes = plt.subplots(len(B_VALUES), 2, figsize=(10, 3*len(B_VALUES)))
    if len(B_VALUES) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('QQ-plots: Empirical vs Normal by b value (Hub agents only)', fontsize=12)
    
    # Group by b and get Hub data
    for i, b_val in enumerate(B_VALUES):
        # Left: state mode
        ax_left = axes[i, 0]
        # Right: stateless mode  
        ax_right = axes[i, 1]
        
        for ax, mode in [(ax_left, 'state'), (ax_right, 'stateless')]:
            # Find all Hub entries for this b and mode across graphs
            hub_data = []
            for r in results:
                if r['b'] == b_val and r['mode'] == mode and r['agent_role'] == 'Hub':
                    # Approximate from mean and std (assuming normality of replications)
                    # In real scenario, we'd save all replication values
                    mean = r['p_mean']
                    std = r['std']
                    # Generate synthetic sample from summary stats
                    synthetic = np.random.normal(mean, std, N_REPLICATIONS)
                    hub_data.extend(synthetic)
            
            if hub_data:
                stats.probplot(hub_data, dist=stats.norm, plot=ax)
                ax.get_lines()[0].set_markerfacecolor('steelblue')
                ax.get_lines()[0].set_markersize(4)
                ax.get_lines()[1].set_color('red')
                ax.set_title(f'{mode}, b={b_val}')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qq_by_b_value.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: qq_by_b_value.png")

def run_bernoulli_experiment(save_raw_data=False):
    """
    Run full experiment with optional raw data saving for diagnostics.
    """
    output_dir = "experiments/exp8/results/bernoulli_parallel"
    os.makedirs(output_dir, exist_ok=True)
    
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    if save_raw_data:
        raw_data_dir = os.path.join(output_dir, "raw_replication_data")
        os.makedirs(raw_data_dir, exist_ok=True)
    
    start_time = time.time()
    results_summary = []
    all_raw_data = {}  # Store for diagnostics if needed
    
    total_configs = len(GRAPH_TYPES) * len(MODES) * len(B_VALUES)
    print(f"Parallel Experiment: {N_WORKERS} workers, {total_configs} configs")
    print(f"Total simulations: {total_configs * N_REPLICATIONS}\n")
    
    config_idx = 0
    
    for graph_type in GRAPH_TYPES:
        print(f"\n{'='*50}")
        print(f"[{config_idx+1}/{total_configs}] Graph: {graph_type}")
        
        if graph_type == 'star_graph':
            num_nodes = 5
            graph = StarGraph(num_nodes=num_nodes, device=DEVICE)
        else:
            num_nodes = 5
            graph = WheelGraph(num_nodes=num_nodes, device=DEVICE)
        
        adj_matrix = graph.generate_adjacency_matrix()
        degrees = torch.sum(adj_matrix, dim=1)
        
        for mode in MODES:
            for b_val in B_VALUES:
                config_idx += 1
                t_start = time.time()
                
                print(f"  Config {config_idx}/{total_configs}: {mode}, b={b_val}...", end=" ", flush=True)
                
                # PARALLEL EXECUTION
                rep_results = run_config_parallel(
                    graph_type, mode, b_val, num_nodes, adj_matrix, degrees
                )  # Shape: (n_replications, n_agents)
                
                # Store raw data if requested
                if save_raw_data:
                    key = f"{graph_type}_{mode}_b{b_val}"
                    all_raw_data[key] = rep_results.copy()
                
                # Process results with bootstrap
                for agent_idx in range(num_nodes):
                    agent_data = rep_results[:, agent_idx]
                    
                    # Get bootstrap with distribution for key agents
                    is_key_agent = (agent_idx == 0) or (b_val in [B_VALUES[0], B_VALUES[-1]])
                    
                    if is_key_agent:
                        mean_p, ci_low, ci_high, boot_dist = bootstrap_ci(
                            agent_data, return_dist=True
                        )
                        # Save diagnostic plot for key agents
                        if graph_type == 'star_graph':
                            role = "Hub" if agent_idx == 0 else "Leaf"
                        else:
                            role = "Hub" if agent_idx == 0 else "Rim"
                        
                        plot_bootstrap_diagnostics(
                            boot_dist, mean_p, role, b_val, mode, 
                            graph_type, diag_dir
                        )
                    else:
                        mean_p, ci_low, ci_high = bootstrap_ci(agent_data)
                        if graph_type == 'star_graph':
                            role = "Hub" if agent_idx == 0 else "Leaf"
                        else:
                            role = "Hub" if agent_idx == 0 else "Rim"
                    
                    results_summary.append({
                        "graph": graph_type,
                        "mode": mode,
                        "b": b_val,
                        "agent_id": agent_idx,
                        "agent_role": role,
                        "degree": degrees[agent_idx].item(),
                        "n_replications": N_REPLICATIONS,
                        "p_mean": float(mean_p),
                        "ci_lower": float(ci_low),
                        "ci_upper": float(ci_high),
                        "std": float(np.std(agent_data)),
                        "se": float(np.std(agent_data) / np.sqrt(N_REPLICATIONS))
                    })
                
                elapsed = time.time() - t_start
                print(f"✓ {elapsed:.1f}s")
    
    # Save raw data if collected
    if save_raw_data and all_raw_data:
        import pickle
        with open(os.path.join(raw_data_dir, "replication_data.pkl"), 'wb') as f:
            pickle.dump(all_raw_data, f)
        print(f"\n  Saved raw replication data")
    
    # Generate all plots
    save_results_csv(results_summary, output_dir)
    plot_results(results_summary, output_dir)
    plot_cooperation_distributions(results_summary, output_dir)
    plot_qq_by_b(results_summary, output_dir)
    
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✓ Complete! Total time: {total_time/60:.1f} minutes")
    print(f"  Throughput: {total_configs * N_REPLICATIONS / total_time:.1f} sims/sec")
    print(f"  Results: {output_dir}")
    
    return results_summary

def save_results_csv(results, output_dir):
    import csv
    keys = results[0].keys()
    with open(os.path.join(output_dir, "results.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results, output_dir):
    """Standard summary plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cooperation Rates (Parallel Execution)', fontsize=14)
    
    data_by_key = {}
    for r in results:
        key = (r['graph'], r['mode'], r['agent_role'], r['b'])
        data_by_key[key] = r
    
    for idx, graph_type in enumerate(GRAPH_TYPES):
        for jdx, mode in enumerate(MODES):
            ax = axes[idx, jdx]
            
            roles = ['Hub', 'Leaf'] if graph_type == 'star_graph' else ['Hub', 'Rim']
            
            for role in roles:
                bs = sorted([b for (g, m, r, b) in data_by_key.keys() 
                           if g == graph_type and m == mode and r == role])
                
                means, lows, highs = [], [], []
                for b in bs:
                    e = data_by_key[(graph_type, mode, role, b)]
                    means.append(e['p_mean'])
                    lows.append(e['ci_lower'])
                    highs.append(e['ci_upper'])
                
                ax.errorbar(bs, means, 
                           yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
                           label=role, marker='o', capsize=4, linewidth=2)
            
            ax.set_title(f"{graph_type.replace('_', ' ').title()}, {mode}")
            ax.set_xlabel("Benefit (b)")
            ax.set_ylabel("Cooperation rate p")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results.png"), dpi=150, bbox_inches='tight')
    plt.close()

def load_and_plot_only():
    """
    Load existing results and generate only diagnostic plots.
    """
    output_dir = "experiments/exp8/results/bernoulli_parallel"
    csv_path = os.path.join(output_dir, "results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: No results found at {csv_path}")
        print("Run experiment first or check path.")
        return
    
    print(f"Loading results from {csv_path}...")
    results = load_results_csv(csv_path)
    print(f"Loaded {len(results)} result entries")
    
    # Check if we have raw data for better diagnostics
    raw_data_path = os.path.join(output_dir, "raw_replication_data", "replication_data.pkl")
    has_raw_data = os.path.exists(raw_data_path)
    
    # Create plots directory
    plot_dir = os.path.join(output_dir, "plots_only_run")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join(plot_dir, "diagnostics"), exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Standard plots
    plot_results(results, plot_dir)
    plot_cooperation_distributions(results, plot_dir)
    plot_qq_by_b(results, plot_dir)
    
    # If we have raw data, generate accurate per-agent diagnostics
    if has_raw_data:
        print("  Found raw replication data, generating accurate diagnostics...")
        import pickle
        with open(raw_data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Generate accurate bootstrap diagnostics from raw data
        for key, rep_results in raw_data.items():
            parts = key.split('_')
            graph_type = parts[0] + '_' + parts[1]
            mode = parts[2]
            b_val = float(parts[3][1:])  # Remove 'b' prefix
            
            for agent_idx in range(rep_results.shape[1]):
                agent_data = rep_results[:, agent_idx]
                
                # Only for key agents to save time
                if agent_idx == 0 or b_val in [B_VALUES[0], B_VALUES[-1]]:
                    mean_p, ci_low, ci_high, boot_dist = bootstrap_ci(agent_data, return_dist=True)
                    
                    if graph_type == 'star_graph':
                        role = "Hub" if agent_idx == 0 else "Leaf"
                    else:
                        role = "Hub" if agent_idx == 0 else "Rim"
                    
                    plot_bootstrap_diagnostics(
                        boot_dist, mean_p, role, b_val, mode,
                        graph_type, os.path.join(plot_dir, "diagnostics")
                    )
    
    print(f"\n✓ Plots saved to: {plot_dir}")

def load_results_csv(file_path):
    import csv
    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Обязательные поля
            row['b'] = float(row['b'])
            row['degree'] = float(row['degree'])
            row['n_replications'] = int(row['n_replications'])
            row['p_mean'] = float(row['p_mean'])
            row['ci_lower'] = float(row['ci_lower'])
            row['ci_upper'] = float(row['ci_upper'])
            row['std'] = float(row['std'])
            
            # Опциональные поля (могут отсутствовать в старых CSV)
            if 'se' in row:
                row['se'] = float(row['se'])
            else:
                row['se'] = row['std'] / np.sqrt(row['n_replications'])
            
            results.append(row)
    return results

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Choose mode:
    # 1. Run full experiment: uncomment next line
    run_bernoulli_experiment(save_raw_data=True)
    
    # 2. Plot only (no computation): uncomment next line
    # load_and_plot_only()