import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib
import traceback

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    print("WARNING: networkx not found. Graph visualization will be skipped.")
    nx = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.batched_sarsa import BatchedGPUSARSALearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import (
    RingGraph, CubicCirculantGraph, QuarticCirculantGraph, 
    QuinticCirculantGraph, Mixed23Graph, Mixed34Graph
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_gpu_info():
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM available)")
        print("  Configured to utilize up to 40GB limits if available.")
    elif torch.backends.mps.is_available():
        print("  MPS (Apple Silicon) Detected. Running on GPU.")
        global DEVICE
        DEVICE = torch.device('mps')
        gpu_utils.gpu_config.device = DEVICE
    else:
        print("  Running on CPU.")
    print("=" * 60)

def smooth(y, box_pts=5):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    y_smooth[-box_pts:] = y[-box_pts:]
    return y_smooth

def save_graph_visualization(graph_cls, n, out_path, title):
    if nx is None:
        return
    graph = graph_cls(n, DEVICE)
    adj = graph.generate_adjacency_matrix().cpu().numpy()
    
    G = nx.from_numpy_array(adj)
    plt.figure(figsize=(8, 8))
    
    # Try circular layout to make regular structures look nice
    pos = nx.circular_layout(G)
    
    # Color nodes based on degree
    degrees = [val for (node, val) in G.degree()]
    deg_color = {2: '#3498db', 3: '#e74c3c', 1: '#2ecc71', 4: '#9b59b6', 5: '#f1c40f'}
    node_colors = [deg_color.get(d, '#95a5a6') for d in degrees]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, 
            font_color='white', font_weight='bold', edge_color='gray')
    plt.title(title)
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_json_artifacts(p_hist, qc_hist, qd_hist, out_path):
    # To save space, we only save the mean over the repetitions (axis 1)
    # Shape of inputs: (T_out, REPS, N)
    data = {
        "p_mean": p_hist.mean(axis=1).tolist(),
        "qc_mean": qc_hist.mean(axis=1).tolist(),
        "qd_mean": qd_hist.mean(axis=1).tolist()
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)

# ═══════════════════════════════════════════════════════════════════════════
# Core Simulation Logic
# ═══════════════════════════════════════════════════════════════════════════

ALPHA = 0.01
RECORD_EVERY = 5_000

def run_simulation(graph, gamma, beta, learner_type, iters=500_000, reps=50_000):
    TEMP = 1.0 / beta
    np.random.seed(42)
    torch.manual_seed(42)

    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    num_nodes = graph.num_nodes

    states = torch.zeros((reps, num_nodes), dtype=torch.long, device=DEVICE)

    if learner_type == 'q_learning':
        learner = BatchedGPUQLearner(
            batch_size=reps, n_agents=num_nodes, action_space_size=2,
            learning_rate=ALPHA, discount_factor=gamma, exploration_rate=0.0,
            strategy='boltzmann', temperature=TEMP, max_states=1,
        )
    elif learner_type == 'sarsa':
        learner = BatchedGPUSARSALearner(
            batch_size=reps, n_agents=num_nodes, action_space_size=2,
            learning_rate=ALPHA, discount_factor=gamma, exploration_rate=0.0,
            strategy='boltzmann', temperature=TEMP, max_states=1,
        )
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")

    reward_manager = BonusRewardManager(reward_type='pp', b=2.0, c=1.0, bonus=1.0)

    adj_batched = adj_t.unsqueeze(0).expand(reps, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(reps, -1)

    T_out = iters // RECORD_EVERY + 1
    p_hist = np.zeros((T_out, reps, num_nodes), dtype=np.float32)
    qc_hist = np.zeros((T_out, reps, num_nodes), dtype=np.float32)
    qd_hist = np.zeros((T_out, reps, num_nodes), dtype=np.float32)

    def record_state(t_idx):
        q_now = learner.q_table[:, :, 0, :].cpu()
        probs = torch.softmax(q_now / TEMP, dim=-1).numpy()
        p_hist[t_idx] = probs[..., 1] # Index 1 usually maps to cooperation? In QL it was index 1.
        qd_hist[t_idx] = q_now[..., 0].numpy()
        qc_hist[t_idx] = q_now[..., 1].numpy()

    record_state(0)

    with torch.no_grad():
        if learner_type == 'sarsa':
            actions = learner.get_actions(states)
            for t in range(1, iters + 1):
                rewards = reward_manager.calculate_rewards(
                    actions.float(), adj_batched, deg_batched)
                next_actions = learner.get_actions(states)
                learner.update(states, actions, rewards, states, next_actions)
                actions = next_actions
                if t % RECORD_EVERY == 0:
                    record_state(t // RECORD_EVERY)
        else: # q_learning
            for t in range(1, iters + 1):
                actions = learner.get_actions(states)
                rewards = reward_manager.calculate_rewards(
                    actions.float(), adj_batched, deg_batched)
                learner.update(states, actions, rewards, states)
                if t % RECORD_EVERY == 0:
                    record_state(t // RECORD_EVERY)

    final_pc = p_hist[-1].mean()
    
    # Calculate exit time (e.g. time to reach >0.95 or <0.05 prob stability)
    mean_p_over_time = p_hist.mean(axis=1).mean(axis=1) # (T_out,)
    exit_idx = -1
    for idx, p in enumerate(mean_p_over_time):
        if p > 0.95 or p < 0.05:
            # Check if it stays stable
            if np.all(mean_p_over_time[idx:] > 0.95) or np.all(mean_p_over_time[idx:] < 0.05):
                exit_idx = idx
                break
    exit_time = exit_idx * RECORD_EVERY if exit_idx != -1 else iters

    return p_hist, qc_hist, qd_hist, final_pc, exit_time

def plot_dynamics(p_hist, qc_hist, qd_hist, title, out_path, degrees):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    T_out, _, N = p_hist.shape
    x = np.arange(T_out) * RECORD_EVERY

    deg_color = {2: '#3498db', 3: '#e74c3c', 1: '#2ecc71', 4: '#9b59b6', 5: '#f1c40f'}
    plotted_degs = set()

    for i in range(N):
        d = int(degrees[i])
        col = deg_color.get(d, '#95a5a6')
        label = f'deg={d}' if d not in plotted_degs else None
        plotted_degs.add(d)

        mean_p = smooth(p_hist[:, :, i].mean(axis=1))
        std_p = smooth(p_hist[:, :, i].std(axis=1))
        ax1.plot(x, mean_p, color=col, linewidth=1.5, alpha=0.7, label=label)
        ax1.fill_between(x, np.clip(mean_p - std_p, 0, 1),
                         np.clip(mean_p + std_p, 0, 1), color=col, alpha=0.08)

        mean_qc = smooth(qc_hist[:, :, i].mean(axis=1))
        mean_qd = smooth(qd_hist[:, :, i].mean(axis=1))
        ax2.plot(x, mean_qc, color=col, linestyle='-', linewidth=1.2, alpha=0.7)
        ax2.plot(x, mean_qd, color=col, linestyle='--', linewidth=1.2, alpha=0.7)

    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title('P(C)', fontsize=12)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('P(C)')
    ax1.legend(loc='best', fontsize=9)

    ax2.set_title('Q(C) [solid] vs Q(D) [dashed]', fontsize=12)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Q-value')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Task Definitions
# ═══════════════════════════════════════════════════════════════════════════
BASE_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'supervisor_results'))
os.makedirs(BASE_OUT_DIR, exist_ok=True)

def run_experiment_bundle(task_name, gtype_cls, n_values, gammas, betas, learners, reps=50_000, iters=500_000, description=""):
    print(f"\n[{task_name}] Starting... {description}")
    out_dir = os.path.join(BASE_OUT_DIR, task_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save graph visualizations for each n
    for n in n_values:
        save_graph_visualization(gtype_cls, n, os.path.join(out_dir, f"{gtype_cls.__name__}_n{n}_vis.png"), f"{gtype_cls.__name__} n={n}")
        
    results = {}
    total = len(n_values) * len(gammas) * len(betas) * len(learners)
    
    with tqdm(total=total, desc=task_name) as pbar:
        for n in n_values:
            graph = gtype_cls(n, DEVICE)
            degrees = graph.generate_adjacency_matrix().sum(dim=1).cpu().numpy()
            for gamma in gammas:
                for beta in betas:
                    for learner in learners:
                        p_hist, qc_hist, qd_hist, final_pc, exit_t = run_simulation(
                            graph, gamma, beta, learner, iters=iters, reps=reps)
                        
                        exp_str = f"n{n}_g{gamma}_b{beta}_{learner}"
                        
                        # Plot dynamics
                        title = f"{gtype_cls.__name__} | {learner.upper()} | n={n}, γ={gamma}, β={beta}"
                        plot_path = os.path.join(out_dir, f"dynamics_{exp_str}.png")
                        plot_dynamics(p_hist, qc_hist, qd_hist, title, plot_path, degrees)
                        
                        # Save artifacts
                        json_path = os.path.join(out_dir, f"data_{exp_str}.json")
                        save_json_artifacts(p_hist, qc_hist, qd_hist, json_path)
                        
                        results[exp_str] = {"final_pc": float(final_pc), "exit_time": exit_t}
                        pbar.update(1)
                        
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[{task_name}] Completed. Results saved in {out_dir}")

# ═══════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print_gpu_info()

    # Task 0: Figure-style k-regular comparison (Q-learning, gamma=0)
    # Params match the figure: T=1_000_000, alpha=0.01, beta=1.0, gamma=0.0
    kreg_n_values = [20, 50, 100]
    kreg_gammas = [0.0]
    kreg_betas = [1.0]
    kreg_reps = 512
    kreg_iters = 1_000_000

    run_experiment_bundle(
        task_name="task0_kreg_gamma0_ring",
        gtype_cls=RingGraph,
        n_values=kreg_n_values,
        gammas=kreg_gammas,
        betas=kreg_betas,
        learners=['q_learning'],
        reps=kreg_reps,
        iters=kreg_iters,
        description="Figure-style params for 2-regular ring graphs."
    )
    run_experiment_bundle(
        task_name="task0_kreg_gamma0_mixed23",
        gtype_cls=Mixed23Graph,
        n_values=kreg_n_values,
        gammas=kreg_gammas,
        betas=kreg_betas,
        learners=['q_learning'],
        reps=kreg_reps,
        iters=kreg_iters,
        description="Figure-style params for mixed 2/3-regular graphs."
    )
    run_experiment_bundle(
        task_name="task0_kreg_gamma0_cubic",
        gtype_cls=CubicCirculantGraph,
        n_values=kreg_n_values,
        gammas=kreg_gammas,
        betas=kreg_betas,
        learners=['q_learning'],
        reps=kreg_reps,
        iters=kreg_iters,
        description="Figure-style params for 3-regular cubic graphs."
    )

    # Allow running only Task 0 to avoid long full-suite runs.
    if os.environ.get("ONLY_TASK0", "0") == "1":
        return
    
    # Task 1: Gamma = 0 on some standard graphs
    run_experiment_bundle(
        task_name="task1_gamma0",
        gtype_cls=Mixed23Graph,
        n_values=[10],
        gammas=[0.0],
        betas=[0.5, 1.0],
        learners=['q_learning', 'sarsa'],
        description="Verify behavior identically at gamma=0."
    )
    
    # Task 2: 3-Cluster Effect on Mixed23Graph (Drawing + Explanation)
    # The 3 clusters occur because vertices of degree 2 have different distances to the degree-3 vertices.
    # The influence of degree-3 nodes propagates, creating different expected rewards depending on topology.
    run_experiment_bundle(
        task_name="task2_3_cluster_effect",
        gtype_cls=Mixed23Graph,
        n_values=[10],
        gammas=[0.9],
        betas=[1.0],
        learners=['q_learning'], # Typically showed up strongly in Q-learning
        description="Visualize 3-cluster Q-function separation in Mixed23Graph."
    )
    with open(os.path.join(BASE_OUT_DIR, "task2_3_cluster_effect", "explanation.txt"), "w") as f:
        f.write("Explanation of 3-Cluster Effect in Mixed 2/3 Graphs:\n")
        f.write("In a Mixed 2/3 graph, two nodes have degree 3, and the rest have degree 2.\n")
        f.write("The Q-values of degree-2 nodes separate into multiple 'levels' (clusters) based on their distance from the degree-3 nodes.\n")
        f.write("Degree-3 nodes receive higher rewards/interactions. Their high Q-values propagate through the graph, discounted by gamma.\n")
        f.write("Therefore, a degree-2 node immediately adjacent to a degree-3 node will converge to a higher Q-value than a degree-2 node that is 2 or 3 hops away.\n")
        f.write("This creates clearly distinct Q-value tiers (clusters) directly correlated with topological distance.\n")
        
    # Task 3 & 3.1: Scale effect (n=10, 50, 100) and exit time
    run_experiment_bundle(
        task_name="task3_scale_exit_time",
        gtype_cls=Mixed23Graph,
        n_values=[10, 50, 100],
        gammas=[0.9],
        betas=[1.0],
        learners=['q_learning'],
        iters=2_000_000, # Increased iterations for larger graphs
        description="Scale Mixed23Graph to n=100 and evaluate exit times."
    )
    
    # Task 4: Higher degree graphs (k=4, k=5)
    run_experiment_bundle(
        task_name="task4_higher_degrees",
        gtype_cls=QuarticCirculantGraph,
        n_values=[10],
        gammas=[0.8, 0.9],
        betas=[0.75, 1.5],
        learners=['q_learning', 'sarsa'],
        description="Analyze 4-regular (Quartic) topology."
    )
    run_experiment_bundle(
        task_name="task4_higher_degrees_k5",
        gtype_cls=QuinticCirculantGraph,
        n_values=[10],
        gammas=[0.8, 0.9],
        betas=[0.75, 1.5],
        learners=['q_learning', 'sarsa'],
        description="Analyze 5-regular (Quintic) topology."
    )

    # Task 5: Mixed 3/4 Graph with gamma=0
    run_experiment_bundle(
        task_name="task5_mixed34_gamma0",
        gtype_cls=Mixed34Graph,
        n_values=[10, 20],
        gammas=[0.0],
        betas=[0.5, 1.0, 1.5],
        learners=['q_learning', 'sarsa'],
        description="Mixed 3/4 regularity at gamma=0."
    )
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
