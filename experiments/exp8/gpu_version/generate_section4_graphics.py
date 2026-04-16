import os
import sys
import torch
import numpy as np
import matplotlib
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
DEVICE = torch.device('cpu')
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph, StarGraph

class CompleteGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.ones((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        return adj

class RingGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        for i in range(self.num_nodes):
            adj[i, (i + 1) % self.num_nodes] = 1.0
            adj[(i + 1) % self.num_nodes, i] = 1.0
        return adj

class ChainGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        for i in range(self.num_nodes - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        return adj

class WheelGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes, device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        for i in range(1, self.num_nodes):
            next_node = i + 1
            if next_node == self.num_nodes:
                next_node = 1
            adj[i, next_node] = 1.0
            adj[next_node, i] = 1.0
        return adj

def smooth(y, box_pts=10):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    y_smooth[-box_pts:] = y[-box_pts:]
    return y_smooth

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../experiments/exp8/papers/Q_Learning_on_graphs/figures'))
os.makedirs(out_dir, exist_ok=True)

GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9]
ALPHA = 0.01
BETA = 0.5
TEMP = 1.0 / BETA
ITERS = 1000000
RECORD_EVERY = 10000
REPS = 32

def run_simulation(graph, gamma):
    np.random.seed(42)
    torch.manual_seed(42)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    num_nodes = graph.num_nodes
    
    states = torch.zeros((REPS, num_nodes), dtype=torch.long, device=DEVICE)
    
    learner = BatchedGPUQLearner(
        batch_size=REPS, n_agents=num_nodes, action_space_size=2,
        learning_rate=ALPHA, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=TEMP, max_states=1,
    )
    reward_manager = BonusRewardManager(reward_type='pp', b=2.0, c=1.0, bonus=1.0)
    
    adj_batched = adj_t.unsqueeze(0).expand(REPS, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(REPS, -1)
    
    T_out = ITERS // RECORD_EVERY + 1
    p_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    qc_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    qd_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    
    def record_state(t_idx):
        q_now = learner.q_table[:, :, 0, :].cpu()
        probs = torch.softmax(q_now / TEMP, dim=-1).numpy()
        p_hist[t_idx] = probs[..., 1]
        qd_hist[t_idx] = q_now[..., 0].numpy()
        qc_hist[t_idx] = q_now[..., 1].numpy()
        
    record_state(0)
    
    with torch.no_grad():
        for t in tqdm(range(1, ITERS + 1), desc=f"Simulating Gamma={gamma}", leave=False, mininterval=1.0):
            actions = learner.get_actions(states) 
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
            if t % RECORD_EVERY == 0:
                record_state(t // RECORD_EVERY)
                
    return p_hist, qc_hist, qd_hist

def plot_dynamics(p_hist, qc_hist, qd_hist, title, out_path, graph_name):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    T_out, _, N = p_hist.shape
    x = np.arange(T_out) * RECORD_EVERY
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    for i in range(N):
        label = f"Agent {i}"
        col = colors[i % len(colors)]
        
        # P(C) plot
        mean_p = p_hist[:, :, i].mean(axis=1)
        std_p = p_hist[:, :, i].std(axis=1)
        
        # smooth
        mean_p = smooth(mean_p, 5)
        std_p = smooth(std_p, 5)
        ax1.plot(x, mean_p, label=label, color=col, linewidth=2)
        ax1.fill_between(x, np.clip(mean_p - std_p, 0, 1), np.clip(mean_p + std_p, 0, 1), color=col, alpha=0.15)
        
        # Q-values plot
        mean_qc = qc_hist[:, :, i].mean(axis=1)
        mean_qd = qd_hist[:, :, i].mean(axis=1)
        
        mean_qc = smooth(mean_qc, 5)
        mean_qd = smooth(mean_qd, 5)
        
        ax2.plot(x, mean_qc, label=f"{label} Q(C)", color=col, linestyle='-', linewidth=2)
        ax2.plot(x, mean_qd, label=f"{label} Q(D)", color=col, linestyle='--', linewidth=2)
        
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Probability of Cooperation P(C)", fontsize=12)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("P(C)")
    #ax1.legend(loc='best')
    
    ax2.set_title("Q-values: Q(C) [solid] vs Q(D) [dashed]", fontsize=12)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Q-value")
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

from multiprocessing import Pool, cpu_count
from functools import partial

def run_single_config(args):
    c_name, graph, g, graph_titles_c_name = args
    p_hist, qc_hist, qd_hist = run_simulation(graph, gamma=g)
    out_file = os.path.join(out_dir, f"{c_name}_gamma{g}.jpg")
    title = f"{graph_titles_c_name}, Gamma={g}"
    plot_dynamics(p_hist, qc_hist, qd_hist, title, out_file, c_name)
    return True

def main():
    print("Generating Section 4 Graphics...")
    configurations = {
        'triangle': CompleteGraph(3, DEVICE),
        'chain3': ChainGraph(3, DEVICE),
        'complete4': CompleteGraph(4, DEVICE),
        'chain4': ChainGraph(4, DEVICE),
        'star4': StarGraph(4, DEVICE),
        'ring4': RingGraph(4, DEVICE),
        'wheel4': WheelGraph(4, DEVICE)
    }
    
    graph_titles = {
        'triangle': 'Triangle (K3)',
        'chain3': 'Chain (3 nodes)',
        'complete4': 'Complete (K4)',
        'chain4': 'Chain (4 nodes)',
        'star4': 'Star (4 nodes)',
        'ring4': 'Ring (4 nodes)',
        'wheel4': 'Wheel (4 nodes)'
    }
    
    tasks = []
    for c_name, graph in configurations.items():
        for g in GAMMAS:
            tasks.append((c_name, graph, g, graph_titles[c_name]))
            
    total_runs = len(tasks)
    print(f"Total experiments to run: {total_runs}")
    
    # Run in parallel
    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} parallel workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_config, tasks), total=total_runs, desc="Overall Progress"))
        
    print('\nAll experiments completed.')

if __name__ == '__main__':
    # Need to set start method for multiprocessing using GPUs/PyTorch
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()