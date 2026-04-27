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
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, BaseGraph

class CompleteGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.ones((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        return adj

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../experiments/exp8/papers/Q_Learning_on_graphs/figures'))
os.makedirs(out_dir, exist_ok=True)

# Grid parameters
GAMMAS = np.linspace(0.0, 0.99, 15)
BETAS = np.linspace(0.1, 3.0, 15)
ALPHA = 0.01
ITERS = 250000  # Shorter simulation for faster grid search
REPS = 32

def run_simulation(graph, gamma, beta):
    TEMP = 1.0 / beta
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
    
    with torch.no_grad():
        for t in range(1, ITERS + 1):
            actions = learner.get_actions(states) 
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
    q_now = learner.q_table[:, :, 0, :].cpu()
    probs = torch.softmax(q_now / TEMP, dim=-1).numpy()
    final_p = probs[..., 1] # Probability of cooperation
    
    return final_p.mean(axis=0) # Mean over reps (shape: num_nodes)

from multiprocessing import Pool, cpu_count

def run_single_config(args):
    g_idx, b_idx, gamma, beta = args
    graph_star = StarGraph(4, DEVICE)
    graph_comp = CompleteGraph(4, DEVICE)
    
    # Run star graph
    p_star = run_simulation(graph_star, gamma, beta)
    # Return metrics for the center (node 0) and leaf (node 1)
    
    # Run complete graph
    p_comp = run_simulation(graph_comp, gamma, beta)
    
    return g_idx, b_idx, p_star[0], p_star[1], p_comp[0]

def main():
    print("Generating Phase Diagrams...")
    tasks = []
    for i, g in enumerate(GAMMAS):
        for j, b in enumerate(BETAS):
            tasks.append((i, j, g, b))
            
    total_runs = len(tasks)
    num_workers = min(cpu_count(), 8)
    
    # We will store results in matrices
    # shape: (len(GAMMAS), len(BETAS))
    heatmap_star_center = np.zeros((len(GAMMAS), len(BETAS)))
    heatmap_star_leaf = np.zeros((len(GAMMAS), len(BETAS)))
    heatmap_complete = np.zeros((len(GAMMAS), len(BETAS)))
    
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(run_single_config, tasks), total=total_runs, desc="Phase Diagram Progress"):
            g_idx, b_idx, p_star_c, p_star_l, p_comp = res
            heatmap_star_center[g_idx, b_idx] = p_star_c
            heatmap_star_leaf[g_idx, b_idx] = p_star_l
            heatmap_complete[g_idx, b_idx] = p_comp
            
    # Plotting
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
        
    X, Y = np.meshgrid(BETAS, GAMMAS)
    
    def plot_heatmap(data, title, filename):
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, data, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
        plt.colorbar(label='Probability of Cooperation P(C)')
        plt.xlabel(r'Exploration Parameter $\beta$')
        plt.ylabel(r'Discount Factor $\gamma$')
        plt.title(title)
        
        # Add a contour line for the 0.1 trap threshold
        plt.contour(X, Y, data, levels=[0.1, 0.5, 0.9], colors=['red', 'black', 'green'], linestyles='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()

    plot_heatmap(heatmap_star_center, "Phase Diagram: Star Graph (Center Agent)", "phase_diagram_star_center.jpg")
    plot_heatmap(heatmap_star_leaf, "Phase Diagram: Star Graph (Peripheral Agent)", "phase_diagram_star_leaf.jpg")
    plot_heatmap(heatmap_complete, "Phase Diagram: Complete Graph (Any Agent)", "phase_diagram_complete.jpg")
    
    print('\\nPhase diagrams generated successfully.')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
