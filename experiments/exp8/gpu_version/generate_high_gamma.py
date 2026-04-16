import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
os.environ['TRAP_DEVICE'] = 'cpu'
DEVICE = torch.device('cpu')

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
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

# Newly Implemented missing 4-node topologies
class DiamondGraph(BaseGraph):
    def __init__(self, num_nodes=4, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        # Square with crossed line = Diamond. Complete minus 1 edge. Let's disconnect 0 and 3.
        adj = torch.ones((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        adj[0, 3] = 0.0
        adj[3, 0] = 0.0
        return adj

class PawGraph(BaseGraph):
    def __init__(self, num_nodes=4, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        # nodes 0,1,2 form a triangle
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 0] = adj[0, 2] = 1.0
        # node 3 connected to node 2
        adj[3, 2] = adj[2, 3] = 1.0
        return adj

def smooth(y, box_pts=10):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    y_smooth[-box_pts:] = y[-box_pts:]
    return y_smooth

base_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../experiments/exp8/results/high_gamma_plots'))

# Simulation function
def run_sim_large_scale(graph, gamma, beta=1.0, b=3.0, c=1.0, bonus=1.0, 
                        n_replications=32, num_iterations=1_000_000, record_every=10_000):
    np.random.seed(42)
    torch.manual_seed(42)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
    num_nodes = graph.num_nodes
    states = torch.zeros((n_replications, num_nodes), dtype=torch.long, device=DEVICE)
    temp = 1.0 / float(beta)
    
    learner = BatchedGPUQLearner(
        batch_size=n_replications, n_agents=num_nodes, action_space_size=2,
        learning_rate=0.02, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=temp, max_states=1,
    )
    reward_manager = BonusRewardManager(reward_type='pp', b=b, c=c, bonus=bonus)
    
    adj_batched = adj_t.unsqueeze(0).expand(n_replications, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(n_replications, -1)
    
    T_out = num_iterations // record_every + 1
    p_hist = np.zeros((T_out, n_replications, num_nodes), dtype=np.float32)
    
    q_now = learner.q_table[:, :, 0, :].cpu()
    p_hist[0] = torch.softmax(q_now / temp, dim=-1).numpy()[..., 1]
    
    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states) # Use random exploration dynamically? Currently e=0.0
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
            if t % record_every == 0:
                q_now = learner.q_table[:, :, 0, :].cpu()
                p_hist[t // record_every] = torch.softmax(q_now / temp, dim=-1).numpy()[..., 1]
                
    end_probs = p_hist[-1]
    trap_threshold = 0.1
    trap_rates = (end_probs < trap_threshold).mean(axis=0)
    
    return p_hist, trap_rates


def plot_probs_mean_std(p_hist, title, out_path, graph_type, record_every=10_000):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    plt.figure(figsize=(10, 5))
    
    T_out, bs, N = p_hist.shape
    x = np.arange(T_out) * record_every
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    for i in range(N):
        mean_y = p_hist[:, :, i].mean(axis=1)
        std_y = p_hist[:, :, i].std(axis=1)
        
        mean_y = smooth(mean_y, box_pts=5)
        std_y = smooth(std_y, box_pts=5)
        
        # Determine labels based on topologies
        if graph_type == "Complete":
            label = f"Агент {i}"
            col = colors[i % len(colors)]
        elif graph_type == "Star":
            if i == 0:
                label = "Центр (Агент 0)"
                col = '#e74c3c'
            else:
                label = f"Периферия (Агент {i})"
                col = colors[i]
        elif graph_type == "Ring":
            label = f"Узел кольца (Агент {i})"
            col = colors[i]
        elif graph_type == "Chain":
            if i in [0, 3]:
                label = f"Крайний (Агент {i})"
                col = '#2ecc71' if i == 0 else '#3498db'
            else:
                label = f"Внутренний (Агент {i})"
                col = '#e74c3c' if i == 1 else '#9b59b6'
        elif graph_type == "Diamond":
            if i in [0, 3]:
                label = f"Степень 2 (Агент {i})" # these lack connection to each other
                col = '#3498db' if i == 0 else '#2ecc71'
            else:
                label = f"Степень 3 (Агент {i})" # these are connected to everyone
                col = '#e74c3c' if i == 1 else '#9b59b6'
        elif graph_type == "Paw":
            if i == 3:
                label = "Хвост (Степень 1, Агент 3)"
                col = '#3498db'
            elif i == 2:
                label = "Центр-Соединитель (Степень 3, Агент 2)"
                col = '#e74c3c'
            else:
                label = f"Основание треугольника (Агент {i})"
                col = '#2ecc71' if i == 0 else '#9b59b6'
        else:
            label = f"Агент {i}"
            col = colors[i % len(colors)]

        # Highlight important nodes with bold line
        lw = 2.5 if ("Центр" in label or "Степень 3" in label or "Внутренний" in label) else 1.8
        alpha = 1.0 if lw > 2.0 else 0.8
        
        plt.plot(x, mean_y, label=label, color=col, alpha=alpha, linewidth=lw)
        plt.fill_between(x, np.clip(mean_y - std_y, 0, 1), np.clip(mean_y + std_y, 0, 1), color=col, alpha=0.15)
        
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.4, label='Зона ловушки (P(C) < 0.1)')
    plt.ylim(-0.02, 0.45) 
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Итерации', fontsize=12)
    plt.ylabel('Средняя Вероятность Кооперации P(C)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    print("Running high-gamma scale simulations (32 reps, 1M iterations)...")
    
    gamma_values = [0.8, 0.9, 0.95, 0.97, 0.99]
    reps = 32
    iters = 1_000_000
    
    graphs = [
        ("Complete", CompleteGraph(4, DEVICE)),
        ("Star", StarGraph(4, DEVICE)),
        ("Ring", RingGraph(4, DEVICE)),
        ("Chain", ChainGraph(4, DEVICE)),
        ("Diamond", DiamondGraph(4, DEVICE)),
        ("Paw", PawGraph(4, DEVICE))
    ]
    
    report_lines = []
    report_lines.append(r"# Исследование Ловушек при Высоком Дисконтировании ($\gamma \to 1$)" + "\n\n")
    report_lines.append("В данном отчете мы исследовали **все 6 возможных связных графов из 4 узлов**. Это покрывает структуры с картинки (Полный, Звезда, Кольцо, Цепь) и дополняет их промежуточными графами (Diamond, Paw).\n")
    report_lines.append(f"Моделирование запущено с высокими степенями дальновидности: $\\gamma \in {{gamma_values}}$.\n\n")
    
    for graph_name, gr in graphs:
        # Create output directory
        topology_dir = os.path.join(base_out_dir, graph_name)
        os.makedirs(topology_dir, exist_ok=True)
        
        report_lines.append(f"## Топология: {graph_name}\n")
        report_lines.append(f"Все графики (Mean ± 1 std) для 32 независимых запусков.\n\n")
        
        for g in gamma_values:
            print(f"Running {graph_name} with Gamma={g}...")
            p_hist, trap_r = run_sim_large_scale(gr, gamma=g, n_replications=reps, num_iterations=iters)
            
            fname = f"{graph_name}_g{g}.png"
            out_path = os.path.join(topology_dir, fname)
            
            title = f"{graph_name} Graph (4 agents) - Gamma={g}"
            plot_probs_mean_std(p_hist, title, out_path, graph_name)
            
            # Formulate markdown relative link properly 
            rel_path = f"/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/high_gamma_plots/{graph_name}/{fname}"
            report_lines.append(f"### Gamma = {g}\n")
            report_lines.append(f"![{graph_name} Gamma {g}]({rel_path})\n\n")
            
    # Write report
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../high_gamma_report.md'))
    with open(report_path, 'w') as f:
        f.write("".join(report_lines))
        
    print(f"Done. Report updated in {report_path}")

if __name__ == "__main__":
    main()
