import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

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

class DiamondGraph(BaseGraph):
    def __init__(self, num_nodes=4, device=None):
        super().__init__(num_nodes=num_nodes, device=device)
    def generate_adjacency_matrix(self):
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
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 0] = adj[0, 2] = 1.0
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

base_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../experiments/exp8/results/high_gamma_plots_single'))

def run_sim_task(args):
    graph_name, gamma, num_iterations, base_dir = args
    print(f"Starting {graph_name} with Gamma={gamma}...")
    
    if graph_name == "Complete": gr = CompleteGraph(4, DEVICE)
    elif graph_name == "Star": gr = StarGraph(4, DEVICE)
    elif graph_name == "Ring": gr = RingGraph(4, DEVICE)
    elif graph_name == "Chain": gr = ChainGraph(4, DEVICE)
    elif graph_name == "Diamond": gr = DiamondGraph(4, DEVICE)
    elif graph_name == "Paw": gr = PawGraph(4, DEVICE)
    
    np.random.seed() # Randomize seed per task
    torch.manual_seed(np.random.randint(0, 100000))
    
    adj_t = gr.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
    num_nodes = gr.num_nodes
    n_replications = 1 # EXACTLY ONE GAME
    states = torch.zeros((n_replications, num_nodes), dtype=torch.long, device=DEVICE)
    temp = 1.0 / 1.0 # beta=1.0
    
    learner = BatchedGPUQLearner(
        batch_size=n_replications, n_agents=num_nodes, action_space_size=2,
        learning_rate=0.02, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=temp, max_states=1,
    )
    b, c, bonus = 3.0, 1.0, 1.0
    reward_manager = BonusRewardManager(reward_type='pp', b=b, c=c, bonus=bonus)
    
    adj_batched = adj_t.unsqueeze(0).expand(n_replications, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(n_replications, -1)
    
    record_every = 10_000
    T_out = num_iterations // record_every + 1
    p_hist = np.zeros((T_out, num_nodes), dtype=np.float32)
    
    q_now = learner.q_table[:, :, 0, :].cpu()
    p_hist[0] = torch.softmax(q_now / temp, dim=-1).numpy()[0, :, 1]
    
    with torch.no_grad():
        for t in tqdm(range(1, num_iterations + 1), desc=f"{graph_name} g={gamma}", leave=False):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
            if t % record_every == 0:
                q_now = learner.q_table[:, :, 0, :].cpu()
                p_hist[t // record_every] = torch.softmax(q_now / temp, dim=-1).numpy()[0, :, 1]
                
    # Now plot directly
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    plt.figure(figsize=(10, 5))
    x = np.arange(T_out) * record_every
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    for i in range(num_nodes):
        raw_y = p_hist[:, i]
        y = smooth(raw_y, box_pts=5)
        
        # Consistent label mappings
        if graph_name == "Complete":
            label = f"Агент {i}"
            col = colors[i % len(colors)]
        elif graph_name == "Star":
            if i == 0:
                label = "Центр (Агент 0)"
                col = '#e74c3c'
            else:
                label = f"Периферия (Агент {i})"
                col = colors[i]
        elif graph_name == "Ring":
            label = f"Узел кольца (Агент {i})"
            col = colors[i]
        elif graph_name == "Chain":
            if i in [0, 3]:
                label = f"Крайний (Агент {i})"
                col = '#2ecc71' if i == 0 else '#3498db'
            else:
                label = f"Внутренний (Агент {i})"
                col = '#e74c3c' if i == 1 else '#9b59b6'
        elif graph_name == "Diamond":
            if i in [0, 3]:
                label = f"Степень 2 (Агент {i})"
                col = '#3498db' if i == 0 else '#2ecc71'
            else:
                label = f"Степень 3 (Агент {i})"
                col = '#e74c3c' if i == 1 else '#9b59b6'
        elif graph_name == "Paw":
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

        lw = 2.5 if ("Центр" in label or "Степень 3" in label or "Внутренний" in label) else 1.8
        alpha = 1.0 if lw > 2.0 else 0.8
        
        plt.plot(x, y, label=label, color=col, alpha=alpha, linewidth=lw)
        
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.4, label='Зона ловушки (P(C) < 0.1)')
    plt.ylim(-0.02, 0.45) 
    plt.title(f"{graph_name} Graph (4 agents) [Одиночный запуск] - Gamma={gamma}", fontsize=14, fontweight='bold')
    plt.xlabel('Итерации', fontsize=12)
    plt.ylabel('Вероятность Кооперации P(C)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    
    fname = f"{graph_name}_g{gamma}.png"
    out_path = os.path.join(base_dir, graph_name, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Finished {graph_name} Gamma={gamma}")
    return graph_name, gamma, fname


def main():
    print("Running sequential simulations (1 rep, 300k iterations)...")
    gamma_values = [0.8, 0.9, 0.95, 0.97, 0.99]
    iters = 300_000
    graphs = ["Complete", "Star", "Ring", "Chain", "Diamond", "Paw"]
    
    os.makedirs(base_out_dir, exist_ok=True)
    for gname in graphs:
        os.makedirs(os.path.join(base_out_dir, gname), exist_ok=True)
        
    # Run sequentially
    for gname in graphs:
        for gam in gamma_values:
            run_sim_task((gname, gam, iters, base_out_dir))
            
    # Build report
    report_lines = []
    report_lines.append(r"# Исследование Ловушек (Единичные симуляции) $\gamma \to 1$" + "\n\n")
    report_lines.append("Ниже представлены наглядные траектории **конкретно взятых одиночных игр** (без усреднений) 300 000 итераций.\n")
    
    for graph_name in graphs:
        report_lines.append(f"## Топология: {graph_name}\n")
        for gamma in gamma_values:
            fname = f"{graph_name}_g{gamma}.png"
            rel_path = f"/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/high_gamma_plots_single/{graph_name}/{fname}"
            report_lines.append(f"### Gamma = {gamma}\n")
            report_lines.append(f"![{graph_name} Gamma {gamma}]({rel_path})\n\n")
            
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../high_gamma_report_single.md'))
    with open(report_path, 'w') as f:
        f.write("".join(report_lines))
        
    print(f"Done! Report saved to {report_path}")


if __name__ == "__main__":
    main()
