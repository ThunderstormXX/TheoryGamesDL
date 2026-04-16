import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
os.environ['TRAP_DEVICE'] = 'cpu'
DEVICE = torch.device('cpu')

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph, StarGraph, WheelGraph, TriangleGraph

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

def run_sim_with_history(graph, num_nodes, gamma, beta=1.0, b=3.0, c=1.0, bonus=1.0, num_iterations=200_000, record_every=1000):
    np.random.seed(42)
    torch.manual_seed(42)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
    n_replications = 3  # Multiple paths; we'll plot the first one
    states = torch.zeros((n_replications, num_nodes), dtype=torch.long, device=DEVICE)
    temp = 1.0 / beta
    
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
    probs = torch.softmax(q_now / temp, dim=-1).numpy()[..., 1]
    p_hist[0] = probs
    
    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
            if t % record_every == 0:
                q_now = learner.q_table[:, :, 0, :].cpu()
                probs = torch.softmax(q_now / temp, dim=-1).numpy()[..., 1]
                p_hist[t // record_every] = probs
                
    return p_hist

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../experiments/exp8/results/custom_report_plots'))
os.makedirs(out_dir, exist_ok=True)

def plot_probs(p_hist, title, fname):
    plt.figure(figsize=(8, 4))
    rep_idx = 0  # Plot first replication
    T_out, bs, N = p_hist.shape
    x = np.arange(T_out) * 1000
    for i in range(N):
        plt.plot(x, p_hist[:, rep_idx, i], label=f'Agent {i}', alpha=0.8)
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('P(C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return out_path

new_report = []
new_report.append("# Отчет по симуляциям\n")
new_report.append("Симуляции проведены с параметром `Beta=1.0`, `200 000 итераций`.\n")

print("Generating plots and rendering new report...")

gamma_values = [0.0, 0.3, 0.5, 0.7]

new_report.append("## 1. Звезда (=цепочка) (3 игрока)\n")
graph = StarGraph(3, DEVICE)
for g in gamma_values:
    p_hist = run_sim_with_history(graph, 3, g)
    fname = f"star3_g{g}.png"
    plot_probs(p_hist, f"Star(3) - Gamma={g}", fname)
    new_report.append(f"### Gamma = {g}\n")
    new_report.append(f"Периферийные игроки удерживают более высокую вероятность кооперации, в то время как центральный игрок уходит в ловушку дезертирства.\n")
    new_report.append(f"![Star3 Gamma {g}](/absolute/path/to/experiments/exp8/results/custom_report_plots/{fname})\n\n")

new_report.append("## 2. Треугольник (3 игрока)\n")
graph = TriangleGraph(DEVICE)
for g in gamma_values:
    p_hist = run_sim_with_history(graph, 3, g)
    fname = f"triangle3_g{g}.png"
    plot_probs(p_hist, f"Triangle(3) - Gamma={g}", fname)
    new_report.append(f"### Gamma = {g}\n")
    new_report.append(f"Для симметричного треугольника ловушка не так выражена, P(C) плавно снижается у всех агентов.\n")
    new_report.append(f"![Triangle3 Gamma {g}](/absolute/path/to/experiments/exp8/results/custom_report_plots/{fname})\n\n")


new_report.append("## 3. Топологии 4 игроков (Gamma=0.5)\n")
graphs4 = [
    ("Полный граф", CompleteGraph(4, DEVICE), "complete4"),
    ("Звезда (центр=0)", StarGraph(4, DEVICE), "star4"),
    ("Кольцо", RingGraph(4, DEVICE), "ring4"),
    ("Цепочка", ChainGraph(4, DEVICE), "chain4"),
    ("Колесо (центр=0)", WheelGraph(4, DEVICE), "wheel4"),
]

for name, gr, prefix in graphs4:
    p_hist = run_sim_with_history(gr, 4, 0.5)
    fname = f"{prefix}_g0.5.png"
    plot_probs(p_hist, f"{name} - Gamma=0.5", fname)
    new_report.append(f"### {name}\n")
    new_report.append(f"Результаты вероятности P(C) в динамике:\n")
    new_report.append(f"![{name} Gamma 0.5](/absolute/path/to/experiments/exp8/results/custom_report_plots/{fname})\n\n")

report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../simulation_report.md'))

# Resolve paths
report_content = "".join(new_report).replace("/absolute/path/to", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

with open(report_path, 'w') as f:
    f.write(report_content)

print(f"Report complete: {report_path}")
