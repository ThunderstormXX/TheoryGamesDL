import os
import sys
import torch
import numpy as np

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

def run_sim(graph, num_nodes, gamma, beta=0.5, b=3.0, c=1.0, bonus=1.0, n_replications=10, num_iterations=1_000_000):
    np.random.seed(42)
    torch.manual_seed(42)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
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
    
    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
    q_now = learner.q_table[:, :, 0, :].cpu().numpy()
    probs = torch.softmax(torch.tensor(q_now) / temp, dim=-1).numpy()[..., 1]
    return probs

with open('igor_report.md', 'w') as f:
    f.write("# Отчет по симуляциям\n\n")
    
    gamma_values = [0.0, 0.3, 0.5, 0.7]
    reps = 10
    
    f.write("## 1. Звезда (=цепочка) (3 игрока)\n")
    graph = StarGraph(3, DEVICE)
    for g in gamma_values:
        probs = run_sim(graph, 3, g, n_replications=reps, num_iterations=200_000, beta=1.0)
        # check if central player (index 0) gets stuck
        center_stuck_count = 0
        min_ps = []
        for p in probs:
            if p[0] < 0.1 or p[0] > 0.9:
                center_stuck_count += 1
            min_ps.append(min(p))
            
        f.write(f"- Gamma = {g}:\n")
        f.write(f"  - Центральный игрок в ловушке в {center_stuck_count}/{reps} запусков.\n")
        f.write(f"  - Пример исходов вероятности P(C) для всех игроков (центр первый): {probs[0]}\n")

    f.write("\n## 2. Треугольник (3 игрока)\n")
    graph = TriangleGraph(DEVICE)
    for g in gamma_values:
        probs = run_sim(graph, 3, g, n_replications=reps, num_iterations=200_000, beta=1.0)
        trapped_2_count = 0
        powers = []
        for p in probs:
            stuck_count = sum(1 for val in p if val < 0.05)
            if stuck_count == 2:
                trapped_2_count += 1
            if stuck_count > 0:
                min_p = min(p)
                if min_p > 0:
                    power = -np.log10(min_p)
                    powers.append(power)
                    
        avg_x = np.mean(powers) if powers else 0
        f.write(f"- Gamma = {g}:\n")
        f.write(f"  - Ровно 2 игрока в ловушке в {trapped_2_count}/{reps} запусков.\n")
        if powers:
            f.write(f"  - Вероятность C у игроков в ловушке: 10^-{avg_x:.2f}\n")
        else:
            f.write(f"  - Ловушки не сформировались достаточно глубоко (<0.05).\n")
        f.write(f"  - Пример P(C): {probs[0]}\n")
        
    f.write("\n## 3. Игры 4 игроков (Топологии), Gamma=0.5, Beta=1.0\n")
    graphs4 = [
        ("Полный граф", CompleteGraph(4, DEVICE)),
        ("Звезда (центр - 0)", StarGraph(4, DEVICE)),
        ("Кольцо", RingGraph(4, DEVICE)),
        ("Цепочка", ChainGraph(4, DEVICE)),
        ("Колесо (центр - 0)", WheelGraph(4, DEVICE)),
    ]
    
    for name, gr in graphs4:
        f.write(f"### Топология: {name}\n")
        probs = run_sim(gr, 4, 0.5, n_replications=reps, num_iterations=200_000, beta=1.0)
        f.write(f"- Запусков: {reps}\n")
        f.write(f"- Примеры финальных вероятностей P(C) (каждая строка - запуск):\n")
        for i in range(min(5, reps)):
            f.write(f"  - {probs[i]}\n")
            
print("Report generated to igor_report.md")
