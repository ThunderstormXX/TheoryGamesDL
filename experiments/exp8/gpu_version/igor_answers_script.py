import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Force CPU since it's just a few nodes and reps, CPU is faster for small graphs
os.environ['TRAP_DEVICE'] = 'cpu'
DEVICE = torch.device('cpu')

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph, StarGraph, WheelGraph, TriangleGraph
from experiments.exp8.gpu_version.trap_effect_experiment import detect_neighbor_gap_trap_intervals


# --- Additional Graph Structures ---

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


# --- Helper to run sim ---

def run_sim(graph, num_nodes, gamma, beta=0.5, b=3.0, c=1.0, bonus=1.0, n_replications=10, num_iterations=100_000):
    np.random.seed(42)
    torch.manual_seed(42)
    
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
    batch_size = n_replications
    states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=DEVICE)
    temp = 1.0 / beta
    
    learner = BatchedGPUQLearner(
        batch_size=batch_size,
        n_agents=num_nodes,
        action_space_size=2,
        learning_rate=0.02,
        discount_factor=gamma,
        exploration_rate=0.0,
        strategy='boltzmann',
        temperature=temp,
        max_states=1,
    )
    
    reward_manager = BonusRewardManager(reward_type='pp', b=b, c=c, bonus=bonus)
    
    record_every = 100
    T_out = num_iterations // record_every + 1
    p_hist = np.zeros((T_out, batch_size, num_nodes), dtype=np.float32)
    
    # initial state
    q_now = learner.q_table[:, :, 0, :]
    probs = torch.softmax(q_now / temp, dim=-1)
    p_hist[0] = probs[..., 1].cpu().numpy()
    
    adj_batched = adj_t.unsqueeze(0).expand(batch_size, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
            if t % record_every == 0:
                q_now = learner.q_table[:, :, 0, :]
                probs = torch.softmax(q_now / temp, dim=-1)
                p_hist[t // record_every] = probs[..., 1].cpu().numpy()
                
    return p_hist, adj_t.cpu().numpy()


# --- Questions ---

def analyze_trap(p_hist, adj, graph_name, gamma):
    # p_hist shape: (T, bs, N)
    T, bs, N = p_hist.shape
    
    min_duration_points = 500 // 100 # RECORD_EVERY=100
    TRAP_NEIGHBOR_GAP = 0.1
    
    traps = 0
    center_stuck = 0
    
    # Track lowest probabilities of C
    min_pcs = []
    trapped_players_count = []
    
    for b in range(bs):
        ints_b = detect_neighbor_gap_trap_intervals(p_hist[:, b, :], adj, TRAP_NEIGHBOR_GAP, min_duration_points)
        if ints_b:
            traps += 1
            # Look at the end of the first trap interval
            t_trap_mid = ints_b[0][1] - 1
            p_trap = p_hist[t_trap_mid, b, :]
            
            if graph_name == "Star3":
                # center is index 0
                if p_trap[0] < 0.1 or p_trap[0] > 0.9:
                    # check if it's the most extreme
                    if abs(0.5 - p_trap[0]) > max(abs(0.5 - p_trap[1]), abs(0.5 - p_trap[2])):
                        center_stuck += 1
                        
            if graph_name == "Triangle3":
                # how many are close to 0 or close to 1
                near_zero = sum(1 for p in p_trap if p < 0.05)
                near_one = sum(1 for p in p_trap if p > 0.95)
                
                # generally the 'trap' in these games is low P(C) for defectors and high P(C) for exploited
                stuck = sum(1 for p in p_trap if p < 0.1) # defectors
                trapped_players_count.append(stuck)
                
                # min p(C):
                min_p = min(p_trap)
                min_pcs.append(min_p)

    return {
        "traps_found": traps,
        "center_stuck": center_stuck,
        "min_pcs": min_pcs,
        "trapped_players_count": trapped_players_count,
        "p_end_sample": p_hist[-1, 0, :]
    }


def main():
    print("--- Phase 1 ---")
    gamma_values = [0.0, 0.3, 0.5, 0.7]
    reps = 10
    
    print("\n1. Star 3 (Chain)")
    graph = StarGraph(3, DEVICE)
    for g in gamma_values:
        p_hist, adj = run_sim(graph, 3, g, n_replications=reps)
        res = analyze_trap(p_hist, adj, "Star3", g)
        print(f"Gamma = {g}: Traps found = {res['traps_found']}/{reps}. Center stuck in {res['center_stuck']}/{res['traps_found']} traps. Sample P(C) end: {res['p_end_sample']}")
        
    print("\n2. Triangle 3")
    graph = TriangleGraph(DEVICE)
    for g in gamma_values:
        p_hist, adj = run_sim(graph, 3, g, n_replications=reps)
        res = analyze_trap(p_hist, adj, "Triangle3", g)
        
        avg_stuck = np.mean(res['trapped_players_count']) if res['trapped_players_count'] else 0
        min_p = np.mean(res['min_pcs']) if res['min_pcs'] else 1.0
        
        # calculate 10^-x
        x = -np.log10(min_p) if min_p > 0 else float('inf')
        
        print(f"Gamma = {g}: Traps found = {res['traps_found']}/{reps}. Avg trapped players = {avg_stuck:.1f}. Avg min P(C) = {min_p:.2e} (10^-{x:.2f}). Sample P(C) end: {res['p_end_sample']}")


    print("\n--- Phase 2: 4 Players ---")
    graphs4 = [
        ("Complete", CompleteGraph(4, DEVICE)),
        ("Star", StarGraph(4, DEVICE)),
        ("Ring", RingGraph(4, DEVICE)),
        ("Chain", ChainGraph(4, DEVICE)),
        ("Wheel", WheelGraph(4, DEVICE)),
    ]
    
    for name, gr in graphs4:
        p_hist, adj = run_sim(gr, 4, 0.5, n_replications=reps) # gamma=0.5
        # just record the end states
        print(f"\n{name} 4 (Gamma=0.5, 10 reps):")
        # How many traps?
        ints_any = 0
        end_states = []
        for b in range(reps):
            ints_b = detect_neighbor_gap_trap_intervals(p_hist[:, b, :], adj, 0.1, 5)
            if ints_b: ints_any += 1
            end_states.append(p_hist[-1, b, :])
            
        print(f" Traps found: {ints_any}/{reps}")
        print(f" Sample P(C) outcomes ends:")
        for b in range(min(5, reps)):
            print(f"   Rep {b}: {end_states[b]}")
            
if __name__ == '__main__':
    main()
