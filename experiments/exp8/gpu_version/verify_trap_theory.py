import os
import sys
import torch
import numpy as np
import math
from scipy.special import comb
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config
from experiments.exp8.gpu_version.core.graph_structure import (
    RingGraph, CubicCirculantGraph, QuarticCirculantGraph, QuinticCirculantGraph
)

def expected_t_breakout(K, b, c, beta):
    p0 = 1.0 / (1.0 + math.exp(beta * c * K))
    m_star = int(math.floor(c * K / b)) + 1
    
    if m_star > K:
        return float('inf')
        
    p_breakout = 0.0
    for m in range(m_star, K + 1):
        p_breakout += comb(K, m) * (p0 ** m) * ((1 - p0) ** (K - m))
        
    p_breakout *= p0  # Agent i must also cooperate
    
    if p_breakout == 0:
        return float('inf')
        
    return 1.0 / p_breakout

def run_experiment(graph_class, n_nodes, K, beta, b, c, gamma, alpha, batch_size=1000, max_steps=100000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    gpu_config.device = device
    
    graph = graph_class(num_nodes=n_nodes, device=device)
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1)
    
    learner = BatchedGPUQLearner(
        batch_size=batch_size,
        n_agents=n_nodes,
        action_space_size=2,
        learning_rate=alpha,
        discount_factor=gamma,
        exploration_rate=0.0,
        strategy='boltzmann',
        temperature=1.0/beta,
        max_states=1
    )
    
    q_d = 1.0 / (1.0 - gamma)
    # Initialize each agent according to its degree
    for i in range(n_nodes):
        q_c_i = 1.0 / (1.0 - gamma) - c * degrees[i].item()
        learner.q_table[:, i, 0, 0] = q_d
        learner.q_table[:, i, 0, 1] = q_c_i
    
    reward_manager = RewardManager(reward_type='pp', b=b, c=c)
    
    states = torch.zeros((batch_size, n_nodes), dtype=torch.long, device=device)
    adj_batched = adj.unsqueeze(0).expand(batch_size, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(batch_size, -1)
    
    breakout_times = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    not_broken_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for t in range(1, max_steps + 1):
            if not not_broken_mask.any():
                break
                
            actions = learner.get_actions(states) # (B, N)
            q_now = learner.q_table[:, :, 0, :] # (B, N, 2)
            
            # Check for breakout of agent 0 specifically to match the theoretical E[T] for a single agent.
            broke_out = (q_now[:, 0, 1] > q_now[:, 0, 0])
            
            new_breakouts = broke_out & not_broken_mask
            if new_breakouts.any():
                breakout_times[new_breakouts] = t
                not_broken_mask[new_breakouts] = False
                
            actions_f = actions.float()
            rewards = reward_manager.calculate_rewards(actions_f, adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
    valid_times = breakout_times[breakout_times > 0].float()
    if len(valid_times) == 0:
        emp_mean = float('inf')
    else:
        emp_mean = valid_times.mean().item()
        
    th_mean = expected_t_breakout(K, b, c, beta)
    return emp_mean, th_mean

if __name__ == '__main__':
    b = 2.0
    c = 1.0
    gamma = 0.95
    alpha = 0.1
    n_nodes = 8 # even number >= 6 for Quintic
    
    graphs_to_test = [
        (RingGraph, 2, "2-Regular (Ring)"),
        (CubicCirculantGraph, 3, "3-Regular"),
        (QuarticCirculantGraph, 4, "4-Regular"),
        (QuinticCirculantGraph, 5, "5-Regular")
    ]
    
    betas = np.linspace(0.5, 2.0, 10)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['r', 'g', 'b', 'm']
    
    for (g_class, K, name), color in zip(graphs_to_test, colors):
        print(f"\nTesting {name} (K={K})")
        emp_means = []
        th_means = []
        
        for beta in betas:
            emp, th = run_experiment(g_class, n_nodes, K, beta, b, c, gamma, alpha, batch_size=10000, max_steps=15000)
            print(f"  Beta: {beta:.2f} | Emp E[T]: {emp:.2f} | Th E[T]: {th:.2f}")
            emp_means.append(emp)
            th_means.append(th)
            
        plt.plot(betas, th_means, label=f'{name} Theory', color=color, linestyle='--')
        plt.plot(betas, emp_means, label=f'{name} Emp', color=color, marker='x')

    plt.yscale('log')
    plt.xlabel('Beta (Inverse Temperature)')
    plt.ylabel('Expected Breakout Time E[T] for Agent 0')
    plt.title('Trap Breakout Time: Theory vs Practice across Topologies')
    plt.legend()
    plt.grid(True)
    
    out_dir = os.path.join(os.path.dirname(__file__), '../../results/trap_theory')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'breakout_time_topologies.png')
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")
