#!/usr/bin/env python3
"""
Comprehensive experiment to verify the drift-model proof of EV-SARSA trap exit time.
Uses the existing GPU framework (BatchedGPUQLearner, graph structures, RewardManager).

Tests all claims from the proof:
  1. Lemma 1: Q-value fixed points in trap (Q*(D), Q*(C), gap Δ*=ck)
  2. Lemma 2: Cooperation probability p0 = 1/(1 + e^{β·ck})
  3. Lemma 3: One-step breakout impossibility & AR(1) drift dynamics
  4. Theorem: E[T] ≈ c·(1 + e^{βck})² / (αb)  [drift model]
  5. Comparison with old one-step model: E[T] ~ (1+e^{βck})^{m*+1}
  6. α-dependence: E[T] ∝ 1/α

KEY DIFFERENCE from verify_trap_theory.py: implements EV-SARSA update
(expected value under Boltzmann policy) instead of Q-learning (max).

NOTE on rewards: The proof uses r(C) = b·Σa_j - ck + 1, r(D) = b·Σa_j + 1.
The RewardManager 'pp' mode gives: b·pool - c·x_i·k.  In addition we add +1.
Since action encoding here is: action=1 → Cooperate, action=0 → Defect,
cooperators = (1-actions) where actions from learner uses 0=best(D), 1=other(C).
Actually the learner encodes: action_space[0] = D, action_space[1] = C.
Boltzmann with Q(D) > Q(C) → samples action=0 (D) mostly.

We carefully match the proof's conventions below.
"""

import os
import sys
import math
import torch
import numpy as np
from scipy.special import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp8.gpu_version.core.graph_structure import (
    RingGraph, CubicCirculantGraph, QuarticCirculantGraph, QuinticCirculantGraph
)
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config

# ═══════════════════════════════════════════════════════════════
# GPU device setup
# ═══════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# ═══════════════════════════════════════════════════════════════
# EV-SARSA simulation (vectorized, batch-parallel)
# ═══════════════════════════════════════════════════════════════

class BatchEVSarsa:
    """
    Batched EV-SARSA on GPU/MPS, exactly matching the proof specification.
    
    Convention: Q-table shape (B, N, 2) where index 0 = D, index 1 = C.
    Boltzmann: π(C) = σ(β·(Q(C) - Q(D))) = 1/(1+e^{β(Q(D)-Q(C))}).
    """
    
    def __init__(self, batch_size, n_agents, alpha, beta, gamma, device):
        self.B = batch_size
        self.N = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        
        # Q-table: (B, N, 2)  — action 0=D, 1=C
        self.q = torch.zeros((batch_size, n_agents, 2), device=device, dtype=torch.float32)
    
    def init_trap(self, degrees):
        """
        Initialize Q-values at trap fixed points (Lemma 1).
        degrees: (N,) tensor of node degrees.
        """
        # Q*(D) = 1/(1-γ),  Q*(C) = 1 - c·k + γ/(1-γ)
        # But we don't know c here, so we parametrize differently:
        # We'll set them externally.
        pass
    
    def init_trap_values(self, q_d_vals, q_c_vals):
        """
        q_d_vals: (N,) or scalar — Q*(D) for each agent
        q_c_vals: (N,) or scalar — Q*(C) for each agent
        """
        if isinstance(q_d_vals, (int, float)):
            self.q[:, :, 0] = q_d_vals
        else:
            self.q[:, :, 0] = q_d_vals.unsqueeze(0)
        
        if isinstance(q_c_vals, (int, float)):
            self.q[:, :, 1] = q_c_vals
        else:
            self.q[:, :, 1] = q_c_vals.unsqueeze(0)
    
    def pi_c(self):
        """π(C) = 1/(1 + e^{β(Q(D)-Q(C))}), returns (B, N)."""
        diff = self.beta * (self.q[:, :, 0] - self.q[:, :, 1])  # β(Q(D)-Q(C))
        diff = torch.clamp(diff, max=80)  # overflow guard
        return 1.0 / (1.0 + torch.exp(diff))
    
    def sample_actions(self):
        """
        Sample actions: 1=C, 0=D. Returns (B, N) int tensor.
        """
        p_c = self.pi_c()  # (B, N)
        return (torch.rand_like(p_c) < p_c).long()
    
    def expected_value(self):
        """V = π(C)Q(C) + π(D)Q(D), returns (B, N)."""
        p_c = self.pi_c()
        return p_c * self.q[:, :, 1] + (1 - p_c) * self.q[:, :, 0]
    
    def update(self, actions, rewards):
        """
        EV-SARSA update:
          Q(a) ← Q(a) + α[r(a) + γV - Q(a)]
        
        actions: (B, N) — 0=D, 1=C
        rewards: (B, N) — reward for the chosen action
        """
        V = self.expected_value()  # (B, N)
        td_target = rewards + self.gamma * V  # (B, N)
        
        # Current Q for chosen action
        actions_oh = actions.unsqueeze(-1)  # (B, N, 1)
        current_q = torch.gather(self.q, 2, actions_oh).squeeze(-1)  # (B, N)
        
        # New Q value
        new_q = current_q + self.alpha * (td_target - current_q)
        
        # Scatter back
        self.q.scatter_(2, actions_oh, new_q.unsqueeze(-1))


def compute_rewards_pp(actions, adj, degrees, b, c):
    """
    Proof reward function:
      r_i(C) = b·Σ_{j∈N_i} a_j - c·k_i + 1
      r_i(D) = b·Σ_{j∈N_i} a_j + 1
    
    actions: (B, N) — 0=D, 1=C
    adj: (N, N) adjacency matrix
    degrees: (N,) degree vector
    
    Returns: (B, N) rewards.
    """
    B, N = actions.shape
    actions_f = actions.float()  # (B, N) — 1.0 for C, 0.0 for D
    
    # Neighbor cooperation count: (B, N)
    neighbor_coop = torch.mm(actions_f.view(B, N), adj.t())  # each row: sum of a_j for neighbors
    # Note: adj is symmetric, so adj.t() == adj. But being explicit.
    # Actually for batch: need bmm or einsum
    # Since adj is same for all batches:
    neighbor_coop = actions_f @ adj  # (B, N) @ (N, N) → (B, N)
    
    # Base reward
    base = b * neighbor_coop + 1.0  # (B, N)
    
    # Cost for cooperators
    cost = c * degrees.unsqueeze(0) * actions_f  # (B, N) — only cooperators pay
    
    return base - cost


# ═══════════════════════════════════════════════════════════════
# Theoretical predictions
# ═══════════════════════════════════════════════════════════════

def theory_p0(beta, c, k):
    return 1.0 / (1.0 + math.exp(beta * c * k))

def theory_drift_ET(alpha, b, c, beta, k):
    p0 = theory_p0(beta, c, k)
    return c / (alpha * b * p0**2)

def theory_onestep_ET(b, c, beta, k):
    p0 = theory_p0(beta, c, k)
    m_star = int(math.floor(c * k / b)) + 1
    if m_star > k:
        return float('inf')
    p_br = 0.0
    for m in range(m_star, k + 1):
        p_br += comb(k, m, exact=True) * (p0**m) * ((1 - p0)**(k - m))
    p_br *= p0
    if p_br <= 0:
        return float('inf')
    return 1.0 / p_br

# ═══════════════════════════════════════════════════════════════
# Experiment A: Verify Lemma 1 — Q-value fixed points
# ═══════════════════════════════════════════════════════════════

def experiment_lemma1(device, out_dir):
    print("=" * 60)
    print("EXPERIMENT A: Verify Lemma 1 — Q-value fixed points in trap")
    print("=" * 60)
    
    b, c, gamma, alpha = 3.0, 1.0, 0.95, 0.1
    beta_trap = 10.0  # high β to keep agents in trap
    n_nodes = 8
    batch_size = 64
    n_steps = 30000
    
    configs = [
        ("k=2 (Ring)", RingGraph(n_nodes, device=device), 2),
        ("k=3 (Cubic)", CubicCirculantGraph(n_nodes, device=device), 3),
        ("k=4 (Quartic)", QuarticCirculantGraph(n_nodes, device=device), 4),
        ("k=5 (Quintic)", QuinticCirculantGraph(n_nodes, device=device), 5),
    ]
    
    results = []
    
    for name, graph, k in configs:
        adj = graph.generate_adjacency_matrix()
        degrees = adj.sum(dim=1)
        
        # Initialize from zero Q-values (agents will converge to trap)
        ev = BatchEVSarsa(batch_size, n_nodes, alpha, beta_trap, gamma, device)
        
        # Run until convergence
        for t in range(n_steps):
            actions = ev.sample_actions()
            rewards = compute_rewards_pp(actions, adj, degrees, b, c)
            ev.update(actions, rewards)
        
        # Measure Q-values (average over batches, agent 0)
        q_d_emp = ev.q[:, 0, 0].mean().item()
        q_c_emp = ev.q[:, 0, 1].mean().item()
        delta_emp = q_d_emp - q_c_emp
        
        q_d_th = 1.0 / (1.0 - gamma)
        q_c_th = 1.0 - c * k + gamma / (1.0 - gamma)
        delta_th = c * k
        
        err_d = abs(q_d_emp - q_d_th) / abs(q_d_th) * 100
        err_c = abs(q_c_emp - q_c_th) / abs(q_c_th) * 100 if q_c_th != 0 else abs(q_c_emp - q_c_th) * 100
        err_delta = abs(delta_emp - delta_th) / delta_th * 100
        
        print(f"\n  {name}:")
        print(f"    Q*(D): theory={q_d_th:.4f}, emp={q_d_emp:.4f}, err={err_d:.2f}%")
        print(f"    Q*(C): theory={q_c_th:.4f}, emp={q_c_emp:.4f}, err={err_c:.2f}%")
        print(f"    Δ*:    theory={delta_th:.4f}, emp={delta_emp:.4f}, err={err_delta:.2f}%")
        
        results.append({'name': name, 'k': k,
                        'q_d_th': q_d_th, 'q_d_emp': q_d_emp,
                        'q_c_th': q_c_th, 'q_c_emp': q_c_emp,
                        'delta_th': delta_th, 'delta_emp': delta_emp})
    
    return results

# ═══════════════════════════════════════════════════════════════
# Experiment B: Verify Lemma 2 — p0 in trap
# ═══════════════════════════════════════════════════════════════

def experiment_lemma2(device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Verify Lemma 2 — Cooperation probability p0")
    print("=" * 60)
    
    b, c, gamma, alpha = 3.0, 1.0, 0.95, 0.1
    n_nodes = 8
    batch_size = 128
    
    betas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    ks_and_graphs = [
        (2, RingGraph(n_nodes, device=device)),
        (3, CubicCirculantGraph(n_nodes, device=device)),
        (4, QuarticCirculantGraph(n_nodes, device=device)),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e63946', '#457b9d', '#2a9d8f']
    
    for (k, graph), color in zip(ks_and_graphs, colors):
        adj = graph.generate_adjacency_matrix()
        degrees = adj.sum(dim=1)
        
        p0_theory = []
        p0_empirical = []
        
        for beta in betas:
            p0_th = theory_p0(beta, c, k)
            p0_theory.append(p0_th)
            
            # Init at trap fixed point
            ev = BatchEVSarsa(batch_size, n_nodes, alpha, beta, gamma, device)
            q_d = 1.0 / (1.0 - gamma)
            q_c_vals = torch.tensor([1.0 - c * degrees[i].item() + gamma / (1.0 - gamma) 
                                     for i in range(n_nodes)], device=device)
            ev.init_trap_values(q_d, q_c_vals)
            
            # Warm-up
            for t in range(3000):
                actions = ev.sample_actions()
                rewards = compute_rewards_pp(actions, adj, degrees, b, c)
                ev.update(actions, rewards)
            
            # Measure
            coop_total = 0
            n_measure = 10000
            for t in range(n_measure):
                actions = ev.sample_actions()
                rewards = compute_rewards_pp(actions, adj, degrees, b, c)
                coop_total += actions[:, 0].float().sum().item()
                ev.update(actions, rewards)
            
            p0_emp = coop_total / (n_measure * batch_size)
            p0_empirical.append(p0_emp)
            
            print(f"  k={k}, β={beta:.1f}: p0_theory={p0_th:.6f}, p0_emp={p0_emp:.6f}")
        
        ax.plot(betas, p0_theory, '--', color=color, linewidth=2, label=f'k={k} theory')
        ax.plot(betas, p0_empirical, 'o', color=color, markersize=8, label=f'k={k} empirical')
    
    ax.set_xlabel('β', fontsize=14)
    ax.set_ylabel('p₀', fontsize=14)
    ax.set_title('Lemma 2: p₀ = 1/(1 + e^{βck})', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lemma2_p0.png'), dpi=150)
    plt.close()
    print(f"  → Saved lemma2_p0.png")

# ═══════════════════════════════════════════════════════════════
# Experiment C: AR(1) drift trajectory
# ═══════════════════════════════════════════════════════════════

def experiment_ar1_drift(device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Verify AR(1) drift dynamics (Lemma 3)")
    print("=" * 60)
    
    b, c, gamma, alpha, beta = 3.0, 1.0, 0.95, 0.1, 2.0
    k = 4
    n_nodes = 8
    graph = QuarticCirculantGraph(n_nodes, device=device)
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1)
    
    batch_size = 1  # single trajectory for visualization
    ev = BatchEVSarsa(batch_size, n_nodes, alpha, beta, gamma, device)
    q_d = 1.0 / (1.0 - gamma)
    q_c_vals = torch.tensor([1.0 - c * degrees[i].item() + gamma / (1.0 - gamma) 
                             for i in range(n_nodes)], device=device)
    ev.init_trap_values(q_d, q_c_vals)
    
    q_star_c = q_c_vals[0].item()
    
    max_steps = 5000
    drift_trace = []
    
    for t in range(max_steps):
        actions = ev.sample_actions()
        rewards = compute_rewards_pp(actions, adj, degrees, b, c)
        ev.update(actions, rewards)
        x = ev.q[0, 0, 1].item() - q_star_c
        drift_trace.append(x)
    
    drift_trace = np.array(drift_trace)
    
    p0 = theory_p0(beta, c, k)
    x_bar = b * k * p0
    barrier = c * k
    
    print(f"  p0 = {p0:.6f}")
    print(f"  Stationary mean x̄ = b·k·p0 = {x_bar:.6f}")
    print(f"  Barrier c·k = {barrier:.2f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax = axes[0]
    steps = np.arange(1, len(drift_trace) + 1)
    ax.plot(steps, drift_trace, linewidth=0.5, alpha=0.8, color='steelblue')
    ax.axhline(y=x_bar, color='orange', linestyle='--', linewidth=2, label=f'x̄ = bkp₀ = {x_bar:.4f}')
    ax.axhline(y=barrier, color='red', linestyle='--', linewidth=2, label=f'Barrier ck = {barrier:.1f}')
    ax.set_xlabel('Step t', fontsize=12)
    ax.set_ylabel('x_n = Q(C) - Q*(C)', fontsize=12)
    ax.set_title(f'AR(1) Drift Trajectory (k={k}, β={beta}, α={alpha})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if len(drift_trace) > 100:
        ax.hist(drift_trace, bins=80, density=True, alpha=0.7, color='steelblue', label='Empirical')
        ax.axvline(x=x_bar, color='orange', linestyle='--', linewidth=2, label=f'Theory E[x] = {x_bar:.4f}')
        emp_mean = np.mean(drift_trace)
        ax.axvline(x=emp_mean, color='green', linestyle=':', linewidth=2,
                   label=f'Emp mean = {emp_mean:.4f}')
        ax.set_xlabel('x_n', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of x_n', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        print(f"  Empirical mean x_n: {emp_mean:.6f} (theory: {x_bar:.6f})")
        print(f"  Empirical std x_n: {np.std(drift_trace):.6f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ar1_drift.png'), dpi=150)
    plt.close()
    print(f"  → Saved ar1_drift.png")

# ═══════════════════════════════════════════════════════════════
# Experiment D: Main theorem — E[T] vs β
# ═══════════════════════════════════════════════════════════════

def experiment_main_theorem(device, out_dir):
    """
    Main experiment: measure E[T] for different k, β.
    Compare with drift model and one-step model.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Main Theorem — E[T] vs β (drift vs one-step)")
    print("=" * 60)
    
    b, c, gamma, alpha = 3.0, 1.0, 0.95, 0.1
    n_nodes = 8
    batch_size = 2000
    max_steps = 50000
    
    configs = [
        ("k=2", RingGraph(n_nodes, device=device), 2),
        ("k=3", CubicCirculantGraph(n_nodes, device=device), 3),
        ("k=4", QuarticCirculantGraph(n_nodes, device=device), 4),
        ("k=5", QuinticCirculantGraph(n_nodes, device=device), 5),
    ]
    
    betas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    all_results = {}
    
    for idx, (name, graph, k) in enumerate(configs):
        adj = graph.generate_adjacency_matrix()
        degrees = adj.sum(dim=1)
        
        emp_means = []
        drift_theory = []
        onestep_theory = []
        
        print(f"\n  {name}:")
        
        for beta in betas:
            # Initialize at trap
            ev = BatchEVSarsa(batch_size, n_nodes, alpha, beta, gamma, device)
            q_d = 1.0 / (1.0 - gamma)
            q_c_vals = torch.tensor([1.0 - c * degrees[i].item() + gamma / (1.0 - gamma) 
                                     for i in range(n_nodes)], device=device)
            ev.init_trap_values(q_d, q_c_vals)
            
            # Track breakout per batch element (agent 0)
            breakout_times = torch.full((batch_size,), max_steps, dtype=torch.long, device=device)
            not_broken = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            for t in range(1, max_steps + 1):
                if not not_broken.any():
                    break
                
                actions = ev.sample_actions()
                rewards = compute_rewards_pp(actions, adj, degrees, b, c)
                ev.update(actions, rewards)
                
                # Check breakout: Q(C) > Q(D) for agent 0
                broke = (ev.q[:, 0, 1] > ev.q[:, 0, 0]) & not_broken
                if broke.any():
                    breakout_times[broke] = t
                    not_broken[broke] = False
            
            valid = breakout_times[breakout_times < max_steps].float()
            if len(valid) > 0:
                emp_mean = valid.mean().item()
                frac = len(valid) / batch_size
            else:
                emp_mean = float(max_steps)
                frac = 0.0
            
            et_drift = theory_drift_ET(alpha, b, c, beta, k)
            et_onestep = theory_onestep_ET(b, c, beta, k)
            
            emp_means.append(emp_mean)
            drift_theory.append(et_drift)
            onestep_theory.append(et_onestep)
            
            print(f"    β={beta:.1f}: emp={emp_mean:.1f} (escaped={frac*100:.0f}%), "
                  f"drift={et_drift:.1f}, onestep={et_onestep:.1f}")
        
        all_results[name] = {
            'betas': betas, 'emp': emp_means, 'drift': drift_theory, 'onestep': onestep_theory
        }
        
        ax = axes[idx]
        ax.semilogy(betas, emp_means, 'ko-', label='Empirical E[T]', markersize=7, linewidth=2)
        ax.semilogy(betas, drift_theory, 'b--', label='Drift model', linewidth=2)
        onestep_plot = [min(v, 1e15) for v in onestep_theory]
        ax.semilogy(betas, onestep_plot, 'r:', label='One-step model', linewidth=2, alpha=0.7)
        ax.set_xlabel('β', fontsize=12)
        ax.set_ylabel('E[T]', fontsize=12)
        ax.set_title(f'{name}: E[T] vs β  (b={b}, c={c}, α={alpha})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Theorem: Trap Exit Time — Drift Model vs One-Step Model vs Experiment',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, 'main_theorem_ET_vs_beta.png'), dpi=150)
    plt.close()
    print(f"\n  → Saved main_theorem_ET_vs_beta.png")
    
    return all_results

# ═══════════════════════════════════════════════════════════════
# Experiment E: α-dependence — E[T] ∝ 1/α
# ═══════════════════════════════════════════════════════════════

def experiment_alpha_dependence(device, out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT E: Verify α-dependence — E[T] ∝ 1/α")
    print("=" * 60)
    
    b, c, gamma, beta = 3.0, 1.0, 0.95, 1.5
    k = 3
    n_nodes = 8
    graph = CubicCirculantGraph(n_nodes, device=device)
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1)
    
    batch_size = 2000
    max_steps = 50000
    
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    emp_means = []
    theory_means = []
    
    for alpha in alphas:
        ev = BatchEVSarsa(batch_size, n_nodes, alpha, beta, gamma, device)
        q_d = 1.0 / (1.0 - gamma)
        q_c_vals = torch.tensor([1.0 - c * degrees[i].item() + gamma / (1.0 - gamma) 
                                 for i in range(n_nodes)], device=device)
        ev.init_trap_values(q_d, q_c_vals)
        
        breakout_times = torch.full((batch_size,), max_steps, dtype=torch.long, device=device)
        not_broken = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for t in range(1, max_steps + 1):
            if not not_broken.any():
                break
            actions = ev.sample_actions()
            rewards = compute_rewards_pp(actions, adj, degrees, b, c)
            ev.update(actions, rewards)
            
            broke = (ev.q[:, 0, 1] > ev.q[:, 0, 0]) & not_broken
            if broke.any():
                breakout_times[broke] = t
                not_broken[broke] = False
        
        valid = breakout_times[breakout_times < max_steps].float()
        if len(valid) > 0:
            emp_mean = valid.mean().item()
            frac = len(valid) / batch_size
        else:
            emp_mean = float(max_steps)
            frac = 0.0
        
        th_mean = theory_drift_ET(alpha, b, c, beta, k)
        emp_means.append(emp_mean)
        theory_means.append(th_mean)
        
        print(f"  α={alpha:.3f}: emp={emp_mean:.1f} (escaped={frac*100:.0f}%), drift_theory={th_mean:.1f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.loglog(alphas, emp_means, 'ko-', label='Empirical', markersize=8, linewidth=2)
    ax.loglog(alphas, theory_means, 'b--', label='Drift model', linewidth=2)
    # Reference 1/α line
    ref_alpha = np.array(alphas)
    ref_idx = 3  # α=0.1
    ax.loglog(alphas, emp_means[ref_idx] * alphas[ref_idx] / ref_alpha, 'r:', 
              label='∝ 1/α reference', linewidth=2, alpha=0.6)
    ax.set_xlabel('α', fontsize=13)
    ax.set_ylabel('E[T]', fontsize=13)
    ax.set_title(f'E[T] vs α  (k={k}, β={beta})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    products_emp = [a * e for a, e in zip(alphas, emp_means)]
    products_th = [a * e for a, e in zip(alphas, theory_means)]
    ax.plot(alphas, products_emp, 'ko-', label='α·E[T] empirical', markersize=8, linewidth=2)
    ax.axhline(y=products_th[0], color='blue', linestyle='--', linewidth=2,
               label=f'α·E[T] theory = {products_th[0]:.2f}')
    ax.set_xlabel('α', fontsize=13)
    ax.set_ylabel('α · E[T]', fontsize=13)
    ax.set_title('α · E[T] should be constant if E[T] ∝ 1/α', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'alpha_dependence.png'), dpi=150)
    plt.close()
    print(f"  → Saved alpha_dependence.png")

# ═══════════════════════════════════════════════════════════════
# Experiment F: Scaling analysis — log₁₀(theory/emp)
# ═══════════════════════════════════════════════════════════════

def experiment_scaling_analysis(out_dir, main_results):
    print("\n" + "=" * 60)
    print("EXPERIMENT F: Scaling Analysis — log₁₀(theory/emp)")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'k=2': '#e63946', 'k=3': '#457b9d', 'k=4': '#2a9d8f', 'k=5': '#e9c46a'}
    
    for name, data in main_results.items():
        betas = data['betas']
        emp = np.array(data['emp'])
        drift = np.array(data['drift'])
        onestep = np.array(data['onestep'])
        
        mask = (emp > 0) & (emp < 49999) & np.isfinite(drift) & np.isfinite(onestep)
        color = colors.get(name, 'gray')
        
        if mask.sum() > 0:
            ratio_drift = np.log10(drift[mask] / emp[mask])
            ratio_onestep = np.log10(np.minimum(onestep[mask], 1e30) / emp[mask])
            
            axes[0].plot(betas[mask], ratio_drift, 'o-', label=name, color=color, markersize=6)
            axes[1].plot(betas[mask], ratio_onestep, 's-', label=name, color=color, markersize=6)
            
            print(f"  {name}: drift ratio range [{ratio_drift.min():.2f}, {ratio_drift.max():.2f}], "
                  f"onestep ratio range [{ratio_onestep.min():.2f}, {ratio_onestep.max():.2f}]")
    
    for ax, title in [(axes[0], 'Drift Model'), (axes[1], 'One-Step Model')]:
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('β', fontsize=13)
        ax.set_ylabel('log₁₀(theory / empirical)', fontsize=13)
        ax.set_title(f'{title} Accuracy', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Accuracy Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, 'scaling_analysis.png'), dpi=150)
    plt.close()
    print(f"  → Saved scaling_analysis.png")

# ═══════════════════════════════════════════════════════════════
# Experiment G: One-step breakout impossibility
# ═══════════════════════════════════════════════════════════════

def experiment_onestep_impossibility(out_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT G: Verify one-step breakout impossibility (Lemma 3, §4.2)")
    print("=" * 60)
    
    configs = [
        (0.1, 3.0, 1.0, 2), (0.1, 3.0, 1.0, 3),
        (0.1, 3.0, 1.0, 4), (0.1, 3.0, 1.0, 5),
        (0.05, 3.0, 1.0, 4), (0.1, 2.0, 1.0, 4),
    ]
    
    print(f"\n  {'α':>6} {'b':>5} {'c':>5} {'k':>5} {'m*_onestep':>12} {'m*_old':>8} {'Possible?':>10}")
    print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*12} {'-'*8} {'-'*10}")
    
    for alpha, b, c, k in configs:
        m_onestep = c * k / (alpha * b)
        m_old = c * k / b
        possible = "YES" if m_onestep <= k else "NO"
        print(f"  {alpha:>6.3f} {b:>5.1f} {c:>5.1f} {k:>5} {m_onestep:>12.2f} {m_old:>8.2f} {possible:>10}")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = get_device()
    gpu_config.device = device
    
    out_dir = os.path.join(SCRIPT_DIR, 'proof_verification_results')
    os.makedirs(out_dir, exist_ok=True)
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  PROOF VERIFICATION: EV-SARSA Trap Exit Time (Drift Model)   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}\n")
    
    # A. Lemma 1
    lemma1_results = experiment_lemma1(device, out_dir)
    
    # B. Lemma 2
    experiment_lemma2(device, out_dir)
    
    # C. AR(1) drift
    experiment_ar1_drift(device, out_dir)
    
    # D. Main theorem
    main_results = experiment_main_theorem(device, out_dir)
    
    # E. α-dependence
    experiment_alpha_dependence(device, out_dir)
    
    # F. Scaling analysis
    experiment_scaling_analysis(out_dir, main_results)
    
    # G. One-step impossibility
    experiment_onestep_impossibility(out_dir)
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  All plots saved to: {out_dir}/")
    print(f"    - lemma2_p0.png               — Lemma 2 verification")
    print(f"    - ar1_drift.png               — AR(1) drift dynamics")
    print(f"    - main_theorem_ET_vs_beta.png — Main theorem: E[T] vs β")
    print(f"    - alpha_dependence.png        — E[T] ∝ 1/α verification")
    print(f"    - scaling_analysis.png        — log₁₀(theory/emp) comparison")
    print(f"\n  Key predictions:")
    print(f"    Drift:    E[T] = c·(1 + e^{{βck}})² / (αb)")
    print(f"    One-step: E[T] = 1/P_br ~ (1+e^{{βck}})^{{m*+1}}")

if __name__ == '__main__':
    main()
