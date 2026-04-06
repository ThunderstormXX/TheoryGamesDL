import os
import sys
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Device selection
DEVICE = torch.device('cpu') 
if torch.backends.mps.is_available():
    print('✓ MPS available')

# Import GPU components
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import EdgeGraph, TriangleGraph, StarGraph
from experiments.exp8.gpu_version.visualization.plotting import (
    plot_two_agent_combined_series,
    plot_three_agent_combined_series
)

# Re-implementing the loop from trap_effect_experiment.py but with BonusRewardManager
from experiments.exp8.gpu_version.trap_effect_experiment import (
    _softmax_probs, _make_pbar, detect_neighbor_gap_trap_intervals
)

# =========================
# CONFIG (Task 2: b=3, c=1, bonus=+1)
# =========================
B = 3.0
C = 1.0
BONUS = 1.0
REWARD_TYPE = 'pp'
ALPHA = 0.02
# Grid search for traps
BETA_VALUES = [0.5]
GAMMA_VALUES = [0.95, 0.97]

NUM_ITERATIONS = 1_000_000
RECORD_EVERY = 10
N_REPLICATIONS = 32
SEED = 0

TRAP_NEIGHBOR_GAP = 0.1
TRAP_MIN_DURATION = 500

# Correct directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = str(PROJECT_ROOT / 'experiments' / 'exp8' / 'results' / 'trap_effect' / 'task2_b3_c1_bonus1')

def run_batched_with_bonus(
    *,
    adjacency: np.ndarray,
    benefit: float,
    cost: float,
    bonus: float,
    reward_type: str,
    beta: float,
    gamma: float,
    action_flip_prob: float,
    n_replications: int,
    seed: int,
    num_iterations: int,
    record_every: int,
    alpha: float,
    progress_desc: str = "iterations",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    adj = torch.from_numpy(adjacency).float().to(DEVICE)
    degrees = adj.sum(dim=1)
    n_agents = int(adj.shape[0])
    batch_size = int(n_replications)

    states = torch.zeros((batch_size, n_agents), dtype=torch.long, device=DEVICE)
    temp = 1.0 / float(beta)

    learner = BatchedGPUQLearner(
        batch_size=batch_size,
        n_agents=n_agents,
        action_space_size=2,
        learning_rate=float(alpha),
        discount_factor=float(gamma),
        exploration_rate=0.0,
        strategy='boltzmann',
        temperature=float(temp),
        max_states=1,
    )

    reward_manager = BonusRewardManager(reward_type=reward_type, b=float(benefit), c=float(cost), bonus=float(bonus))

    T_out = num_iterations // record_every + 1
    q_hist_t = torch.empty((T_out, batch_size, n_agents, 2), device=DEVICE, dtype=torch.float32)
    p_hist_t = torch.empty((T_out, batch_size, n_agents), device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        q_now = learner.q_table[:, :, 0, :]
        probs = _softmax_probs(q_now, temp=temp)
        q_hist_t[0] = q_now
        p_hist_t[0] = probs[..., 1]

    pbar = _make_pbar(total=num_iterations, desc=progress_desc)
    adj_batched = adj.unsqueeze(0).expand(batch_size, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)

            if t % record_every == 0:
                out_idx = t // record_every
                q_now = learner.q_table[:, :, 0, :]
                probs = _softmax_probs(q_now, temp=temp)
                q_hist_t[out_idx] = q_now
                p_hist_t[out_idx] = probs[..., 1]

            if t % 200 == 0:
                pbar.update(200)

    pbar.close()
    return {
        'q_hist': q_hist_t.cpu().numpy(),
        'p_hist': p_hist_t.cpu().numpy(),
        'degrees': degrees.cpu().numpy(),
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    graphs = [
        ('edge2', EdgeGraph(device=DEVICE)),
        ('triangle3', TriangleGraph(device=DEVICE)),
        ('star3', StarGraph(num_nodes=3, device=DEVICE)),
    ]

    summary = {
        'timestamp': timestamp,
        'benefit': B,
        'cost': C,
        'bonus': BONUS,
        'num_iterations': NUM_ITERATIONS,
        'grid': {'beta': BETA_VALUES, 'gamma': GAMMA_VALUES},
        'runs': []
    }

    for graph_name, graph in graphs:
        adj_t = graph.generate_adjacency_matrix().detach().cpu().numpy().astype(np.float32)
        n_agents = int(adj_t.shape[0])

        for beta in BETA_VALUES:
            for gamma in GAMMA_VALUES:
                run_id = f"{graph_name}_beta{beta}_g{gamma}"
                print(f"Running {run_id} with b={B}, c={C}, bonus={BONUS}")
                
                sim = run_batched_with_bonus(
                    adjacency=adj_t,
                    benefit=B,
                    cost=C,
                    bonus=BONUS,
                    reward_type=REWARD_TYPE,
                    beta=beta,
                    gamma=gamma,
                    action_flip_prob=0.0,
                    n_replications=N_REPLICATIONS,
                    seed=SEED,
                    num_iterations=NUM_ITERATIONS,
                    record_every=RECORD_EVERY,
                    alpha=ALPHA,
                    progress_desc=run_id
                )

                q_hist = sim['q_hist']
                p_hist = sim['p_hist']
                
                run_dir = os.path.join(session_dir, run_id)
                os.makedirs(run_dir, exist_ok=True)

                min_duration_points = max(1, int(np.ceil(TRAP_MIN_DURATION / float(RECORD_EVERY))))
                
                trap_reps = []
                for b in range(N_REPLICATIONS):
                    ints_b = detect_neighbor_gap_trap_intervals(p_hist[:, b, :], adj_t, TRAP_NEIGHBOR_GAP, min_duration_points)
                    if ints_b:
                        trap_reps.append((b, ints_b))
                
                plot_idx = trap_reps[0][0] if trap_reps else 0
                
                if n_agents == 2:
                    plot_two_agent_combined_series(
                        q_hist=q_hist, p_hist=p_hist, record_every=RECORD_EVERY,
                        title_prefix=f"{run_id} | b={B} c={C} bonus={BONUS} (rep {plot_idx})",
                        rep_idx=plot_idx, smooth_window=1, benefit=B, cost=C, reward_type=REWARD_TYPE, save_dir=run_dir
                    )
                else:
                    plot_three_agent_combined_series(
                        q_hist=q_hist, p_hist=p_hist, record_every=RECORD_EVERY,
                        title_prefix=f"{run_id} | b={B} c={C} bonus={BONUS} (rep {plot_idx})",
                        rep_idx=plot_idx, smooth_window=1, benefit=B, cost=C, reward_type=REWARD_TYPE, save_dir=run_dir
                    )

                run_summary = {
                    'graph': graph_name,
                    'beta': beta,
                    'gamma': gamma,
                    'trap_fraction': len(trap_reps) / N_REPLICATIONS,
                    'num_reps_with_traps': len(trap_reps)
                }
                summary['runs'].append(run_summary)
                
    with open(os.path.join(session_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Task 2 complete. Results in {session_dir}")

if __name__ == '__main__':
    main()
