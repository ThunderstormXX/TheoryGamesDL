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
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.core.graph_structure import EdgeGraph, TriangleGraph, StarGraph
from experiments.exp8.gpu_version.visualization.plotting import (
    plot_three_agent_combined_series
)

# Reuse logic from trap_effect_experiment.py
from experiments.exp8.gpu_version.trap_effect_experiment import (
    _softmax_probs, _make_pbar, detect_neighbor_gap_trap_intervals,
    run_batched_stateless_trap
)

# =========================
# CONFIG (Task 1: b=4, c=1)
# =========================
B = 4.0
C = 1.0
REWARD_TYPE = 'pp'
ALPHA = 0.02
# Grid search for traps
BETA_VALUES = [0.5]
GAMMA_VALUES = [0.95]
ACTION_FLIP_PROBS = [0.0]

NUM_ITERATIONS = 1000000
RECORD_EVERY = 10
N_REPLICATIONS = 32
SEED = 0

TRAP_NEIGHBOR_GAP = 0.1
TRAP_MIN_DURATION = 500
TRAP_EPS = 0.02

# Correct directory as per existing experiment structure
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = str(PROJECT_ROOT / 'experiments' / 'exp8' / 'results' / 'trap_effect' / 'task1_b4_c1')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    graphs = [
        ('triangle3', TriangleGraph(device=DEVICE)),
        ('star3', StarGraph(num_nodes=3, device=DEVICE)),
    ]

    summary = {
        'timestamp': timestamp,
        'benefit': B,
        'cost': C,
        'num_iterations': NUM_ITERATIONS,
        'grid': {'beta': BETA_VALUES, 'gamma': GAMMA_VALUES},
        'runs': []
    }

    for graph_name, graph in graphs:
        adj_t = graph.generate_adjacency_matrix().detach().cpu().numpy().astype(np.float32)
        
        for beta in BETA_VALUES:
            for gamma in GAMMA_VALUES:
                run_id = f"{graph_name}_beta{beta}_g{gamma}"
                print(f"Running {run_id} with b={B}, c={C}")
                
                sim = run_batched_stateless_trap(
                    adjacency=adj_t,
                    benefit=B,
                    cost=C,
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
                
                # Detect traps across all replications
                trap_reps = []
                for b in range(N_REPLICATIONS):
                    ints_b = detect_neighbor_gap_trap_intervals(
                        p_hist[:, b, :], adj_t, TRAP_NEIGHBOR_GAP, min_duration_points
                    )
                    if ints_b:
                        trap_reps.append((b, ints_b))
                
                # Plot the first replica with traps, or 0 if none
                plot_idx = trap_reps[0][0] if trap_reps else 0
                
                plot_three_agent_combined_series(
                    q_hist=q_hist,
                    p_hist=p_hist,
                    record_every=RECORD_EVERY,
                    title_prefix=f"{run_id} | b={B} c={C}",
                    rep_idx=plot_idx,
                    smooth_window=1,
                    benefit=B,
                    cost=C,
                    reward_type=REWARD_TYPE,
                    save_dir=run_dir
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
    print(f"Task 1 complete. Results in {session_dir}")

if __name__ == '__main__':
    main()
