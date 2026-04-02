import os
import sys
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# Device selection
# NOTE: On macOS, MPS can be slower than CPU for very small tensors (e.g., N<=3).
# You can force the device with env var: TRAP_DEVICE=cpu or TRAP_DEVICE=mps
_trap_device_env = os.environ.get('TRAP_DEVICE', '').strip().lower()
_mps_available = torch.backends.mps.is_available()

if _trap_device_env in {'cpu'}:
    DEVICE = torch.device('cpu')
elif _trap_device_env in {'mps', 'gpu'}:
    DEVICE = torch.device('mps') if _mps_available else torch.device('cpu')
else:
    # Default to CPU: for tiny graphs (2–3 agents) this is typically faster than MPS.
    DEVICE = torch.device('cpu')

if _mps_available:
    if _trap_device_env in {'mps', 'gpu'}:
        print('✓ MPS available (forced via TRAP_DEVICE)')
    elif _trap_device_env in {'cpu'}:
        print('✓ MPS available (but forced CPU via TRAP_DEVICE)')
    else:
        print('✓ MPS available (defaulting to CPU for small graphs; set TRAP_DEVICE=mps to override)')

# Override gpu_config
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE
print(f'Experiment will run on: {gpu_utils.gpu_config.device}')

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.reward_models import RewardManager
from experiments.exp8.gpu_version.core.graph_structure import EdgeGraph, TriangleGraph, StarGraph
from experiments.exp8.gpu_version.visualization.plotting import (
    plot_trap_q_and_p,
    plot_trap_delta_q,
    plot_trap_probabilities_combined,
    plot_two_agent_combined_series,
    plot_three_agent_combined_series,
    plot_deltaq_increment_distribution,
    plot_volatility_acf,
    plot_volatility_clustering_timeseries,
)


# =========================
# CONFIG (edit if needed)
# =========================

# Use the payoff matrix notation from the presentation (row player's payoff):
#   CC=a, CD=b, DC=c, DD=d
# The benefit-cost decomposition on 1-edge graph implies (after shifting by d):
#   T=c-d = benefit, S=b-d = -cost, R=a-d = benefit - cost, P=0
PAYOFF_MATRIX: Optional[Dict[str, float]] = None
# Example (standard benefit-cost PD with cost=1): a=b-c -> here: a=1, b=-1, c=2, d=0
# PAYOFF_MATRIX = {"a": 1.0, "b": -1.0, "c": 2.0, "d": 0.0}

REWARD_TYPE = 'pp'  # one of: pp,pf,ff,fp

# Learning params
ALPHA = 0.02
# Keep defaults small (3 runs total: edge/triangle/star).
# Expand these lists to reproduce phase diagrams / noise sweeps.
BETA_VALUES = [1]  # inverse temperature (noise): higher beta = less noise
GAMMA_VALUES = [0.97] # discount factor: 0.0 = myopic, 1.0 = far-sighted

# Softmax sampling already provides stochasticity.
# ACTION_FLIP_PROBS is an *extra* external noise knob: after sampling an action,
# we flip it with this probability (set to [0.0] to disable).
ACTION_FLIP_PROBS = [0.0]

# Output controls
# Plotting (Matplotlib) can be a large fraction of runtime on macOS.
# You said Q(C), Q(D) and probabilities are required; keep those on.
SAVE_QP_PLOT = True

# For edge2 plots: optional simple moving average window (1 disables smoothing).
TWO_AGENT_SMOOTH_WINDOW = 1

# For 3-agent combined plots: optional simple moving average window (1 disables smoothing).
THREE_AGENT_SMOOTH_WINDOW = 1

# Everything below is optional and can be expensive for large sweeps.
# Turn on only when you need δ(ΔQ) distribution / volatility clustering.
RUN_EXTRA_ANALYSIS = True
SAVE_EXTRA_PLOTS = True

# Extra-analysis controls
ACF_MAX_LAG = 50
ACF_SUM_LAGS = 10  # summarize clustering strength as sum_{lag=1..K} ACF(lag)
# Rolling window (in recorded points of δ(ΔQ), i.e. on the diff'ed series).
VOL_ROLLING_WINDOW = 200

# Simulation length
NUM_ITERATIONS = 100_000
RECORD_EVERY = 10
WARMUP_FRAC = 0.0  # for analysis plots (keep 0.0 to see trap onset)

# Replications per config (batched)
N_REPLICATIONS = 32
SEED = 0

# Trap detection
TRAP_EPS = 0.02
# Neighbor-gap threshold for trap detection:
# trap at time t if exists agent i with p_i(C) - mean_{j in N(i)} p_j(C) >= threshold.
TRAP_NEIGHBOR_GAP = 0.1
# Interpreted in *iterations* (not recorded points). Internally converted using RECORD_EVERY.
TRAP_MIN_DURATION = 500

# Results
OUTPUT_DIR = os.path.join('experiments', 'exp8', 'results', 'trap_effect')

B = 2.0
C = 1.0

# =========================
# Helpers
# =========================

@dataclass(frozen=True)
class CalibratedBC:
    benefit: float
    cost: float
    d_shift: float
    r_residual: float


def _softmax_probs(q: torch.Tensor, temp: float) -> torch.Tensor:
    # q: (..., 2)
    return torch.softmax(q / temp, dim=-1)


def calibrate_benefit_cost_from_matrix(payoff: Dict[str, float]) -> CalibratedBC:
    a = float(payoff['a']); b = float(payoff['b']); c = float(payoff['c']); d = float(payoff['d'])
    # shift by d so P becomes 0 (reward manager has no offset; constant shift doesn't change policy)
    a0, b0, c0 = a - d, b - d, c - d
    benefit = c0
    cost = -b0
    r_residual = a0 - (benefit - cost)
    return CalibratedBC(benefit=benefit, cost=cost, d_shift=d, r_residual=r_residual)


def detect_trap_intervals(p_traj: np.ndarray, eps: float, min_duration: int) -> List[Tuple[int, int]]:
    # Legacy: low-cooperation trap (all agents p(C) < eps).
    # p_traj: (T_out, N)
    below = np.all(p_traj < eps, axis=1)
    intervals: List[Tuple[int, int]] = []
    t = 0
    T = len(below)
    while t < T:
        if below[t]:
            t0 = t
            while t < T and below[t]:
                t += 1
            if t - t0 >= min_duration:
                intervals.append((t0, t))
        else:
            t += 1
    return intervals


def detect_neighbor_gap_trap_intervals(
    p_traj: np.ndarray,
    adjacency: np.ndarray,
    gap_threshold: float,
    min_duration: int,
) -> List[Tuple[int, int]]:
    """Trap intervals based on neighbor-gap asymmetry.

    Trap at time t if there exists an agent i such that:
        p_i(C) - mean_{j in N(i)} p_j(C) >= gap_threshold

    Example: [0.0, 0.0, 0.1] on a 3-node graph is a trap for gap_threshold<=0.1.
    """
    p_traj = np.asarray(p_traj, dtype=float)
    adj = np.asarray(adjacency, dtype=float)
    if p_traj.ndim != 2:
        raise ValueError(f"p_traj must have shape (T_out, N), got {p_traj.shape}")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square, got {adj.shape}")
    if int(adj.shape[0]) != int(p_traj.shape[1]):
        raise ValueError(f"adjacency N={adj.shape[0]} doesn't match p_traj N={p_traj.shape[1]}")

    neighbors = [np.where(adj[i] > 0)[0] for i in range(adj.shape[0])]
    gap_threshold = float(gap_threshold)

    # p_traj: (T_out, N)
    trap_mask = np.zeros(p_traj.shape[0], dtype=bool)
    for t in range(p_traj.shape[0]):
        p_t = p_traj[t]
        is_trap_t = False
        for i, nbr_idx in enumerate(neighbors):
            if nbr_idx.size == 0:
                continue
            nbr_mean = float(np.mean(p_t[nbr_idx]))
            if float(p_t[i]) - nbr_mean >= gap_threshold:
                is_trap_t = True
                break
        trap_mask[t] = is_trap_t

    intervals: List[Tuple[int, int]] = []
    t = 0
    T = len(trap_mask)
    while t < T:
        if trap_mask[t]:
            t0 = t
            while t < T and trap_mask[t]:
                t += 1
            if t - t0 >= min_duration:
                intervals.append((t0, t))
        else:
            t += 1
    return intervals


def acf(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    # x: (T,)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = int(window)
    if w <= 1:
        return np.full_like(x, np.nan)
    if x.size < w:
        return np.full_like(x, float(np.std(x)) if x.size else np.nan)
    kernel = np.ones(w, dtype=float)
    m1 = np.convolve(x, kernel, mode='valid') / w
    m2 = np.convolve(x * x, kernel, mode='valid') / w
    var = np.maximum(m2 - m1 * m1, 0.0)
    std = np.sqrt(var)
    out = np.full_like(x, np.nan)
    out[w - 1:] = std
    return out


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float('nan')
    mu = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return 0.0
    m3 = float(np.mean((x - mu) ** 3))
    return m3 / (s ** 3)


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float('nan')
    mu = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return 0.0
    m4 = float(np.mean((x - mu) ** 4))
    return m4 / (s ** 4) - 3.0


def summarize_deltaq(deltaq_inc: np.ndarray) -> Dict[str, Any]:
    # deltaq_inc: (B, N, T-1)
    vals = deltaq_inc.reshape(-1)
    abs_vals = np.abs(vals)
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'p01': float(np.percentile(vals, 1)),
        'p50': float(np.percentile(vals, 50)),
        'p99': float(np.percentile(vals, 99)),
        'skew': float(skewness(vals)),
        'excess_kurtosis': float(excess_kurtosis(vals)),
        'abs_mean': float(np.mean(abs_vals)),
        'abs_std': float(np.std(abs_vals)),
        'abs_p50': float(np.percentile(abs_vals, 50)),
        'abs_p90': float(np.percentile(abs_vals, 90)),
        'abs_p99': float(np.percentile(abs_vals, 99)),
    }


class _NullPbar:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc

    def update(self, n: int) -> None:
        return None

    def close(self) -> None:
        return None


def _make_pbar(*, total: int, desc: str):
    if tqdm is None:
        return _NullPbar(total=total, desc=desc)
    return tqdm(total=total, desc=desc, leave=False, mininterval=0.3)


# =========================
# Core simulation
# =========================

def run_batched_stateless_trap(
    *,
    adjacency: np.ndarray,
    benefit: float,
    cost: float,
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

    # stateless: single state index 0
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

    reward_manager = RewardManager(reward_type=reward_type, b=float(benefit), c=float(cost))

    T_out = num_iterations // record_every + 1
    # Store histories on device and transfer to CPU only once at the end.
    q_hist_t = torch.empty((T_out, batch_size, n_agents, 2), device=DEVICE, dtype=torch.float32)
    p_hist_t = torch.empty((T_out, batch_size, n_agents), device=DEVICE, dtype=torch.float32)

    out_i = 0
    with torch.no_grad():
        q_now = learner.q_table[:, :, 0, :]  # (B,N,2)
        probs = _softmax_probs(q_now, temp=temp)  # (B,N,2)
        q_hist_t[out_i] = q_now
        p_hist_t[out_i] = probs[..., 1]  # action 1 is C (cooperate)

    pbar = _make_pbar(total=num_iterations, desc=progress_desc)
    update_every = 200
    adj_batched = adj.unsqueeze(0).expand(batch_size, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(batch_size, -1)

    # No autograd needed anywhere in this simulation.
    with torch.no_grad():
        for t in range(1, num_iterations + 1):
            actions = learner.get_actions(states)  # (B,N) in {0,1}

            if action_flip_prob > 0:
                flip = (torch.rand_like(actions, dtype=torch.float32) < action_flip_prob)
                actions = torch.where(flip, 1 - actions, actions)

            actions_f = actions.float()
            rewards = reward_manager.calculate_rewards(
                actions_f,
                adj_batched,
                deg_batched,
            )

            next_states = states  # stateless
            learner.update(states, actions, rewards, next_states)

            if t % record_every == 0:
                out_i += 1
                q_now = learner.q_table[:, :, 0, :]
                probs = _softmax_probs(q_now, temp=temp)
                q_hist_t[out_i] = q_now
                p_hist_t[out_i] = probs[..., 1]

            if t % update_every == 0:
                pbar.update(update_every)

    remainder = num_iterations % update_every
    if remainder:
        pbar.update(remainder)
    pbar.close()

    q_hist = q_hist_t.detach().cpu().numpy()  # (T_out,B,N,2)
    p_hist = p_hist_t.detach().cpu().numpy()  # (T_out,B,N)
    return {
        'q_hist': q_hist,
        'p_hist': p_hist,
        'degrees': degrees.detach().cpu().numpy(),
    }


# =========================
# Main
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Calibrate benefit/cost
    if PAYOFF_MATRIX is not None:
        calib = calibrate_benefit_cost_from_matrix(PAYOFF_MATRIX)
        benefit = calib.benefit
        cost = calib.cost
        print(f'Calibrated benefit={benefit:.4f}, cost={cost:.4f}, shift d={calib.d_shift:.4f}, R residual={calib.r_residual:.4e}')
        if abs(calib.r_residual) > 1e-6:
            print('WARNING: matrix not exactly benefit-cost decomposable (R != T+S-P after shift). Using T and S to set benefit/cost.')
    else:
        # Default: c=1 cost, b=3.0 benefit
        benefit = B
        cost = C
        calib = None
        print(f'Using default benefit={benefit}, cost={cost}')

    summary: Dict[str, Any] = {
        'timestamp': timestamp,
        'reward_type': REWARD_TYPE,
        'alpha': ALPHA,
        'beta_values': BETA_VALUES,
        'gamma_values': GAMMA_VALUES,
        'action_flip_probs': ACTION_FLIP_PROBS,
        'num_iterations': NUM_ITERATIONS,
        'record_every': RECORD_EVERY,
        'n_replications': N_REPLICATIONS,
        'seed': SEED,
        'trap_eps': TRAP_EPS,
        'trap_neighbor_gap': TRAP_NEIGHBOR_GAP,
        'trap_min_duration': TRAP_MIN_DURATION,
        'benefit': benefit,
        'cost': cost,
        'save_qp_plot': SAVE_QP_PLOT,
        'run_extra_analysis': RUN_EXTRA_ANALYSIS,
        'save_extra_plots': SAVE_EXTRA_PLOTS,
        'payoff_matrix': PAYOFF_MATRIX,
        'calibration': None if calib is None else {
            'benefit': calib.benefit,
            'cost': calib.cost,
            'd_shift': calib.d_shift,
            'r_residual': calib.r_residual,
        },
        'runs': [],
    }

    # Output structure:
    #   OUTPUT_DIR/
    #     bc_b{benefit}_c{cost}/
    #       run_<timestamp>/
    #         <config folders...>
    def _fmt_num(x: float) -> str:
        # stable folder names (avoid dots and minus signs causing issues)
        s = f"{float(x):.6g}"
        return s.replace('-', 'm').replace('.', 'p')

    bc_dirname = f"bc_b{_fmt_num(benefit)}_c{_fmt_num(cost)}"
    session_dir = os.path.join(OUTPUT_DIR, bc_dirname, f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    graphs = [
        ('edge2', EdgeGraph(device=DEVICE)),
        ('triangle3', TriangleGraph(device=DEVICE)),
        ('star3', StarGraph(num_nodes=3, device=DEVICE)),
    ]

    run_grid = []
    for graph_name, _graph in graphs:
        for beta in BETA_VALUES:
            for gamma in GAMMA_VALUES:
                for flip_p in ACTION_FLIP_PROBS:
                    run_grid.append((graph_name, beta, gamma, flip_p))

    grid_pbar = _make_pbar(total=len(run_grid), desc='configs')

    # For session-level comparison plots (filled only when RUN_EXTRA_ANALYSIS is enabled)
    analysis_rows: List[Dict[str, Any]] = []

    for graph_name, beta, gamma, flip_p in run_grid:
        graph = next(g for name, g in graphs if name == graph_name)
        adj_t = graph.generate_adjacency_matrix().detach().cpu().numpy().astype(np.float32)

        run_name = (
            f"{graph_name}_rt{REWARD_TYPE}_beta{beta}_g{gamma}_flip{flip_p}_"
            f"T{NUM_ITERATIONS}_R{N_REPLICATIONS}"
        )
        run_dir = os.path.join(session_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        sim = run_batched_stateless_trap(
            adjacency=adj_t,
            benefit=benefit,
            cost=cost,
            reward_type=REWARD_TYPE,
            beta=beta,
            gamma=gamma,
            action_flip_prob=flip_p,
            n_replications=N_REPLICATIONS,
            seed=SEED,
            num_iterations=NUM_ITERATIONS,
            record_every=RECORD_EVERY,
            alpha=ALPHA,
            progress_desc=f"{graph_name} beta={beta} g={gamma} flip={flip_p}",
        )

        q_hist = sim['q_hist']
        p_hist = sim['p_hist']

        stats = None
        acf_mean = None
        acf_summary = None
        vol_series = None
        vol_roll = None
        t_inc = None
        if RUN_EXTRA_ANALYSIS or SAVE_EXTRA_PLOTS:
            warmup = int(WARMUP_FRAC * q_hist.shape[0])
            q_w = q_hist[warmup:]

            # ΔQ = Q(C) - Q(D) (action 1 - action 0)
            delta_q = q_w[..., 1] - q_w[..., 0]  # (T,B,N)
            deltaq_inc = np.diff(delta_q, axis=0).transpose(1, 2, 0)  # (B,N,T-1)

            stats = summarize_deltaq(deltaq_inc)

            # volatility clustering proxy.
            # Fast version: compute ACF once for the aggregated mean |δ(ΔQ)| series.
            abs_inc = np.abs(deltaq_inc)
            mean_abs_series = abs_inc.mean(axis=(0, 1))  # (T-1,)
            acf_mean = acf(mean_abs_series, max_lag=int(ACF_MAX_LAG))
            K = int(min(ACF_SUM_LAGS, len(acf_mean) - 1))
            acf_summary = {
                'acf1': float(acf_mean[1]) if len(acf_mean) > 1 else float('nan'),
                'acf5': float(acf_mean[5]) if len(acf_mean) > 5 else float('nan'),
                'acf_sum_1_to_K': float(np.sum(acf_mean[1:K + 1])) if K >= 1 else float('nan'),
                'K': int(K),
            }

            vol_series = mean_abs_series
            vol_roll = rolling_std(mean_abs_series, window=int(VOL_ROLLING_WINDOW))
            # time axis for increments (in iterations)
            t_inc = (np.arange(mean_abs_series.shape[0]) + (warmup + 1)) * int(RECORD_EVERY)

        # Trap detection
        # 1) Per-replication (more faithful for metastability)
        min_duration_points = max(1, int(np.ceil(TRAP_MIN_DURATION / float(RECORD_EVERY))))
        trap_any = 0
        trap_exit_any = 0
        trap_rep0: List[Tuple[int, int]] = []
        first_trap_rep_idx: Optional[int] = None
        for b in range(p_hist.shape[1]):
            ints_b = detect_neighbor_gap_trap_intervals(
                p_hist[:, b, :],
                adjacency=adj_t,
                gap_threshold=TRAP_NEIGHBOR_GAP,
                min_duration=min_duration_points,
            )
            if ints_b:
                trap_any += 1
                # Exit means a trap interval ends before the final recorded point.
                # If the last trap runs until the end, we treat it as "no exit".
                if any(t1 < (p_hist.shape[0] - 1) for (_, t1) in ints_b):
                    trap_exit_any += 1
                if first_trap_rep_idx is None:
                    first_trap_rep_idx = b
            if b == 0:
                trap_rep0 = ints_b
        trap_fraction = float(trap_any) / float(p_hist.shape[1])
        trap_exit_fraction = float(trap_exit_any) / float(p_hist.shape[1])
        trap_exit_fraction_given_trap = float(trap_exit_any) / float(trap_any) if trap_any > 0 else 0.0

        # 2) Mean over reps (kept for backwards-compat / sanity)
        p_mean = p_hist.mean(axis=1)  # (T,N)
        trap_int_mean = detect_neighbor_gap_trap_intervals(
            p_mean,
            adjacency=adj_t,
            gap_threshold=TRAP_NEIGHBOR_GAP,
            min_duration=min_duration_points,
        )

        rep_idx_for_plot = 0 if first_trap_rep_idx is None else int(first_trap_rep_idx)
        trap_intervals_for_plot = detect_neighbor_gap_trap_intervals(
            p_hist[:, rep_idx_for_plot, :],
            adjacency=adj_t,
            gap_threshold=TRAP_NEIGHBOR_GAP,
            min_duration=min_duration_points,
        )

        plot_paths: Dict[str, Any] = {}
        plot_paths_three: Dict[str, Any] = {}
        p_combined_plot_path: Optional[str] = None

        if SAVE_QP_PLOT:
            if int(adj_t.shape[0]) == 2:
                plot_paths = plot_two_agent_combined_series(
                    q_hist=q_hist,
                    p_hist=p_hist,
                    record_every=RECORD_EVERY,
                    title_prefix=f"Trap experiment | {graph_name} | beta={beta} gamma={gamma} flip_p={flip_p}",
                    rep_idx=rep_idx_for_plot,
                    smooth_window=int(TWO_AGENT_SMOOTH_WINDOW),
                    benefit=benefit,
                    cost=cost,
                    reward_type=REWARD_TYPE,
                    save_dir=run_dir,
                )
            else:
                plot_trap_q_and_p(
                    q_hist=q_hist,
                    p_hist=p_hist,
                    record_every=RECORD_EVERY,
                    trap_eps=TRAP_EPS,
                    title=f"Trap experiment | {graph_name} | beta={beta} gamma={gamma} flip_p={flip_p}",
                    benefit=benefit,
                    cost=cost,
                    reward_type=REWARD_TYPE,
                    rep_idx=rep_idx_for_plot,
                    overlay_mean=True,
                    save_path=os.path.join(run_dir, 'q_and_p.png'),
                )

                plot_trap_delta_q(
                    q_hist=q_hist,
                    record_every=RECORD_EVERY,
                    title=f"Trap experiment (Delta Q) | {graph_name} | beta={beta} gamma={gamma} flip_p={flip_p}",
                    rep_idx=rep_idx_for_plot,
                    trap_intervals=trap_intervals_for_plot,
                    save_path=os.path.join(run_dir, 'delta_q.png'),
                )

                p_combined_plot_path = plot_trap_probabilities_combined(
                    p_hist=p_hist,
                    record_every=RECORD_EVERY,
                    trap_neighbor_gap=TRAP_NEIGHBOR_GAP,
                    title=f"Trap experiment (p(C)) | {graph_name} | beta={beta} gamma={gamma} flip_p={flip_p}",
                    rep_idx=rep_idx_for_plot,
                    trap_intervals=trap_intervals_for_plot,
                    save_path=os.path.join(run_dir, 'p_combined.png'),
                )

                # Additional combined plot for 3-agent runs (requested)
                if int(adj_t.shape[0]) == 3:
                    plot_paths_three = plot_three_agent_combined_series(
                        q_hist=q_hist,
                        p_hist=p_hist,
                        record_every=RECORD_EVERY,
                        title_prefix=f"Trap experiment | {graph_name} | beta={beta} gamma={gamma} flip_p={flip_p}",
                        rep_idx=rep_idx_for_plot,
                        smooth_window=int(THREE_AGENT_SMOOTH_WINDOW),
                        benefit=benefit,
                        cost=cost,
                        reward_type=REWARD_TYPE,
                        save_dir=run_dir,
                    )

        if int(adj_t.shape[0]) == 2:
            two_agent_plot_path = None
            try:
                two_agent_plot_path = plot_paths.get('two_agent_qp') if SAVE_QP_PLOT else None
            except Exception:
                two_agent_plot_path = None
        else:
            two_agent_plot_path = None

        if int(adj_t.shape[0]) == 3:
            three_agent_plot_path = None
            try:
                three_agent_plot_path = plot_paths_three.get('three_agent_qp') if SAVE_QP_PLOT else None
            except Exception:
                three_agent_plot_path = None
        else:
            three_agent_plot_path = None

        if SAVE_EXTRA_PLOTS:
            if not RUN_EXTRA_ANALYSIS:
                raise RuntimeError('SAVE_EXTRA_PLOTS requires RUN_EXTRA_ANALYSIS=True')
            plot_deltaq_increment_distribution(
                deltaq_inc=deltaq_inc,
                title='Distribution of δ(ΔQ) where ΔQ=Q(C)-Q(D)',
                save_path=os.path.join(run_dir, 'deltaq_increment_dist.png'),
            )
            plot_volatility_acf(
                acf_vals=acf_mean,
                title='ACF of |δ(ΔQ)| (volatility clustering proxy)',
                save_path=os.path.join(run_dir, 'volatility_acf.png'),
            )
            plot_volatility_clustering_timeseries(
                t=t_inc,
                mean_abs_inc=vol_series,
                rolling_std=vol_roll,
                title=f'Volatility clustering: mean |δ(ΔQ)| and rolling std (W={int(VOL_ROLLING_WINDOW)})',
                save_path=os.path.join(run_dir, 'volatility_clustering.png'),
            )

        run_summary = {
            'graph': graph_name,
            'n_agents': int(adj_t.shape[0]),
            'degrees': sim['degrees'].tolist(),
            'beta': beta,
            'gamma': gamma,
            'action_flip_prob': flip_p,
            'trap_min_duration_points': int(min_duration_points),
            'trap_neighbor_gap': TRAP_NEIGHBOR_GAP,
            'deltaq_inc_stats': stats,
            'volatility_acf_summary': acf_summary,
            'volatility_acf_abs_deltaq': None if acf_mean is None else acf_mean.tolist(),
            'trap_fraction_any_rep': trap_fraction,
            'trap_exit_fraction_any_rep': trap_exit_fraction,
            'trap_exit_fraction_given_trap': trap_exit_fraction_given_trap,
            'trap_intervals_rep0': trap_rep0,
            'trap_rep_idx_used_for_plot': rep_idx_for_plot,
            'trap_intervals_rep_used_for_plot': trap_intervals_for_plot,
            'trap_intervals_on_mean_p': trap_int_mean,
            'two_agent_qp_plot': two_agent_plot_path,
            'three_agent_qp_plot': three_agent_plot_path,
            'p_combined_plot': p_combined_plot_path,
        }
        summary['runs'].append(run_summary)

        if RUN_EXTRA_ANALYSIS:
            analysis_rows.append({
                'graph': graph_name,
                'n_agents': int(adj_t.shape[0]),
                'beta': float(beta),
                'gamma': float(gamma),
                'action_flip_prob': float(flip_p),
                'std_deltaq_inc': None if stats is None else float(stats['std']),
                'excess_kurtosis_deltaq_inc': None if stats is None else float(stats['excess_kurtosis']),
                'abs_mean_deltaq_inc': None if stats is None else float(stats['abs_mean']),
                'acf_sum_1_to_K': None if acf_summary is None else float(acf_summary['acf_sum_1_to_K']),
                'acf1': None if acf_summary is None else float(acf_summary['acf1']),
                'trap_fraction_any_rep': float(trap_fraction),
                'trap_exit_fraction_any_rep': float(trap_exit_fraction),
                'trap_exit_fraction_given_trap': float(trap_exit_fraction_given_trap),
            })

        (Path(run_dir) / 'run_summary.json').write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        print(
            f'Done: {run_name} | trap_fraction_any_rep={trap_fraction:.3f} '
            f'| trap_mean_intervals={len(trap_int_mean)} | trap_rep0_intervals={len(trap_rep0)}'
        )
        grid_pbar.update(1)

    grid_pbar.close()

    out_path = os.path.join(session_dir, f'summary_{timestamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Session-level comparison plots: 2-player baseline vs 3-player topologies across noise (beta) and gamma.
    if RUN_EXTRA_ANALYSIS and SAVE_EXTRA_PLOTS and analysis_rows:
        import matplotlib.pyplot as _plt

        def _plot_metric_sweep(metric_key: str, ylabel: str, filename: str) -> None:
            rows = [r for r in analysis_rows if r.get(metric_key) is not None]
            if not rows:
                return

            graphs_order = ['edge2', 'triangle3', 'star3']
            fig, axes = _plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
            fig.suptitle(f'{ylabel} vs noise (beta) for different gamma', fontsize=12)

            gammas = sorted({float(r['gamma']) for r in rows})
            flips = sorted({float(r['action_flip_prob']) for r in rows})
            colors = _plt.cm.get_cmap('tab10')(np.linspace(0, 1, max(len(gammas), 1)))
            markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

            for ax, gname in zip(axes, graphs_order):
                sub = [r for r in rows if r['graph'] == gname]
                if not sub:
                    ax.set_title(gname)
                    ax.axis('off')
                    continue
                for gi, gamma_val in enumerate(gammas):
                    for fi, flip_val in enumerate(flips):
                        pts = [r for r in sub if float(r['gamma']) == gamma_val and float(r['action_flip_prob']) == flip_val]
                        if not pts:
                            continue
                        pts = sorted(pts, key=lambda r: float(r['beta']))
                        xs = [float(r['beta']) for r in pts]
                        ys = [float(r[metric_key]) for r in pts]
                        label = f"g={gamma_val:g}" + (f" flip={flip_val:g}" if len(flips) > 1 else '')
                        ax.plot(xs, ys, color=colors[gi], marker=markers[fi % len(markers)], linewidth=1.4, markersize=4, label=label)

                ax.set_title(gname)
                ax.set_xlabel('beta (less noise → higher beta)')
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)

            # Single legend (avoid duplicates)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 5), fontsize=9, frameon=False)
            _plt.tight_layout(rect=(0, 0, 1, 0.9))
            fig.savefig(os.path.join(session_dir, filename), dpi=160, bbox_inches='tight')
            _plt.close(fig)

        def _plot_n2_vs_n3(metric_key: str, ylabel: str, filename: str) -> None:
            rows = [r for r in analysis_rows if r.get(metric_key) is not None]
            if not rows:
                return
            gammas = sorted({float(r['gamma']) for r in rows})
            flips = sorted({float(r['action_flip_prob']) for r in rows})
            betas = sorted({float(r['beta']) for r in rows})

            fig, ax = _plt.subplots(figsize=(8.5, 5.0))
            fig.suptitle(f'2 players vs 3 players: {ylabel} (edge2 vs triangle/star)', fontsize=12)
            colors = _plt.cm.get_cmap('tab10')(np.linspace(0, 1, max(len(gammas), 1)))

            for gi, gamma_val in enumerate(gammas):
                for flip_val in flips:
                    def _get(graph: str, beta_val: float) -> Optional[float]:
                        for r in rows:
                            if r['graph'] == graph and float(r['gamma']) == gamma_val and float(r['action_flip_prob']) == float(flip_val) and float(r['beta']) == float(beta_val):
                                return float(r[metric_key])
                        return None

                    y_edge = []
                    y_tri = []
                    y_star = []
                    x_ok = []
                    for bval in betas:
                        ve = _get('edge2', bval)
                        vt = _get('triangle3', bval)
                        vs = _get('star3', bval)
                        if ve is None or vt is None or vs is None:
                            continue
                        x_ok.append(bval)
                        y_edge.append(ve)
                        y_tri.append(vt)
                        y_star.append(vs)

                    if not x_ok:
                        continue
                    base_label = f"g={gamma_val:g}" + (f" flip={flip_val:g}" if len(flips) > 1 else '')
                    ax.plot(x_ok, y_edge, color=colors[gi], linestyle='-', marker='o', linewidth=1.5, markersize=4, label=base_label + ' edge2')
                    ax.plot(x_ok, y_tri, color=colors[gi], linestyle='--', marker='s', linewidth=1.3, markersize=4, label=base_label + ' triangle3')
                    ax.plot(x_ok, y_star, color=colors[gi], linestyle=':', marker='D', linewidth=1.3, markersize=4, label=base_label + ' star3')

            ax.set_xlabel('beta (less noise → higher beta)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)
            _plt.tight_layout()
            fig.savefig(os.path.join(session_dir, filename), dpi=160, bbox_inches='tight')
            _plt.close(fig)

        _plot_metric_sweep('std_deltaq_inc', 'std(δ(ΔQ))', 'sweep_std_deltaq_inc.png')
        _plot_metric_sweep('excess_kurtosis_deltaq_inc', 'excess kurtosis of δ(ΔQ)', 'sweep_kurtosis_deltaq_inc.png')
        _plot_metric_sweep('acf_sum_1_to_K', f'sum ACF[1..{int(ACF_SUM_LAGS)}] of |δ(ΔQ)|', 'sweep_volatility_clustering_strength.png')
        _plot_metric_sweep('trap_exit_fraction_any_rep', 'trap exit fraction (any rep)', 'sweep_trap_exit_fraction_any_rep.png')
        _plot_metric_sweep('trap_exit_fraction_given_trap', 'trap exit fraction | trap', 'sweep_trap_exit_fraction_given_trap.png')

        _plot_n2_vs_n3('acf_sum_1_to_K', f'sum ACF[1..{int(ACF_SUM_LAGS)}] of |δ(ΔQ)|', 'compare_n2_vs_n3_volatility_clustering.png')
        _plot_n2_vs_n3('std_deltaq_inc', 'std(δ(ΔQ))', 'compare_n2_vs_n3_std_deltaq_inc.png')
        _plot_n2_vs_n3('trap_exit_fraction_any_rep', 'trap exit fraction (any rep)', 'compare_n2_vs_n3_trap_exit_fraction_any_rep.png')
        _plot_n2_vs_n3('trap_exit_fraction_given_trap', 'trap exit fraction | trap', 'compare_n2_vs_n3_trap_exit_fraction_given_trap.png')

    print(f'All done. Summary: {out_path}')


if __name__ == '__main__':
    main()
