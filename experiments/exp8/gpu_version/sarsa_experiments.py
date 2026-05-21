"""
SARSA experiments — mirrors all Q-Learning experiments from earlier sessions.

Covers:
  Part 1  — Section 4 graphs:  Triangle, Chain3, Complete4, Chain4,
                                Star4, Ring4, Wheel4
  Part 2  — Custom 6-node topology (A-B-C-{D,E,F})
  Part 3  — k-regular graphs:  Ring(n), Cubic(n), Mixed(n)

All output goes to  results/sarsa/<subdir>/
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils

# Auto-detect device, optimizing for WSL & 3080 Ti
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f"CUDA detected. Optimizing for: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("MPS detected. Using Apple Silicon GPU.")
else:
    DEVICE = torch.device('cpu')
    print("No GPU detected. Falling back to CPU.")

gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_sarsa import BatchedGPUSARSALearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph, StarGraph


# ═══════════════════════════════════════════════════════════════════════════
# Graph definitions (copied from previous scripts for self-containedness)
# ═══════════════════════════════════════════════════════════════════════════

class CompleteGraph(BaseGraph):
    def __init__(self, n, device=None):
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.ones((self.num_nodes, self.num_nodes),
                         device=self.device, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        return adj

class RingGraph(BaseGraph):
    def __init__(self, n, device=None):
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
        return adj

class ChainGraph(BaseGraph):
    def __init__(self, n, device=None):
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        return adj

class WheelGraph(BaseGraph):
    def __init__(self, n, device=None):
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        for i in range(1, n):
            nxt = i + 1 if i + 1 < n else 1
            adj[i, nxt] = 1.0
            adj[nxt, i] = 1.0
        return adj

class CustomSixNodeGraph(BaseGraph):
    """A-B-C-{D,E,F}"""
    def __init__(self, device=None):
        super().__init__(num_nodes=6, device=device)
    def generate_adjacency_matrix(self):
        adj = torch.zeros((6, 6), device=self.device, dtype=torch.float32)
        for u, v in [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)]:
            adj[u, v] = adj[v, u] = 1.0
        return adj

class CubicCirculantGraph(BaseGraph):
    """3-regular circulant C(n, {1, n/2}). n must be even >= 4."""
    def __init__(self, n, device=None):
        assert n % 2 == 0 and n >= 4
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        h = n // 2
        for i in range(n):
            adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
            adj[i, (i + h) % n] = adj[(i + h) % n, i] = 1.0
        return adj

class MixedGraph(BaseGraph):
    """Ring + one antipodal chord. 2 nodes deg-3, rest deg-2."""
    def __init__(self, n, device=None):
        assert n >= 4
        super().__init__(num_nodes=n, device=device)
    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        for i in range(n):
            adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
        h = n // 2
        adj[0, h] = adj[h, 0] = 1.0
        return adj


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def smooth(y, box_pts=5):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    ys = np.convolve(y, box, mode='same')
    ys[:box_pts] = y[:box_pts]
    ys[-box_pts:] = y[-box_pts:]
    return ys

RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../results/sarsa'))

GAMMAS = [0.8, 0.9, 0.95, 0.99]
BETAS = [0.7, 0.8, 0.9, 1.0]
ALPHA = 0.01
ITERS = 1_000_000
RECORD_EVERY = 10_000
REPS = 128  # Increased from 32 to 128 to better saturate the 3080 Ti's CUDA cores


# ═══════════════════════════════════════════════════════════════════════════
# SARSA Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_sarsa_simulation(graph, gamma, beta):
    """
    Run stateless SARSA.  Returns P(C) history, Q-value histories, and
    final mean P(C).
    """
    TEMP = 1.0 / beta
    np.random.seed(42)
    torch.manual_seed(42)

    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    num_nodes = graph.num_nodes

    states = torch.zeros((REPS, num_nodes), dtype=torch.long, device=DEVICE)

    learner = BatchedGPUSARSALearner(
        batch_size=REPS, n_agents=num_nodes, action_space_size=2,
        learning_rate=ALPHA, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=TEMP, max_states=1,
    )
    reward_manager = BonusRewardManager(
        reward_type='pp', b=2.0, c=1.0, bonus=1.0)

    adj_batched = adj_t.unsqueeze(0).expand(REPS, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(REPS, -1)

    T_out = ITERS // RECORD_EVERY + 1
    p_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    qc_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    qd_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)

    def record(t_idx):
        q_now = learner.q_table[:, :, 0, :].cpu()
        probs = torch.softmax(q_now / TEMP, dim=-1).numpy()
        p_hist[t_idx] = probs[..., 1]
        qd_hist[t_idx] = q_now[..., 0].numpy()
        qc_hist[t_idx] = q_now[..., 1].numpy()

    # SARSA: need initial actions before entering the loop
    actions = learner.get_actions(states)
    record(0)

    with torch.no_grad():
        for t in range(1, ITERS + 1):
            rewards = reward_manager.calculate_rewards(
                actions.float(), adj_batched, deg_batched)

            next_states = states          # stateless
            next_actions = learner.get_actions(next_states)

            learner.update(states, actions, rewards,
                           next_states, next_actions)

            states = next_states
            actions = next_actions

            if t % RECORD_EVERY == 0:
                record(t // RECORD_EVERY)

    final_pc = float(p_hist[-1].mean())
    return p_hist, qc_hist, qd_hist, final_pc


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_dynamics(p_hist, qc_hist, qd_hist, title, out_path, degrees,
                  node_labels=None):
    """
    Two-panel plot: P(C) dynamics (left) + Q-values per agent (right).
    If node_labels is provided, use them; otherwise colour by degree.
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    T_out, _, N = p_hist.shape
    x = np.arange(T_out) * RECORD_EVERY

    palette = ['#e74c3c', '#f39c12', '#2c3e50', '#2ecc71',
               '#3498db', '#9b59b6', '#1abc9c', '#e67e22',
               '#34495e', '#e84393', '#00b894', '#6c5ce7',
               '#fdcb6e', '#fab1a0', '#74b9ff', '#a29bfe',
               '#55efc4', '#ff7675', '#636e72', '#dfe6e9']
    deg_color = {1: '#2ecc71', 2: '#3498db', 3: '#e74c3c', 4: '#9b59b6'}

    plotted = set()
    for i in range(N):
        d = int(degrees[i])
        if node_labels is not None:
            col = palette[i % len(palette)]
            lbl = node_labels[i]
            lw = 2.5 if i in (0, 3) else 1.2  # highlight A vs D
        else:
            col = deg_color.get(d, '#95a5a6')
            lbl = f'deg={d}' if d not in plotted else None
            plotted.add(d)
            lw = 1.5

        mean_p = smooth(p_hist[:, :, i].mean(axis=1))
        std_p = smooth(p_hist[:, :, i].std(axis=1))
        ax1.plot(x, mean_p, color=col, linewidth=lw, alpha=0.8, label=lbl)
        ax1.fill_between(x, np.clip(mean_p - std_p, 0, 1),
                         np.clip(mean_p + std_p, 0, 1),
                         color=col, alpha=0.08)

        mean_qc = smooth(qc_hist[:, :, i].mean(axis=1))
        mean_qd = smooth(qd_hist[:, :, i].mean(axis=1))
        ax2.plot(x, mean_qc, color=col, linestyle='-', linewidth=lw,
                 alpha=0.8, label=f'{lbl} Q(C)' if lbl else None)
        ax2.plot(x, mean_qd, color=col, linestyle='--', linewidth=lw,
                 alpha=0.8)

    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title('P(C)', fontsize=12)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('P(C)')
    ax1.legend(loc='best', fontsize=7)

    ax2.set_title('Q(C) [solid] vs Q(D) [dashed]', fontsize=12)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Q-value')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_qvalues_per_agent(qc_hist, qd_hist, title, out_path, degrees,
                           node_labels=None):
    """
    One subplot per agent showing Q(C) and Q(D) over time.
    Makes it easy to inspect individual agent behaviour.
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    T_out, _, N = qc_hist.shape
    x = np.arange(T_out) * RECORD_EVERY

    ncols = min(N, 4)
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             sharex=True)
    if nrows == 1:
        axes = [axes] if N == 1 else list(axes)
    else:
        axes = list(axes.flatten())

    for i in range(N):
        ax = axes[i]
        d = int(degrees[i])
        lbl = node_labels[i] if node_labels else f'Agent {i} (deg={d})'

        mean_qc = smooth(qc_hist[:, :, i].mean(axis=1))
        mean_qd = smooth(qd_hist[:, :, i].mean(axis=1))
        std_qc = smooth(qc_hist[:, :, i].std(axis=1))
        std_qd = smooth(qd_hist[:, :, i].std(axis=1))

        ax.plot(x, mean_qc, color='#2ecc71', linewidth=2, label='Q(C)')
        ax.fill_between(x, mean_qc - std_qc, mean_qc + std_qc,
                        color='#2ecc71', alpha=0.15)
        ax.plot(x, mean_qd, color='#e74c3c', linewidth=2, label='Q(D)')
        ax.fill_between(x, mean_qd - std_qd, mean_qd + std_qd,
                        color='#e74c3c', alpha=0.15)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Q-value')
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_summary_heatmaps(results, graph_type_keys, out_dir, graph_labels):
    """Heatmap of mean final P(C) vs (n, gamma) for each beta."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    n_panels = len(graph_type_keys)
    for beta in BETAS:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                                 sharey=True)
        if n_panels == 1:
            axes = [axes]
        n_vals = sorted({k[1] for k in results})

        for ax, gtype in zip(axes, graph_type_keys):
            data = np.full((len(n_vals), len(GAMMAS)), np.nan)
            for i, n in enumerate(n_vals):
                for j, g in enumerate(GAMMAS):
                    data[i, j] = results.get((gtype, n, g, beta), np.nan)

            im = ax.imshow(data, aspect='auto', origin='lower',
                           cmap='RdYlGn', vmin=0, vmax=1,
                           extent=[-0.5, len(GAMMAS) - 0.5,
                                   -0.5, len(n_vals) - 0.5])
            ax.set_xticks(range(len(GAMMAS)))
            ax.set_xticklabels([f'{g}' for g in GAMMAS])
            ax.set_yticks(range(len(n_vals)))
            ax.set_yticklabels([str(n) for n in n_vals])
            ax.set_xlabel(r'$\gamma$')
            ax.set_title(graph_labels.get(gtype, gtype), fontsize=12)

            for i in range(len(n_vals)):
                for j in range(len(GAMMAS)):
                    v = data[i, j]
                    if np.isnan(v):
                        continue
                    c = 'white' if v < 0.4 or v > 0.8 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, color=c, fontweight='bold')

        axes[0].set_ylabel('n (nodes)')
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04,
                     label='Mean Final P(C)')
        fig.suptitle(f'SARSA — Mean Cooperation  |  '
                     fr'$\beta$={beta}, $\alpha$={ALPHA}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(out_dir, f'summary_heatmap_beta{beta}.jpg')
        plt.savefig(path, dpi=150)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Task definitions for each experiment part
# ═══════════════════════════════════════════════════════════════════════════

# Part 1 — Section 4 graphs
SEC4_CONFIGS = {
    'triangle':  ('Triangle (K3)',     lambda: CompleteGraph(3, DEVICE)),
    'chain3':    ('Chain (3)',         lambda: ChainGraph(3, DEVICE)),
    'complete4': ('Complete (K4)',     lambda: CompleteGraph(4, DEVICE)),
    'chain4':    ('Chain (4)',         lambda: ChainGraph(4, DEVICE)),
    'star4':     ('Star (4)',          lambda: StarGraph(4, DEVICE)),
    'ring4':     ('Ring (4)',          lambda: RingGraph(4, DEVICE)),
    'wheel4':    ('Wheel (4)',         lambda: WheelGraph(4, DEVICE)),
}

# Part 2 — Custom 6-node
CUSTOM6_LABELS = [
    'A (deg=1, nbr_deg=2)', 'B (deg=2)', 'C (deg=4, hub)',
    'D (deg=1, nbr_deg=4)', 'E (deg=1, nbr_deg=4)', 'F (deg=1, nbr_deg=4)',
]

# Part 3 — k-regular
KREG_CONFIGS = {
    'ring':  ('Ring (2-reg)',  RingGraph),
    'cubic': ('Cubic (3-reg)', CubicCirculantGraph),
    'mixed': ('Mixed (2/3)',   MixedGraph),
}
N_VALUES = [4, 10, 20, 50, 100]


# ═══════════════════════════════════════════════════════════════════════════
# Worker functions
# ═══════════════════════════════════════════════════════════════════════════

from multiprocessing import Pool, cpu_count


def _worker_sec4(args):
    """Section 4 experiment worker."""
    key, gamma, beta = args
    label, graph_fn = SEC4_CONFIGS[key]
    graph = graph_fn()
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1).cpu().numpy()

    p, qc, qd, fpc = run_sarsa_simulation(graph, gamma, beta)

    out = os.path.join(RESULTS_ROOT, 'section4')
    os.makedirs(out, exist_ok=True)

    title = f'SARSA — {label}, γ={gamma}, β={beta}'
    plot_dynamics(p, qc, qd, title,
                  os.path.join(out, f'{key}_g{gamma}_b{beta}.jpg'),
                  degrees)
    plot_qvalues_per_agent(
        qc, qd, f'SARSA Q-values — {label}, γ={gamma}, β={beta}',
        os.path.join(out, f'{key}_qvals_g{gamma}_b{beta}.jpg'),
        degrees)

    return ('sec4', key, gamma, beta, fpc)


def _worker_custom6(args):
    """Custom 6-node worker."""
    gamma, beta = args
    graph = CustomSixNodeGraph(DEVICE)
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1).cpu().numpy()

    p, qc, qd, fpc = run_sarsa_simulation(graph, gamma, beta)

    out = os.path.join(RESULTS_ROOT, 'custom6')
    os.makedirs(out, exist_ok=True)

    title = f'SARSA — Custom 6-Node, γ={gamma}, β={beta}'
    plot_dynamics(p, qc, qd, title,
                  os.path.join(out, f'custom6_g{gamma}_b{beta}.jpg'),
                  degrees, node_labels=CUSTOM6_LABELS)
    plot_qvalues_per_agent(
        qc, qd, f'SARSA Q-values — Custom 6-Node, γ={gamma}, β={beta}',
        os.path.join(out, f'custom6_qvals_g{gamma}_b{beta}.jpg'),
        degrees, node_labels=CUSTOM6_LABELS)

    return ('custom6', 'custom6', gamma, beta, fpc)


def _worker_kreg(args):
    """k-regular graph worker."""
    gtype, n, gamma, beta = args
    label, cls = KREG_CONFIGS[gtype]
    graph = cls(n, DEVICE)
    adj = graph.generate_adjacency_matrix()
    degrees = adj.sum(dim=1).cpu().numpy()

    p, qc, qd, fpc = run_sarsa_simulation(graph, gamma, beta)

    out = os.path.join(RESULTS_ROOT, 'k_regular')
    os.makedirs(out, exist_ok=True)

    title = f'SARSA — {label}, n={n}, γ={gamma}, β={beta}'
    plot_dynamics(p, qc, qd, title,
                  os.path.join(out, f'{gtype}_n{n}_g{gamma}_b{beta}.jpg'),
                  degrees)

    return ('kreg', gtype, n, gamma, beta, fpc)


def _dispatch(args):
    """Route task to the appropriate worker."""
    kind, payload = args
    if kind == 'sec4':
        return _worker_sec4(payload)
    elif kind == 'c6':
        return _worker_custom6(payload)
    else:
        return _worker_kreg(payload)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print('=' * 60)
    print('  SARSA Experiments — Full Suite')
    print('=' * 60)

    # ── build all task lists ──
    tasks_sec4 = []
    for key in SEC4_CONFIGS:
        for g in GAMMAS:
            for b in BETAS:
                tasks_sec4.append((key, g, b))

    tasks_c6 = [(g, b) for g in GAMMAS for b in BETAS]

    tasks_kreg = []
    for gtype in KREG_CONFIGS:
        for n in N_VALUES:
            for g in GAMMAS:
                for b in BETAS:
                    tasks_kreg.append((gtype, n, g, b))

    total = len(tasks_sec4) + len(tasks_c6) + len(tasks_kreg)
    print(f'  Section 4:     {len(tasks_sec4)} experiments')
    print(f'  Custom 6-node: {len(tasks_c6)} experiments')
    print(f'  k-regular:     {len(tasks_kreg)} experiments')
    print(f'  TOTAL:         {total} experiments')
    print(f'  Iters/exp:     {ITERS}')
    print(f'  Reps:          {REPS}')
    print()

    # Use more workers to feed the 3080 Ti, which is very fast
    num_workers = min(cpu_count(), 12)
    print(f'Using {num_workers} parallel workers...\n')

    results_kreg = {}

    # Wrap all tasks into a single pool for simplicity
    all_tasks = (
        #[('sec4', t) for t in tasks_sec4] +
        #[('c6', t) for t in tasks_c6] +
        [('kreg', t) for t in tasks_kreg]
    )

    results_kreg = {}
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(_dispatch, all_tasks),
                        total=total, desc='SARSA Experiments'):
            if res[0] == 'kreg':
                _, gtype, n, gamma, beta, fpc = res
                results_kreg[(gtype, n, gamma, beta)] = fpc

    elapsed = time.time() - t0
    print(f'\nAll {total} simulations completed in {elapsed:.0f}s '
          f'({elapsed / 60:.1f} min)')

    # ── Summary plots for k-regular ──
    if results_kreg:
        print('\nGenerating k-regular summary heatmaps...')
        kreg_out = os.path.join(RESULTS_ROOT, 'k_regular')
        graph_labels = {k: v[0] for k, v in KREG_CONFIGS.items()}
        plot_summary_heatmaps(
            results_kreg, list(KREG_CONFIGS.keys()), kreg_out, graph_labels)

    print(f'\nAll outputs saved under: {RESULTS_ROOT}')
    print(f'Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
