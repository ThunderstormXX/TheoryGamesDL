"""
Stateful Q-Learning on the Custom 6-Node Graph.

Graph topology:
    A --- B --- C --- D
                |
                E
                |
                F

Node degrees:  A=1, B=2, C=4, D=1, E=1, F=1
State = number of cooperating neighbours (integer).
max_states per agent = degree + 1.

We use a global max_states = max(degree) + 1 = 5 for the Q-table,
and each agent only visits states 0..degree_i.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
DEVICE = torch.device('cpu')
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import BaseGraph


# ── Graph definition ────────────────────────────────────────────────────────
class CustomSixNodeGraph(BaseGraph):
    """
    A --- B --- C --- D
                |
                E
                |
                F
    """
    def __init__(self, device=None):
        super().__init__(num_nodes=6, device=device)

    def generate_adjacency_matrix(self):
        adj = torch.zeros((6, 6), device=self.device, dtype=torch.float32)
        edges = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)]
        for u, v in edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        return adj


# ── Helpers ─────────────────────────────────────────────────────────────────
NODE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
NODE_LABELS = [
    'A (deg=1, nbr_deg=2)',
    'B (deg=2)',
    'C (deg=4, hub)',
    'D (deg=1, nbr_deg=4)',
    'E (deg=1, nbr_deg=4)',
    'F (deg=1, nbr_deg=4)',
]
NODE_COLORS = ['#e74c3c', '#f39c12', '#2c3e50', '#2ecc71', '#3498db', '#9b59b6']

out_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../../experiments/exp8/papers/Q_Learning_on_graphs/figures'))
os.makedirs(out_dir, exist_ok=True)

# ── Experiment parameters ───────────────────────────────────────────────────
GAMMAS = [0.0, 0.5, 0.9, 0.95]
BETAS = [0.5, 1.0, 2.0]
ALPHA = 0.01
ITERS = 1_000_000
RECORD_EVERY = 10_000
REPS = 32


def smooth(y, box_pts=5):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    y_smooth[-box_pts:] = y[-box_pts:]
    return y_smooth


# ── Simulation ──────────────────────────────────────────────────────────────
def compute_states(actions_float, adj_batched):
    """State for each agent = number of cooperating neighbours."""
    # actions_float: 1 = cooperate, 0 = defect  (cooperator indicator)
    coop = actions_float.unsqueeze(2)                # (B, N, 1)
    n_coop_nbrs = torch.bmm(adj_batched, coop).squeeze(2)  # (B, N)
    return n_coop_nbrs.long()


def run_simulation(graph, gamma, beta):
    TEMP = 1.0 / beta
    np.random.seed(42)
    torch.manual_seed(42)

    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    num_nodes = graph.num_nodes
    max_deg = int(degrees.max().item())
    max_states = max_deg + 1  # states 0 .. max_deg

    adj_batched = adj_t.unsqueeze(0).expand(REPS, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(REPS, -1)

    learner = BatchedGPUQLearner(
        batch_size=REPS, n_agents=num_nodes, action_space_size=2,
        learning_rate=ALPHA, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=TEMP, max_states=max_states,
    )
    reward_manager = BonusRewardManager(reward_type='pp', b=2.0, c=1.0, bonus=1.0)

    # Initial state: random actions → derive state
    actions = torch.randint(0, 2, (REPS, num_nodes), device=DEVICE)
    cooperators = 1 - actions  # 1=coop, 0=defect
    states = compute_states(cooperators.float(), adj_batched)
    states = torch.clamp(states, 0, max_states - 1)

    T_out = ITERS // RECORD_EVERY + 1

    # P(C) history: averaged over states via actual Boltzmann policy
    p_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.float32)
    # Q-table snapshots for the final heatmap
    # shape: (T_out, REPS, num_nodes, max_states, 2)
    q_snapshots = np.zeros((T_out, REPS, num_nodes, max_states, 2), dtype=np.float32)
    # Track which state each agent is actually in
    state_hist = np.zeros((T_out, REPS, num_nodes), dtype=np.int32)

    def record(t_idx, cur_states):
        q_now = learner.q_table.cpu().numpy()            # (B, N, S, 2)
        q_snapshots[t_idx] = q_now
        state_hist[t_idx] = cur_states.cpu().numpy()
        # P(C) in current state via Boltzmann
        for i in range(num_nodes):
            s = cur_states[:, i].cpu().numpy()            # (B,)
            for b in range(REPS):
                qv = q_now[b, i, s[b]]                   # (2,)
                exp_q = np.exp(qv / TEMP - np.max(qv / TEMP))
                p_hist[t_idx, b, i] = exp_q[1] / exp_q.sum()

    record(0, states)

    with torch.no_grad():
        for t in tqdm(range(1, ITERS + 1),
                      desc=f'γ={gamma}, β={beta}', leave=False, mininterval=1.0):
            actions = learner.get_actions(states)
            cooperators = (1 - actions).float()           # 1=coop
            rewards = reward_manager.calculate_rewards(cooperators, adj_batched, deg_batched)
            next_states = compute_states(cooperators, adj_batched)
            next_states = torch.clamp(next_states, 0, max_states - 1)
            learner.update(states, actions, rewards, next_states)
            states = next_states

            if t % RECORD_EVERY == 0:
                record(t // RECORD_EVERY, states)

    return p_hist, q_snapshots, state_hist, max_states, degrees.cpu().numpy().astype(int)


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_full_dashboard(p_hist, q_snapshots, state_hist, max_states, degrees,
                        gamma, beta, out_path):
    """
    3-row figure:
      Row 1: P(C) dynamics for all 6 agents (one panel)
      Row 2: Per-agent Q(C)−Q(D) heatmap over (state, time)
      Row 3: Final Q-table bar chart per agent (Q(C) vs Q(D) for each valid state)
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    TEMP = 1.0 / beta
    T_out, _, N = p_hist.shape
    x_iter = np.arange(T_out) * RECORD_EVERY

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.45)

    # ── Row 1: P(C) dynamics ────────────────────────────────────────────
    ax_pc = fig.add_subplot(gs[0, :])
    for i in range(N):
        mean_p = p_hist[:, :, i].mean(axis=1)
        std_p  = p_hist[:, :, i].std(axis=1)
        mean_p = smooth(mean_p)
        std_p  = smooth(std_p)
        lw = 2.5 if i in (0, 3) else 1.2
        ax_pc.plot(x_iter, mean_p, label=NODE_LABELS[i],
                   color=NODE_COLORS[i], linewidth=lw)
        if i in (0, 3):
            ax_pc.fill_between(x_iter,
                               np.clip(mean_p - std_p, 0, 1),
                               np.clip(mean_p + std_p, 0, 1),
                               color=NODE_COLORS[i], alpha=0.12)
    ax_pc.set_ylim(-0.02, 1.02)
    ax_pc.set_xlabel('Iterations')
    ax_pc.set_ylabel('P(C) in current state')
    ax_pc.set_title('Probability of Cooperation (stateful)', fontsize=13)
    ax_pc.legend(loc='best', fontsize=8)

    # ── Row 2: Q(C)−Q(D) heatmaps per agent ────────────────────────────
    for i in range(N):
        ax = fig.add_subplot(gs[1, i])
        deg_i = degrees[i]
        # mean over REPS: Q(C) - Q(D) for each (state, time)
        qc = q_snapshots[:, :, i, :deg_i + 1, 1].mean(axis=1)  # (T, S)
        qd = q_snapshots[:, :, i, :deg_i + 1, 0].mean(axis=1)  # (T, S)
        diff = qc - qd  # positive → prefers C

        im = ax.imshow(diff.T, aspect='auto', origin='lower',
                       cmap='RdYlGn', vmin=-2, vmax=2,
                       extent=[0, ITERS, -0.5, deg_i + 0.5])
        ax.set_yticks(range(deg_i + 1))
        ax.set_ylabel('State (# coop nbrs)')
        ax.set_xlabel('Iter')
        ax.set_title(f'{NODE_NAMES[i]} (deg={deg_i})', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 3: Final Q-table bar chart per agent ─────────────────────────
    for i in range(N):
        ax = fig.add_subplot(gs[2, i])
        deg_i = degrees[i]
        states_range = np.arange(deg_i + 1)
        # Mean over REPS of final Q-values
        final_qc = q_snapshots[-1, :, i, :deg_i + 1, 1].mean(axis=0)
        final_qd = q_snapshots[-1, :, i, :deg_i + 1, 0].mean(axis=0)
        bar_w = 0.35
        ax.bar(states_range - bar_w / 2, final_qc, bar_w,
               color='#2ecc71', label='Q(C)', edgecolor='white')
        ax.bar(states_range + bar_w / 2, final_qd, bar_w,
               color='#e74c3c', label='Q(D)', edgecolor='white')
        ax.set_xticks(states_range)
        ax.set_xlabel('State')
        ax.set_ylabel('Q-value')
        ax.set_title(f'{NODE_NAMES[i]} (deg={deg_i})', fontsize=10)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f'Stateful Q-Learning on Custom 6-Node Graph  |  '
        f'γ={gamma}, β={beta}, α={ALPHA}, T={ITERS}',
        fontsize=15, fontweight='bold', y=0.99)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Runner ──────────────────────────────────────────────────────────────────
from multiprocessing import Pool, cpu_count


def run_single_config(args):
    g, beta = args
    graph = CustomSixNodeGraph(DEVICE)
    p_hist, q_snapshots, state_hist, max_states, degrees = \
        run_simulation(graph, gamma=g, beta=beta)
    out_file = os.path.join(out_dir,
                            f'custom6_stateful_gamma{g}_beta{beta}.jpg')
    plot_full_dashboard(p_hist, q_snapshots, state_hist, max_states, degrees,
                        g, beta, out_file)
    return True


def main():
    print('Generating Stateful Q-Learning Graphics for Custom 6-Node Graph...')
    tasks = [(g, b) for g in GAMMAS for b in BETAS]
    total = len(tasks)
    print(f'Total experiments: {total}')

    num_workers = min(cpu_count(), 8)
    print(f'Using {num_workers} parallel workers...')

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(run_single_config, tasks),
                  total=total, desc='Overall'))

    print('\nAll stateful experiments completed.')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
