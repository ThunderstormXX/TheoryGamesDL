import matplotlib.pyplot as plt
import numpy as np
import os


def _robust_ylim(
    ys: list[np.ndarray],
    *,
    q_low: float = 1.0,
    q_high: float = 99.0,
    pad_frac: float = 0.06,
    fallback_pad: float = 1.0,
) -> tuple[float, float]:
    """Compute informative y-limits from multiple series.

    Uses percentiles to avoid a single spike flattening the plot.
    """
    vals = np.concatenate([np.asarray(y, dtype=float).reshape(-1) for y in ys if y is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-fallback_pad, fallback_pad)

    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-fallback_pad, fallback_pad)

    if hi <= lo:
        mid = float(np.mean(vals))
        return (mid - fallback_pad, mid + fallback_pad)

    span = hi - lo
    pad = max(1e-9, pad_frac * span)
    return (lo - pad, hi + pad)


def _pc_ylim(ys: list[np.ndarray], *, pad: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(y, dtype=float).reshape(-1) for y in ys if y is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-0.05, 1.05)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    lo = max(-0.05, lo - pad)
    hi = min(1.05, hi + pad)
    if hi <= lo:
        return (-0.05, 1.05)
    return (lo, hi)

def plot_cooperation_with_std(history_data, labels, title="Cooperation Rate", save_path="results/plot.png", 
                               theory_values=None, experiment_info=None):
    """
    Plots mean cooperation rates with standard deviation shading.
    """
    plt.figure(figsize=(12, 7))
    
    rounds = np.arange(history_data[0].shape[0])
    
    for history, label in zip(history_data, labels):
        mean_coop = np.mean(history, axis=1)
        std_coop = np.std(history, axis=1)
        
        plt.plot(rounds, mean_coop, label=f"Mean {label}", linewidth=2)
        plt.fill_between(rounds, mean_coop - std_coop, mean_coop + std_coop, alpha=0.15)
        
    # Add theoretical horizontal lines
    if theory_values:
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(theory_values)))
        for i, (label, val) in enumerate(theory_values.items()):
            plt.axhline(y=val, color=colors[i], linestyle='--', alpha=0.8, 
                        label=f"Theory {label}: {val:.3f}")
        
    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.title(title, fontsize=14)
    
    # Add box with experiment info
    if experiment_info:
        info_text = "Experiment Parameters:\n" + "-"*20 + "\n"
        info_text += "\n".join([f"{k}: {v}" for k, v in experiment_info.items()])
        
        # Change position to upper left or lower right with better styling
        plt.text(0.98, 0.05, info_text, transform=plt.gca().transAxes, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='black'),
                 fontsize=10, family='monospace')
        
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    abs_save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
    plt.savefig(abs_save_path, dpi=150)
    print(f"Plot saved to: {abs_save_path}")
    plt.close()

def plot_q_table(q_table, k, agent_idx=0, batch_idx=0, save_path="results/q_table.png"):
    """
    Plots Q-values for a single agent.
    """
    q_np = q_table[batch_idx, agent_idx].cpu().detach().numpy()
    valid_states = k + 1
    q_subset = q_np[:valid_states, :]
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_subset[:, 0], label='Action 0 (Cooperate)', marker='o')
    plt.plot(q_subset[:, 1], label='Action 1 (Defect)', marker='x')
    plt.xlabel('State (Number of Cooperating Neighbors)')
    plt.ylabel('Q-Value')
    plt.title(f'Q-Function for Agent {agent_idx} (Batch {batch_idx})')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_trap_q_and_p(
    *,
    q_hist: np.ndarray,
    p_hist: np.ndarray,
    record_every: int,
    trap_eps: float,
    title: str,
    benefit: float | None = None,
    cost: float | None = None,
    reward_type: str | None = None,
    rep_idx: int = 0,
    overlay_mean: bool = True,
    save_path: str,
) -> str:
    """Plot mean Q(D)/Q(C) and p(C) per agent.

    q_hist: (T_out, B, N, 2)
    p_hist: (T_out, B, N)   where p(C) corresponds to action=1
    """
    if q_hist.ndim != 4:
        raise ValueError(f"q_hist must have shape (T_out,B,N,2), got {q_hist.shape}")
    if p_hist.ndim != 3:
        raise ValueError(f"p_hist must have shape (T_out,B,N), got {p_hist.shape}")

    T_out, B, n_agents, _ = q_hist.shape
    rep_idx = int(np.clip(rep_idx, 0, max(B - 1, 0)))

    q_rep = q_hist[:, rep_idx]  # (T_out,N,2)
    p_rep = p_hist[:, rep_idx]  # (T_out,N)

    q_mean = q_hist.mean(axis=1) if (overlay_mean and B > 1) else None
    p_mean = p_hist.mean(axis=1) if (overlay_mean and B > 1) else None

    x = np.arange(T_out) * int(record_every)

    fig, axes = plt.subplots(n_agents, 2, figsize=(14, 3.5 * n_agents), sharex=True)
    if n_agents == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=12)

    # Parameter annotation (requested): show b and c on the figure itself.
    info_parts = []
    if reward_type is not None:
        info_parts.append(f"reward={reward_type}")
    if benefit is not None:
        info_parts.append(f"b={benefit:.4g}")
    if cost is not None:
        info_parts.append(f"c={cost:.4g}")
    if info_parts:
        fig.text(
            0.01,
            0.99,
            " | ".join(info_parts),
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='0.7'),
        )

    for i in range(n_agents):
        ax_q = axes[i, 0]
        ax_p = axes[i, 1]

        if q_mean is not None:
            ax_q.plot(x, q_mean[:, i, 0], color='C0', alpha=0.35, linewidth=1.2, label='Q(D) mean')
            ax_q.plot(x, q_mean[:, i, 1], color='C1', alpha=0.35, linewidth=1.2, label='Q(C) mean')

        ax_q.plot(x, q_rep[:, i, 0], color='C0', linewidth=1.8, label=f'Q(D) rep{rep_idx}')
        ax_q.plot(x, q_rep[:, i, 1], color='C1', linewidth=1.8, label=f'Q(C) rep{rep_idx}')
        ax_q.set_ylabel(f'Agent {i}')
        # Informative scaling for Q
        qlo, qhi = _robust_ylim(
            [q_rep[:, i, 0], q_rep[:, i, 1]] + ([] if q_mean is None else [q_mean[:, i, 0], q_mean[:, i, 1]]),
            q_low=1.0,
            q_high=99.0,
            pad_frac=0.06,
        )
        ax_q.set_ylim(qlo, qhi)
        ax_q.grid(True, alpha=0.3)
        if i == 0:
            ax_q.legend(loc='best', fontsize=8)

        if p_mean is not None:
            ax_p.plot(x, p_mean[:, i], color='purple', alpha=0.35, linewidth=1.2, label='p(C) mean')
        ax_p.plot(x, p_rep[:, i], color='purple', linewidth=1.8, label=f'p(C) rep{rep_idx}')
        ax_p.axhline(trap_eps, color='red', linestyle='--', alpha=0.6, label='trap eps')
        # Informative scaling for probabilities (still bounded to [-0.05, 1.05])
        plo, phi = _pc_ylim([p_rep[:, i]] + ([] if p_mean is None else [p_mean[:, i]]))
        ax_p.set_ylim(plo, phi)
        ax_p.grid(True, alpha=0.3)
        if i == 0:
            ax_p.legend(loc='best', fontsize=8)

    axes[-1, 0].set_xlabel('Iterations')
    axes[-1, 1].set_xlabel('Iterations')
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=160, bbox_inches='tight')
    plt.close()
    return abs_path


def plot_two_agent_combined_series(
    *,
    q_hist: np.ndarray,
    p_hist: np.ndarray,
    record_every: int,
    title_prefix: str,
    rep_idx: int,
    smooth_window: int = 1,
    benefit: float | None = None,
    cost: float | None = None,
    reward_type: str | None = None,
    save_dir: str,
) -> dict:
    """Create a single figure with 3 subplots for a 2-agent system.

    Subplots:
      1) Q(C) for both agents (agent0 red, agent1 blue)
      2) Q(D) for both agents
      3) p(C) for both agents

    If smooth_window > 1, applies a simple moving average for readability.
    Returns a dict of absolute paths.
    """
    if q_hist.ndim != 4 or q_hist.shape[-1] != 2:
        raise ValueError(f"q_hist must have shape (T_out,B,N,2), got {q_hist.shape}")
    if p_hist.ndim != 3:
        raise ValueError(f"p_hist must have shape (T_out,B,N), got {p_hist.shape}")

    T_out, B, n_agents, _ = q_hist.shape
    if n_agents != 2:
        raise ValueError(f"plot_two_agent_combined_series expects exactly 2 agents, got {n_agents}")

    rep_idx = int(np.clip(rep_idx, 0, max(B - 1, 0)))
    q_rep = q_hist[:, rep_idx]  # (T_out,2,2)
    p_rep = p_hist[:, rep_idx]  # (T_out,2)
    x = np.arange(T_out) * int(record_every)

    info_parts = []
    if reward_type is not None:
        info_parts.append(f"reward={reward_type}")
    if benefit is not None:
        info_parts.append(f"b={benefit:.4g}")
    if cost is not None:
        info_parts.append(f"c={cost:.4g}")
    info_text = " | ".join(info_parts) if info_parts else None

    def _sma(y: np.ndarray, w: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        w = int(w)
        if w <= 1 or y.size == 0:
            return y
        w = min(w, int(y.size))
        k = np.ones(w, dtype=float) / float(w)
        return np.convolve(y, k, mode='same')

    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    out: dict = {}

    lw = 1.2
    alpha = 0.95

    q_c0 = _sma(q_rep[:, 0, 1], smooth_window)
    q_c1 = _sma(q_rep[:, 1, 1], smooth_window)
    q_d0 = _sma(q_rep[:, 0, 0], smooth_window)
    q_d1 = _sma(q_rep[:, 1, 0], smooth_window)
    p0 = _sma(p_rep[:, 0], smooth_window)
    p1 = _sma(p_rep[:, 1], smooth_window)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True)
    fig.suptitle(title_prefix, fontsize=12)

    ax = axes[0]
    ax.plot(x, q_c0, color='red', linewidth=lw, alpha=alpha, label='Agent 0')
    ax.plot(x, q_c1, color='blue', linewidth=lw, alpha=alpha, label='Agent 1')
    ax.set_ylabel('Q(C)')
    lo, hi = _robust_ylim([q_c0, q_c1], q_low=1.0, q_high=99.0, pad_frac=0.06)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    ax = axes[1]
    ax.plot(x, q_d0, color='red', linewidth=lw, alpha=alpha)
    ax.plot(x, q_d1, color='blue', linewidth=lw, alpha=alpha)
    ax.set_ylabel('Q(D)')
    lo, hi = _robust_ylim([q_d0, q_d1], q_low=1.0, q_high=99.0, pad_frac=0.06)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(x, p0, color='red', linewidth=lw, alpha=alpha)
    ax.plot(x, p1, color='blue', linewidth=lw, alpha=alpha)
    ax.set_ylabel('p(C)')
    ax.set_xlabel('Iterations')
    plo, phi = _pc_ylim([p0, p1])
    ax.set_ylim(plo, phi)
    ax.grid(True, alpha=0.25)

    if info_text:
        fig.text(
            0.01,
            0.99,
            info_text + (f" | SMA={int(smooth_window)}" if int(smooth_window) > 1 else ""),
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='0.7'),
        )
    elif int(smooth_window) > 1:
        fig.text(
            0.01,
            0.99,
            f"SMA={int(smooth_window)}",
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='0.7'),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.abspath(os.path.join(save_dir, 'two_agent_qp.png'))
    plt.savefig(combined_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    out['two_agent_qp'] = combined_path
    return out


def plot_three_agent_combined_series(
    *,
    q_hist: np.ndarray,
    p_hist: np.ndarray,
    record_every: int,
    title_prefix: str,
    rep_idx: int,
    smooth_window: int = 1,
    benefit: float | None = None,
    cost: float | None = None,
    reward_type: str | None = None,
    save_dir: str,
) -> dict:
    """Create a single figure with 3 subplots for a 3-agent system.

    Subplots:
      1) Q(C) for all 3 agents
      2) Q(D) for all 3 agents
      3) p(C) for all 3 agents

    Colors: agent0 red, agent1 blue, agent2 green.
    If smooth_window > 1, applies a simple moving average for readability.
    """
    if q_hist.ndim != 4 or q_hist.shape[-1] != 2:
        raise ValueError(f"q_hist must have shape (T_out,B,N,2), got {q_hist.shape}")
    if p_hist.ndim != 3:
        raise ValueError(f"p_hist must have shape (T_out,B,N), got {p_hist.shape}")

    T_out, B, n_agents, _ = q_hist.shape
    if n_agents != 3:
        raise ValueError(f"plot_three_agent_combined_series expects exactly 3 agents, got {n_agents}")

    rep_idx = int(np.clip(rep_idx, 0, max(B - 1, 0)))
    q_rep = q_hist[:, rep_idx]  # (T_out,3,2)
    p_rep = p_hist[:, rep_idx]  # (T_out,3)
    x = np.arange(T_out) * int(record_every)

    info_parts = []
    if reward_type is not None:
        info_parts.append(f"reward={reward_type}")
    if benefit is not None:
        info_parts.append(f"b={benefit:.4g}")
    if cost is not None:
        info_parts.append(f"c={cost:.4g}")
    info_text = " | ".join(info_parts) if info_parts else None

    def _sma(y: np.ndarray, w: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        w = int(w)
        if w <= 1 or y.size == 0:
            return y
        w = min(w, int(y.size))
        k = np.ones(w, dtype=float) / float(w)
        return np.convolve(y, k, mode='same')

    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    out: dict = {}

    lw = 1.2
    alpha = 0.95
    colors = ['red', 'blue', 'green']
    labels = ['Agent 0', 'Agent 1', 'Agent 2']

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.6), sharex=True)
    fig.suptitle(title_prefix, fontsize=12)

    # Q(C)
    ax = axes[0]
    for i in range(3):
        ax.plot(x, _sma(q_rep[:, i, 1], smooth_window), color=colors[i], linewidth=lw, alpha=alpha, label=labels[i])
    ax.set_ylabel('Q(C)')
    qc_series = [_sma(q_rep[:, 0, 1], smooth_window), _sma(q_rep[:, 1, 1], smooth_window), _sma(q_rep[:, 2, 1], smooth_window)]
    lo, hi = _robust_ylim(qc_series, q_low=1.0, q_high=99.0, pad_frac=0.06)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    # Q(D)
    ax = axes[1]
    for i in range(3):
        ax.plot(x, _sma(q_rep[:, i, 0], smooth_window), color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_ylabel('Q(D)')
    qd_series = [_sma(q_rep[:, 0, 0], smooth_window), _sma(q_rep[:, 1, 0], smooth_window), _sma(q_rep[:, 2, 0], smooth_window)]
    lo, hi = _robust_ylim(qd_series, q_low=1.0, q_high=99.0, pad_frac=0.06)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)

    # p(C)
    ax = axes[2]
    for i in range(3):
        ax.plot(x, _sma(p_rep[:, i], smooth_window), color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_ylabel('p(C)')
    ax.set_xlabel('Iterations')
    pc_series = [_sma(p_rep[:, 0], smooth_window), _sma(p_rep[:, 1], smooth_window), _sma(p_rep[:, 2], smooth_window)]
    plo, phi = _pc_ylim(pc_series)
    ax.set_ylim(plo, phi)
    ax.grid(True, alpha=0.25)

    header = None
    if info_text:
        header = info_text
    if int(smooth_window) > 1:
        header = (header + " | " if header else "") + f"SMA={int(smooth_window)}"
    if header:
        fig.text(
            0.01,
            0.99,
            header,
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='0.7'),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.abspath(os.path.join(save_dir, 'three_agent_qp.png'))
    plt.savefig(combined_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    out['three_agent_qp'] = combined_path
    return out


def plot_deltaq_increment_distribution(*, deltaq_inc: np.ndarray, title: str, save_path: str) -> str:
    """Histogram of δ(ΔQ) where ΔQ=Q(C)-Q(D).

    deltaq_inc: (B, N, T-1)
    """
    vals = np.asarray(deltaq_inc).reshape(-1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel('δ(ΔQ)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=160, bbox_inches='tight')
    plt.close()
    return abs_path


def plot_volatility_acf(*, acf_vals: np.ndarray, title: str, save_path: str) -> str:
    """Plot ACF curve (e.g., for |δ(ΔQ)|)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(acf_vals, marker='o', markersize=3, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=160, bbox_inches='tight')
    plt.close()
    return abs_path


def plot_volatility_clustering_timeseries(
    *,
    t: np.ndarray,
    mean_abs_inc: np.ndarray,
    rolling_std: np.ndarray,
    title: str,
    save_path: str,
) -> str:
    """Show volatility clustering over time.

    Plots mean |δ(ΔQ)| time series and its rolling std. The rolling std may contain
    NaNs at the beginning (window warm-up), which are handled by Matplotlib.
    """
    t = np.asarray(t)
    mean_abs_inc = np.asarray(mean_abs_inc)
    rolling_std = np.asarray(rolling_std)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    ax.plot(t, mean_abs_inc, color='black', linewidth=1.0)
    ax.set_ylabel('mean |δ(ΔQ)|')
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(t, rolling_std, color='darkred', linewidth=1.2)
    ax.set_ylabel('rolling std')
    ax.set_xlabel('Iterations')
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=160, bbox_inches='tight')
    plt.close()
    return abs_path
