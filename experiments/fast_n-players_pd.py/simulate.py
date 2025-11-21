#!/usr/bin/env python3
# simulate.py
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm

from bots import BoltzmannAgent
from environment import GameFactory

# safe XDG_RUNTIME_DIR (как раньше)
_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
if not _runtime_dir or not os.path.isdir(_runtime_dir) or (os.stat(_runtime_dir).st_mode & 0o777) != 0o700:
    tmp_runtime = os.path.join(tempfile.gettempdir(), f"runtime-{os.getuid()}")
    os.makedirs(tmp_runtime, exist_ok=True)
    try:
        os.chmod(tmp_runtime, 0o700)
    except PermissionError:
        pass
    os.environ["XDG_RUNTIME_DIR"] = tmp_runtime

def smooth(x: np.ndarray, window: int = 201) -> np.ndarray:
    if window <= 1:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode='same')

def detect_traps(p_traj: np.ndarray, eps: float = 0.02, min_duration: int = 200):
    n_players, T = p_traj.shape
    below = np.all(p_traj < eps, axis=0)
    intervals = []
    t = 0
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



def _python_run_loop(T, agents, game, record_every, store_q_traj, out_len):
    """Pure-Python run (fallback)."""
    n_players = len(agents)
    p_traj = np.empty((n_players, out_len), dtype=float)
    q_traj = np.empty((n_players, out_len, 2), dtype=float) if store_q_traj else None

    mean_r = 0.0
    out_idx = 0
    for t in range(T):
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))
        mean_r = (t) / (t + 1) * mean_r + rewards[0] / (t + 1)
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
        if (t % record_every) == 0:
            for i, agent in enumerate(agents):
                p_traj[i, out_idx] = agent.current_p_cooperate()
                if store_q_traj:
                    q_traj[i, out_idx, :] = agent.get_q()
            out_idx += 1
    if out_idx < out_len:
        p_traj = p_traj[:, :out_idx]
        if store_q_traj:
            q_traj = q_traj[:, :out_idx, :]
    return agents, p_traj, q_traj, mean_r

def run_sim(
        T=20000,
        alpha=0.01,
        beta=1.0,
        gamma=0.9,
        seed=42,
        n_players=2,
        max_keep=100_000,
        use_tqdm=True,
    ):
    """
    Запускает симуляцию с сохранением НЕ больше max_keep последних точек.
    Возвращает агентов, p_traj, q_traj, mean_q, meta.
    """

    rng = np.random.default_rng(seed)
    game = GameFactory.create_generalized_prisoners_dilemma(n_players, 3, 4, 1, 0)
    agents = [
        BoltzmannAgent(name=f"A{i+1}", alpha=alpha, beta=beta, gamma=gamma, rng=rng)
        for i in range(n_players)
    ]

    M = max_keep
    p_traj = np.zeros((n_players, M), dtype=float)
    q_traj = np.zeros((n_players, M, 2), dtype=float)

    mean_q = 0.0

    iterator = range(T)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Simulating", ncols=80)

    for t in iterator:
        actions = [agent.choose_action() for agent in agents]

        rewards = game.get_payoffs(tuple(actions))
        mean_q = (t / (t + 1)) * mean_q + rewards[0] / (t + 1)

        idx = t % M

        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])

            p_traj[i, idx] = agent.current_p_cooperate()
            q_traj[i, idx, :] = agent.get_q()

    # ---- восстановить корректный порядок ----
    if T <= M:
        p_final = p_traj[:, :T].copy()
        q_final = q_traj[:, :T, :].copy()
    else:
        start = T % M
        p_final = np.concatenate([p_traj[:, start:], p_traj[:, :start]], axis=1)
        q_final = np.concatenate([q_traj[:, start:], q_traj[:, :start]], axis=1)

    meta = {
        "T": T,
        "used_T": p_final.shape[1],
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "n_players": n_players,
    }

    return agents, p_final, q_final, mean_q, meta



# ---------- utility plotting functions ----------
def plot_results(p_traj, q_traj, agents, title_suffix="", smooth=False, meta=None):

    # --- формируем строку параметров ---
    if meta is not None:
        param_string = (
            f"α={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
            f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}"
        )
    else:
        param_string = ""

    n_players, T = p_traj.shape
    time = np.arange(T)
    smooth_win = max(101, T // 200)
    subsample = max(1, T // 2000)
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(12, 6 + 2 * n_players))
    gs = fig.add_gridspec(2 + n_players, 1, height_ratios=[3, 0.2] + [1]*n_players, hspace=0.6)
    ax_top = fig.add_subplot(gs[0, 0])

    # ---- TITLE with params ----
    title = "Probability of Cooperation over time"
    if title_suffix:
        title += f" — {title_suffix}"
    if param_string:
        title += f"\n{param_string}"
    ax_top.set_title(title)

    # ---- plot p(C) ----
    for i in range(n_players):
        raw = p_traj[i]
        sm = smooth(raw, window=smooth_win)
        ax_top.scatter(time[::subsample], raw[::subsample], s=4, alpha=0.15, color=colors[i % len(colors)])
        ax_top.plot(time, sm, label=f"{agents[i].name} (smoothed)", lw=1.6, color=colors[i % len(colors)])
        last = int(0.8 * T)
        mean_p = np.mean(raw[last:]) if last < T else np.mean(raw)
        ax_top.axhline(mean_p, color=colors[i % len(colors)], linestyle=':', alpha=0.8)
        ax_top.text(T * 0.99, mean_p, f"{mean_p:.2f}", ha='right', va='center',
                    fontsize=8, color=colors[i % len(colors)])

    ax_top.set_xlabel("Timestep")
    ax_top.set_ylabel("P(C)")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.legend(loc='upper right', fontsize=9)
    ax_top.grid(alpha=0.3)

    # ---- info block ----
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    summary_lines = []
    for i in range(n_players):
        mean_last = np.mean(p_traj[i, int(0.8*T):]) if int(0.8*T) < T else np.mean(p_traj[i])
        summary_lines.append(f"{agents[i].name}: mean p(C) last 20% = {mean_last:.3f}")
    ax_info.text(0.01, 0.5, "\n".join(summary_lines), fontsize=10, va='center')

    # ---- Q-values ----
    for i in range(n_players):
        ax_q = fig.add_subplot(gs[2 + i, 0])
        qC = q_traj[i, :, 0]
        qD = q_traj[i, :, 1]
        ax_q.plot(time, qC, label=f"{agents[i].name} Q(C)", color=colors[2*i % len(colors)])
        ax_q.plot(time, qD, label=f"{agents[i].name} Q(D)", linestyle='--', color=colors[(2*i+1) % len(colors)])

        last = int(0.8 * T)
        mean_qC = np.mean(qC[last:]) if last < T else np.mean(qC)
        mean_qD = np.mean(qD[last:]) if last < T else np.mean(qD)
        ax_q.axhline(mean_qC, color=colors[2*i % len(colors)], linestyle=':', alpha=0.8)
        ax_q.axhline(mean_qD, color=colors[(2*i+1) % len(colors)], linestyle=':', alpha=0.8)
        ax_q.text(T * 0.99, mean_qC, f"{mean_qC:.2f}", ha='right', va='center', fontsize=8,
                  color=colors[2*i % len(colors)])
        ax_q.text(T * 0.99, mean_qD, f"{mean_qD:.2f}", ha='right', va='center', fontsize=8,
                  color=colors[(2*i+1) % len(colors)])

        ax_q.set_ylabel("Q-value")
        ax_q.set_xlabel("Timestep")
        ax_q.legend(loc='upper right', fontsize=9)
        ax_q.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_p_traj_separate(p_traj: np.ndarray, agents, title_suffix="", meta=None):
    n_players, T = p_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    if meta is not None:
        title_suffix += (
            f"\nα={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
            f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}"
        )

    plt.figure(figsize=(10, 6))
    for i in range(n_players):
        plt.plot(time, p_traj[i], label=f"{agents[i].name} P(C)", color=colors[i % len(colors)], alpha=0.7)

    plt.title("P(C) Over Time" + (f" — {title_suffix}" if title_suffix else ""))
    plt.xlabel("Timestep")
    plt.ylabel("P(C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_q_values_separate(q_traj: np.ndarray, agents, title_suffix="", meta=None):
    n_players, T, _ = q_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    if meta is not None:
        title_suffix += (
            f"\nα={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
            f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}"
        )

    for i in range(n_players):
        qC = q_traj[i, :, 0]
        qD = q_traj[i, :, 1]

        plt.plot(time, qC, label=f"{agents[i].name} Q(C)", color=colors[2*i % len(colors)])
        plt.plot(time, qD, label=f"{agents[i].name} Q(D)", linestyle='--', color=colors[(2*i+1) % len(colors)])

        last = int(0.8 * T)
        mean_qC = np.mean(qC[last:])
        mean_qD = np.mean(qD[last:])

        plt.axhline(mean_qC, color=colors[2*i % len(colors)], linestyle=':', alpha=0.8)
        plt.axhline(mean_qD, color=colors[(2*i+1) % len(colors)], linestyle=':', alpha=0.8)

        plt.text(T * 0.99, mean_qC, f"{mean_qC:.2f}", ha='right', va='center',
                 color=colors[2*i % len(colors)], fontsize=8)
        plt.text(T * 0.99, mean_qD, f"{mean_qD:.2f}", ha='right', va='center',
                 color=colors[(2*i+1) % len(colors)], fontsize=8)

        plt.title("Q-values Over Time" + (f" — {title_suffix}" if title_suffix else ""))
        plt.xlabel("Timestep")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    agents, p_traj, q_traj, mean_r, meta = run_sim(
        T=200000,
        alpha=0.01,
        beta=2.00,
        gamma=0.7,
        n_players=50,
        max_keep=200000,
    )

    # рисуем графики
    plot_p_traj_separate(p_traj, agents, meta = meta)
    plot_q_values_separate(q_traj, agents, meta = meta)
