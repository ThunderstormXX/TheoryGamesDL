#!/usr/bin/env python3
"""
simulate_3p.py

Запускает симуляцию Boltzmann Q-learning для n_players (2 или 3),
строит информативные графики (p(C) с сглаживанием и Q-values по агентам),
и включает детектор "ловушек" (все p(C) малы долгое время).

"""

import os
import tempfile
import stat
import numpy as np
import matplotlib.pyplot as plt


from bots import BoltzmannAgent
from environment import GameFactory



# ------------------------------------------------------------------------
# Утилиты
def smooth(x: np.ndarray, window: int = 201) -> np.ndarray:
    """Сглаживание скользящим средним (одномерный, центрированное)."""
    if window <= 1:
        return x
    w = np.ones(window) / window
    # 'same' -> output длиной x, do convolution with padding via mode='reflect'
    return np.convolve(x, w, mode='same')

def detect_traps(p_traj: np.ndarray, eps: float = 0.02, min_duration: int = 200):
    """
    Находит интервалы (start, end) где все игроки имеют p(C) < eps подряд не менее min_duration.
    p_traj shape: (n_players, T)
    """
    n_players, T = p_traj.shape
    below = np.all(p_traj < eps, axis=0)  # shape (T,)
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

# ------------------------------------------------------------------------
# Симулятор
def run_sim(T=20000, alpha=0.01, beta=1.0, gamma=0.9, seed=42, n_players=3):
    """
    Запускает симуляцию n_players (2 или 3) с BoltzmannAgent.
    Возвращает агентов, p_traj (n_players x T), q_traj (n_players x T x 2).
    """
    assert n_players in (2, 3), "supported n_players = 2 or 3"
    rng = np.random.default_rng(seed)
    game = GameFactory.create_generalized_prisoners_dilemma(n_players)
    agents = [BoltzmannAgent(name=f"A{i+1}", alpha=alpha, beta=beta, gamma=gamma, rng=rng)
              for i in range(n_players)]

    p_traj = np.empty((n_players, T), dtype=float)
    q_traj = np.empty((n_players, T, 2), dtype=float)

    for t in range(T):
        # 1) агенты выбирают действия
        actions = [agent.choose_action() for agent in agents]

        # 2) получаем выплату каждому игроку
        rewards = game.get_payoffs(tuple(actions))

        # 3) обновляем агента (Q-learning)
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            p_traj[i, t] = agent.current_p_cooperate()
            q_traj[i, t, :] = agent.get_q()

    return agents, p_traj, q_traj

# ------------------------------------------------------------------------
# Улучшённая визуализация
def plot_results(p_traj: np.ndarray, q_traj: np.ndarray, agents, title_suffix=""):
    """
    1) Верхний большой график: p(C) для каждого игрока, сглаженные кривые + полупрозрачные сырые точки.
    2) Нижняя панель: по одному сабплоту на агента с Q(C) и Q(D).
    """
    n_players, T = p_traj.shape
    time = np.arange(T)

    # Параметры визуализации
    smooth_win = max(101, T // 200)  # окно в зависимости от T, чтобы не очень мелко
    subsample = max(1, T // 2000)    # если T большой, рисуем точки с шагом subsample
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(12, 6 + 2 * n_players))
    gs = fig.add_gridspec(2 + n_players, 1, height_ratios=[3, 0.2] + [1]*n_players, hspace=0.6)

    # --- Верх: p(C) raw + smoothed
    ax_top = fig.add_subplot(gs[0, 0])
    for i in range(n_players):
        raw = p_traj[i]
        sm = smooth(raw, window=smooth_win)
        # plot raw as faint dots (subsampled)
        ax_top.scatter(time[::subsample], raw[::subsample], s=4, alpha=0.15, color=colors[i % len(colors)])
        # plot smoothed line
        ax_top.plot(time, sm, label=f"{agents[i].name} (smoothed)", lw=1.6, color=colors[i % len(colors)])
        # also mark final mean value
        final_mean = np.mean(raw[int(0.8*T):])
        ax_top.hlines(final_mean, xmin=0, xmax=T, colors=colors[i % len(colors)], linestyles='dotted', alpha=0.7)
        ax_top.text(T*0.98, final_mean, f"{final_mean:.2f}", va='center', ha='right', fontsize=9, color=colors[i % len(colors)])

    ax_top.set_xlabel("Timestep")
    ax_top.set_ylabel("P(C)")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.set_title("Probability of Cooperation over time" + (f" — {title_suffix}" if title_suffix else ""))
    ax_top.legend(loc='upper right', fontsize=9)
    ax_top.grid(alpha=0.3)

    # --- краткая сводка (подпись)
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    summary_lines = []
    for i in range(n_players):
        mean_last = np.mean(p_traj[i, int(0.8*T):])
        summary_lines.append(f"{agents[i].name}: mean p(C) last 20% = {mean_last:.3f}")
    ax_info.text(0.01, 0.5, "\n".join(summary_lines), fontsize=10, va='center')

    # --- Нижние: Q-values per agent (по одной строке на агента)
    for i in range(n_players):
        ax_q = fig.add_subplot(gs[2 + i, 0])
        ax_q.plot(time, q_traj[i, :, 0], label=f"{agents[i].name} Q(C)", color=colors[2*i % len(colors)])
        ax_q.plot(time, q_traj[i, :, 1], label=f"{agents[i].name} Q(D)", linestyle='--', color=colors[(2*i+1) % len(colors)])
        ax_q.set_ylabel("Q-value")
        ax_q.set_xlabel("Timestep")
        ax_q.legend(loc='upper right', fontsize=9)
        ax_q.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

def plot_p_traj_separate(p_traj: np.ndarray, agents, title_suffix=""):
    """
    Строит ОТДЕЛЬНО PDF-подобный график P(C) по каждому игроку.
    Увеличенная фигура, удобна для анализа.
    """
    n_players, T = p_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

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

def plot_q_values_separate(q_traj: np.ndarray, agents, title_suffix=""):
    """
    Рисует по ОДНОМУ ОТДЕЛЬНОМУ окну Q(C) и Q(D) для каждого агента.
    Очень удобно, когда нужно изучать поведение отдельно.
    """
    n_players, T, _ = q_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    for i in range(n_players):
        plt.figure(figsize=(9, 5))
        plt.plot(time, q_traj[i, :, 0], label=f"{agents[i].name} Q(C)", color=colors[2*i % len(colors)])
        plt.plot(time, q_traj[i, :, 1], label=f"{agents[i].name} Q(D)", linestyle='--', color=colors[(2*i+1) % len(colors)])
        plt.title(f"Q-values for {agents[i].name}" + (f" — {title_suffix}" if title_suffix else ""))
        plt.xlabel("Timestep")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------
# Примеры запуска (если файл запускается как скрипт)
if __name__ == "__main__":
    # Параметры
    T = 100000
    alpha = 0.01
    beta = 1.0
    gamma = 0.9
    seed = 42
    n_players = 3  # можно 2 или 3

    agents, p_traj, q_traj = run_sim(T=T, alpha=alpha, beta=beta, gamma=gamma, seed=seed, n_players=n_players)

    # Найти ловушки
    traps = detect_traps(p_traj, eps=0.02, min_duration=200)
    print("Found traps (start, end):", traps)

    # Напечатать статистику агентов
    for a in agents:
        a.print_stats()

    # Визуализировать
    plot_p_traj_separate(p_traj, agents)
    plot_q_values_separate(q_traj, agents)
