#!/usr/bin/env python3
"""Симуляция многоагентной дилеммы заключенного с Q-обучением."""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Optional
from tqdm import tqdm
from datetime import datetime

from bots import BoltzmannAgent
from environment import GameFactory

# Настройка безопасной директории для runtime
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
    """Сглаживание временного ряда скользящим средним."""
    if window <= 1:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode='same')

def detect_traps(p_traj: np.ndarray, eps: float = 0.02, min_duration: int = 200):
    """Обнаружение ловушек дефекта (периоды низкой кооперации)."""
    n_players, T = p_traj.shape
    below = np.all(p_traj < eps, axis=0)  # Все игроки ниже порога
    intervals = []
    t = 0
    
    while t < T:
        if below[t]:
            t0 = t
            while t < T and below[t]:
                t += 1
            if t - t0 >= min_duration:  # Достаточно длинный период
                intervals.append((t0, t))
        else:
            t += 1
    return intervals



def _python_run_loop(T, agents, game, record_every, store_q_traj, out_len):
    """Основной цикл симуляции (Python fallback)."""
    n_players = len(agents)
    p_traj = np.empty((n_players, out_len), dtype=float)
    q_traj = np.empty((n_players, out_len, 2), dtype=float) if store_q_traj else None

    mean_r = 0.0
    out_idx = 0
    
    for t in range(T):
        # Выбор действий всеми агентами
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))
        mean_r = (t) / (t + 1) * mean_r + rewards[0] / (t + 1)
        
        # Обучение агентов
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            
        # Запись данных
        if (t % record_every) == 0:
            for i, agent in enumerate(agents):
                p_traj[i, out_idx] = agent.current_p_cooperate()
                if store_q_traj:
                    q_traj[i, out_idx, :] = agent.get_q()
            out_idx += 1
            
    # Обрезка до фактического размера
    if out_idx < out_len:
        p_traj = p_traj[:, :out_idx]
        if store_q_traj:
            q_traj = q_traj[:, :out_idx, :]
    return agents, p_traj, q_traj, mean_r

def run_sim(T=20000, alpha=0.01, beta=1.0, gamma=0.9, seed=42, 
           n_players=2, max_keep=100_000, use_tqdm=True):
    """Запуск симуляции с ограничением памяти.
    
    Args:
        T: Количество шагов симуляции
        alpha: Скорость обучения
        beta: Температура Больцмана
        gamma: Коэффициент дисконтирования
        seed: Случайное зерно
        n_players: Количество игроков
        max_keep: Максимум точек для сохранения
        use_tqdm: Показывать прогресс-бар
    
    Returns:
        agents, p_traj, q_traj, mean_q, meta
    """
    # Инициализация
    rng = np.random.default_rng(seed)
    game = GameFactory.create_generalized_prisoners_dilemma(args.n_players, benefit=args.benefit, cost=args.cost, reward_offset=args.reward_offset)
    agents = [BoltzmannAgent(name=f"A{i+1}", alpha=alpha, beta=beta, 
                           gamma=gamma, rng=rng) for i in range(n_players)]

    # Буферы с циклической записью
    M = max_keep
    p_traj = np.zeros((n_players, M), dtype=float)
    q_traj = np.zeros((n_players, M, 2), dtype=float)
    mean_q = 0.0

    iterator = tqdm(range(T), desc="Simulating", ncols=80) if use_tqdm else range(T)

    # Основной цикл симуляции
    for t in iterator:
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))
        mean_q = (t / (t + 1)) * mean_q + rewards[0] / (t + 1)
        
        idx = t % M  # Циклический индекс
        
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            p_traj[i, idx] = agent.current_p_cooperate()
            q_traj[i, idx, :] = agent.get_q()

    # Восстановление правильного порядка данных
    if T <= M:
        p_final = p_traj[:, :T].copy()
        q_final = q_traj[:, :T, :].copy()
    else:
        start = T % M
        p_final = np.concatenate([p_traj[:, start:], p_traj[:, :start]], axis=1)
        q_final = np.concatenate([q_traj[:, start:], q_traj[:, :start]], axis=1)

    meta = {
        "T": T, "used_T": p_final.shape[1], "alpha": alpha,
        "beta": beta, "gamma": gamma, "n_players": n_players
    }

    return agents, p_final, q_final, mean_q, meta



# Функции визуализации
def plot_results(p_traj, q_traj, agents, title_suffix="", smooth_plot=False, meta=None):
    """Комплексная визуализация результатов симуляции."""
    # Формирование строки параметров
    param_string = ""
    if meta is not None:
        param_string = (f"α={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
                       f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}")

    n_players, T = p_traj.shape
    time = np.arange(T)
    smooth_win = max(101, T // 200)
    subsample = max(1, T // 2000)
    colors = plt.cm.tab10.colors

    # Создание фигуры с подграфиками
    fig = plt.figure(figsize=(12, 6 + 2 * n_players))
    gs = fig.add_gridspec(2 + n_players, 1, height_ratios=[3, 0.2] + [1]*n_players, hspace=0.6)
    ax_top = fig.add_subplot(gs[0, 0])

    # Заголовок с параметрами
    title = "Вероятность кооперации во времени"
    if title_suffix:
        title += f" — {title_suffix}"
    if param_string:
        title += f"\n{param_string}"
    ax_top.set_title(title, fontsize=14, fontweight='bold')

    # График вероятности кооперации
    for i in range(n_players):
        raw = p_traj[i]
        sm = smooth(raw, window=smooth_win)
        
        # Исходные данные (точки)
        ax_top.scatter(time[::subsample], raw[::subsample], s=4, alpha=0.15, 
                      color=colors[i % len(colors)])
        # Сглаженная кривая
        ax_top.plot(time, sm, label=f"{agents[i].name} (smoothed)", 
                   lw=1.6, color=colors[i % len(colors)])
        
        # Среднее значение за последние 20%
        last = int(0.8 * T)
        mean_p = np.mean(raw[last:]) if last < T else np.mean(raw)
        ax_top.axhline(mean_p, color=colors[i % len(colors)], linestyle=':', alpha=0.8)
        ax_top.text(T * 0.99, mean_p, f"{mean_p:.2f}", ha='right', va='center',
                   fontsize=8, color=colors[i % len(colors)])

    ax_top.set_xlabel("Шаг симуляции", fontsize=11)
    ax_top.set_ylabel("Вероятность кооперации P(C)", fontsize=11)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.legend(loc='upper right', fontsize=9)
    ax_top.grid(alpha=0.3)

    # Информационный блок
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    summary_lines = []
    for i in range(n_players):
        mean_last = np.mean(p_traj[i, int(0.8*T):]) if int(0.8*T) < T else np.mean(p_traj[i])
        summary_lines.append(f"{agents[i].name}: mean p(C) last 20% = {mean_last:.3f}")
    ax_info.text(0.01, 0.5, "\n".join(summary_lines), fontsize=10, va='center')

    # Q-значения для каждого игрока
    for i in range(n_players):
        ax_q = fig.add_subplot(gs[2 + i, 0])
        qC, qD = q_traj[i, :, 0], q_traj[i, :, 1]
        
        ax_q.plot(time, qC, label=f"{agents[i].name} Q(C)", 
                 color=colors[2*i % len(colors)])
        ax_q.plot(time, qD, label=f"{agents[i].name} Q(D)", 
                 linestyle='--', color=colors[(2*i+1) % len(colors)])

        # Средние значения Q
        last = int(0.8 * T)
        mean_qC = np.mean(qC[last:]) if last < T else np.mean(qC)
        mean_qD = np.mean(qD[last:]) if last < T else np.mean(qD)
        
        ax_q.axhline(mean_qC, color=colors[2*i % len(colors)], linestyle=':', alpha=0.8)
        ax_q.axhline(mean_qD, color=colors[(2*i+1) % len(colors)], linestyle=':', alpha=0.8)
        
        ax_q.text(T * 0.99, mean_qC, f"{mean_qC:.2f}", ha='right', va='center', 
                 fontsize=8, color=colors[2*i % len(colors)])
        ax_q.text(T * 0.99, mean_qD, f"{mean_qD:.2f}", ha='right', va='center', 
                 fontsize=8, color=colors[(2*i+1) % len(colors)])

        ax_q.set_ylabel("Q-значение", fontsize=10)
        ax_q.set_xlabel("Шаг симуляции", fontsize=10)
        ax_q.legend(loc='upper right', fontsize=9)
        ax_q.grid(alpha=0.25)

    plt.tight_layout()
    
    # Сохранение в results текущего эксперимента
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"full_results_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {filename}")


def plot_p_traj_separate(p_traj: np.ndarray, agents, title_suffix="", meta=None):
    """Отдельный график вероятностей кооперации."""
    n_players, T = p_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    if meta is not None:
        title_suffix += (f"\nα={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
                        f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}")

    plt.figure(figsize=(10, 6))
    for i in range(n_players):
        plt.plot(time, p_traj[i], label=f"{agents[i].name} P(C)", 
                color=colors[i % len(colors)], alpha=0.7)

    title = "Динамика вероятности кооперации"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Шаг симуляции", fontsize=11)
    plt.ylabel("Вероятность кооперации P(C)", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Сохранение в results текущего эксперимента
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"p_cooperation_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {filename}")


def plot_q_values_separate(q_traj: np.ndarray, agents, title_suffix="", meta=None):
    """Отдельный график Q-значений."""
    n_players, T, _ = q_traj.shape
    time = np.arange(T)
    colors = plt.cm.tab10.colors

    if meta is not None:
        title_suffix += (f"\nα={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']}   "
                        f"players={meta['n_players']}   steps={meta['used_T']}/{meta['T']}")

    plt.figure(figsize=(10, 6))
    for i in range(n_players):
        qC = q_traj[i, :, 0]
        qD = q_traj[i, :, 1]

        plt.plot(time, qC, label=f"{agents[i].name} Q(C)", 
                color=colors[2*i % len(colors)])
        plt.plot(time, qD, label=f"{agents[i].name} Q(D)", 
                linestyle='--', color=colors[(2*i+1) % len(colors)])

    title = "Динамика Q-значений"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Шаг симуляции", fontsize=11)
    plt.ylabel("Q-значение", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Сохранение в results текущего эксперимента
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"q_values_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {filename}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Симуляция многоагентной дилеммы заключенного")
    parser.add_argument("--T", type=int, default=2000000, help="Количество шагов")
    parser.add_argument("--alpha", type=float, default=0.01, help="Скорость обучения")
    parser.add_argument("--beta", type=float, default=1, help="Температура Больцмана")
    parser.add_argument("--gamma", type=float, default=0.7, help="Коэффициент дисконтирования")
    parser.add_argument("--n_players", type=int, default=5, help="Количество игроков")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--benefit", type=float, default=6.0, help="Параметр benefit (b)")
    parser.add_argument("--cost", type=float, default=4.0, help="Параметр cost (c)")
    parser.add_argument("--reward_offset", type=float, default=1.0, help="Смещение наград")
    parser.add_argument("--no_plots", action="store_true", help="Не строить графики")
    parser.add_argument("--output_json", type=str, help="Путь для сохранения результатов в JSON")
    
    args = parser.parse_args()
    
    agents, p_traj, q_traj, mean_reward, meta = run_sim(
        T=args.T,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        n_players=args.n_players,
        seed=args.seed,
        max_keep=min(args.T, 300000),
        use_tqdm=not args.no_plots
    )
    
    # Вычисление метрик
    last_20_percent = int(0.8 * meta['used_T'])
    mean_coop = np.mean(p_traj[:, last_20_percent:])
    
    results = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "n_players": args.n_players,
        "T": args.T,
        "mean_reward": float(mean_reward),
        "mean_cooperation": float(mean_coop),
        "final_cooperation": [float(np.mean(p_traj[i, last_20_percent:])) for i in range(args.n_players)]
    }
    
    # Сохранение результатов в JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Результаты сохранены: {args.output_json}")
    
    # Построение графиков
    if not args.no_plots:
        plot_p_traj_separate(p_traj, agents, meta=meta)
        plot_q_values_separate(q_traj, agents, meta=meta)
    
    # Вывод основных метрик
    print(f"\nРезультаты: β={args.beta:.2f}, γ={args.gamma:.2f}")
    print(f"Средняя кооперация: {mean_coop:.4f}")
    print(f"Средний reward: {mean_reward:.4f}")
