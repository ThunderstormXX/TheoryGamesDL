"""Комплексная визуализация результатов симуляции."""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from simulation.metrics import smooth


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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"full_results_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {filename}")
