"""Визуализация временных рядов вероятностей кооперации и Q-значений."""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from simulation.metrics import smooth


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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"q_values_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {filename}")
