import numpy as np
import matplotlib.pyplot as plt
from typing import List


def smooth(arr: List[float], window: int = 50):
    """Скользящее среднее для сглаживания кривых. Возвращает исходный массив, если он короче окна."""
    if len(arr) < window + 1:
        return arr
    return [float(np.mean(arr[i:i + window])) for i in range(len(arr) - window)]


def plot_rewards_and_coop(rewards: List[float], actions: List[int], title_prefix: str = "Агент", window: int = 50) -> None:
    """Строит 2 графика: сглаженные награды и частоту сотрудничества (0=сотрудничать, 1=предать)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # Награды
    axes[0].plot(smooth(rewards, window))
    axes[0].set_title(f"Награды {title_prefix}")
    axes[0].set_xlabel('Игры')
    axes[0].set_ylabel('Награда')
    axes[0].grid(True)

    # Частота сотрудничества = 1 - среднее по окну действий
    if len(actions) >= window + 1:
        coop_rate = [1 - float(np.mean(actions[i:i + window])) for i in range(len(actions) - window)]
    else:
        w = max(1, min(window, len(actions)))
        coop_rate = [1 - float(np.mean(actions[max(0, i - w + 1):i + 1])) for i in range(len(actions))]

    axes[1].plot(coop_rate)
    axes[1].set_title('Частота сотрудничества')
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Игры')
    axes[1].set_ylabel('Доля сотрудничества (0-1)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
