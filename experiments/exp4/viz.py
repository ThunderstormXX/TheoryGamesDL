import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_policy_heatmap_over_time(policies: List[np.ndarray], title: str = "Policy evolution", cmap: str = "viridis"):
    """Строит теплокарту эволюции распределения действий во времени.

    policies: список массивов вероятностей по действиям, длиной T.
              Каждый элемент shape (A,), где A — число дискретных действий.
    На графике: ось X — время (шаг), ось Y — индекс действия, цвет — вероятность.
    """
    if len(policies) == 0:
        raise ValueError("Empty policies list")
    P = np.stack(policies, axis=0)  # (T, A)
    P = P.T  # (A, T) для теплокарты: строки — действия, столбцы — время

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(P, aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel('Время (шаги)')
    ax.set_ylabel('Индекс действия')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='P(action)')
    plt.tight_layout()
    plt.show()


def plot_two_policies_heatmaps_over_time(policies_a: List[np.ndarray], policies_b: List[np.ndarray],
                                         title_a: str = "Agent A", title_b: str = "Agent B"):
    if len(policies_a) == 0 or len(policies_b) == 0:
        raise ValueError("Empty policies list")
    A = np.stack(policies_a, axis=0).T
    B = np.stack(policies_b, axis=0).T
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    im0 = axes[0].imshow(A, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(title_a); axes[0].set_xlabel('Время'); axes[0].set_ylabel('Действие')
    im1 = axes[1].imshow(B, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(title_b); axes[1].set_xlabel('Время'); axes[1].set_ylabel('Действие')
    fig.colorbar(im0, ax=axes[0], label='P(action)')
    fig.colorbar(im1, ax=axes[1], label='P(action)')
    plt.tight_layout()
    plt.show()


def plot_policy_heatmap(policy: np.ndarray, actions: np.ndarray | None = None,
                        title: str = "Policy heatmap", cmap: str = "viridis") -> None:
    """Теплокарта политики.

    Поддерживает:
    - 1D вектор вероятностей формы (A,) — рисуется как одна строка (1 x A),
      ось X — индекс/значение действия.
    - 2D матрицу формы (A, B) — рисуется как полноценная теплокарта (например, совместная политика).

    Args:
        policy: np.ndarray, (A,) или (A, B)
        actions: массив значений действий (A,) для подписей оси X (и Y при 2D)
        title: заголовок графика
        cmap: colormap
    """
    arr = np.array(policy, dtype=float)
    if arr.ndim == 1:
        data = arr.reshape(1, -1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
        im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap)
        ax.set_yticks([0])
        ax.set_yticklabels(["prob"])
        ax.set_xlabel('Действие')
        if actions is not None and len(actions) == data.shape[1]:
            ax.set_xticks(np.arange(len(actions)))
            ax.set_xticklabels([f"{x:.2f}" for x in actions], rotation=45)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='P(action)')
        plt.tight_layout()
        plt.show()
    elif arr.ndim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap)
        ax.set_xlabel('Действие (ось X)')
        ax.set_ylabel('Действие (ось Y)')
        if actions is not None and len(actions) == arr.shape[1]:
            ax.set_xticks(np.arange(len(actions)))
            ax.set_xticklabels([f"{x:.2f}" for x in actions], rotation=45)
        if actions is not None and len(actions) == arr.shape[0]:
            ax.set_yticks(np.arange(len(actions)))
            ax.set_yticklabels([f"{x:.2f}" for x in actions])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='value')
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("policy must be 1D or 2D ndarray")


def plot_joint_policy_heatmap(policy_a: np.ndarray, policy_b: np.ndarray, actions: np.ndarray | None = None,
                              title: str = "Joint policy heatmap", cmap: str = "viridis") -> None:
    """Теплокарта совместной политики двух агентов как внешнее произведение.

    Args:
        policy_a: (A,) — распределение по действиям агента A (ось Y)
        policy_b: (B,) — распределение по действиям агента B (ось X)
        actions: (A,) (и/или (B,)) — значения дискретных действий для подписей
        title: заголовок
        cmap: colormap
    """
    pa = np.array(policy_a, dtype=float)
    pb = np.array(policy_b, dtype=float)
    assert pa.ndim == 1 and pb.ndim == 1, "policy_a and policy_b must be 1D"
    M = np.outer(pa, pb)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(M, aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel('Agent B action')
    ax.set_ylabel('Agent A action')
    if actions is not None and len(actions) == M.shape[1]:
        ax.set_xticks(np.arange(len(actions)))
        ax.set_xticklabels([f"{x:.2f}" for x in actions], rotation=45)
    if actions is not None and len(actions) == M.shape[0]:
        ax.set_yticks(np.arange(len(actions)))
        ax.set_yticklabels([f"{x:.2f}" for x in actions])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='P(a)*P(b)')
    plt.tight_layout()
    plt.show()
