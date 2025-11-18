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


# -----------------------------
# PMF / CDF визуализации
# -----------------------------

def _default_width_from_actions(actions: np.ndarray) -> float:
    actions = np.array(actions, dtype=float)
    if actions.ndim != 1 or len(actions) == 0:
        return 0.4
    if len(actions) == 1:
        return 0.4
    # предполагаем равномерную сетку
    return float((actions[1] - actions[0]) * 0.8)


def plot_two_policies_bar(policy_a: np.ndarray, policy_b: np.ndarray,
                          actions: np.ndarray | None = None,
                          title_a: str = "Гистограмма политики — Agent A",
                          title_b: str = "Гистограмма политики — Agent B",
                          color_a: str = 'steelblue', color_b: str = 'salmon') -> None:
    """Столбчатые диаграммы для двух политик (только теоретические PMF).

    Если задан actions (значения на [0,1]), столбики ставятся по этим X; иначе по индексам.
    """
    pa = np.array(policy_a, dtype=float)
    pb = np.array(policy_b, dtype=float)
    A = len(pa)
    assert pb.shape == pa.shape, "policy_a and policy_b must have same shape"

    if actions is None:
        xs = np.arange(A)
        width = 0.8
        xlim = (xs.min() - 0.5, xs.max() + 0.5)
        xticks = xs
        xticklabels = [str(i) for i in xs]
    else:
        xs = np.array(actions, dtype=float)
        width = _default_width_from_actions(xs)
        pad = (xs.max() - xs.min()) * 0.05 if A > 1 else 0.05
        xlim = (xs.min() - pad, xs.max() + pad)
        xticks = xs
        xticklabels = [f"{x:.2f}" for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    axes[0].bar(xs, pa, width=width, color=color_a, edgecolor='black')
    axes[0].set_title(title_a)
    axes[0].set_xlabel('Действие')
    axes[0].set_ylabel('Вероятность')
    axes[0].set_xlim(xlim)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=45)

    axes[1].bar(xs, pb, width=width, color=color_b, edgecolor='black')
    axes[1].set_title(title_b)
    axes[1].set_xlabel('Действие')
    axes[1].set_xlim(xlim)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, rotation=45)

    plt.tight_layout()
    plt.show()


def plot_two_policies_pmf_vs_empirical(policy_a: np.ndarray, policy_b: np.ndarray,
                                       actions: np.ndarray | None = None, N: int = 20000,
                                       rng: np.random.Generator | None = None,
                                       title_a: str = 'Распределение действия — Agent A',
                                       title_b: str = 'Распределение действия — Agent B',
                                       color_emp_a: str = 'steelblue', color_emp_b: str = 'salmon',
                                       color_the: str = 'orange') -> None:
    """Сравнение эмпирической PMF (по сэмплам) и теоретической PMF (политика) для двух агентов.

    Bars = эмпирические частоты, Line = теоретические вероятности политики.
    Если actions задан, ось X — значения действий; иначе индексы.
    """
    pa = np.array(policy_a, dtype=float)
    pb = np.array(policy_b, dtype=float)
    A = len(pa)
    assert pb.shape == pa.shape, "policy_a and policy_b must have same shape"

    if rng is None:
        rng = np.random.default_rng(0)

    idx_a = rng.choice(A, size=int(N), p=pa)
    idx_b = rng.choice(A, size=int(N), p=pb)
    emp_a = np.bincount(idx_a, minlength=A) / float(N)
    emp_b = np.bincount(idx_b, minlength=A) / float(N)

    if actions is None:
        xs = np.arange(A)
        width = 0.8
        xlim = (xs.min() - 0.5, xs.max() + 0.5)
        xticks = xs
        xticklabels = [str(i) for i in xs]
    else:
        xs = np.array(actions, dtype=float)
        width = _default_width_from_actions(xs)
        pad = (xs.max() - xs.min()) * 0.05 if A > 1 else 0.05
        xlim = (xs.min() - pad, xs.max() + pad)
        xticks = xs
        xticklabels = [f"{x:.2f}" for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    axes[0].bar(xs, emp_a, width=width, color=color_emp_a, edgecolor='black', label='эмпирика')
    axes[0].plot(xs, pa, 'o-', color=color_the, label='теория (политика)')
    axes[0].set_title(title_a)
    axes[0].set_xlabel('Действие')
    axes[0].set_ylabel('Вероятность')
    axes[0].set_xlim(xlim)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=45)
    axes[0].legend()

    axes[1].bar(xs, emp_b, width=width, color=color_emp_b, edgecolor='black', label='эмпирика')
    axes[1].plot(xs, pb, 'o-', color=color_the, label='теория (политика)')
    axes[1].set_title(title_b)
    axes[1].set_xlabel('Действие')
    axes[1].set_xlim(xlim)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_policy_cdf(policy: np.ndarray, actions: np.ndarray | None = None,
                    N: int | None = None, rng: np.random.Generator | None = None,
                    ax: plt.Axes | None = None, title: str = 'Функция распределения F(x)',
                    color_emp: str = 'steelblue', color_the: str = 'orange') -> plt.Axes:
    """Строит CDF политики. Если задан N, добавляет эмпирическую CDF по сэмплам.

    Возвращает Axes с нарисованным графиком.
    """
    p = np.array(policy, dtype=float)
    A = len(p)
    if actions is None:
        xs = np.arange(A)
    else:
        xs = np.array(actions, dtype=float)

    cdf_the = np.cumsum(p)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    if N is not None and N > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(A, size=int(N), p=p)
        emp = np.bincount(idx, minlength=A) / float(N)
        cdf_emp = np.cumsum(emp)
        ax.step(xs, cdf_emp, where='post', color=color_emp, label='эмпирическая CDF')

    ax.step(xs, cdf_the, where='post', color=color_the, linestyle='--', label='теоретическая CDF')
    ax.set_title(title)
    ax.set_xlabel('Действие')
    ax.set_ylabel('F(x)')
    # Автопределение границ по actions/индексам
    if actions is not None:
        pad = (xs.max() - xs.min()) * 0.05 if A > 1 else 0.5
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
    else:
        ax.set_xlim(-0.5, A - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_two_policies_cdf(policy_a: np.ndarray, policy_b: np.ndarray,
                          actions: np.ndarray | None = None,
                          N: int | None = None, rng: np.random.Generator | None = None,
                          title_a: str = 'Функция распределения F_A(x)',
                          title_b: str = 'Функция распределения F_B(x)') -> None:
    """Строит CDF для двух политик на соседних подграфиках. При N добавляет эмпирическую CDF.
    """
    pa = np.array(policy_a, dtype=float)
    pb = np.array(policy_b, dtype=float)
    assert pa.shape == pb.shape, "policy_a and policy_b must have same shape"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plot_policy_cdf(pa, actions=actions, N=N, rng=rng, ax=axes[0], title=title_a)
    plot_policy_cdf(pb, actions=actions, N=N, rng=rng, ax=axes[1], title=title_b)
    plt.tight_layout()
    plt.show()


def plot_comprehensive_results(rewards_a: List[float], rewards_b: List[float],
                               q_values_a: List[np.ndarray], q_values_b: List[np.ndarray],
                               policies_a: List[np.ndarray], policies_b: List[np.ndarray],
                               action_values: np.ndarray | None = None,
                               window_size: int = 100, save_path: str | None = None) -> None:
    """Комплексная визуализация результатов эксперимента для двух агентов.
    
    Args:
        rewards_a, rewards_b: списки наград во времени
        q_values_a, q_values_b: списки Q-функций (np.ndarray) во времени
        policies_a, policies_b: списки распределений политик во времени
        action_values: значения действий [0, 1] для подписей осей (если None, используются индексы)
        window_size: размер окна для скользящего среднего наград
        save_path: если указан, сохраняет график в файл вместо показа
    """
    if len(rewards_a) == 0:
        raise ValueError("Empty rewards list")
    
    # Скользящее среднее для наград
    def moving_average(data, window):
        if len(data) < window:
            window = len(data)
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    rewards_a_smooth = moving_average(rewards_a, window_size)
    rewards_b_smooth = moving_average(rewards_b, window_size)
    
    # Преобразуем списки в массивы для теплокарт
    Q_a = np.stack(q_values_a, axis=0).T  # (A, T)
    Q_b = np.stack(q_values_b, axis=0).T  # (A, T)
    P_a = np.stack(policies_a, axis=0).T  # (A, T)
    P_b = np.stack(policies_b, axis=0).T  # (A, T)
    
    # Создаем большую фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1, 0.8])
    
    # 1. Награды во времени
    ax1 = fig.add_subplot(gs[0, :])
    time_steps = np.arange(len(rewards_a))
    ax1.plot(time_steps, rewards_a, alpha=0.3, color='steelblue', linewidth=0.5)
    ax1.plot(time_steps, rewards_b, alpha=0.3, color='salmon', linewidth=0.5)
    if len(rewards_a_smooth) > 0:
        ax1.plot(time_steps[window_size-1:], rewards_a_smooth, 
                label=f'Agent A (MA-{window_size})', color='darkblue', linewidth=2)
        ax1.plot(time_steps[window_size-1:], rewards_b_smooth, 
                label=f'Agent B (MA-{window_size})', color='darkred', linewidth=2)
    ax1.set_xlabel('Время (шаги)')
    ax1.set_ylabel('Награда')
    ax1.set_title('Эволюция наград')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Настройка меток для оси Y (действия от 0 до 1)
    num_actions = Q_a.shape[0]
    if action_values is not None and len(action_values) == num_actions:
        # Выбираем ~10 меток для читаемости
        num_ticks = min(10, num_actions)
        tick_indices = np.linspace(0, num_actions - 1, num_ticks, dtype=int)
        tick_labels = [f"{action_values[i]:.2f}" for i in tick_indices]
    else:
        tick_indices = np.arange(0, num_actions, max(1, num_actions // 10))
        tick_labels = [str(i) for i in tick_indices]
    
    # 2. Q-функции Agent A
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(Q_a, aspect='auto', origin='lower', cmap='RdYlGn', extent=[0, Q_a.shape[1], 0, 1])
    ax2.set_title('Q-функция Agent A')
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Действие (непрерывное)')
    ax2.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax2.set_yticklabels(tick_labels)
    plt.colorbar(im2, ax=ax2, label='Q-value')
    
    # 3. Q-функции Agent B
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(Q_b, aspect='auto', origin='lower', cmap='RdYlGn', extent=[0, Q_b.shape[1], 0, 1])
    ax3.set_title('Q-функция Agent B')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Действие (непрерывное)')
    ax3.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax3.set_yticklabels(tick_labels)
    plt.colorbar(im3, ax=ax3, label='Q-value')
    
    # 4. Политики Agent A
    ax4 = fig.add_subplot(gs[2, 0])
    im4 = ax4.imshow(P_a, aspect='auto', origin='lower', cmap='viridis', extent=[0, P_a.shape[1], 0, 1])
    ax4.set_title('Политика Agent A')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('Действие (непрерывное)')
    ax4.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax4.set_yticklabels(tick_labels)
    plt.colorbar(im4, ax=ax4, label='P(action)')
    
    # 5. Политики Agent B
    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(P_b, aspect='auto', origin='lower', cmap='viridis', extent=[0, P_b.shape[1], 0, 1])
    ax5.set_title('Политика Agent B')
    ax5.set_xlabel('Время')
    ax5.set_ylabel('Действие (непрерывное)')
    ax5.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax5.set_yticklabels(tick_labels)
    plt.colorbar(im5, ax=ax5, label='P(action)')
    
    # 6. Начальные распределения политик (гистограммы)
    ax6 = fig.add_subplot(gs[3, 0])
    initial_policy_a = policies_a[0]
    initial_policy_b = policies_b[0]
    
    if action_values is not None:
        x_vals = action_values
        width = (action_values[1] - action_values[0]) * 0.35 if len(action_values) > 1 else 0.015
    else:
        x_vals = np.arange(num_actions)
        width = 0.35
    
    ax6.bar(x_vals - width/2, initial_policy_a, width=width, label='Agent A', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.bar(x_vals + width/2, initial_policy_b, width=width, label='Agent B', 
            color='salmon', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_xlabel('Действие')
    ax6.set_ylabel('Вероятность')
    ax6.set_title('Начальные распределения политик')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    if action_values is not None:
        ax6.set_xlim(-0.05, 1.05)
    
    # 7. Финальные распределения политик (гистограммы)
    ax7 = fig.add_subplot(gs[3, 1])
    final_policy_a = policies_a[-1]
    final_policy_b = policies_b[-1]
    
    ax7.bar(x_vals - width/2, final_policy_a, width=width, label='Agent A', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.bar(x_vals + width/2, final_policy_b, width=width, label='Agent B', 
            color='salmon', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.set_xlabel('Действие')
    ax7.set_ylabel('Вероятность')
    ax7.set_title('Финальные распределения политик')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    if action_values is not None:
        ax7.set_xlim(-0.05, 1.05)
    
    plt.suptitle('Комплексный анализ обучения агентов', fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в: {save_path}")
    else:
        plt.show()
    
    # Дополнительная статистика
    print("\n" + "="*60)
    print("СТАТИСТИКА ОБУЧЕНИЯ")
    print("="*60)
    print(f"Средняя награда Agent A: {np.mean(rewards_a):.4f} ± {np.std(rewards_a):.4f}")
    print(f"Средняя награда Agent B: {np.mean(rewards_b):.4f} ± {np.std(rewards_b):.4f}")
    print(f"\nПоследние 100 шагов:")
    print(f"  Agent A: {np.mean(rewards_a[-100:]):.4f}")
    print(f"  Agent B: {np.mean(rewards_b[-100:]):.4f}")
    print(f"\nФинальные Q-значения:")
    print(f"  Agent A: min={Q_a[:,-1].min():.3f}, max={Q_a[:,-1].max():.3f}, mean={Q_a[:,-1].mean():.3f}")
    print(f"  Agent B: min={Q_b[:,-1].min():.3f}, max={Q_b[:,-1].max():.3f}, mean={Q_b[:,-1].mean():.3f}")
    print(f"\nФинальное распределение политик:")
    final_policy_a = policies_a[-1]
    final_policy_b = policies_b[-1]
    print(f"  Agent A: энтропия={-np.sum(final_policy_a * np.log(final_policy_a + 1e-10)):.3f}")
    print(f"  Agent B: энтропия={-np.sum(final_policy_b * np.log(final_policy_b + 1e-10)):.3f}")
    if action_values is not None:
        print(f"  Agent A: макс. вероятность={final_policy_a.max():.3f} (действие {action_values[final_policy_a.argmax()]:.3f})")
        print(f"  Agent B: макс. вероятность={final_policy_b.max():.3f} (действие {action_values[final_policy_b.argmax()]:.3f})")
    else:
        print(f"  Agent A: макс. вероятность={final_policy_a.max():.3f} (индекс {final_policy_a.argmax()})")
        print(f"  Agent B: макс. вероятность={final_policy_b.max():.3f} (индекс {final_policy_b.argmax()})")
    print("="*60 + "\n")
