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


# -----------------------------
# Анализ сходимости
# -----------------------------

def check_convergence(policies: List[np.ndarray],
                      window: int = 1000,
                      threshold_converged: float = 1e-3,
                      actions: np.ndarray | None = None) -> dict:
    """Проверяет, сошлась ли политика и куда.

    Args:
        policies: список массивов политик по времени (T, A)
        window: окно для оценки стабильности (последние `window` шагов)
        threshold_converged: порог вариации для признания сходимости (std по действиям)
        actions: массив значений действий (для отображения в результатах)

    Returns:
        dict с ключами:
            - 'converged': bool - сошлось ли (True) или нет (False)
            - 'argmax_idx': int - индекс действия с максимальной вероятностью
            - 'argmax_value': float - значение действия (из actions) или индекс
            - 'max_prob': float - максимальная вероятность в финальной политике
            - 'mean_policy': ndarray - средняя политика в окне
            - 'max_std': float - максимальное std (мера нестабильности)
    """
    if len(policies) < window:
        window = len(policies)

    # Берём последнее окно политик
    tail = np.array(policies[-window:], dtype=float)  # (window, A)
    
    # Средняя политика в хвосте
    mean_policy = tail.mean(axis=0)  # (A,)
    
    # Стандартное отклонение по времени для каждого действия
    std_over_time = tail.std(axis=0)  # (A,)
    
    # Максимальная вариация (признак нестабильности)
    max_std = float(std_over_time.max())
    
    # Аргмакс конечной политики
    argmax_idx = int(mean_policy.argmax())
    max_prob = float(mean_policy[argmax_idx])
    
    # Определяем сходимость
    converged = max_std <= threshold_converged
    
    # Значение действия
    if actions is not None and len(actions) > argmax_idx:
        argmax_value = float(actions[argmax_idx])
    else:
        argmax_value = float(argmax_idx)
    
    return {
        'converged': converged,
        'argmax_idx': argmax_idx,
        'argmax_value': argmax_value,
        'max_prob': max_prob,
        'mean_policy': mean_policy,
        'max_std': max_std
    }


def plot_convergence_heatmap(results_grid: dict | List[dict],
                             beta_values: List[float] | None = None,
                             gamma_values: List[float] | None = None,
                             title: str = 'Карта сходимости: Beta vs Gamma') -> None:
    """Строит heat-map сходимости по сетке beta × gamma.

    Args:
        results_grid: либо словарь вида {(beta, gamma): status}, 
                      либо список dict с ключами 'beta', 'gamma', 'status'
        beta_values: список значений beta (опционально, если results_grid — список)
        gamma_values: список значений gamma (опционально)
        title: заголовок графика

    Цвета:
        - Зелёный (2): converged_to_zero
        - Синий (1): converged_elsewhere
        - Красный (0): not_converged
    """
    # Преобразуем к единому виду: словарь (beta, gamma) -> status
    if isinstance(results_grid, list):
        grid_dict = {(r['beta'], r['gamma']): r['status'] for r in results_grid}
    else:
        grid_dict = results_grid

    # Определяем все уникальные beta/gamma
    if beta_values is None:
        beta_values = sorted(set(k[0] for k in grid_dict.keys()))
    if gamma_values is None:
        gamma_values = sorted(set(k[1] for k in grid_dict.keys()))

    # Создаём матрицу для отображения
    n_beta = len(beta_values)
    n_gamma = len(gamma_values)
    matrix = np.full((n_beta, n_gamma), np.nan, dtype=float)

    status_to_code = {
        'not_converged': 0,
        'converged_elsewhere': 1,
        'converged_to_zero': 2
    }

    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            status = grid_dict.get((beta, gamma), 'not_converged')
            matrix[i, j] = status_to_code.get(status, 0)

    # Цветовая схема
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#d62728', '#1f77b4', '#2ca02c'])  # красный, синий, зелёный
    bounds = [-0.5, 0.5, 1.5, 2.5]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm)

    # Подписи осей
    ax.set_xticks(np.arange(n_gamma))
    ax.set_yticks(np.arange(n_beta))
    ax.set_xticklabels([f"{g:.2f}" for g in gamma_values])
    ax.set_yticklabels([f"{b:.2f}" for b in beta_values])
    ax.set_xlabel('Gamma (discount factor)', fontsize=12)
    ax.set_ylabel('Beta (inverse temperature)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Colorbar с подписями
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('Статус сходимости', fontsize=12)
    cbar.ax.set_yticklabels(['Не сошлось', 'Сошлось куда-то', 'Сошлось к 0'])

    # Сетка
    ax.set_xticks(np.arange(n_gamma) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_beta) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    plt.tight_layout()
    plt.show()


def plot_joint_convergence_heatmap(results: List[dict],
                                   beta_values: List[float],
                                   gamma_values: List[float],
                                   actions: np.ndarray | None = None,
                                   title: str = 'Совместная сходимость агентов A и B',
                                   save_path: str | None = None) -> None:
    """Строит heat-map показывающий, куда сходятся политики обоих агентов.
    
    Каждая ячейка содержит информацию о сходимости и значениях действий для обоих игроков.
    Цвет показывает статус сходимости, текст - значения действий.
    
    Args:
        results: список словарей с результатами экспериментов
                 (должны содержать 'beta', 'gamma', 'conv_a', 'conv_b')
        beta_values: список значений beta
        gamma_values: список значений gamma  
        actions: массив значений действий (опционально)
        title: заголовок графика
        save_path: путь для сохранения графика (если None, показывает интерактивно)
    """
    n_beta = len(beta_values)
    n_gamma = len(gamma_values)
    
    # Создаём словарь для быстрого доступа
    grid_dict = {(r['beta'], r['gamma']): r for r in results}
    
    # Матрицы для хранения статусов сходимости
    # Кодируем: 0 - оба не сошлись, 1 - один сошёлся, 2 - оба сошлись
    matrix = np.full((n_beta, n_gamma), 0, dtype=float)
    
    # Текстовые аннотации
    annotations = [['' for _ in range(n_gamma)] for _ in range(n_beta)]
    
    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            r = grid_dict.get((beta, gamma))
            if r is None:
                continue
            
            conv_a = r.get('conv_a', {})
            conv_b = r.get('conv_b', {})
            
            converged_a = conv_a.get('converged', False)
            converged_b = conv_b.get('converged', False)
            
            # Статус сходимости
            if converged_a and converged_b:
                matrix[i, j] = 2  # оба сошлись
            elif converged_a or converged_b:
                matrix[i, j] = 1  # один сошёлся
            else:
                matrix[i, j] = 0  # оба не сошлись
            
            # Формируем текстовую аннотацию
            val_a = conv_a.get('argmax_value', conv_a.get('argmax_idx', '?'))
            val_b = conv_b.get('argmax_value', conv_b.get('argmax_idx', '?'))
            
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Если значения близки к 0 или 1, показываем как integer, иначе 2 знака
                def fmt(v):
                    if abs(v - round(v)) < 0.01:
                        return f"{int(round(v))}"
                    else:
                        return f"{v:.2f}"
                
                text = f"A:{fmt(val_a)}\nB:{fmt(val_b)}"
            else:
                text = f"A:{val_a}\nB:{val_b}"
            
            annotations[i][j] = text
    
    # Цветовая схема
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#d62728', '#ffcc00', '#2ca02c'])  # красный, жёлтый, зелёный
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Адаптивный размер графика и шрифта
    cell_width = 1.0 if n_gamma <= 6 else 0.8 if n_gamma <= 10 else 0.6
    cell_height = 0.7 if n_beta <= 7 else 0.5 if n_beta <= 12 else 0.4
    figwidth = max(10, n_gamma * cell_width)
    figheight = max(6, n_beta * cell_height)
    
    # Размер шрифта зависит от размера сетки
    if n_gamma <= 6 and n_beta <= 7:
        fontsize_annot = 9
        fontsize_tick = 11
        fontsize_label = 13
        fontsize_title = 15
    elif n_gamma <= 10 and n_beta <= 10:
        fontsize_annot = 7
        fontsize_tick = 9
        fontsize_label = 11
        fontsize_title = 13
    else:
        fontsize_annot = 5
        fontsize_tick = 7
        fontsize_label = 9
        fontsize_title = 11
    
    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figheight))
    im = ax.imshow(matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm)
    
    # Подписи осей
    ax.set_xticks(np.arange(n_gamma))
    ax.set_yticks(np.arange(n_beta))
    ax.set_xticklabels([f"{g:.2f}" for g in gamma_values], fontsize=fontsize_tick)
    ax.set_yticklabels([f"{b:.2f}" for b in beta_values], fontsize=fontsize_tick)
    ax.set_xlabel('Gamma (discount factor)', fontsize=fontsize_label)
    ax.set_ylabel('Beta (inverse temperature)', fontsize=fontsize_label)
    ax.set_title(title, fontsize=fontsize_title)
    
    # Аннотации в ячейках (только если сетка не слишком большая)
    show_annotations = n_gamma <= 15 and n_beta <= 15
    if show_annotations:
        for i in range(n_beta):
            for j in range(n_gamma):
                text = annotations[i][j]
                # Цвет текста зависит от фона
                color = 'white' if matrix[i, j] == 0 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontsize=fontsize_annot, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('Статус сходимости', fontsize=12)
    cbar.ax.set_yticklabels(['Оба не сошлись', 'Один сошёлся', 'Оба сошлись'], fontsize=10)
    
    # Сетка
    ax.set_xticks(np.arange(n_gamma) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_beta) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 График сохранён: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_action_value_heatmaps(results: List[dict],
                               beta_values: List[float],
                               gamma_values: List[float],
                               actions: np.ndarray | None = None,
                               save_path: str | None = None) -> None:
    """Строит две heat-map: куда сходится агент A и куда сходится агент B.
    
    Показывает значение действия (из actions), к которому сошлась политика.
    
    Args:
        results: список словарей с результатами экспериментов
        beta_values: список значений beta
        gamma_values: список значений gamma
        actions: массив значений действий
        save_path: путь для сохранения графика (если None, показывает интерактивно)
    """
    n_beta = len(beta_values)
    n_gamma = len(gamma_values)
    
    grid_dict = {(r['beta'], r['gamma']): r for r in results}
    
    matrix_a = np.full((n_beta, n_gamma), np.nan, dtype=float)
    matrix_b = np.full((n_beta, n_gamma), np.nan, dtype=float)
    
    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            r = grid_dict.get((beta, gamma))
            if r is None:
                continue
            
            conv_a = r.get('conv_a', {})
            conv_b = r.get('conv_b', {})
            
            if conv_a.get('converged', False):
                matrix_a[i, j] = conv_a.get('argmax_value', conv_a.get('argmax_idx', np.nan))
            
            if conv_b.get('converged', False):
                matrix_b[i, j] = conv_b.get('argmax_value', conv_b.get('argmax_idx', np.nan))
    
    # Адаптивный размер
    cell_width = 1.3 if n_gamma <= 6 else 1.0 if n_gamma <= 10 else 0.7
    cell_height = 0.6 if n_beta <= 7 else 0.45 if n_beta <= 12 else 0.35
    figwidth = max(14, n_gamma * cell_width * 2)
    figheight = max(5, n_beta * cell_height)
    
    # Размер шрифта
    if n_gamma <= 6 and n_beta <= 7:
        fontsize_tick = 10
        fontsize_label = 12
        fontsize_title = 13
    elif n_gamma <= 10 and n_beta <= 10:
        fontsize_tick = 8
        fontsize_label = 10
        fontsize_title = 11
    else:
        fontsize_tick = 6
        fontsize_label = 8
        fontsize_title = 9
    
    fig, axes = plt.subplots(1, 2, figsize=(figwidth, figheight))
    
    # Agent A
    im0 = axes[0].imshow(matrix_a, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_xticks(np.arange(n_gamma))
    axes[0].set_yticks(np.arange(n_beta))
    axes[0].set_xticklabels([f"{g:.2f}" for g in gamma_values], fontsize=fontsize_tick)
    axes[0].set_yticklabels([f"{b:.2f}" for b in beta_values], fontsize=fontsize_tick)
    axes[0].set_xlabel('Gamma', fontsize=fontsize_label)
    axes[0].set_ylabel('Beta', fontsize=fontsize_label)
    axes[0].set_title('Куда сходится Agent A (значение действия)', fontsize=fontsize_title)
    plt.colorbar(im0, ax=axes[0], label='Значение действия')
    
    # Agent B
    im1 = axes[1].imshow(matrix_b, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_xticks(np.arange(n_gamma))
    axes[1].set_yticks(np.arange(n_beta))
    axes[1].set_xticklabels([f"{g:.2f}" for g in gamma_values], fontsize=fontsize_tick)
    axes[1].set_yticklabels([f"{b:.2f}" for b in beta_values], fontsize=fontsize_tick)
    axes[1].set_xlabel('Gamma', fontsize=fontsize_label)
    axes[1].set_ylabel('Beta', fontsize=fontsize_label)
    axes[1].set_title('Куда сходится Agent B (значение действия)', fontsize=fontsize_title)
    plt.colorbar(im1, ax=axes[1], label='Значение действия')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 График сохранён: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_comprehensive_results(rewards_a: List[float], rewards_b: List[float],
                               q_values_a: List[np.ndarray], q_values_b: List[np.ndarray],
                               policies_a: List[np.ndarray], policies_b: List[np.ndarray],
                               action_values: np.ndarray | None = None,
                               window_size: int = 100, save_path: str | None = None,
                               base_function = None,
                               beta: float | None = None,
                               gamma: float | None = None,
                               steps: int | None = None,
                               alpha: float | None = None,
                               n: int | None = None,
                               game_params: dict | None = None) -> None:
    """Комплексная визуализация результатов эксперимента для двух агентов.
    
    Args:
        rewards_a, rewards_b: списки наград во времени
        q_values_a, q_values_b: списки Q-функций (np.ndarray) во времени
        policies_a, policies_b: списки распределений политик во времени
        action_values: значения действий [0, 1] для подписей осей (если None, используются индексы)
        window_size: размер окна для скользящего среднего наград
        save_path: если указан, сохраняет график в файл вместо показа
        base_function: функция base_function(x) для отображения на графике
        beta: параметр inverse temperature
        gamma: параметр discount factor
        steps: количество шагов обучения
        alpha: learning rate
        n: размер дискретизации
        game_params: параметры игры (r, p, q, s)
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
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1, 1, 0.8])
    
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
    im2 = ax2.imshow(Q_a, aspect='auto', origin='lower', cmap='RdYlGn', extent=[0, Q_a.shape[1], 0, 1]) # type: ignore
    ax2.set_title('Q-функция Agent A')
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Действие (непрерывное)')
    ax2.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax2.set_yticklabels(tick_labels)
    plt.colorbar(im2, ax=ax2, label='Q-value')
    
    # 3. Q-функции Agent B
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(Q_b, aspect='auto', origin='lower', cmap='RdYlGn', extent=[0, Q_b.shape[1], 0, 1]) # type: ignore
    ax3.set_title('Q-функция Agent B')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Действие (непрерывное)')
    ax3.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax3.set_yticklabels(tick_labels)
    plt.colorbar(im3, ax=ax3, label='Q-value')
    
    # 4. Политики Agent A
    ax4 = fig.add_subplot(gs[2, 0])
    im4 = ax4.imshow(P_a, aspect='auto', origin='lower', cmap='viridis', extent=[0, P_a.shape[1], 0, 1]) # type: ignore
    ax4.set_title('Политика Agent A')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('Действие (непрерывное)')
    ax4.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax4.set_yticklabels(tick_labels)
    plt.colorbar(im4, ax=ax4, label='P(action)')
    
    # 5. Политики Agent B
    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(P_b, aspect='auto', origin='lower', cmap='viridis', extent=[0, P_b.shape[1], 0, 1]) # type: ignore
    ax5.set_title('Политика Agent B')
    ax5.set_xlabel('Время')
    ax5.set_ylabel('Действие (непрерывное)')
    ax5.set_yticks(np.linspace(0, 1, len(tick_indices)))
    ax5.set_yticklabels(tick_labels)
    plt.colorbar(im5, ax=ax5, label='P(action)')
    
    # 6. Динамика Q-значений для ключевых действий - Agent A
    ax6 = fig.add_subplot(gs[3, 0])
    if action_values is not None and len(action_values) > 0:
        key_actions = np.arange(0.0, 1.01, 0.1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(key_actions))) # type: ignore
        
        for i, action in enumerate(key_actions):
            # Находим ближайший индекс к этому действию
            idx = np.argmin(np.abs(action_values - action))
            q_trajectory = Q_a[idx, :]  # Q-значения этого действия во времени
            ax6.plot(time_steps, q_trajectory, color=colors[i], 
                    label=f'{action:.1f}', linewidth=1.5, alpha=0.8)
        
        ax6.set_xlabel('Время (шаги)')
        ax6.set_ylabel('Q-значение')
        ax6.set_title('Динамика Q-значений Agent A для ключевых действий')
        ax6.legend(title='Действие', ncol=3, fontsize=8, loc='best')
        ax6.grid(True, alpha=0.3)
    
    # 7. Динамика Q-значений для ключевых действий - Agent B
    ax7 = fig.add_subplot(gs[3, 1])
    if action_values is not None and len(action_values) > 0:
        key_actions = np.arange(0.0, 1.01, 0.1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(key_actions))) # type: ignore
        
        for i, action in enumerate(key_actions):
            # Находим ближайший индекс к этому действию
            idx = np.argmin(np.abs(action_values - action))
            q_trajectory = Q_b[idx, :]  # Q-значения этого действия во времени
            ax7.plot(time_steps, q_trajectory, color=colors[i], 
                    label=f'{action:.1f}', linewidth=1.5, alpha=0.8)
        
        ax7.set_xlabel('Время (шаги)')
        ax7.set_ylabel('Q-значение')
        ax7.set_title('Динамика Q-значений Agent B для ключевых действий')
        ax7.legend(title='Действие', ncol=3, fontsize=8, loc='best')
        ax7.grid(True, alpha=0.3)
    
    # 8. Начальные распределения политик (гистограммы)
    ax8 = fig.add_subplot(gs[4, 0])
    initial_policy_a = policies_a[0]
    initial_policy_b = policies_b[0]
    
    if action_values is not None:
        x_vals = action_values
        width = (action_values[1] - action_values[0]) * 0.35 if len(action_values) > 1 else 0.015
    else:
        x_vals = np.arange(num_actions)
        width = 0.35
    
    ax8.bar(x_vals - width/2, initial_policy_a, width=width, label='Agent A', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax8.bar(x_vals + width/2, initial_policy_b, width=width, label='Agent B', 
            color='salmon', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax8.set_xlabel('Действие')
    ax8.set_ylabel('Вероятность')
    ax8.set_title('Начальные распределения политик')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    if action_values is not None:
        ax8.set_xlim(-0.05, 1.05)
    
    # 9. Финальные распределения политик (гистограммы)
    ax9 = fig.add_subplot(gs[4, 1])
    final_policy_a = policies_a[-1]
    final_policy_b = policies_b[-1]
    
    ax9.bar(x_vals - width/2, final_policy_a, width=width, label='Agent A', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax9.bar(x_vals + width/2, final_policy_b, width=width, label='Agent B', 
            color='salmon', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Добавляем base_function если предоставлена
    if base_function is not None and action_values is not None:
        base_values = np.array([base_function(a) for a in action_values])
        # Нормализуем для сравнения с распределением вероятностей
        base_values_norm = base_values / base_values.sum()
        ax9.plot(action_values, base_values_norm, 'k--', linewidth=2, 
                label='Base function (normalized)', alpha=0.7)
    
    ax9.set_xlabel('Действие')
    ax9.set_ylabel('Вероятность')
    ax9.set_title('Финальные распределения политик')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    if action_values is not None:
        ax9.set_xlim(-0.05, 1.05)
    
    # Формируем заголовок с параметрами
    title_parts = ['Комплексный анализ обучения агентов']
    
    # Строка 1: основные параметры обучения
    if beta is not None or gamma is not None or steps is not None:
        params_str = []
        if beta is not None:
            params_str.append(f'β={beta:.3f}')
        if gamma is not None:
            params_str.append(f'γ={gamma:.2f}')
        if steps is not None:
            params_str.append(f'steps={steps:,}')
        title_parts.append(f"({', '.join(params_str)})")
    
    # Строка 2: дополнительные параметры (alpha, n, game_params)
    extra_params = []
    if alpha is not None:
        extra_params.append(f'α={alpha:.3f}')
    if n is not None:
        extra_params.append(f'n={n}')
    if game_params is not None:
        game_str = ', '.join([f'{k}={v}' for k, v in game_params.items()])
        extra_params.append(f'game: [{game_str}]')
    
    if extra_params:
        title_parts.append(f"\n{', '.join(extra_params)}")
    
    plt.suptitle(' '.join(title_parts), fontsize=14, y=0.995)
    
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
    
    # Финальные Q-значения для ключевых точек 0, 0.1, 0.2, ..., 1
    if action_values is not None and len(action_values) > 0:
        print(f"\nФинальные Q-значения для ключевых действий:")
        key_actions = np.arange(0.0, 1.01, 0.1)
        final_q_a = q_values_a[-1]
        final_q_b = q_values_b[-1]
        
        print(f"  {'Действие':>10} | {'Q(A)':>8} | {'Q(B)':>8}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}")
        
        for action in key_actions:
            # Находим ближайший индекс к этому действию
            idx = np.argmin(np.abs(action_values - action))
            actual_action = action_values[idx]
            q_a_val = final_q_a[idx]
            q_b_val = final_q_b[idx]
            print(f"  {actual_action:>10.3f} | {q_a_val:>8.4f} | {q_b_val:>8.4f}")
    
    print("="*60 + "\n")
