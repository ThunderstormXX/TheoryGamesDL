"""Утилиты для анализа метрик симуляции."""

import numpy as np


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
