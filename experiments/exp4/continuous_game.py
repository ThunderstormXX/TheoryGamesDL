from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PayoffParams:
    r: float  # коэффициент для a*b
    p: float  # коэффициент для (1-a)*b
    q: float  # коэффициент для a*(1-b)
    s: float  # коэффициент для (1-a)*(1-b)

    def is_prisoners_dilemma(self) -> bool:
        """Грубая проверка PD-подобной структуры: T > R > P > S.
        Тут сопоставим: T ~ q (выгодно, когда ты a=1, другой b=0),
        R ~ r (оба сотрудничают a=b=1), P ~ s (оба не сотрудничают a=b=0),
        S ~ p (ты a=0, другой b=1)."""
        T, R, P, S = self.q, self.r, self.s, self.p
        return T > R > P > S


class ContinuousBimatrixGame:
    """Двухигроковая игра с действиями a,b ∈ [0,1], дискретизированными на n+1 точек.

    Вознаграждение игрока 0: r*a*b + p*(1-a)*b + q*a*(1-b) + s*(1-a)*(1-b).
    Для симметрии игрока 1 считаем такую же форму, но с перестановкой ролей (a<->b):
    r*a*b + p*(1-b)*a + q*b*(1-a) + s*(1-b)*(1-a).
    """

    def __init__(self, params: PayoffParams, n: int):
        assert n >= 1
        self.params = params
        self.n = n
        # сетка действий: 0, 1/n, 2/n, ..., 1
        self.actions: np.ndarray = np.linspace(0.0, 1.0, n + 1)

    def payoff_player0(self, a_idx: int, b_idx: int) -> float:
        a = self.actions[a_idx]
        b = self.actions[b_idx]
        r, p, q, s = self.params.r, self.params.p, self.params.q, self.params.s
        return float(r * a * b + p * (1 - a) * b + q * a * (1 - b) + s * (1 - a) * (1 - b))

    def payoff_player1(self, a_idx: int, b_idx: int) -> float:
        # симметрично, меняем роли a и b
        a = self.actions[a_idx]
        b = self.actions[b_idx]
        r, p, q, s = self.params.r, self.params.p, self.params.q, self.params.s
        return float(r * a * b + p * (1 - b) * a + q * b * (1 - a) + s * (1 - b) * (1 - a))

    def num_actions(self) -> int:
        return len(self.actions)

    def grid(self) -> np.ndarray:
        return self.actions.copy()
