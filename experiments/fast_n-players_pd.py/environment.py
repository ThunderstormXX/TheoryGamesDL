# environment.py
import numpy as np
from typing import Callable, Sequence, Optional, Tuple, List

class Game:
    """
    Game with n players, binary actions {0,1}.

    Design:
      - If payoff_tensor is provided, use it.
      - If payoff_func is provided, DO NOT build full tensor by default (avoids 2**n memory).
      - For known symmetric games (Generalized PD) we provide a fast O(n) payoff computation path.
      - To force tensor construction (if you really need it), pass build_tensor=True.
    """
    def __init__(self, n_players: int,
                 payoff_func: Optional[Callable[[Sequence[int]], float]] = None,
                 payoff_tensor: Optional[np.ndarray] = None,
                 name: str = "Untitled Game",
                 build_tensor: bool = False):
        self.n_players = int(n_players)
        self.name = name
        self.shape = (2,) * self.n_players

        self.payoff_tensor: Optional[np.ndarray] = None
        self.payoff_func: Optional[Callable[[Sequence[int]], float]] = None
        self._fast_pd_params: Optional[dict] = None

        if payoff_tensor is not None:
            self.payoff_tensor = np.asarray(payoff_tensor, dtype=float)
        elif payoff_func is not None:
            self.payoff_func = payoff_func
            if build_tensor:
                self.payoff_tensor = self._create_payoff_tensor(payoff_func)
        else:
            raise ValueError("Either payoff_func or payoff_tensor must be provided")

    def enable_fast_public_goods_pd(self, *, benefit: float, cost: float, reward_offset: float = 0.0):
        self._fast_pd_params = {
            "b": float(benefit),
            "c": float(cost),
            "n": self.n_players,
            "offset": float(reward_offset),
        }

    def _create_payoff_tensor(self, payoff_func: Callable[[Sequence[int]], float]) -> np.ndarray:
        # builds full tensor (exponential) — only on explicit request
        tensor = np.zeros(self.shape, dtype=float)
        for indices in np.ndindex(self.shape):
            tensor[indices] = payoff_func(indices)
        return tensor

    def get_payoff(self, choices: Sequence[int]) -> float:
        if self.payoff_tensor is not None:
            return float(self.payoff_tensor[tuple(choices)])
        if self.payoff_func is not None:
            return float(self.payoff_func(tuple(choices)))
        raise RuntimeError("No payoff available")

    def get_payoffs(self, choices: Sequence[int]) -> Tuple[float, ...]:
        if len(choices) != self.n_players:
            raise ValueError("Length of choices must be n_players")

        # 1) if tensor present -> index
        if self.payoff_tensor is not None:
            choices_np = np.asarray(choices, dtype=int)
            payoffs = []
            for i in range(self.n_players):
                rolled = tuple(choices_np[(i + j) % self.n_players] for j in range(self.n_players))
                payoffs.append(float(self.payoff_tensor[rolled]))
            return tuple(payoffs)

        # 2) fast public goods PD path
        if self._fast_pd_params is not None:
            params = self._fast_pd_params
            b, c, n, offset = params["b"], params["c"], params["n"], params["offset"]
            k = sum(1 for choice in choices if choice == 0)  # total cooperators
            payoffs = []
            for action in choices:
                if action == 0:  # Cooperate
                    payoff = (b * k) / n - c + offset
                else:  # Defect
                    payoff = (b * k) / n + offset
                payoffs.append(float(payoff))
            return tuple(payoffs)

        # 3) fallback: call payoff_func rotated for each player (cost O(n^2) for large n; fallback only)
        if self.payoff_func is not None:
            arr = list(choices)
            n = self.n_players
            payoffs = []
            for i in range(n):
                rotated = tuple(arr[(i + j) % n] for j in range(n))
                payoffs.append(float(self.payoff_func(rotated)))
            return tuple(payoffs)

        raise RuntimeError("No way to compute payoffs")

    def analyze(self) -> None:
        print(f"=== Анализ игры '{self.name}' ===")
        print(f"Количество игроков: {self.n_players}")
        if self.payoff_tensor is not None:
            print(f"Форма тензора: {self.payoff_tensor.shape}, итого {self.payoff_tensor.size} исходов")
            print(f"Средний/макс/мин: {np.mean(self.payoff_tensor):.3f}/{np.max(self.payoff_tensor):.3f}/{np.min(self.payoff_tensor):.3f}")
        else:
            print("Выплаты вычисляются на лету (tensor not built).")

        print("\n--- Анализ по игрокам (оценка) ---")
        for i in range(self.n_players):
            if self.payoff_tensor is not None:
                axis_to_keep = [j for j in range(self.n_players) if j != i]
                if axis_to_keep:
                    payoff_when_0 = np.mean(self.payoff_tensor, axis=tuple(axis_to_keep))[0]
                    payoff_when_1 = np.mean(self.payoff_tensor, axis=tuple(axis_to_keep))[1]
                else:
                    payoff_when_0 = self.payoff_tensor[0]
                    payoff_when_1 = self.payoff_tensor[1]
            else:
                # don't attempt exhaustive enumeration for large n
                if self.n_players > 20:
                    payoff_when_0 = float('nan'); payoff_when_1 = float('nan')
                else:
                    vals0=[]; vals1=[]
                    for idx in np.ndindex(self.shape):
                        if idx[i] == 0:
                            vals0.append(self.get_payoff(idx))
                        else:
                            vals1.append(self.get_payoff(idx))
                    payoff_when_0 = float(np.mean(vals0)) if vals0 else float('nan')
                    payoff_when_1 = float(np.mean(vals1)) if vals1 else float('nan')
            print(f"Игрок {i}: mean when 0 = {payoff_when_0:.3f}, when 1 = {payoff_when_1:.3f}")

    def find_pure_nash_equilibria(self) -> List[Tuple[int, ...]]:
        # Warning: exponential in n. Keep for small n only.
        equilibria = []
        for idx in np.ndindex(self.shape):
            current_payoff = self.get_payoff(idx)
            is_eq = True
            idx_list = list(idx)
            for player in range(self.n_players):
                alt = idx_list.copy()
                alt[player] = 1 - alt[player]
                alt_payoff = self.get_payoff(tuple(alt))
                if alt_payoff > current_payoff:
                    is_eq = False
                    break
            if is_eq:
                equilibria.append(tuple(idx))
        return equilibria

    def __str__(self) -> str:
        return f"Game '{self.name}' ({self.n_players} players, tensor_built={self.payoff_tensor is not None})"


class GameFactory:
    @staticmethod
    def create_generalized_prisoners_dilemma(
        n_players: int,
        benefit: float = 3.0,
        cost: float = 1.0,
        reward_offset: float = 0.0,
        build_tensor: bool = False
    ) -> Game:
        # Validate PD constraints: b/n < c < b
        if not (benefit / n_players < cost < benefit):
            raise ValueError(f"Invalid PD parameters: need b/n < c < b, got b={benefit}, c={cost}, n={n_players}")
        
        def public_goods_pd_payoff(choices: Sequence[int]) -> float:
            k = sum(1 for choice in choices if choice == 0)  # total cooperators
            player0_action = choices[0]
            if player0_action == 0:  # Cooperate
                return (benefit * k) / n_players - cost + reward_offset
            else:  # Defect
                return (benefit * k) / n_players + reward_offset

        g = Game(n_players, payoff_func=public_goods_pd_payoff, name=f"Public Goods PD ({n_players})", build_tensor=build_tensor)
        g.enable_fast_public_goods_pd(benefit=benefit, cost=cost, reward_offset=reward_offset)
        return g
