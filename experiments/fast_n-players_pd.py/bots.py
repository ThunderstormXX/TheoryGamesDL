# bots.py
import numpy as np
from typing import List, Optional
from numpy.random import Generator

class BoltzmannAgent:
    """
    Boltzmann (softmax) Q-learning agent.

    Action space: {0 = Cooperate, 1 = Defect}.
    - record_history: отключено по умолчанию (экономия памяти).
    - rng: ожидается np.random.Generator; если не передан, создаётся.
    """
    __slots__ = ("name","alpha","beta","gamma","q_values","history","p_history","_rng","record_history")

    def __init__(self,
                 name: str = "BoltzmannAgent",
                 alpha: float = 0.01,
                 beta: float = 1.0,
                 gamma: float = 0.9,
                 init_q: Optional[List[float]] = None,
                 rng: Optional[Generator] = None,
                 seed: Optional[int] = None,
                 record_history: bool = False):
        self.name = name
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.q_values = np.array([0.0, 0.0], dtype=float) if init_q is None else np.array(init_q, dtype=float)
        self.record_history = bool(record_history)
        self.history: List[int] = [] if self.record_history else None
        self.p_history: List[float] = [] if self.record_history else None
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def policy_probs(self) -> np.ndarray:
        z = self.beta * (self.q_values - np.max(self.q_values))
        ex = np.exp(z)
        probs = ex / np.sum(ex)
        return probs

    def choose_action(self) -> int:
        probs = self.policy_probs()
        pC = probs[0]
        a = 0 if self._rng.random() < pC else 1
        if self.record_history:
            self.history.append(int(a))
            self.p_history.append(float(pC))
        return int(a)

    def learn(self, action: int, reward: float) -> None:
        best_future = float(np.max(self.q_values))
        target = float(reward) + self.gamma * best_future
        td = target - float(self.q_values[action])
        self.q_values[action] += self.alpha * td

    def current_p_cooperate(self) -> float:
        return float(self.policy_probs()[0])

    def reset(self, q_init: Optional[List[float]] = None):
        self.q_values = np.array([0.0, 0.0], dtype=float) if q_init is None else np.array(q_init, dtype=float)
        if self.record_history:
            self.history.clear()
            self.p_history.clear()

    def get_q(self) -> np.ndarray:
        return self.q_values.copy()

    def print_stats(self) -> None:
        q0, q1 = self.q_values
        if self.record_history and self.history:
            coop_rate = (sum(1 for a in self.history if a == 0) / len(self.history))
        else:
            coop_rate = 0.0
        print(f"Agent {self.name} | alpha={self.alpha} beta={self.beta} gamma={self.gamma}")
        print(f"  Q: C={q0:.3f}, D={q1:.3f} | empirical coop={coop_rate:.3f}")
