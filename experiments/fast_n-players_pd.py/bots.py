import random
from typing import List, Optional, Callable
import numpy as np

class SimpleBot:
    """Боты с фиксированной стратегией: cooperate | betray | random"""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.history: List[int] = []

    def choose_action(self) -> int:
        if self.strategy == "cooperate":
            action = 0
        elif self.strategy == "betray":
            action = 1
        elif self.strategy == "random":
            action = random.choice([0, 1])
        else:
            action = random.choice([0, 1])

        self.history.append(action)
        return action

class SmartAgent:
    """Умный агент, который учится на своих ошибках с использованием простого обновления Q-значений для двух действий.

    Действия: 0 = сотрудничать, 1 = предать.
    """

    def __init__(self, name: str = "Агент"):
        self.name = name
        # Q-таблица - память о выгодности действий (для 2 действий)
        self.q_values: List[float] = [0.0, 0.0]
        self.learning_rate: float = 0.1
        self.exploration: float = 0.1
        self.history: List[int] = []

    def choose_action(self) -> int:
        """Выбрать действие — иногда исследуем, иногда выбираем лучшее."""
        if random.random() < self.exploration:
            action = random.choice([0, 1])
        else:
            action = 0 if self.q_values[0] > self.q_values[1] else 1

        self.history.append(action)
        return action

    def learn(self, action: int, reward: float) -> None:
        """Обновить Q-значение для выбранного действия по формуле экспоненциального сглаживания."""
        self.q_values[action] = (1 - self.learning_rate) * self.q_values[action] + self.learning_rate * reward

    def print_stats(self) -> None:
        """Показать статистику агента."""
        print(f"\n📊 {self.name}:")
        print(f"   Q-значения: сотрудничать={self.q_values[0]:.2f}, предать={self.q_values[1]:.2f}")
        print(f"   Любимое действие: {'сотрудничать' if self.q_values[0] > self.q_values[1] else 'предать'}")
        if len(self.history) > 0:
            coop_rate = sum(1 for a in self.history if a == 0) / len(self.history) * 100
            print(f"   Частота сотрудничества: {coop_rate:.1f}%")
        else:
            print("   Пока нет истории действий")


class BoltzmannAgent:
    """
    Boltzmann (softmax) Q-learning agent.

    Action space: {0 = Cooperate, 1 = Defect}.

    Параметры:
      - alpha: learning rate
      - beta: inverse temperature (softmax)
      - gamma: discount factor
    """

    def __init__(self,
                 name: str = "BoltzmannAgent",
                 alpha: float = 0.01,
                 beta: float = 1.0,
                 gamma: float = 0.9,
                 init_q: Optional[List[float]] = None,
                 rng: Optional[Callable] = None,
                 seed: Optional[int] = None):
        self.name = name
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.q_values = np.array([0.0, 0.0], dtype=float) if init_q is None else np.array(init_q, dtype=float)
        self.history: List[int] = []
        self.p_history: List[float] = []
        # RNG per-agent — даёт воспроизводимость, если передан seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def policy_probs(self) -> np.ndarray:
        """Возвращает вектор [p(C), p(D)] — softmax от beta * Q."""
        # численно устойчивый softmax
        z = self.beta * (self.q_values - np.max(self.q_values))
        ex = np.exp(z)
        probs = ex / np.sum(ex)
        return probs

    def choose_action(self) -> int:
        """Выбрать действие согласно softmax-распределению."""
        probs = self.policy_probs()
        a = self._rng.choice([0,1], p=probs)
        self.history.append(int(a))
        self.p_history.append(float(probs[0]))
        return int(a)

    def learn(self, action: int, reward: float) -> None:
        """Q-learning обновление (off-policy): Q(a) <- Q(a) + alpha*(r + gamma*max(Q) - Q(a))."""
        best_future = float(np.max(self.q_values))
        target = float(reward) + self.gamma * best_future
        td = target - float(self.q_values[action])
        self.q_values[action] += self.alpha * td

    def current_p_cooperate(self) -> float:
        return float(self.policy_probs()[0])

    def reset(self, q_init: Optional[List[float]] = None):
        self.q_values = np.array([0.0, 0.0], dtype=float) if q_init is None else np.array(q_init, dtype=float)
        self.history.clear()
        self.p_history.clear()

    def get_q(self) -> np.ndarray:
        return self.q_values.copy()

    def print_stats(self) -> None:
        q0, q1 = self.q_values
        coop_rate = (sum(1 for a in self.history if a == 0) / len(self.history)) if self.history else 0.0
        print(f"Agent {self.name} | alpha={self.alpha} beta={self.beta} gamma={self.gamma}")
        print(f"  Q: C={q0:.3f}, D={q1:.3f} | empirical coop={coop_rate:.3f}")