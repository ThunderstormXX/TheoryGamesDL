from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class SoftmaxSARSAAgent:
    """SARSA агент с softmax-политикой для дискретного множества действий.

    - Политика: pi(a|s) = softmax(beta * Q[s, a])
    - Обновление SARSA: Q[s,a] += alpha * (r + gamma * Q[s', a'] - Q[s,a])
    - Здесь состояние одномерное (s=0), фокус на распределении по действиям
    """

    def __init__(self, num_actions: int, alpha: float = 0.1, gamma: float = 0.95, beta: float = 5.0,
                 seed: Optional[int] = None):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((1, num_actions), dtype=float)
        self.state = 0  # единственное состояние
        self.last_action = None  # type: Optional[int]

    def policy(self) -> np.ndarray:
        logits = self.beta * self.Q[self.state]
        # стабилизуем softmax
        m = np.max(logits)
        exps = np.exp(logits - m)
        probs = exps / np.sum(exps)
        return probs

    def choose_action(self) -> int:
        probs = self.policy()
        a = int(self.rng.choice(self.num_actions, p=probs))
        return a

    def step(self, reward: float, next_action: Optional[int] = None) -> Tuple[int, float]:
        """Выполнить SARSA-обновление для текущего действия и перейти к следующему.
        Возвращает (chosen_action, reward)."""
        s = self.state
        assert self.last_action is not None, "Call start_episode() before step()"
        a = self.last_action
        r = reward
        s_next = self.state  # состояние не меняется
        if next_action is None:
            next_action = self.choose_action()
        td_target = r + self.gamma * self.Q[s_next, next_action]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        self.last_action = next_action
        return a, r

    def start_episode(self) -> int:
        """Начало эпизода: выбрать действие из softmax-политики и запомнить его."""
        a = self.choose_action()
        self.last_action = a
        return a

    def get_action_probs(self) -> np.ndarray:
        return self.policy()
