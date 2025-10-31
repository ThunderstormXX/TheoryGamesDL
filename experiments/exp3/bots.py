import random
from typing import List


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
