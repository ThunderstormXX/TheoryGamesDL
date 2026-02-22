import numpy as np
from itertools import product
from typing import Callable, Iterable, List, Sequence, Tuple, Optional


class Game:
    """
    Игра с n игроками с бинарными действиями (0 или 1).

    По умолчанию ожидается payoff_func, возвращающая выплату текущего игрока 0
    для данного профиля действий (используется для обучения первого агента).
    Можно также передать готовый тензор выплат payoff_tensor той же формы.
    """

    def __init__(self, n_players: int, payoff_func: Optional[Callable[[Sequence[int]], float]] = None,
                 payoff_tensor: Optional[np.ndarray] = None, name: str = "Untitled Game"):
        self.n_players = n_players
        self.name = name
        self.shape = (2,) * n_players

        if payoff_tensor is not None:
            self.payoff_tensor = payoff_tensor
        elif payoff_func is not None:
            self.payoff_tensor = self._create_payoff_tensor(payoff_func)
        else:
            raise ValueError("Необходимо указать либо payoff_func, либо payoff_tensor")

    def _create_payoff_tensor(self, payoff_func: Callable[[Sequence[int]], float]) -> np.ndarray:
        """Создает тензор выигрышей, итерируясь по всем профилям действий."""
        tensor = np.zeros(self.shape)
        for indices in np.ndindex(self.shape):
            tensor[indices] = payoff_func(indices)
        return tensor

    def get_payoff(self, choices: Sequence[int]) -> float:
        """Возвращает выигрыш для заданной комбинации выборов (игрок 0 по умолчанию)."""
        return float(self.payoff_tensor[tuple(choices)])
    
    def get_payoffs(self, choices: Sequence[int]) -> Tuple[float, ...]:
        """
        Возвращает кортеж выплат (длиной n_players) для данной комбинации choices.
        Использует self.payoff_tensor, который по факту даёт выплату для 'игрока 0'.
        Для получения выплат остальных игроков мы циклически сдвигаем профиль действий.
        """
        if len(choices) != self.n_players:
            raise ValueError("Length of choices must be n_players")

        payoffs = []
        # для каждого игрока i: сделаем ротацию так, чтобы i стал позицией 0 и возьмём значение
        for i in range(self.n_players):
            # циклический сдвиг: new_choices[0] = choices[i], new_choices[1] = choices[i+1], ...
            new_choices = tuple(choices[(i + j) % self.n_players] for j in range(self.n_players))
            pay = float(self.payoff_tensor[new_choices])
            payoffs.append(pay)
        return tuple(payoffs)    

    def analyze(self) -> None:
        """Печатает базовую сводку по игре и средним выигрышам при выборе 0/1 каждым игроком."""
        print(f"=== Анализ игры '{self.name}' ===")
        print(f"Количество игроков: {self.n_players}")
        print(f"Форма тензора выигрышей: {self.payoff_tensor.shape}")
        print(f"Общее количество исходов: {self.payoff_tensor.size}")
        print(f"Средний выигрыш: {np.mean(self.payoff_tensor):.3f}")
        print(f"Максимальный выигрыш: {np.max(self.payoff_tensor):.3f}")
        print(f"Минимальный выигрыш: {np.min(self.payoff_tensor):.3f}")

        print("\n--- Анализ по игрокам ---")
        for i in range(self.n_players):
            axis_to_keep = [j for j in range(self.n_players) if j != i]
            if axis_to_keep:
                payoff_when_0 = np.mean(self.payoff_tensor, axis=tuple(axis_to_keep))[0]
                payoff_when_1 = np.mean(self.payoff_tensor, axis=tuple(axis_to_keep))[1]
            else:
                payoff_when_0 = self.payoff_tensor[0]
                payoff_when_1 = self.payoff_tensor[1]
            print(f"Игрок {i}: средний выигрыш при выборе 0 = {payoff_when_0:.3f}, при выборе 1 = {payoff_when_1:.3f}")

    def find_pure_nash_equilibria(self) -> List[Tuple[int, ...]]:
        """Находит чистые равновесия Нэша в игре (по выгоде игрока 0)."""
        equilibria: List[Tuple[int, ...]] = []
        for strategy in product([0, 1], repeat=self.n_players):
            is_equilibrium = True
            for player in range(self.n_players):
                current_payoff = self.get_payoff(strategy)
                alternative_strategy = list(strategy)
                alternative_strategy[player] = 1 - alternative_strategy[player]
                alternative_payoff = self.get_payoff(alternative_strategy)
                if alternative_payoff > current_payoff:
                    is_equilibrium = False
                    break
            if is_equilibrium:
                equilibria.append(tuple(strategy))
        return equilibria

    def __str__(self) -> str:
        return f"Game '{self.name}' ({self.n_players} players, shape {self.shape})"


class GameFactory:
    """Фабрика для создания стандартных типов игр."""

    @staticmethod
    def create_majority_game(n_players: int) -> Game:
        """Игра 'Большинство' — выигрыш при совпадении с большинством (для игрока 0)."""
        def majority_payoff(choices: Sequence[int]) -> float:
            sum_ones = sum(choices)
            majority = 1 if sum_ones > n_players / 2 else 0
            return 1.0 if choices[0] == majority else -1.0
        return Game(n_players, majority_payoff, name=f"Majority Game ({n_players} players)")

    @staticmethod
    def create_coordination_game(n_players: int) -> Game:
        """Координационная игра — все выигрывают при одинаковом выборе."""
        def coordination_payoff(choices: Sequence[int]) -> float:
            return 1.0 if all(c == choices[0] for c in choices) else -1.0
        return Game(n_players, coordination_payoff, name=f"Coordination Game ({n_players} players)")

    @staticmethod
    def create_public_goods_game(n_players: int, cost: float = 0.5, multiplier: float = 2) -> Game:
        """Игра общественных благ для игрока 0."""
        def public_goods_payoff(choices: Sequence[int]) -> float:
            contributions = sum(choices)
            total_return = contributions * multiplier
            individual_return = total_return / n_players
            return float(individual_return - (cost * choices[0]))
        return Game(n_players, public_goods_payoff, name=f"Public Goods Game ({n_players} players)")

    @staticmethod
    def create_random_game(n_players: int, low: float = -1, high: float = 1) -> Game:
        """Игра со случайными выигрышами."""
        payoff_tensor = np.random.uniform(low, high, (2,) * n_players)
        return Game(n_players, payoff_tensor=payoff_tensor, name=f"Random Game ({n_players} players)")

    @staticmethod
    def create_generalized_prisoners_dilemma(
        n_players: int,
        cooperation_reward: float = 3,
        defection_temptation: float = 4,
        mutual_defection_punishment: float = 1,
        sucker_payoff: float = 0,
    ) -> Game:
        """
        Обобщенная дилемма заключенного для n игроков (возврат для игрока 0).
        """
        def prisoners_dilemma_payoff(choices: Sequence[int]) -> float:
            player0 = choices[0]
            others = choices[1:]

            coop_others = sum(1 for c in others if c == 0)

            if player0 == 0:  # cooperate
                if coop_others == len(others):
                    return cooperation_reward      # (C,C,...)
                else:
                    return sucker_payoff           # (C,D,...)
            else:  # defect
                if coop_others == 0:
                    return mutual_defection_punishment  # (D,D,...)
                else:
                    return defection_temptation         # (D,C,...)
        return Game(n_players, prisoners_dilemma_payoff, name=f"Generalized Prisoner's Dilemma ({n_players} players)")

    @staticmethod
    def create_custom_three_player_game() -> Game:
        """Конкретная 3-игроковая игра из примера."""
        payoff_dict = {
            (0, 0, 0): -10,
            (0, 1, 0): -5,
            (1, 0, 0): 25,
            (1, 1, 0): 8,
            (0, 0, 1): -5,
            (0, 1, 1): -15,
            (1, 0, 1): 8,
            (1, 1, 1): -20,
        }
        def custom_payoff(choices: Sequence[int]) -> float:
            # Явно формируем ключ фиксированной длины (3) для статической типизации
            key = (int(choices[0]), int(choices[1]), int(choices[2]))
            return float(payoff_dict[key])
        return Game(3, custom_payoff, name="Custom 3-Player Game")

    @staticmethod
    def create_generalized_custom_game(
        n_players: int,
        all_defect_payoff: float = -10,
        mixed_payoff_cooperator: float = -5,
        mixed_payoff_defector: float = 25,
        two_cooperators_payoff: float = 8,
        all_cooperate_payoff: float = -20,
        lone_cooperator_payoff: float = -15,
    ) -> Game:
        """
        Обобщение пользовательской игры на n игроков (возвращается выигрыш игрока 0).
        """
        def generalized_custom_payoff(choices: Sequence[int]) -> float:
            cooperators = sum(choices)
            defectors = n_players - cooperators
            current_player_choice = choices[0]
            if current_player_choice == 0:  # игрок 0 предает
                if defectors == n_players:
                    return float(all_defect_payoff)
                else:
                    return float(mixed_payoff_defector)
            else:  # игрок 0 сотрудничает
                if cooperators == 1:
                    return float(lone_cooperator_payoff)
                elif cooperators == n_players:
                    return float(all_cooperate_payoff)
                else:
                    return float(two_cooperators_payoff)
        return Game(n_players, generalized_custom_payoff, name=f"Generalized Custom Game ({n_players} players)")


if __name__ == "__main__":
    custom_game = GameFactory.create_custom_three_player_game()
    custom_game.analyze()

    print("\n" + "=" * 50 + "\n")

    generalized_game = GameFactory.create_generalized_custom_game(4)
    generalized_game.analyze()

    print("\n" + "=" * 50 + "\n")

    nash_eq = custom_game.find_pure_nash_equilibria()
    print(f"Равновесия Нэша в кастомной игре: {nash_eq}")

    pd_game = GameFactory.create_generalized_prisoners_dilemma(3)
    pd_nash = pd_game.find_pure_nash_equilibria()
    print(f"Равновесия Нэша в дилемме заключенного: {pd_nash}")
