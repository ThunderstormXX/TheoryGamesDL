"""
Агент для Q-learning на рынке двух продавцов с больцмановским распределением
"""

import numpy as np
from dataclasses import dataclass


def softmax(Q, beta):
    """
    Больцмановское распределение для выбора действий.
    
    Args:
        Q (np.ndarray): Q-значения для всех действий
        beta (float): Параметр температуры (обратная температура)
        
    Returns:
        np.ndarray: Вероятности выбора каждого действия
    """
    expQ = np.exp(beta * (Q - np.max(Q)))  # стабильность через вычитание максимума
    return expQ / expQ.sum()


@dataclass
class MarketAgent:
    """
    Агент-продавец на рынке с Q-learning и больцмановским выбором действий.
    
    Attributes:
        name (str): Имя агента
        c (float): Себестоимость продукта
        eta (float): Параметр перекрестной эластичности (степень замещаемости товаров)
        beta (float): Параметр температуры для больцмановского распределения
        alpha (float): Скорость обучения
        gamma (float): Коэффициент дисконтирования
        n_grid (int): Количество точек в сетке цен
    """
    name: str
    c: float
    eta: float
    beta: float = 2.0
    alpha: float = 0.01
    gamma: float = 0.9
    n_grid: int = 100

    def __post_init__(self):
        """Инициализация после создания объекта"""
        # Создаем сетку возможных цен от 0 до 1
        self.p_grid = np.linspace(0, 1, self.n_grid)
        
        # Инициализируем Q-функцию нулями
        self.Q = np.zeros(self.n_grid)
        
        # Начальная стратегия - равномерное распределение
        self.pi = np.ones(self.n_grid) / self.n_grid

    def choose_action(self):
        """
        Выбрать действие (цену) согласно текущей политике.
        
        Returns:
            tuple: (выбранная цена, индекс в сетке)
        """
        idx = np.random.choice(self.n_grid, p=self.pi)
        return self.p_grid[idx], idx

    def update_policy(self):
        """
        Обновить распределение действий через softmax на основе Q-значений.
        """
        self.pi = softmax(self.Q, self.beta)

    def learn(self, idx, reward):
        """
        Обновить Q-функцию по правилу временной разницы (TD-learning).
        
        Args:
            idx (int): Индекс выбранного действия
            reward (float): Полученное вознаграждение
        """
        # TD-цель: reward + gamma * max(Q)
        target = reward + self.gamma * np.max(self.Q)
        
        # Обновление Q-значения
        self.Q[idx] += self.alpha * (target - self.Q[idx])
        
        # Обновляем политику после изменения Q
        self.update_policy()

    def expected_price(self):
        """
        Вычислить среднюю ожидаемую цену по текущей политике.
        
        Returns:
            float: Математическое ожидание цены
        """
        return np.sum(self.p_grid * self.pi)
    
    def get_policy_entropy(self):
        """
        Вычислить энтропию текущей политики (мера разнообразия).
        
        Returns:
            float: Энтропия политики
        """
        # Избегаем log(0)
        pi_safe = self.pi + 1e-10
        return -np.sum(self.pi * np.log(pi_safe))
    
    def get_most_probable_price(self):
        """
        Получить наиболее вероятную цену из текущей политики.
        
        Returns:
            float: Цена с максимальной вероятностью
        """
        idx = np.argmax(self.pi)
        return self.p_grid[idx]
    
    def reset(self):
        """
        Сбросить агента в начальное состояние.
        """
        self.Q = np.zeros(self.n_grid)
        self.pi = np.ones(self.n_grid) / self.n_grid

