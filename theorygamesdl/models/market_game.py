"""
Модель рынка с двумя продавцами (дуополия Бертрана с дифференцированными продуктами)
"""

import numpy as np


def reward_fn(p_i, p_j, c, eta):
    """
    Функция выплаты продавца i при цене p_i и цене конкурента p_j.
    
    Модель основана на линейном спросе с перекрестной эластичностью:
    q_i = 1 - p_i + eta * p_j
    
    Прибыль = (цена - себестоимость) * объем продаж
    
    Args:
        p_i (float): Цена продавца i
        p_j (float): Цена конкурента j
        c (float): Себестоимость продукта
        eta (float): Параметр перекрестной эластичности (0 < eta < 1)
                     eta = 0: товары независимы
                     eta → 1: товары близки к совершенным заменителям
        
    Returns:
        float: Прибыль продавца i
    """
    # Объем спроса
    demand = 1 - p_i + eta * p_j
    
    # Прибыль = маржа * спрос
    profit = (p_i - c) * demand
    
    return profit


class MarketGame:
    """
    Игра двух торговцев с Q-обучением и больцмановским выбором действий.
    
    Моделирует дуополию Бертрана, где два продавца конкурируют по ценам,
    а их товары являются частичными заменителями.
    """

    def __init__(self, agent1, agent2, T=50000, track_convergence=False):
        """
        Инициализация игры.
        
        Args:
            agent1: Первый агент-продавец
            agent2: Второй агент-продавец
            T (int): Количество итераций симуляции
            track_convergence (bool): Отслеживать ли метрики сходимости
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.T = T
        self.track_convergence = track_convergence
        
        # История игры
        self.history = {
            "p1": [],      # Цены агента 1
            "p2": [],      # Цены агента 2
            "r1": [],      # Награды агента 1
            "r2": [],      # Награды агента 2
        }
        
        # Метрики сходимости (если включено отслеживание)
        if track_convergence:
            self.history["ep1"] = []  # Ожидаемые цены агента 1
            self.history["ep2"] = []  # Ожидаемые цены агента 2
            self.history["entropy1"] = []  # Энтропия политики агента 1
            self.history["entropy2"] = []  # Энтропия политики агента 2

    def step(self):
        """
        Выполнить один шаг игры.
        
        Процесс:
        1. Оба агента выбирают цены согласно своим политикам
        2. Вычисляются награды (прибыли) для обоих агентов
        3. Агенты обучаются на основе полученных наград
        4. Результаты сохраняются в историю
        """
        # Выбираем действия (цены)
        p1, idx1 = self.agent1.choose_action()
        p2, idx2 = self.agent2.choose_action()

        # Вычисляем награды (прибыли)
        r1 = reward_fn(p1, p2, self.agent1.c, self.agent1.eta)
        r2 = reward_fn(p2, p1, self.agent2.c, self.agent2.eta)

        # Агенты обучаются
        self.agent1.learn(idx1, r1)
        self.agent2.learn(idx2, r2)

        # Логирование
        self.history["p1"].append(p1)
        self.history["p2"].append(p2)
        self.history["r1"].append(r1)
        self.history["r2"].append(r2)
        
        if self.track_convergence:
            self.history["ep1"].append(self.agent1.expected_price())
            self.history["ep2"].append(self.agent2.expected_price())
            self.history["entropy1"].append(self.agent1.get_policy_entropy())
            self.history["entropy2"].append(self.agent2.get_policy_entropy())

    def simulate(self, verbose=True, log_interval=None):
        """
        Запустить полный цикл обучения.
        
        Args:
            verbose (bool): Выводить ли информацию о прогрессе
            log_interval (int): Интервал логирования (по умолчанию T//10)
            
        Returns:
            dict: История игры с ценами и наградами
        """
        if log_interval is None:
            log_interval = max(1, self.T // 10)
        
        for t in range(self.T):
            self.step()
            
            if verbose and (t + 1) % log_interval == 0:
                m1 = self.agent1.expected_price()
                m2 = self.agent2.expected_price()
                avg_r1 = np.mean(self.history["r1"][-log_interval:])
                avg_r2 = np.mean(self.history["r2"][-log_interval:])
                
                print(f"Iter {t+1:6d}: E[p1]={m1:.3f}, E[p2]={m2:.3f}, "
                      f"Avg_R1={avg_r1:.4f}, Avg_R2={avg_r2:.4f}")

        return self.history
    
    def get_nash_equilibrium_theory(self):
        """
        Вычислить теоретическое равновесие Нэша для симметричного случая.
        
        Для симметричных агентов (одинаковые c и eta), равновесие Нэша:
        p* = (1 + c) / (2 - eta)
        
        Returns:
            float: Теоретическая равновесная цена
        """
        if self.agent1.c == self.agent2.c and self.agent1.eta == self.agent2.eta:
            c = self.agent1.c
            eta = self.agent1.eta
            p_star = (1 + c) / (2 - eta)
            return p_star
        else:
            return None
    
    def compute_statistics(self, burn_in=0):
        """
        Вычислить статистику по истории игры.
        
        Args:
            burn_in (int): Количество начальных шагов, которые нужно пропустить
            
        Returns:
            dict: Статистики (средние, стандартные отклонения и т.д.)
        """
        if burn_in >= len(self.history["p1"]):
            burn_in = 0
        
        p1 = np.array(self.history["p1"][burn_in:])
        p2 = np.array(self.history["p2"][burn_in:])
        r1 = np.array(self.history["r1"][burn_in:])
        r2 = np.array(self.history["r2"][burn_in:])
        
        stats = {
            "mean_p1": np.mean(p1),
            "mean_p2": np.mean(p2),
            "std_p1": np.std(p1),
            "std_p2": np.std(p2),
            "mean_r1": np.mean(r1),
            "mean_r2": np.mean(r2),
            "std_r1": np.std(r1),
            "std_r2": np.std(r2),
            "expected_p1": self.agent1.expected_price(),
            "expected_p2": self.agent2.expected_price(),
            "most_probable_p1": self.agent1.get_most_probable_price(),
            "most_probable_p2": self.agent2.get_most_probable_price(),
        }
        
        # Добавляем теоретическое равновесие, если возможно
        nash_price = self.get_nash_equilibrium_theory()
        if nash_price is not None:
            stats["nash_equilibrium"] = nash_price
            stats["deviation_p1"] = abs(stats["expected_p1"] - nash_price)
            stats["deviation_p2"] = abs(stats["expected_p2"] - nash_price)
        
        return stats

