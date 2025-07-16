"""
Функции для запуска симуляций и экспериментов
"""

import numpy as np
import matplotlib.pyplot as plt
from ..models import SocialDilemma
from ..agents import sd_qlearning


def simulate(pd=[3, 1, 0, 4], time=100000, gamma=0.9, alpha=0.01, beta=1, show_q=False, show_plots=True):
    """
    Запускает симуляцию социальной дилеммы с Q-обучением
    
    Args:
        pd (list): Выплаты для дилеммы заключенного [CC, DD, DC, CD]
        time (int): Количество эпизодов
        gamma (float): Коэффициент дисконтирования
        alpha (float): Скорость обучения
        beta (float): Параметр температуры для softmax-стратегии
        show_q (bool): Показывать ли графики Q-значений
        show_plots (bool): Показывать ли графики политик
        
    Returns:
        tuple: (политика игрока 1, политика игрока 2, Q-значения, история)
    """
    n_step_pd = SocialDilemma(pd=pd, dilemma_type="pd", steps_number=1)
    pol1, pol2, payoff_matrix1, payoff_matrix2, errs, history, Q1, Q2, h_rew = sd_qlearning(
        n_step_pd, time, gamma=gamma, alpha=alpha, beta=beta, noise_prob=0
    )

    # Преобразование формата политик для удобства анализа
    pol1 = np.moveaxis(pol1, 0, 1)
    pol2 = np.moveaxis(pol2, 0, 1)
    
    # Извлечение вероятностей сотрудничества (действие 1)
    pol1_y1 = [row[1] for row in pol1[0]]
    pol2_y1 = [row[1] for row in pol2[0]]

    # Извлечение Q-значений
    q1c = [row[1] for row in Q1[0]]  # Q-значения для сотрудничества (C) игрока 1
    q1d = [row[0] for row in Q1[0]]  # Q-значения для предательства (D) игрока 1

    q2c = [row[1] for row in Q2[0]]  # Q-значения для сотрудничества (C) игрока 2
    q2d = [row[0] for row in Q2[0]]  # Q-значения для предательства (D) игрока 2
    
    # Визуализация результатов
    if show_plots:
        plt.plot(pol1_y1)
        plt.plot(pol2_y1)
        plt.show()
        
        if show_q:
            plt.plot(q1c, label="q1c")
            plt.plot(q1d, label="q1d")
            plt.legend()
            plt.show()
            
            plt.plot(q2c, label="q2c")
            plt.plot(q2d, label="q2d")
            plt.legend()
            plt.show()
            
    return pol1_y1, pol2_y1, q1c, q1d, q2c, q2d, history[0], h_rew[0]