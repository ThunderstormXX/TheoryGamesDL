"""
Функции для анализа результатов экспериментов
"""

import numpy as np


def if_pd(payoff_matrix1, payoff_matrix2, step):
    """
    Проверяет, является ли игра дилеммой заключенного
    
    Args:
        payoff_matrix1: Матрица выплат первого игрока
        payoff_matrix2: Матрица выплат второго игрока
        step: Шаг игры
        
    Returns:
        bool: True, если игра является дилеммой заключенного
    """
    return (payoff_matrix1[step][0][0] > payoff_matrix1[step][1][0] and
            payoff_matrix1[step][1][1] > payoff_matrix1[step][0][0] and
            payoff_matrix1[step][0][1] > payoff_matrix1[step][1][1]) and (payoff_matrix2[step][0][0] > 
                                                                          payoff_matrix2[step][1][0] and
            payoff_matrix2[step][1][1] > payoff_matrix2[step][0][0] and
            payoff_matrix2[step][0][1] > payoff_matrix2[step][1][1])


def if_sh(payoff_matrix1, payoff_matrix2, step):
    """
    Проверяет, является ли игра охотой на оленя
    
    Args:
        payoff_matrix1: Матрица выплат первого игрока
        payoff_matrix2: Матрица выплат второго игрока
        step: Шаг игры
        
    Returns:
        bool: True, если игра является охотой на оленя
    """
    return (payoff_matrix1[step][0][0] > payoff_matrix1[step][1][0] and
            payoff_matrix1[step][0][1] > payoff_matrix1[step][0][0] and
            payoff_matrix1[step][1][1] > payoff_matrix1[step][0][1]) and (payoff_matrix2[step][0][0] > 
                                                                          payoff_matrix2[step][1][0] and
            payoff_matrix2[step][0][1] > payoff_matrix2[step][0][0] and
            payoff_matrix2[step][1][1] > payoff_matrix2[step][0][1])


def if_bs(payoff_matrix1, payoff_matrix2, step):
    """
    Проверяет, является ли игра битвой полов
    
    Args:
        payoff_matrix1: Матрица выплат первого игрока
        payoff_matrix2: Матрица выплат второго игрока
        step: Шаг игры
        
    Returns:
        bool: True, если игра является битвой полов
    """
    return (payoff_matrix1[step][1][1] > payoff_matrix1[step][0][0] and
            payoff_matrix1[step][0][0] > payoff_matrix1[step][1][0] and
            payoff_matrix1[step][0][0] > payoff_matrix1[step][0][1]) and (payoff_matrix2[step][0][0] > 
                                                                          payoff_matrix2[step][1][1] and
            payoff_matrix2[step][1][1] > payoff_matrix2[step][1][0] and
            payoff_matrix2[step][1][1] > payoff_matrix2[step][0][1])


def cc_check(pol1, pol2, epsilon=0.01, delta=0.7, episode_count=100000):
    """
    Проверяет, сходится ли игра к стратегии (C,C) (сотрудничество обоих игроков)
    
    Args:
        pol1: Политика первого игрока
        pol2: Политика второго игрока
        epsilon: Порог для определения сотрудничества
        delta: Доля эпизодов, необходимая для подтверждения сходимости
        episode_count: Количество эпизодов
        
    Returns:
        bool: True, если игра сходится к (C,C)
    """
    return len([elem for elem in pol1 if elem > (1 - epsilon)]) > episode_count * delta and len(
                [elem for elem in pol2 if elem > (1 - epsilon)]) > episode_count * delta


def dd_check(pol1, pol2, epsilon=0.01, delta=0.7, episode_count=100000):
    """
    Проверяет, сходится ли игра к стратегии (D,D) (предательство обоих игроков)
    
    Args:
        pol1: Политика первого игрока
        pol2: Политика второго игрока
        epsilon: Порог для определения предательства
        delta: Доля эпизодов, необходимая для подтверждения сходимости
        episode_count: Количество эпизодов
        
    Returns:
        bool: True, если игра сходится к (D,D)
    """
    return len([elem for elem in pol1 if elem < epsilon]) > episode_count * delta and len(
                [elem for elem in pol2 if elem < epsilon]) > episode_count * delta


def cd_dc_check(pol1, pol2, epsilon=0.01, delta=0.7, episode_count=100000):
    """
    Проверяет, сходится ли игра к стратегии (C,D) или (D,C)
    
    Args:
        pol1: Политика первого игрока
        pol2: Политика второго игрока
        epsilon: Порог для определения стратегии
        delta: Доля эпизодов, необходимая для подтверждения сходимости
        episode_count: Количество эпизодов
        
    Returns:
        bool: True, если игра сходится к (C,D) или (D,C)
    """
    return (len([elem for elem in pol1 if elem > (1 - epsilon)]) > episode_count * delta and len(
                [elem for elem in pol2 if 1 - elem > (1 - epsilon)]) > episode_count * delta) or (len(
                [elem for elem in pol1 if 1 - elem > (1 - epsilon)]) > episode_count * delta and len(
                [elem for elem in pol2 if elem > (1 - epsilon)]) > episode_count * delta)


def mixed_equilibria_check(payoff_matrix, pol1, pol2, step, epsilon, delta, episode_count):
    """
    Проверяет, сходится ли игра к смешанному равновесию
    
    Args:
        payoff_matrix: Матрица выплат
        pol1: Политика первого игрока
        pol2: Политика второго игрока
        step: Шаг игры
        epsilon: Порог для определения стратегии
        delta: Доля эпизодов, необходимая для подтверждения сходимости
        episode_count: Количество эпизодов
        
    Returns:
        bool: True, если игра сходится к смешанному равновесию
    """
    eq = (payoff_matrix[0] - payoff_matrix[3]) / (payoff_matrix[0] - payoff_matrix[3] + payoff_matrix[1] - payoff_matrix[2])
    return (len([elem for elem in pol1[step] if elem[1] > (eq - epsilon) or elem[1] < (eq + epsilon)]) > episode_count * delta 
            and len([elem for elem in pol2[step] if elem[0] > (eq - epsilon) or elem[0] < (eq + epsilon)])
            > episode_count * delta) or (len([elem for elem in pol1[step] if elem[0] > (eq - epsilon) or elem[0]
                                              < (eq + epsilon)]) > episode_count * delta
            and len([elem for elem in pol2[step] if elem[1] > (eq - epsilon) or elem[1] < (eq + epsilon)])
                                         > episode_count * delta)