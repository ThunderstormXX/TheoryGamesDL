"""
Реализация Q-обучения для социальных дилемм
"""

import numpy as np
import math


def calc_q(game, Q, curr_state, next_state, your_act, enemy_act, alpha, gamma, player, noise_prob, prob):
    """
    Вычисление Q-значения для текущего состояния и действия
    
    Args:
        game: Объект игры
        Q: Матрица Q-значений
        curr_state: Текущее состояние
        next_state: Следующее состояние
        your_act: Ваше действие
        enemy_act: Действие противника
        alpha: Скорость обучения
        gamma: Коэффициент дисконтирования
        player: Номер игрока (0 или 1)
        noise_prob: Вероятность шума
        prob: Вероятность выбора действия
        
    Returns:
        tuple: (новое Q-значение, вознаграждение, наличие ошибки)
    """
    # Используем максимальное Q-значение для следующего состояния
    next_q = np.max(Q[curr_state])

    # Получаем вознаграждение и информацию о наличии шума
    reward, err = game.reward_func(your_act, enemy_act, player, noise_prob)
    
    # Обновляем Q-значение по формуле Q-обучения
    res = Q[curr_state, your_act] + alpha * (reward + gamma * next_q - Q[curr_state, your_act])
    
    # Альтернативная формула с учетом вероятности выбора действия
    # res = Q[curr_state, your_act] + (1/prob) * alpha * (reward + gamma * next_q - Q[curr_state, your_act])
    
    return res, reward, err


def sd_qlearning(game, episode_count=100, alpha=0.5, gamma=0.5, eps=0.1, beta=1, noise_prob=0):
    """
    Q-обучение для социальных дилемм
    
    Args:
        game: Объект игры
        episode_count: Количество эпизодов
        alpha: Скорость обучения
        gamma: Коэффициент дисконтирования
        eps: Параметр исследования для epsilon-жадной стратегии
        beta: Параметр температуры для softmax-стратегии
        noise_prob: Вероятность шума
        
    Returns:
        tuple: (политика игрока 1, политика игрока 2, матрицы выплат, ошибки, история, Q-значения, история вознаграждений)
    """
    policy1 = []  # Политика игрока 1 (форма: сессия-шаг-действие)
    policy2 = []  # Политика игрока 2

    # Инициализация структур для хранения выплат
    payoff1 = [[] for _ in range(len(game.states))]
    payoff2 = [[] for _ in range(len(game.states))]
    
    # Инициализация структур для хранения истории
    history = [[] for _ in range(len(game.states))]
    history_rew = [[] for _ in range(len(game.states))]
    history_Q1 = [[] for _ in range(len(game.states))]
    history_Q2 = [[] for _ in range(len(game.states))]

    # Счетчик ошибок для каждого состояния
    errs = [0] * len(game.states)

    # Инициализация Q-таблиц для обоих игроков
    Q1 = np.zeros([len(game.states), len(game.actions)]) 
    Q2 = np.zeros([len(game.states), len(game.actions)]) 
    
    terminal_state = len(game.states) - 1

    for i in range(episode_count):
        p1_s = 0  # Начальное состояние игрока 1
        p2_s = 0  # Начальное состояние игрока 2

        j = 0  # Счетчик шагов
        
        while game.curr_state <= terminal_state:  
            # Выбор действий на основе политики
            if (len(policy1) > 0):
                # Проверка на NaN и исправление
                if math.isnan(policy1[-1][game.curr_state][0]):
                    policy1[-1][game.curr_state][0] = 0.99999999999999
                    policy1[-1][game.curr_state][1] = 1.0 - 0.99999999999999
                if math.isnan(policy1[-1][game.curr_state][1]):
                    policy1[-1][game.curr_state][1] = 0.99999999999999
                    policy1[-1][game.curr_state][0] = 1.0 - 0.99999999999999
                if math.isnan(policy2[-1][game.curr_state][0]):
                    policy2[-1][game.curr_state][0] = 0.99999999999999
                    policy2[-1][game.curr_state][1] = 1.0 - 0.99999999999999
                if math.isnan(policy2[-1][game.curr_state][1]):
                    policy2[-1][game.curr_state][1] = 0.99999999999999
                    policy2[-1][game.curr_state][0] = 1.0 - 0.99999999999999
                
                # Выбор действий на основе вероятностей из политики
                p1_a = np.random.choice([0, 1], p=[policy1[-1][game.curr_state][0], policy1[-1][game.curr_state][1]])
                p2_a = np.random.choice([0, 1], p=[policy2[-1][game.curr_state][0], policy2[-1][game.curr_state][1]])
            else:
                # Случайный выбор действий в начале обучения
                p1_a = np.random.choice([1, 0])
                p2_a = np.random.choice([1, 0])

            # Выполнение шага в игре
            game.action()
            p1_s1 = game.curr_state
            p2_s1 = game.curr_state
            
            # Определение вероятностей выбора действий
            if len(policy1) == 0:
                p1p = 0.5
                p2p = 0.5
            else:
                if p1_a == 0:
                    p1p = policy1[-1][0][0]
                else:
                    p1p = policy1[-1][0][1]

                if p2_a == 0:
                    p2p = policy2[-1][0][0]
                else:
                    p2p = policy2[-1][0][1]

            # Обновление Q-значений
            Q1[p1_s, p1_a], rew1, err1 = calc_q(game, Q1, p1_s, p1_s1, p1_a, p2_a, alpha, gamma, 0, noise_prob, p1p)
            Q2[p2_s, p2_a], rew2, err2 = calc_q(game, Q2, p2_s, p2_s1, p2_a, p1_a, alpha, gamma, 1, noise_prob, p2p)
            
            # Учет ошибок
            errs[j] += max(err1, err2)
            
            # Обновление матриц выплат
            step_payoff1 = np.matrix([[np.nan] * 2] * 2)
            step_payoff2 = np.matrix([[np.nan] * 2] * 2)
            step_payoff1[p1_a, p2_a] = rew1
            step_payoff2[p2_a, p1_a] = rew2
            
            payoff1[j].append(step_payoff1)
            payoff2[j].append(step_payoff2)
            
            # Сохранение истории
            history[j].append([p1_a, p2_a])
            history_rew[j].append([rew1, rew2])
            history_Q1[j].append(Q1[j].copy())
            history_Q2[j].append(Q2[j].copy())

            # Переход к следующему состоянию
            p1_s = p1_s1
            p2_s = p2_s1
            
            j += 1
        
        # Вычисление softmax-политики
        session1 = []
        session2 = []
        for j in range(len(game.states)):
            pol1 = [None] * 2
            pol2 = [None] * 2
            pol1[0] = np.exp(beta * Q1[j][0]) / (np.exp(beta * Q1[j][0]) + np.exp(beta * Q1[j][1]))
            pol1[1] = np.exp(beta * Q1[j][1]) / (np.exp(beta * Q1[j][0]) + np.exp(beta * Q1[j][1]))
            pol2[0] = np.exp(beta * Q2[j][0]) / (np.exp(beta * Q2[j][0]) + np.exp(beta * Q2[j][1]))
            pol2[1] = np.exp(beta * Q2[j][1]) / (np.exp(beta * Q2[j][0]) + np.exp(beta * Q2[j][1]))

            session1.append(pol1)
            session2.append(pol2)

        policy1.append(session1)
        policy2.append(session2)
        
        # Сброс игры после последнего состояния
        game.reset()
      
    # Вычисление средних матриц выплат
    payoff_matrixes1 = [np.nanmean(x, axis=0) for x in payoff1]
    payoff_matrixes2 = [np.nanmean(x, axis=0) for x in payoff2]

    # Нормализация ошибок
    errs = np.array(errs) / episode_count
    
    return policy1, policy2, payoff_matrixes1, payoff_matrixes2, errs, history, history_Q1, history_Q2, history_rew