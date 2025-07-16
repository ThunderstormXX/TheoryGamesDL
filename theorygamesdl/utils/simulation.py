"""
Функции для запуска симуляций и экспериментов
"""

import numpy as np
import matplotlib.pyplot as plt
from ..models import SocialDilemma
from ..agents import sd_qlearning, train_dqn_agents, train_a2c_agents, evaluate_neural_agents


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
        plt.figure(figsize=(10, 6))
        plt.plot(pol1_y1, label='Игрок 1')
        plt.plot(pol2_y1, label='Игрок 2')
        plt.title('Вероятность сотрудничества (Классическое Q-обучение)')
        plt.xlabel('Эпизод')
        plt.ylabel('Вероятность сотрудничества')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        if show_q:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(q1c, label="Q(сотрудничество)")
            plt.plot(q1d, label="Q(предательство)")
            plt.title('Q-значения игрока 1')
            plt.xlabel('Эпизод')
            plt.ylabel('Q-значение')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(q2c, label="Q(сотрудничество)")
            plt.plot(q2d, label="Q(предательство)")
            plt.title('Q-значения игрока 2')
            plt.xlabel('Эпизод')
            plt.ylabel('Q-значение')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
    return pol1_y1, pol2_y1, q1c, q1d, q2c, q2d, history[0], h_rew[0]


def simulate_neural(dilemma_type="pd", payoffs=None, agent_type="dqn", episodes=1000, 
                   eval_episodes=100, show_plots=True):
    """
    Запускает симуляцию социальной дилеммы с нейросетевыми агентами
    
    Args:
        dilemma_type (str): Тип дилеммы ('pd', 'sh', 'ch', 'bs', 'hs', 'sp', 'wmp')
        payoffs (list): Выплаты для дилеммы (если None, используются значения по умолчанию)
        agent_type (str): Тип агента ('dqn' или 'a2c')
        episodes (int): Количество эпизодов обучения
        eval_episodes (int): Количество эпизодов для оценки
        show_plots (bool): Показывать ли графики
        
    Returns:
        tuple: (вероятности сотрудничества игрока 1, вероятности сотрудничества игрока 2, история)
    """
    # Создаем игру с заданными параметрами
    if payoffs is not None and dilemma_type == "pd":
        game = SocialDilemma(pd=payoffs, dilemma_type=dilemma_type, steps_number=1)
    else:
        game = SocialDilemma(dilemma_type=dilemma_type, steps_number=1)
    
    # Обучаем агентов
    if agent_type.lower() == "dqn":
        agent1, agent2, train_history, train_rewards = train_dqn_agents(
            game, episodes=episodes, state_size=2, action_size=2
        )
    elif agent_type.lower() == "a2c":
        agent1, agent2, train_history, train_rewards = train_a2c_agents(
            game, episodes=episodes, state_size=2, action_size=2
        )
    else:
        raise ValueError("Неизвестный тип агента. Используйте 'dqn' или 'a2c'.")
    
    # Оцениваем обученных агентов
    history, rewards, coop_probs1, coop_probs2 = evaluate_neural_agents(
        game, agent1, agent2, episodes=eval_episodes, state_size=2
    )
    
    # Визуализация результатов
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(coop_probs1, label='Player 1')
        plt.plot(coop_probs2, label='Player 2')
        plt.title(f'Вероятность сотрудничества ({agent_type.upper()} агенты)')
        plt.xlabel('Шаг')
        plt.ylabel('Вероятность сотрудничества')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return coop_probs1, coop_probs2, history, rewards