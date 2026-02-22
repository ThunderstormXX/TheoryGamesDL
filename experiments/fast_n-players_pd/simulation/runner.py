"""Модуль для запуска симуляций многоагентной дилеммы заключенного."""

import numpy as np
from tqdm import tqdm

from bots import BoltzmannAgent
from environment import GameFactory


def _python_run_loop(T, agents, game, record_every, store_q_traj, out_len):
    """Основной цикл симуляции (Python fallback)."""
    n_players = len(agents)
    p_traj = np.empty((n_players, out_len), dtype=float)
    q_traj = np.empty((n_players, out_len, 2), dtype=float) if store_q_traj else None

    mean_r = 0.0
    out_idx = 0
    
    for t in range(T):
        # Выбор действий всеми агентами
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))
        mean_r = (t) / (t + 1) * mean_r + rewards[0] / (t + 1)
        
        # Обучение агентов
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            
        # Запись данных
        if (t % record_every) == 0:
            for i, agent in enumerate(agents):
                p_traj[i, out_idx] = agent.current_p_cooperate()
                if store_q_traj:
                    q_traj[i, out_idx, :] = agent.get_q()
            out_idx += 1
            
    # Обрезка до фактического размера
    if out_idx < out_len:
        p_traj = p_traj[:, :out_idx]
        if store_q_traj:
            q_traj = q_traj[:, :out_idx, :]
    return agents, p_traj, q_traj, mean_r


def run_sim(T=20000, alpha=0.01, beta=1.0, gamma=0.9, seed=42, 
           n_players=2, benefit=6.0, cost=4.0, reward_offset=1.0,
           max_keep=100_000, q_init=1, use_tqdm=True):
    """Запуск симуляции с ограничением памяти.
    
    Args:
        T: Количество шагов симуляции
        alpha: Скорость обучения (learning rate)
        beta: Обратная температура Больцмана (inverse temperature): чем выше, тем детерминированнее политика
        gamma: Коэффициент дисконтирования (discount factor)
        seed: Случайное зерно
        n_players: Количество игроков
        benefit: Параметр benefit (b) для Public Goods PD
        cost: Параметр cost (c) для Public Goods PD
        reward_offset: Смещение наград
        max_keep: Максимум точек для сохранения
        use_tqdm: Показывать прогресс-бар
    
    Returns:
        agents, p_traj, q_traj, mean_q, meta
    """
    # Инициализация
    rng = np.random.default_rng(seed)
    game = GameFactory.create_generalized_prisoners_dilemma(n_players, benefit=benefit, cost=cost, reward_offset=reward_offset)
    agents = [BoltzmannAgent(name=f"A{i+1}", alpha=alpha, beta=beta, 
                           gamma=gamma, rng=rng) for i in range(n_players)]

    # Буферы с циклической записью
    M = max_keep
    p_traj = np.zeros((n_players, M), dtype=float)
    q_traj = np.zeros((n_players, M, 2), dtype=float)
    mean_q = 0.0

    iterator = tqdm(range(T), desc="Simulating", ncols=80) if use_tqdm else range(T)

    # Основной цикл симуляции
    for t in iterator:
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))
        mean_q = (t / (t + 1)) * mean_q + rewards[0] / (t + 1)
        
        idx = t % M  # Циклический индекс
        
        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            p_traj[i, idx] = agent.current_p_cooperate()
            q_traj[i, idx, :] = agent.get_q()

    # Восстановление правильного порядка данных
    if T <= M:
        p_final = p_traj[:, :T].copy()
        q_final = q_traj[:, :T, :].copy()
    else:
        start = T % M
        p_final = np.concatenate([p_traj[:, start:], p_traj[:, :start]], axis=1)
        q_final = np.concatenate([q_traj[:, start:], q_traj[:, :start]], axis=1)

    meta = {
        "T": T, "used_T": p_final.shape[1], "alpha": alpha,
        "beta": beta, "gamma": gamma, "n_players": n_players
    }

    return agents, p_final, q_final, mean_q, meta
