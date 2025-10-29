"""
Эксперимент exp2: Q-learning для игры двух продавцов на рынке

Этот эксперимент моделирует дуополию Бертрана с дифференцированными продуктами,
где два продавца обучаются устанавливать оптимальные цены с помощью Q-learning
и больцмановского выбора действий.
"""

import numpy as np
import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from theorygamesdl.agents.market_qlearning import MarketAgent
from theorygamesdl.models.market_game import MarketGame


def run_basic_experiment():
    """
    Базовый эксперимент: симметричные агенты на рынке
    """
    print("=" * 60)
    print("Эксперимент exp2: Q-learning для рынка двух продавцов")
    print("=" * 60)
    
    # Параметры игры
    c = 0.2      # Себестоимость
    eta = 0.7    # Перекрестная эластичность (0.7 означает высокую замещаемость)
    
    # Параметры обучения
    beta = 3.0   # Температура для больцмановского распределения
    alpha = 0.01 # Скорость обучения
    gamma = 0.9  # Коэффициент дисконтирования
    
    # Создаём агентов
    agent1 = MarketAgent(
        name="Продавец A",
        c=c,
        eta=eta,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        n_grid=100
    )
    
    agent2 = MarketAgent(
        name="Продавец B",
        c=c,
        eta=eta,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        n_grid=100
    )
    
    # Создаём игру
    game = MarketGame(agent1, agent2, T=20000, track_convergence=True)
    
    # Вычисляем теоретическое равновесие Нэша
    nash_price = game.get_nash_equilibrium_theory()
    print(f"\n📊 Параметры:")
    print(f"   Себестоимость (c): {c}")
    print(f"   Эластичность (eta): {eta}")
    print(f"   Теоретическое равновесие Нэша: p* = {nash_price:.3f}")
    print(f"\n🎯 Параметры обучения:")
    print(f"   Скорость обучения (alpha): {alpha}")
    print(f"   Дисконтирование (gamma): {gamma}")
    print(f"   Температура (beta): {beta}")
    print(f"   Размер сетки цен: {agent1.n_grid}")
    
    print(f"\n🚀 Начинаем симуляцию ({game.T} итераций)...")
    print("-" * 60)
    
    # Запускаем симуляцию
    history = game.simulate(verbose=True, log_interval=2000)
    
    # Анализируем результаты
    print("\n" + "=" * 60)
    print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    # Статистика за последние 20% итераций (после сходимости)
    burn_in = int(0.8 * game.T)
    stats = game.compute_statistics(burn_in=burn_in)
    
    print(f"\n🎯 Ожидаемые цены (по политике):")
    print(f"   {agent1.name}: {stats['expected_p1']:.3f}")
    print(f"   {agent2.name}: {stats['expected_p2']:.3f}")
    
    print(f"\n📊 Средние цены (последние 20% итераций):")
    print(f"   {agent1.name}: {stats['mean_p1']:.3f} ± {stats['std_p1']:.3f}")
    print(f"   {agent2.name}: {stats['mean_p2']:.3f} ± {stats['std_p2']:.3f}")
    
    print(f"\n💰 Средняя прибыль (последние 20% итераций):")
    print(f"   {agent1.name}: {stats['mean_r1']:.4f} ± {stats['std_r1']:.4f}")
    print(f"   {agent2.name}: {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}")
    
    if 'nash_equilibrium' in stats:
        print(f"\n🎲 Отклонение от равновесия Нэша (p* = {stats['nash_equilibrium']:.3f}):")
        print(f"   {agent1.name}: {stats['deviation_p1']:.4f}")
        print(f"   {agent2.name}: {stats['deviation_p2']:.4f}")
    
    print(f"\n🎲 Наиболее вероятные цены:")
    print(f"   {agent1.name}: {stats['most_probable_p1']:.3f}")
    print(f"   {agent2.name}: {stats['most_probable_p2']:.3f}")
    
    print("\n" + "=" * 60)
    
    return game, history, stats


def run_asymmetric_experiment():
    """
    Эксперимент с асимметричными агентами (разная себестоимость)
    """
    print("\n" + "=" * 60)
    print("Эксперимент: Асимметричные продавцы")
    print("=" * 60)
    
    # Параметры игры
    c1 = 0.15    # Низкая себестоимость у продавца 1
    c2 = 0.25    # Высокая себестоимость у продавца 2
    eta = 0.7
    
    # Параметры обучения
    beta = 3.0
    alpha = 0.01
    gamma = 0.9
    
    # Создаём агентов
    agent1 = MarketAgent(name="Продавец A (низкие издержки)", c=c1, eta=eta,
                        beta=beta, alpha=alpha, gamma=gamma)
    agent2 = MarketAgent(name="Продавец B (высокие издержки)", c=c2, eta=eta,
                        beta=beta, alpha=alpha, gamma=gamma)
    
    # Создаём игру
    game = MarketGame(agent1, agent2, T=20000, track_convergence=True)
    
    print(f"\n📊 Параметры:")
    print(f"   Себестоимость A: {c1}")
    print(f"   Себестоимость B: {c2}")
    print(f"   Эластичность (eta): {eta}")
    
    print(f"\n🚀 Начинаем симуляцию...")
    print("-" * 60)
    
    # Запускаем симуляцию
    history = game.simulate(verbose=True, log_interval=2000)
    
    # Анализируем результаты
    print("\n" + "=" * 60)
    print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ (асимметричный случай)")
    print("=" * 60)
    
    burn_in = int(0.8 * game.T)
    stats = game.compute_statistics(burn_in=burn_in)
    
    print(f"\n🎯 Ожидаемые цены:")
    print(f"   {agent1.name}: {stats['expected_p1']:.3f}")
    print(f"   {agent2.name}: {stats['expected_p2']:.3f}")
    
    print(f"\n💰 Средняя прибыль:")
    print(f"   {agent1.name}: {stats['mean_r1']:.4f}")
    print(f"   {agent2.name}: {stats['mean_r2']:.4f}")
    
    print("\n" + "=" * 60)
    
    return game, history, stats


def run_elasticity_comparison():
    """
    Сравнение разных значений эластичности замещения
    """
    print("\n" + "=" * 60)
    print("Эксперимент: Влияние эластичности замещения")
    print("=" * 60)
    
    eta_values = [0.3, 0.5, 0.7, 0.9]
    c = 0.2
    
    results = []
    
    for eta in eta_values:
        print(f"\n--- Эластичность eta = {eta} ---")
        
        agent1 = MarketAgent(name="A", c=c, eta=eta, beta=3.0, alpha=0.01, gamma=0.9)
        agent2 = MarketAgent(name="B", c=c, eta=eta, beta=3.0, alpha=0.01, gamma=0.9)
        
        game = MarketGame(agent1, agent2, T=10000, track_convergence=False)
        nash_price = game.get_nash_equilibrium_theory()
        
        history = game.simulate(verbose=False)
        
        burn_in = int(0.8 * game.T)
        stats = game.compute_statistics(burn_in=burn_in)
        
        print(f"Теоретическое равновесие: {nash_price:.3f}")
        print(f"Ожидаемая цена A: {stats['expected_p1']:.3f}")
        print(f"Ожидаемая цена B: {stats['expected_p2']:.3f}")
        print(f"Средняя прибыль A: {stats['mean_r1']:.4f}")
        print(f"Средняя прибыль B: {stats['mean_r2']:.4f}")
        
        results.append({
            'eta': eta,
            'nash_price': nash_price,
            'stats': stats
        })
    
    print("\n" + "=" * 60)
    print("Сводка: Влияние эластичности")
    print("=" * 60)
    print(f"{'eta':<8} {'Nash p*':<10} {'Средняя p':<12} {'Средняя прибыль':<15}")
    print("-" * 60)
    
    for r in results:
        avg_p = (r['stats']['expected_p1'] + r['stats']['expected_p2']) / 2
        avg_r = (r['stats']['mean_r1'] + r['stats']['mean_r2']) / 2
        print(f"{r['eta']:<8.1f} {r['nash_price']:<10.3f} {avg_p:<12.3f} {avg_r:<15.4f}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Запускаем основной эксперимент
    game, history, stats = run_basic_experiment()
    
    # Дополнительные эксперименты
    print("\n\n")
    game_asym, history_asym, stats_asym = run_asymmetric_experiment()
    
    print("\n\n")
    elasticity_results = run_elasticity_comparison()
    
    print("\n✅ Все эксперименты завершены!")
    print("\n💡 Для визуализации результатов используйте visualize.py")

