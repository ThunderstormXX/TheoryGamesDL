"""
Быстрый тест эксперимента exp2
"""

import numpy as np
import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from theorygamesdl.agents.market_qlearning import MarketAgent
from theorygamesdl.models.market_game import MarketGame


def quick_test():
    """
    Быстрый тест с малым количеством итераций
    """
    print("=" * 60)
    print("Быстрый тест exp2")
    print("=" * 60)
    
    # Параметры
    c = 0.2
    eta = 0.7
    beta = 3.0
    alpha = 0.01
    gamma = 0.9
    
    # Создаём агентов
    agent1 = MarketAgent(name="Продавец A", c=c, eta=eta, beta=beta, alpha=alpha, gamma=gamma, n_grid=50)
    agent2 = MarketAgent(name="Продавец B", c=c, eta=eta, beta=beta, alpha=alpha, gamma=gamma, n_grid=50)
    
    # Создаём игру
    game = MarketGame(agent1, agent2, T=1000, track_convergence=True)
    
    # Теоретическое равновесие
    nash_price = game.get_nash_equilibrium_theory()
    print(f"\n📊 Параметры:")
    print(f"   Себестоимость (c): {c}")
    print(f"   Эластичность (eta): {eta}")
    print(f"   Теоретическое равновесие Нэша: p* = {nash_price:.3f}")
    
    print(f"\n🚀 Запуск симуляции (1000 итераций)...")
    
    # Запускаем симуляцию
    history = game.simulate(verbose=True, log_interval=200)
    
    # Результаты
    stats = game.compute_statistics(burn_in=800)
    
    print("\n" + "=" * 60)
    print("📈 РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Ожидаемая цена A: {stats['expected_p1']:.3f}")
    print(f"Ожидаемая цена B: {stats['expected_p2']:.3f}")
    print(f"Отклонение от Nash p* (A): {stats.get('deviation_p1', 'N/A'):.4f}")
    print(f"Отклонение от Nash p* (B): {stats.get('deviation_p2', 'N/A'):.4f}")
    print(f"Средняя прибыль A: {stats['mean_r1']:.4f}")
    print(f"Средняя прибыль B: {stats['mean_r2']:.4f}")
    print("=" * 60)
    
    print("\n✅ Тест успешно завершен!")
    return game, history, stats


if __name__ == "__main__":
    game, history, stats = quick_test()


