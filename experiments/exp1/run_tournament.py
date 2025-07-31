#!/usr/bin/env python3
"""
Эксперимент 1: Турнирная система с AlphaRank анализом

Запускает турниры между множественными нейросетевыми агентами,
вычисляет матрицы переходов и анализирует эволюцию стационарного распределения.
"""

import sys
import os

# Добавляем путь к папке эксперимента
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from tournament_system import TournamentSystem
import numpy as np

def run_prisoner_dilemma_tournament():
    """Турнир в дилемме заключенного"""
    print("=== Турнир: Дилемма заключенного ===")
    
    tournament = TournamentSystem(
        n_agents=6,
        game_payoffs=[3, 1, 0, 4],  # [CC, DD, DC, CD]
        games_per_pair=50
    )
    
    tournament.run_evolution(rounds=15)
    tournament.plot_evolution("prisoners_dilemma")
    filename = tournament.save_results("pd_tournament_results.json")
    
    return tournament, filename

def run_stag_hunt_tournament():
    """Турнир в охоте на оленя"""
    print("\n=== Турнир: Охота на оленя ===")
    
    tournament = TournamentSystem(
        n_agents=6,
        game_payoffs=[3, 1, 0, 2],  # [CC, DD, DC, CD]
        games_per_pair=50
    )
    
    tournament.run_evolution(rounds=15)
    tournament.plot_evolution("stag_hunt")
    filename = tournament.save_results("sh_tournament_results.json")
    
    return tournament, filename

def run_large_tournament():
    """Большой турнир с 8 агентами"""
    print("\n=== Большой турнир: 8 агентов ===")
    
    tournament = TournamentSystem(
        n_agents=8,
        game_payoffs=[3, 1, 0, 4],  # Дилемма заключенного
        games_per_pair=30
    )
    
    tournament.run_evolution(rounds=20)
    tournament.plot_evolution("large_tournament")
    filename = tournament.save_results("large_tournament_results.json")
    
    return tournament, filename

def analyze_results(tournament):
    """Анализирует результаты турнира"""
    print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
    
    # Финальные данные
    final_stationary = np.array(tournament.history['stationary_distributions'][-1])
    final_rewards = np.array(tournament.history['mean_rewards'][-1])
    
    print(f"Финальное стационарное распределение:")
    for i, prob in enumerate(final_stationary):
        print(f"  Agent {i}: {prob:.4f}")
    
    print(f"\nФинальные средние награды:")
    for i, reward in enumerate(final_rewards):
        print(f"  Agent {i}: {reward:.4f}")
    
    # Корреляция
    correlation = np.corrcoef(final_stationary, final_rewards)[0, 1]
    print(f"\nКорреляция между стационарным распределением и наградами: {correlation:.4f}")
    
    # Топ агенты
    stat_ranking = np.argsort(final_stationary)[::-1]
    reward_ranking = np.argsort(final_rewards)[::-1]
    
    print(f"\nРейтинг по стационарному распределению: {stat_ranking}")
    print(f"Рейтинг по наградам: {reward_ranking}")
    
    # Проверяем, совпадают ли топ-3
    top3_stat = set(stat_ranking[:3])
    top3_reward = set(reward_ranking[:3])
    overlap = len(top3_stat.intersection(top3_reward))
    
    print(f"Совпадение в топ-3: {overlap}/3 агентов")
    
    return {
        'final_stationary': final_stationary.tolist(),
        'final_rewards': final_rewards.tolist(),
        'correlation': float(correlation),
        'stat_ranking': stat_ranking.tolist(),
        'reward_ranking': reward_ranking.tolist(),
        'top3_overlap': overlap
    }

def main():
    """Главная функция"""
    print("Запуск турнирной системы с AlphaRank анализом...")
    
    experiments = [
        ("Prisoner's Dilemma", run_prisoner_dilemma_tournament),
        ("Stag Hunt", run_stag_hunt_tournament),
        ("Large Tournament", run_large_tournament)
    ]
    
    results = {}
    
    for name, experiment_func in experiments:
        print(f"\n{'='*60}")
        print(f"Запуск эксперимента: {name}")
        print(f"{'='*60}")
        
        try:
            tournament, filename = experiment_func()
            analysis = analyze_results(tournament)
            
            results[name] = {
                'tournament': tournament,
                'filename': filename,
                'analysis': analysis
            }
            
            print(f"✓ Эксперимент {name} завершен успешно")
            
        except Exception as e:
            print(f"✗ Ошибка в эксперименте {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Общая сводка
    print(f"\n{'='*60}")
    print("ОБЩАЯ СВОДКА")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"\n{name}: ОШИБКА - {result['error']}")
        else:
            analysis = result['analysis']
            print(f"\n{name}:")
            print(f"  Файл: {result['filename']}")
            print(f"  Корреляция: {analysis['correlation']:.4f}")
            print(f"  Совпадение топ-3: {analysis['top3_overlap']}/3")
            print(f"  Лучший агент (статистика): Agent {analysis['stat_ranking'][0]}")
            print(f"  Лучший агент (награды): Agent {analysis['reward_ranking'][0]}")

if __name__ == "__main__":
    main()