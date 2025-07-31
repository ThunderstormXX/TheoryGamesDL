#!/usr/bin/env python3
"""
Эксперимент 1: AlphaRank-based RL для матричных игр

Цель: обучить нейросетевых агентов в матричной игре с RL, 
где reward основан на AlphaRank.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from alpharank_simulation import AlphaRankSimulation
import numpy as np

def run_prisoner_dilemma():
    """Запуск эксперимента с дилеммой заключенного"""
    print("=== Эксперимент: Дилемма заключенного с AlphaRank ===")
    
    # Классические выплаты дилеммы заключенного
    payoffs = [3, 1, 0, 4]  # [CC, DD, DC, CD]
    
    sim = AlphaRankSimulation(
        n_agents=2,
        game_payoffs=payoffs,
        lr=0.001
    )
    
    sim.run(episodes=1000)
    filename = sim.save_results("pd_alpharank_results.json")
    sim.plot_results()
    
    return sim, filename

def run_stag_hunt():
    """Запуск эксперимента с охотой на оленя"""
    print("=== Эксперимент: Охота на оленя с AlphaRank ===")
    
    # Выплаты для охоты на оленя
    payoffs = [3, 1, 0, 2]  # [CC, DD, DC, CD]
    
    sim = AlphaRankSimulation(
        n_agents=2,
        game_payoffs=payoffs,
        lr=0.001
    )
    
    sim.run(episodes=1000)
    filename = sim.save_results("sh_alpharank_results.json")
    sim.plot_results()
    
    return sim, filename

def run_multi_agent_experiment():
    """Запуск эксперимента с несколькими агентами"""
    print("=== Эксперимент: Многоагентная система с AlphaRank ===")
    
    payoffs = [3, 1, 0, 4]  # Дилемма заключенного
    
    sim = AlphaRankSimulation(
        n_agents=4,
        game_payoffs=payoffs,
        lr=0.0005
    )
    
    sim.run(episodes=1500)
    filename = sim.save_results("multi_agent_alpharank_results.json")
    sim.plot_results()
    
    return sim, filename

def main():
    """Главная функция эксперимента"""
    print("Запуск экспериментов с AlphaRank...")
    
    experiments = [
        ("Prisoner's Dilemma", run_prisoner_dilemma),
        ("Stag Hunt", run_stag_hunt),
        ("Multi-Agent", run_multi_agent_experiment)
    ]
    
    results = {}
    
    for name, experiment_func in experiments:
        print(f"\n{'='*50}")
        print(f"Запуск эксперимента: {name}")
        print(f"{'='*50}")
        
        try:
            sim, filename = experiment_func()
            results[name] = {
                'simulation': sim,
                'filename': filename,
                'final_strategies': sim.history['strategies'][-1],
                'final_rewards': sim.history['rewards'][-1]
            }
            print(f"✓ Эксперимент {name} завершен успешно")
            
        except Exception as e:
            print(f"✗ Ошибка в эксперименте {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Сводка результатов
    print(f"\n{'='*50}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*50}")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ОШИБКА - {result['error']}")
        else:
            final_strategies = np.array(result['final_strategies'])
            coop_probs = final_strategies[:, 0]
            print(f"{name}:")
            print(f"  Файл: {result['filename']}")
            print(f"  Финальные стратегии (P(Coop)): {coop_probs}")
            print(f"  Финальные rewards: {result['final_rewards']}")

if __name__ == "__main__":
    main()