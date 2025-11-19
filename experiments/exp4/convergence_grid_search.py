#!/usr/bin/env python3
"""
Grid search по параметрам beta и gamma для анализа сходимости.

Запускает обучение SARSA для разных комбинаций параметров,
проверяет сходимость и сохраняет результаты в JSON.
"""
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from continuous_game import ContinuousBimatrixGame, PayoffParams
from softmax_sarsa_agent import SoftmaxSARSAAgent
from viz import check_convergence


def run_single_experiment(beta: float, gamma: float,
                          params: PayoffParams,
                          n: int = 100,
                          steps: int = 50000,
                          alpha: float = 0.01,
                          init_mode: str = 'uniform',
                          seed: int = 42) -> Dict:
    """Запускает один эксперимент и возвращает статус сходимости.
    
    Returns:
        dict с ключами: beta, gamma, status_a, status_b, argmax_a, argmax_b
    """
    game = ContinuousBimatrixGame(params, n=n)
    agent_a = SoftmaxSARSAAgent(game.num_actions(), alpha=alpha, gamma=gamma,
                                beta=beta, init_mode=init_mode, seed=seed)
    agent_b = SoftmaxSARSAAgent(game.num_actions(), alpha=alpha, gamma=gamma,
                                beta=beta, init_mode=init_mode, seed=seed + 1)
    
    policies_a = []
    policies_b = []
    
    a = agent_a.start_episode()
    b = agent_b.start_episode()
    
    for _ in range(steps):
        r_a = game.payoff_player0(a, b)
        r_b = game.payoff_player1(a, b)
        next_a = agent_a.choose_action()
        next_b = agent_b.choose_action()
        agent_a.step(r_a, next_action=next_a)
        agent_b.step(r_b, next_action=next_b)
        policies_a.append(agent_a.get_action_probs())
        policies_b.append(agent_b.get_action_probs())
        a, b = next_a, next_b
    
    # Проверяем сходимость
    status_a, argmax_a = check_convergence(policies_a, window=5000,
                                           threshold_converged=1e-3,
                                           threshold_zero=0.05)
    status_b, argmax_b = check_convergence(policies_b, window=5000,
                                           threshold_converged=1e-3,
                                           threshold_zero=0.05)
    
    return {
        'beta': beta,
        'gamma': gamma,
        'status_a': status_a,
        'status_b': status_b,
        'argmax_a': float(argmax_a),
        'argmax_b': float(argmax_b)
    }


def main():
    # Параметры сетки
    beta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    gamma_values = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    
    # Параметры игры (Prisoner's Dilemma-like)
    params = PayoffParams(r=3.0, p=4.0, q=0.0, s=1.0)
    
    # Параметры обучения
    n = 100  # дискретизация [0,1] на 101 точку
    steps = 50000
    alpha = 0.01
    init_mode = 'uniform'
    
    results = []
    total = len(beta_values) * len(gamma_values)
    
    print(f"🔬 Запускаем grid search: {len(beta_values)} beta × {len(gamma_values)} gamma = {total} экспериментов")
    print(f"Параметры: n={n}, steps={steps}, alpha={alpha}, init_mode={init_mode}")
    print(f"Игра: r={params.r}, p={params.p}, q={params.q}, s={params.s}")
    print()
    
    with tqdm(total=total, desc="Grid search") as pbar:
        for beta in beta_values:
            for gamma in gamma_values:
                result = run_single_experiment(beta=beta, gamma=gamma, params=params,
                                              n=n, steps=steps, alpha=alpha,
                                              init_mode=init_mode, seed=42)
                results.append(result)
                pbar.set_postfix({'beta': beta, 'gamma': gamma,
                                 'status_a': result['status_a'][:12]})
                pbar.update(1)
    
    # Сохраняем результаты
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'convergence_grid_search.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'params': {
                'beta_values': beta_values,
                'gamma_values': gamma_values,
                'n': n,
                'steps': steps,
                'alpha': alpha,
                'init_mode': init_mode,
                'game_params': {'r': params.r, 'p': params.p, 'q': params.q, 's': params.s}
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Результаты сохранены в {output_file}")
    
    # Краткая статистика
    status_counts = {}
    for r in results:
        s = r['status_a']
        status_counts[s] = status_counts.get(s, 0) + 1
    
    print("\n📊 Статистика сходимости (Agent A):")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(results)
        print(f"  {status:25s}: {count:3d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
