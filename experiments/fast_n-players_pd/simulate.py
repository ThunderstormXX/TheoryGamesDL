#!/usr/bin/env python3
"""CLI для симуляции многоагентной дилеммы заключенного с Q-обучением."""

import os
import sys
import tempfile
import numpy as np
import json
import argparse

# Настройка безопасной директории для runtime
_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
if not _runtime_dir or not os.path.isdir(_runtime_dir) or (os.stat(_runtime_dir).st_mode & 0o777) != 0o700:
    tmp_runtime = os.path.join(tempfile.gettempdir(), f"runtime-{os.getuid()}")
    os.makedirs(tmp_runtime, exist_ok=True)
    try:
        os.chmod(tmp_runtime, 0o700)
    except PermissionError:
        pass
    os.environ["XDG_RUNTIME_DIR"] = tmp_runtime

# Импорты из модулей
from simulation.runner import run_sim
from visualization.time_series import plot_p_traj_separate, plot_q_values_separate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Симуляция многоагентной дилеммы заключенного")
    parser.add_argument("--T", type=int, default=2000000, help="Количество шагов")
    parser.add_argument("--alpha", type=float, default=0.01, help="Скорость обучения")
    parser.add_argument("--beta", type=float, default=3.0, help="Обратная температура Больцмана (inverse temperature): чем выше, тем детерминированнее политика")
    parser.add_argument("--gamma", type=float, default=0.7, help="Коэффициент дисконтирования")
    parser.add_argument("--n_players", type=int, default=5, help="Количество игроков")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--benefit", type=float, default=6.0, help="Параметр benefit (b)")
    parser.add_argument("--cost", type=float, default=4.0, help="Параметр cost (c)")
    parser.add_argument("--reward_offset", type=float, default=1.0, help="Смещение наград")
    parser.add_argument("--no_plots", action="store_true", help="Не строить графики")
    parser.add_argument("--output_json", type=str, help="Путь для сохранения результатов в JSON")
    
    args = parser.parse_args()
    
    # Основная симуляция
    agents, p_traj, q_traj, mean_reward, meta = run_sim(
        T=args.T,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        n_players=args.n_players,
        seed=args.seed,
        benefit=args.benefit,
        cost=args.cost,
        reward_offset=args.reward_offset,
        max_keep=min(args.T, 300000),
        use_tqdm=not args.no_plots
    )
    
    # Вычисление метрик
    last_20_percent = int(0.8 * meta['used_T'])
    mean_coop = np.mean(p_traj[:, last_20_percent:])
    
    results = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "n_players": args.n_players,
        "T": args.T,
        "mean_reward": float(mean_reward),
        "mean_cooperation": float(mean_coop),
        "final_cooperation": [float(np.mean(p_traj[i, last_20_percent:])) for i in range(args.n_players)]
    }
    
    # Сохранение результатов в JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Результаты сохранены: {args.output_json}")
    
    # Построение графиков
    if not args.no_plots:
        plot_p_traj_separate(p_traj, agents, meta=meta)
        plot_q_values_separate(q_traj, agents, meta=meta)
    
    # Вывод основных метрик
    print(f"\nРезультаты: β={args.beta:.2f}, γ={args.gamma:.2f}")
    print(f"Средняя кооперация: {mean_coop:.4f}")
    print(f"Средний reward: {mean_reward:.4f}")
