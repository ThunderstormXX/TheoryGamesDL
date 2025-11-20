#!/usr/bin/env python3
"""
Скрипт для запуска grid search по параметрам beta и gamma
с 200000 шагами и анализом сходимости.

Повторяет функциональность ноутбука convergence_analysis.ipynb
с расширенным количеством шагов для более точного анализа.

ОПТИМИЗАЦИИ ПАМЯТИ:
- Сэмплирование данных для графиков (~1000 точек вместо 200000)
- Хранение только последних 5000 шагов для проверки сходимости
- Принудительная очистка памяти (gc.collect) каждые 10 экспериментов
- Закрытие matplotlib фигур после сохранения
- Удаление больших массивов после использования

ВЫВОД:
- JSON файл с результатами сходимости для всех экспериментов
- Две общие heat-map визуализации (совместная и раздельная)
- Индивидуальные графики для каждой пары (beta, gamma) в individual_plots/

ВРЕМЯ ВЫПОЛНЕНИЯ:
Примерно 2-5 часов для 168 экспериментов (зависит от железа)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm # type: ignore

# Настройка путей для импорта
exp4_path = Path(__file__).parent.absolute()
if str(exp4_path) not in sys.path:
    sys.path.insert(0, str(exp4_path))

r_base = 3
p_base = 4
q_base = 0
s_base = 1

from continuous_game import ContinuousBimatrixGame, PayoffParams # type: ignore
from softmax_sarsa_agent import SoftmaxSARSAAgent # type: ignore
from viz import check_convergence, plot_joint_convergence_heatmap, plot_action_value_heatmaps, plot_comprehensive_results # type: ignore
import gc


def base_function(x: float, B: float) -> float:
    """
    Базовая функция: B * exp(B*x) / (exp(B) - 1)
    
    Args:
        x: значение аргумента
        B: параметр (beta)
    
    Returns:
        Значение функции
    """

    exp_B = np.exp(B)
    return B * np.exp(B * x) / (exp_B - 1)


def solve_beta_newton(R: float, S: float, T: float, P: float, 
                      beta: float,
                      initial_guess: float = 1.0,
                      max_iter: int = 100,
                      tol: float = 1e-8) -> float:
    """
    Решает уравнение для B методом Ньютона:
    B = β * [(R - S - T + P) * ((B-1)*e^B + 1) / (B*(e^B - 1)) + (S - P)]
    
    где числитель = (B-1)*e^B + 1, знаменатель = B*(e^B - 1)
    
    Перепишем как: f(B) = B - β * [(R - S - T + P) * ((B-1)*e^B + 1) / (B*(e^B - 1)) + (S - P)] = 0
    
    Args:
        R, S, T, P: параметры игры
        beta: заданное значение β (inverse temperature)
        initial_guess: начальное приближение для B
        max_iter: максимальное количество итераций
        tol: точность
        
    Returns:
        Решение B
    """
    
    def f(B: float) -> float:
        """Уравнение f(B) = 0"""
        if abs(B) < 1e-10:
            # Предельный случай при B→0
            return B
        
        try:
            exp_B = np.exp(B)
            
            # Числитель: (B-1)*e^B + 1
            numerator = (B - 1) * exp_B + 1
            
            # Знаменатель: B * (e^B - 1)
            denominator = B * (exp_B - 1)
            
            if abs(denominator) < 1e-15:
                # Избегаем деления на ноль
                return B
            
            # Вычисляем дробь
            fraction = numerator / denominator
            
            # f(B) = B - β * [(R - S - T + P) * fraction + (S - P)]
            rhs = beta * ((R - S - T + P) * fraction + (S - P))
            
            return B - rhs
            
        except (OverflowError, RuntimeWarning):
            # При очень больших значениях
            return B
    
    def df(B: float, h: float = 1e-7) -> float:
        """Численная производная f'(B)"""
        return (f(B + h) - f(B - h)) / (2 * h)
    
    B = initial_guess
    
    for iteration in range(max_iter):
        fB = f(B)
        
        if abs(fB) < tol:
            return B
        
        dfB = df(B)
        
        if abs(dfB) < 1e-15:
            # Производная слишком мала, попробуем другое начальное приближение
            print(f"  ⚠️ Производная близка к нулю на итерации {iteration}, B={B:.6f}")
            B = initial_guess * 0.5 if initial_guess > 0 else -0.5
            continue
        
        # Шаг метода Ньютона
        B_new = B - fB / dfB
        
        # Проверка на сходимость
        if abs(B_new - B) < tol:
            return B_new
        
        B = B_new
    
    print(f"  ⚠️ Метод Ньютона не сошёлся за {max_iter} итераций. Возвращаем B={B:.6f}")
    return B


def run_single_experiment(beta: float, gamma: float,
                          params: PayoffParams,
                          n: int = 100,
                          steps: int = 200000,
                          alpha: float = 0.01,
                          init_mode: str = 'uniform',
                          seed: int = 42,
                          save_plot: bool = True,
                          output_dir: Path = None, # type: ignore
                          base_function = None) -> dict:
    """Запускает один эксперимент и возвращает детальную информацию о сходимости.
    
    Args:
        beta: inverse temperature для softmax
        gamma: discount factor
        params: параметры игры
        n: размер дискретизации пространства действий
        steps: количество шагов обучения
        alpha: learning rate
        init_mode: режим инициализации Q-значений
        seed: random seed
        save_plot: создавать ли визуализацию для этого эксперимента
        base_function: опциональная функция base_function(x) для использования в эксперименте
        output_dir: директория для сохранения графиков
        
    Returns:
        dict с результатами эксперимента
    """
    game = ContinuousBimatrixGame(params, n=n)
    agent_a = SoftmaxSARSAAgent(game.num_actions(), alpha=alpha, gamma=gamma,
                                beta=beta, init_mode=init_mode, seed=seed)
    agent_b = SoftmaxSARSAAgent(game.num_actions(), alpha=alpha, gamma=gamma,
                                beta=beta, init_mode=init_mode, seed=seed + 1)
    
    # Для оптимизации памяти: храним только каждый N-й элемент для графиков
    # и последние M элементов для проверки сходимости
    sample_interval = max(1, steps // 1000)  # Сохраняем ~1000 точек для графиков
    convergence_window = 5000
    
    policies_a_full = []  # Только последние для сходимости
    policies_b_full = []
    
    # Для графиков - сэмплированные данные
    policies_a_sampled = []
    policies_b_sampled = []
    rewards_a = []
    rewards_b = []
    q_values_a = []
    q_values_b = []
    
    a = agent_a.start_episode()
    b = agent_b.start_episode()
    
    for step in range(steps):
        r_a = game.payoff_player0(a, b)
        r_b = game.payoff_player1(a, b)
        next_a = agent_a.choose_action()
        next_b = agent_b.choose_action()
        agent_a.step(r_a, next_action=next_a)
        agent_b.step(r_b, next_action=next_b)
        
        # Сохраняем для проверки сходимости (только последние convergence_window)
        if step >= steps - convergence_window:
            policies_a_full.append(agent_a.get_action_probs())
            policies_b_full.append(agent_b.get_action_probs())
        
        # Сэмплированные данные для графиков
        if save_plot and (step % sample_interval == 0 or step == steps - 1):
            policies_a_sampled.append(agent_a.get_action_probs())
            policies_b_sampled.append(agent_b.get_action_probs())
            rewards_a.append(r_a)
            rewards_b.append(r_b)
            q_values_a.append(agent_a.Q[0].copy())
            q_values_b.append(agent_b.Q[0].copy())
        
        a, b = next_a, next_b
    
    # Проверяем сходимость с передачей сетки действий
    actions = game.grid()
    conv_a = check_convergence(policies_a_full, window=min(convergence_window, len(policies_a_full)),
                               threshold_converged=1e-3,
                               actions=actions)
    conv_b = check_convergence(policies_b_full, window=min(convergence_window, len(policies_b_full)),
                               threshold_converged=1e-3,
                               actions=actions)
    
    # Создаём визуализацию, если нужно
    if save_plot and output_dir:
        plot_dir = output_dir / 'individual_plots'
        plot_dir.mkdir(exist_ok=True)
        
        filename = f"beta{beta}_gamma{gamma}.png"
        save_path = plot_dir / filename

        try:
            plot_comprehensive_results(
                rewards_a=rewards_a,
                rewards_b=rewards_b,
                q_values_a=q_values_a,
                q_values_b=q_values_b,
                policies_a=policies_a_sampled,
                policies_b=policies_b_sampled,
                action_values=actions,
                window_size=min(100, len(rewards_a) // 10),
                save_path=str(save_path),
                base_function=base_function,
                beta=beta,
                gamma=gamma,
                steps=steps,
                alpha=alpha,
                n=n,
                game_params={'r': params.r, 'p': params.p, 'q': params.q, 's': params.s}
            )
            plt.close('all')  # Освобождаем память
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании графика для β={beta}, γ={gamma}: {e}")
    
    # Очищаем большие массивы
    del policies_a_full, policies_b_full
    del policies_a_sampled, policies_b_sampled
    del rewards_a, rewards_b, q_values_a, q_values_b
    gc.collect()
    
    return {
        'beta': beta,
        'gamma': gamma,
        'conv_a': conv_a,
        'conv_b': conv_b
    }


def make_serializable(obj):
    """Рекурсивно преобразует numpy arrays в списки для JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def print_statistics(results: list):
    """Выводит детальную статистику по результатам экспериментов."""
    
    # Краткая статистика сходимости
    converged_a_count = sum(1 for r in results if r['conv_a']['converged'])
    converged_b_count = sum(1 for r in results if r['conv_b']['converged'])
    both_converged = sum(1 for r in results if r['conv_a']['converged'] and r['conv_b']['converged'])
    
    print("\n" + "="*70)
    print("📊 СТАТИСТИКА СХОДИМОСТИ")
    print("="*70)
    print(f"Agent A сошёлся: {converged_a_count}/{len(results)} ({100*converged_a_count/len(results):.1f}%)")
    print(f"Agent B сошёлся: {converged_b_count}/{len(results)} ({100*converged_b_count/len(results):.1f}%)")
    print(f"Оба сошлись:     {both_converged}/{len(results)} ({100*both_converged/len(results):.1f}%)")
    
    # Детальная статистика для сошедшихся экспериментов
    converged_both = [r for r in results if r['conv_a']['converged'] and r['conv_b']['converged']]
    
    if len(converged_both) > 0:
        print(f"\n🔬 Анализ {len(converged_both)} экспериментов, где оба агента сошлись:\n")
        
        # === СТАТИСТИКА 1: Распределение по значениям действий ===
        print("1️⃣ К каким действиям сходятся агенты?")
        print("-" * 70)
        
        actions_a = [r['conv_a']['argmax_value'] for r in converged_both]
        actions_b = [r['conv_b']['argmax_value'] for r in converged_both]
        
        print(f"Agent A:")
        print(f"  Среднее действие: {np.mean(actions_a):.3f}")
        print(f"  Медиана:          {np.median(actions_a):.3f}")
        print(f"  Стд. отклонение:  {np.std(actions_a):.3f}")
        print(f"  Диапазон:         [{np.min(actions_a):.3f}, {np.max(actions_a):.3f}]")
        
        print(f"\nAgent B:")
        print(f"  Среднее действие: {np.mean(actions_b):.3f}")
        print(f"  Медиана:          {np.median(actions_b):.3f}")
        print(f"  Стд. отклонение:  {np.std(actions_b):.3f}")
        print(f"  Диапазон:         [{np.min(actions_b):.3f}, {np.max(actions_b):.3f}]")
        
        # === СТАТИСТИКА 2: Качество сходимости ===
        print("\n\n2️⃣ Насколько уверенно сходятся? (вероятность доминирующего действия)")
        print("-" * 70)
        
        probs_a = [r['conv_a']['max_prob'] for r in converged_both]
        probs_b = [r['conv_b']['max_prob'] for r in converged_both]
        
        print(f"Agent A:")
        print(f"  Средняя max_prob: {np.mean(probs_a):.4f}")
        print(f"  Медиана:          {np.median(probs_a):.4f}")
        print(f"  Мин/Макс:         [{np.min(probs_a):.4f}, {np.max(probs_a):.4f}]")
        print(f"  Слабая сходимость (p<0.5): {sum(1 for p in probs_a if p < 0.5)}/{len(probs_a)}")
        print(f"  Сильная сходимость (p>0.8): {sum(1 for p in probs_a if p > 0.8)}/{len(probs_a)}")
        
        print(f"\nAgent B:")
        print(f"  Средняя max_prob: {np.mean(probs_b):.4f}")
        print(f"  Медиана:          {np.median(probs_b):.4f}")
        print(f"  Мин/Макс:         [{np.min(probs_b):.4f}, {np.max(probs_b):.4f}]")
        print(f"  Слабая сходимость (p<0.5): {sum(1 for p in probs_b if p < 0.5)}/{len(probs_b)}")
        print(f"  Сильная сходимость (p>0.8): {sum(1 for p in probs_b if p > 0.8)}/{len(probs_b)}")
        
        # === СТАТИСТИКА 3: Стабильность ===
        print("\n\n3️⃣ Стабильность политики (std в окне сходимости)")
        print("-" * 70)
        
        stds_a = [r['conv_a']['max_std'] for r in converged_both]
        stds_b = [r['conv_b']['max_std'] for r in converged_both]
        
        print(f"Agent A:")
        print(f"  Средний std: {np.mean(stds_a):.6f}")
        print(f"  Медиана:     {np.median(stds_a):.6f}")
        print(f"  Макс. std:   {np.max(stds_a):.6f}")
        
        print(f"\nAgent B:")
        print(f"  Средний std: {np.mean(stds_b):.6f}")
        print(f"  Медиана:     {np.median(stds_b):.6f}")
        print(f"  Макс. std:   {np.max(stds_b):.6f}")
        
        # === СТАТИСТИКА 4: Кооперация vs Дефект ===
        print("\n\n4️⃣ Анализ стратегий: кооперация vs дефект")
        print("-" * 70)
        
        cooperation_threshold = 0.5
        
        coop_a = sum(1 for r in converged_both if r['conv_a']['argmax_value'] > cooperation_threshold)
        defect_a = sum(1 for r in converged_both if r['conv_a']['argmax_value'] <= cooperation_threshold)
        
        coop_b = sum(1 for r in converged_both if r['conv_b']['argmax_value'] > cooperation_threshold)
        defect_b = sum(1 for r in converged_both if r['conv_b']['argmax_value'] <= cooperation_threshold)
        
        print(f"Agent A (порог кооперации = {cooperation_threshold}):")
        print(f"  Кооперация (action > {cooperation_threshold}): {coop_a}/{len(converged_both)} ({100*coop_a/len(converged_both):.1f}%)")
        print(f"  Дефект (action ≤ {cooperation_threshold}):     {defect_a}/{len(converged_both)} ({100*defect_a/len(converged_both):.1f}%)")
        
        print(f"\nAgent B (порог кооперации = {cooperation_threshold}):")
        print(f"  Кооперация (action > {cooperation_threshold}): {coop_b}/{len(converged_both)} ({100*coop_b/len(converged_both):.1f}%)")
        print(f"  Дефект (action ≤ {cooperation_threshold}):     {defect_b}/{len(converged_both)} ({100*defect_b/len(converged_both):.1f}%)")
        
        # Совместные стратегии
        both_coop = sum(1 for r in converged_both 
                        if r['conv_a']['argmax_value'] > cooperation_threshold 
                        and r['conv_b']['argmax_value'] > cooperation_threshold)
        both_defect = sum(1 for r in converged_both 
                          if r['conv_a']['argmax_value'] <= cooperation_threshold 
                          and r['conv_b']['argmax_value'] <= cooperation_threshold)
        mixed = len(converged_both) - both_coop - both_defect
        
        print(f"\nСовместные стратегии:")
        print(f"  Взаимная кооперация (оба > {cooperation_threshold}):  {both_coop}/{len(converged_both)} ({100*both_coop/len(converged_both):.1f}%)")
        print(f"  Взаимный дефект (оба ≤ {cooperation_threshold}):      {both_defect}/{len(converged_both)} ({100*both_defect/len(converged_both):.1f}%)")
        print(f"  Смешанные стратегии:               {mixed}/{len(converged_both)} ({100*mixed/len(converged_both):.1f}%)")
        
        # === ПРИМЕРЫ СХОДИМОСТИ ===
        print("\n\n🎯 Примеры сходимости (первые 10 случаев):")
        print("-" * 70)
        count = 0
        for r in converged_both:
            if count >= 10:
                break
            beta, gamma = r['beta'], r['gamma']
            val_a = r['conv_a']['argmax_value']
            val_b = r['conv_b']['argmax_value']
            prob_a = r['conv_a']['max_prob']
            prob_b = r['conv_b']['max_prob']
            
            # Вычисляем base_function для сравнения с финальным распределением
            base_func_val = base_function(val_a, beta)
            
            print(f"  β={beta:.2f}, γ={gamma:.2f}: A→{val_a:.3f} (p={prob_a:.3f}), B→{val_b:.3f} (p={prob_b:.3f})")
            print(f"    base_function({val_a:.3f}, β={beta:.2f}) = {base_func_val:.6f}")
            count += 1
    
    print("\n" + "="*70)


def main():
    """Главная функция скрипта."""
    
    B_sample = solve_beta_newton(3, 0, 5, 1, 0.8, max_iter=10000)
    print(B_sample)

    print("="*70)
    print("🔬 GRID SEARCH ПО ПАРАМЕТРАМ BETA И GAMMA")
    print("="*70)

    # === ПАРАМЕТРЫ СЕТКИ ===
    # Расширенная сетка (как в ноутбуке)
    # Добавляем оптимальное beta в список
    beta_values = [0.8, 0.9, 1.0, 2.0, 5.0]
    
    gamma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    
    # === ПАРАМЕТРЫ ИГРЫ (Prisoner's Dilemma-like) ===
    params = PayoffParams(r=r_base, p=p_base, q=q_base, s=s_base)
    
    # === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
    n = 100  # дискретизация [0,1]
    steps = 200000  # УВЕЛИЧЕНО с 50000 до 200000
    alpha = 0.01
    init_mode = 'uniform'
    seed = 42
    
    # === НАСТРОЙКИ ВЫВОДА ===
    create_individual_plots = True  # Создавать ли графики для каждой пары
    
    print(f"\n📋 Параметры grid search:")
    print(f"  Beta values ({len(beta_values)}): {beta_values}")
    print(f"  Gamma values ({len(gamma_values)}): {gamma_values}")
    print(f"  Всего экспериментов: {len(beta_values) * len(gamma_values)}")
    
    print(f"\n📋 Параметры обучения:")
    print(f"  Дискретизация: n={n}")
    print(f"  Шаги обучения: steps={steps:,}")
    print(f"  Learning rate: alpha={alpha}")
    print(f"  Init mode: {init_mode}")
    print(f"  Random seed: {seed}")
    
    print(f"\n📋 Параметры игры:")
    print(f"  r={params.r}, p={params.p}, q={params.q}, s={params.s}")
    
    print(f"\n📊 Визуализация:")
    print(f"  Индивидуальные графики: {'✓ Да' if create_individual_plots else '✗ Нет'}")
    
    # Создаём директорию для результатов
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    if create_individual_plots:
        plot_dir = output_dir / 'individual_plots'
        plot_dir.mkdir(exist_ok=True)
        print(f"  Графики сохраняются в: {plot_dir.absolute()}")
    
    # === ЗАПУСК GRID SEARCH ===
    print(f"\n🚀 Запуск grid search...")
    print(f"⏱️  Это может занять значительное время (200000 шагов на эксперимент)")
    print(f"💾 Оптимизация памяти: сэмплирование данных для графиков\n")
    
    results = []
    total = len(beta_values) * len(gamma_values)
    
    pbar = tqdm(total=total, desc='Grid search', unit='exp', leave=True)
    converged_a = 0
    converged_b = 0
    both_converged = 0
    
    for beta_param in beta_values:
        for gamma in gamma_values:
            # Решаем уравнение для текущего beta, чтобы найти B
            if gamma > 0.5:
                steps_ = 10 * steps
            else:
                steps_ = steps
            B = solve_beta_newton(R=r_base, S=q_base, T=p_base, P=s_base, 
                                beta=beta_param, initial_guess=beta_param)
            
            def base_B_function(x: float) -> float:
                return base_function(x, B)

            result = run_single_experiment(
                beta=beta_param, gamma=gamma, params=params,
                n=n, steps=steps_, alpha=alpha,
                init_mode=init_mode, seed=seed,
                save_plot=create_individual_plots,
                output_dir=output_dir,
                base_function=base_B_function
            )
            results.append(result)
            
            a_conv = bool(result['conv_a']['converged'])
            b_conv = bool(result['conv_b']['converged'])
            
            if a_conv:
                converged_a += 1
            if b_conv:
                converged_b += 1
            if a_conv and b_conv:
                both_converged += 1
            
            status = f"A:{'✓' if a_conv else '✗'} B:{'✓' if b_conv else '✗'}"
            pbar.update(1)
            pbar.set_description(f"β={beta_param:.2f} γ={gamma:.2f} B={B:.3f}")
            pbar.set_postfix({
                'status': status,
                'A_done': f"{converged_a}/{len(results)}",
                'B_done': f"{converged_b}/{len(results)}",
                'both': both_converged
            })
            
            # Принудительная сборка мусора каждые 10 экспериментов
            if len(results) % 10 == 0:
                gc.collect()
    
    pbar.close()
    
    print(f"\n✅ Завершено {len(results)} экспериментов")
    
    # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
    output_file = output_dir / f'convergence_grid_search_steps{steps}.json'
    
    results_serializable = make_serializable(results)
    
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
            'results': results_serializable
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в {output_file}")
    
    # === СТАТИСТИКА ===
    print_statistics(results)
    
    # === ВИЗУАЛИЗАЦИЯ ===
    print(f"\n📊 Создание общих heat-map визуализаций...")
    
    # 1. Совместная heat-map
    joint_heatmap_path = output_dir / f'joint_convergence_heatmap_steps{steps}.png'
    plot_joint_convergence_heatmap(results, beta_values=beta_values, gamma_values=gamma_values,
                                   title=f'Совместная сходимость агентов A и B\n(steps={steps:,}, аннотации: A:значение, B:значение)',
                                   save_path=str(joint_heatmap_path))
    print(f"  ✅ Сохранено: {joint_heatmap_path}")
    
    # 2. Отдельные heat-map для A и B
    action_heatmaps_path = output_dir / f'action_value_heatmaps_steps{steps}.png'
    plot_action_value_heatmaps(results, beta_values=beta_values, gamma_values=gamma_values,
                              save_path=str(action_heatmaps_path))
    print(f"  ✅ Сохранено: {action_heatmaps_path}")
    
    print("\n" + "="*70)
    print("✨ ЗАВЕРШЕНО!")
    print("="*70)
    print(f"\n📁 Результаты сохранены в директории: {output_dir.absolute()}")
    print(f"\n📊 Файлы:")
    print(f"  • JSON с данными: {output_file.name}")
    print(f"  • Совместная heat-map: {joint_heatmap_path.name}")
    print(f"  • Отдельные heat-map: {action_heatmaps_path.name}")
    if create_individual_plots:
        plot_dir = output_dir / 'individual_plots'
        num_plots = len(list(plot_dir.glob('*.png'))) if plot_dir.exists() else 0
        print(f"  • Индивидуальные графики: {num_plots} файлов в {plot_dir.name}/")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
