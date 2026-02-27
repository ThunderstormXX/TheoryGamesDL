import sys
import os
import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Оптимизации для A100 (Ampere)
# Включаем TensorFloat-32 (TF32) для значительного ускорения матричных вычислений на A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Включаем авто-тюнинг алгоритмов cuDNN
torch.backends.cudnn.benchmark = True

# Добавляем текущую директорию в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_structure import SmallWorldGraph
from gpu_version.gpu_learner import GPUQLearner
from gpu_version.gpu_reward_model import GPUPPReward
from gpu_version.gpu_game_launcher import GPUMonteKarloPairGame
from gpu_utils import gpu_config

def run_single_simulation(b, gamma, episodes=10000, n_nodes=100, k_neighbors=4, p_rewiring=0.1, k_anchors=1, seed=None):
    """Запускает одну симуляцию с заданными параметрами и возвращает историю доли кооперации."""
    # Устанавливаем seed для воспроизводимости или разнообразия
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    graph = SmallWorldGraph(n=n_nodes, k=k_neighbors, p=p_rewiring)
    
    learners = [GPUQLearner(
        action_space_size=2,
        learning_rate=0.2,
        discount_factor=gamma,
        strategy='boltzmann',
        max_states=n_nodes + 1 
    ) for _ in range(n_nodes)]
    
    reward_model = GPUPPReward(b=b, c=1.0)
    game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=k_anchors)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    history = []
    # Если запуск не в параллельном режиме (или для отладки), можно добавить tqdm тут
    # Но так как мы запускаем много процессов, лучше tqdm снаружи
    for _ in range(episodes):
        game.round()
        # Сохраняем долю кооперации на каждом шаге
        coop_rate = float(game.strategies.mean().item())
        history.append(coop_rate)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    elapsed_time = time.time() - start_time
    
    # Очистка памяти GPU после симуляции
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return history, elapsed_time

def run_single_simulation_wrapper(args):
    """Обертка для запуска в ProcessPoolExecutor"""
    b, gamma, episodes, n_nodes, k_neighbors, p_rewiring, k_anchors, seed = args
    history, elapsed = run_single_simulation(b, gamma, episodes, n_nodes, k_neighbors, p_rewiring, k_anchors, seed)
    return b, gamma, history, elapsed

def plot_dynamics_with_std(results_dict, param_name, save_path, title):
    """Отрисовывает графики динамики кооперации с усреднением и стандартным отклонением"""
    plt.figure(figsize=(12, 8))
    
    sorted_items = sorted(results_dict.items(), key=lambda x: x[0])
    
    for param_val, histories in sorted_items:
        # histories - список списков (N_repeats x episodes)
        if not histories:
            continue
            
        histories_arr = np.array(histories) # Shape: (100, 10000)
        
        mean_history = np.mean(histories_arr, axis=0)
        std_history = np.std(histories_arr, axis=0)
        
        # Для сглаживания графика
        window = max(1, len(mean_history) // 100)
        if window > 1:
            # Сглаживаем среднее
            smoothed_mean = np.convolve(mean_history, np.ones(window)/window, mode='valid')
            smoothed_std = np.convolve(std_history, np.ones(window)/window, mode='valid')
            
            x_range = np.arange(len(smoothed_mean))
            # Отрисовка
            line, = plt.plot(x_range, smoothed_mean, label=f'{param_name} = {param_val}')
            plt.fill_between(x_range, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.2, color=line.get_color())
        else:
            x_range = np.arange(len(mean_history))
            line, = plt.plot(x_range, mean_history, label=f'{param_name} = {param_val}')
            plt.fill_between(x_range, mean_history - std_history, mean_history + std_history, alpha=0.2, color=line.get_color())
            
    plt.title(title, fontsize=14)
    plt.xlabel('Эпизоды', fontsize=12)
    plt.ylabel('Доля кооперации', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_gamma_experiment(n_nodes=1000, episodes=10000, n_repeats=100):
    print("\n" + "="*70)
    print(f"ЭКСПЕРИМЕНТ 1: Зависимость от gamma (Mean ± Std)")
    print(f"Параметры: N={n_nodes}, episodes={episodes}, b=3.0, Repeats={n_repeats}")
    print("="*70)
    
    gammas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    results = {g: [] for g in gammas}
    final_stats = {}
    
    # Формируем задачи: перебираем gamma, и для каждого делаем n_repeats повторений
    tasks = []
    for g in gammas:
        for i in range(n_repeats):
            # Уникальный seed для каждой задачи
            seed = int(time.time()) + i * 1000 + int(g * 100)
            tasks.append((3.0, g, episodes, n_nodes, 4, 0.1, 1, seed))
            
    start_total = time.time()
    
    # 40GB VRAM / ~4GB per task ~= 10 runs max. Safe with 8.
    max_workers = 8 
    
    print(f"Запуск {len(tasks)} задач на {max_workers} процессах...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation_wrapper, task): task for task in tasks}
        
        # Используем tqdm для отображения прогресса
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations (Gamma)"):
            b_val, gamma_val, history, elapsed = future.result()
            results[gamma_val].append(history)
            
    print(f"Общее время эксперимента 1: {time.time() - start_total:.2f} сек")
    
    # Сохранение результатов и статистики
    os.makedirs("results/N_anchors/gamma_exp_stat", exist_ok=True)
    
    # Вычисляем финальные средние значения кооперации (последний эпизод)
    for g, histories in results.items():
        if not histories:
            continue
        finals = [h[-1] for h in histories]
        final_stats[g] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals))
        }
    
    with open("results/N_anchors/gamma_exp_stat/gamma_stats.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        
    # Отрисовка графика
    plot_dynamics_with_std(
        results, 
        'gamma', 
        "results/N_anchors/gamma_exp_stat/gamma_dynamics_std.png",
        f"Динамика кооперации (mean ± std) при разных gamma (b=3.0, N={n_nodes}, {n_repeats} runs)"
    )
    print("Результаты сохранены в results/N_anchors/gamma_exp_stat/")

def run_b_experiment(n_nodes=1000, episodes=10000, n_repeats=100):
    print("\n" + "="*70)
    print(f"ЭКСПЕРИМЕНТ 2: Зависимость от b (Mean ± Std)")
    print(f"Параметры: N={n_nodes}, episodes={episodes}, gamma=0.9, Repeats={n_repeats}")
    print("="*70)
    
    bs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {b_: [] for b_ in bs}
    final_stats = {}
    
    tasks = []
    for b_val in bs:
        for i in range(n_repeats):
            seed = int(time.time()) + i * 2000 + int(b_val * 100)
            tasks.append((b_val, 0.9, episodes, n_nodes, 4, 0.1, 1, seed))
    
    start_total = time.time()
    
    max_workers = 8
    
    print(f"Запуск {len(tasks)} задач на {max_workers} процессах...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation_wrapper, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations (b)"):
            b_res, gamma_res, history, elapsed = future.result()
            results[b_res].append(history)
            
    print(f"Общее время эксперимента 2: {time.time() - start_total:.2f} сек")
    
    os.makedirs("results/N_anchors/b_exp_stat", exist_ok=True)
    
    for b_val, histories in results.items():
        if not histories:
            continue
        finals = [h[-1] for h in histories]
        final_stats[b_val] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals))
        }
        
    with open("results/N_anchors/b_exp_stat/b_stats.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        
    plot_dynamics_with_std(
        results, 
        'b', 
        "results/N_anchors/b_exp_stat/b_dynamics_std.png",
        f"Динамика кооперации (mean ± std) при разных b (gamma=0.9, N={n_nodes}, {n_repeats} runs)"
    )
    print("Результаты сохранены в results/N_anchors/b_exp_stat/")

if __name__ == "__main__":
    # Обязательно для PyTorch + Multiprocessing с CUDA
    mp.set_start_method('spawn', force=True)
    
    gpu_config.print_info()
    
    # Запуск экспериментов
    # n_nodes=1000 для реального эксперимента
    run_gamma_experiment(n_nodes=1000, episodes=10000, n_repeats=100)
    run_b_experiment(n_nodes=1000, episodes=10000, n_repeats=100)
    
    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
