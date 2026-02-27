import sys
import os
import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def run_single_simulation(b, gamma, episodes=10000, n_nodes=1000, k_neighbors=1, p_rewiring=0.1, k_anchors=1):
    """Запускает одну симуляцию с заданными параметрами и возвращает историю доли кооперации."""
    # Устанавливаем seed для каждого процесса, чтобы избежать одинаковых случайных чисел
    torch.manual_seed(int(time.time() * 1000) % 100000)
    np.random.seed(int(time.time() * 1000) % 100000)
    
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
    b, gamma, episodes, n_nodes, k_neighbors, p_rewiring, k_anchors = args
    history, elapsed = run_single_simulation(b, gamma, episodes, n_nodes, k_neighbors, p_rewiring, k_anchors)
    return b, gamma, history, elapsed

def plot_dynamics(results_dict, param_name, save_path, title):
    """Отрисовывает графики динамики кооперации"""
    plt.figure(figsize=(12, 8))
    
    for param_val, history in sorted(results_dict.items()):
        # Для сглаживания графика можно использовать скользящее среднее, если эпизодов много
        window = max(1, len(history) // 100)
        if window > 1:
            smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=f'{param_name} = {param_val}')
        else:
            plt.plot(history, label=f'{param_name} = {param_val}')
            
    plt.title(title, fontsize=14)
    plt.xlabel('Эпизоды', fontsize=12)
    plt.ylabel('Доля кооперации', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_gamma_experiment(n_nodes=1000, episodes=10000):
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: Зависимость от gamma (ПАРАЛЛЕЛЬНО НА A100)")
    print(f"Параметры: Small World, n_nodes={n_nodes}, k_anchors=1, episodes={episodes}, b=3.0")
    print("="*70)
    
    gammas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    results = {}
    final_results = {}
    
    # Подготавливаем аргументы для параллельного запуска
    tasks = [(3.0, g, episodes, n_nodes, 1, 0.1, 1) for g in gammas]
    
    start_total = time.time()
    
    # Запускаем параллельно (A100 40GB легко потянет 6-10 процессов с графами по 1000-5000 узлов)
    max_workers = min(len(gammas), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation_wrapper, task): task for task in tasks}
        
        for future in as_completed(futures):
            b, gamma, history, elapsed = future.result()
            final_coop = history[-1]
            print(f"  [Gamma={gamma}] -> Доля кооперации: {final_coop:.4f} (Время: {elapsed:.2f} сек)")
            results[gamma] = history
            final_results[gamma] = final_coop
            
    print(f"Общее время эксперимента 1: {time.time() - start_total:.2f} сек")
    
    # Сохранение результатов
    os.makedirs("results/N_anchors/gamma_exp", exist_ok=True)
    
    # Сохраняем финальные значения
    with open("results/N_anchors/gamma_exp/gamma_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
        
    # Сохраняем полную историю
    with open("results/N_anchors/gamma_exp/gamma_history.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Отрисовка графика
    plot_dynamics(
        results, 
        'gamma', 
        "results/N_anchors/gamma_exp/gamma_dynamics.png",
        f"Динамика кооперации при разных значениях gamma (b=3.0, N={n_nodes})"
    )
    print("Результаты и графики сохранены в results/N_anchors/gamma_exp/")

def run_b_experiment(n_nodes=1000, episodes=10000):
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: Зависимость от b (ПАРАЛЛЕЛЬНО НА A100)")
    print(f"Параметры: Small World, n_nodes={n_nodes}, k_anchors=1, episodes={episodes}, gamma=0.9")
    print("="*70)
    
    bs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {}
    final_results = {}
    
    tasks = [(b, 0.9, episodes, n_nodes, 1, 0.1, 1) for b in bs]
    
    start_total = time.time()
    
    max_workers = min(len(bs), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation_wrapper, task): task for task in tasks}
        
        for future in as_completed(futures):
            b, gamma, history, elapsed = future.result()
            final_coop = history[-1]
            print(f"  [b={b}] -> Доля кооперации: {final_coop:.4f} (Время: {elapsed:.2f} сек)")
            results[b] = history
            final_results[b] = final_coop
            
    print(f"Общее время эксперимента 2: {time.time() - start_total:.2f} сек")
    
    # Сохранение результатов
    os.makedirs("results/N_anchors/b_exp", exist_ok=True)
    
    # Сохраняем финальные значения
    with open("results/N_anchors/b_exp/b_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
        
    # Сохраняем полную историю
    with open("results/N_anchors/b_exp/b_history.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Отрисовка графика
    plot_dynamics(
        results, 
        'b', 
        "results/N_anchors/b_exp/b_dynamics.png",
        f"Динамика кооперации при разных значениях b (gamma=0.9, N={n_nodes})"
    )
    print("Результаты и графики сохранены в results/N_anchors/b_exp/")

if __name__ == "__main__":
    # Обязательно для PyTorch + Multiprocessing с CUDA
    mp.set_start_method('spawn', force=True)
    
    gpu_config.print_info()
    
    # Запуск обоих экспериментов (увеличили n_nodes до 1000, чтобы загрузить A100)
    run_gamma_experiment(n_nodes=1000, episodes=10000)
    run_b_experiment(n_nodes=1000, episodes=10000)
    
    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
