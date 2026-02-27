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

from gpu_version.batched_gpu import BatchedGPUMonteKarloPairGame
from gpu_utils import gpu_config

def run_batched_experiment(b, gamma, episodes, n_nodes, k_anchors, batch_size, seed=None):
    """
    Запускает BATCH_SIZE параллельных симуляций на одном GPU процессе.
    Returns: (batch_size, episodes) history array
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    start_time = time.time()
    
    # Конфигурация
    graph_params = {'n': n_nodes, 'k': 4, 'p': 0.1}
    learner_params = {
        'learning_rate': 0.2,
        'discount_factor': gamma,
        'strategy': 'boltzmann',
        'max_states': n_nodes + 1
    }
    reward_params = {'b': b, 'c': 1.0}
    
    game = BatchedGPUMonteKarloPairGame(
        batch_size=batch_size,
        n_agents_per_sim=n_nodes,
        graph_params=graph_params,
        learner_params=learner_params,
        reward_params=reward_params,
        k_anchors=k_anchors
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Основной цикл обучения
    # Используем tqdm (или нет, если он внутри ProcessPoolExecutor?)
    # Лучше без tqdm внутри процесса
    for _ in range(episodes):
        game.round()
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    elapsed = time.time() - start_time
    history = game.get_history() # (batch_size, episodes)
    
    # Очистка памяти?
    del game
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return b, gamma, history, elapsed

def run_batched_experiment_wrapper(args):
    return run_batched_experiment(*args)

def plot_dynamics_with_std(results_dict, param_name, save_path, title):
    """Отрисовывает графики динамики кооперации с усреднением и стандартным отклонением"""
    plt.figure(figsize=(12, 8))
    
    sorted_items = sorted(results_dict.items(), key=lambda x: x[0])
    
    for param_val, histories in sorted_items:
        # histories: список списков или один большой массив.
        # run_batched_experiment возвращает массив.
        # Мы их собираем в список массивов.
        if not histories:
            continue
            
        # Concatenate arrays along batch dimension (dim 0)
        # Each item in histories is (batch_size, episodes)
        full_history = np.vstack(histories) # (total_sims, episodes)
        
        mean_history = np.mean(full_history, axis=0)
        std_history = np.std(full_history, axis=0)
        
        window = max(1, len(mean_history) // 100)
        if window > 1:
            smoothed_mean = np.convolve(mean_history, np.ones(window)/window, mode='valid')
            smoothed_std = np.convolve(std_history, np.ones(window)/window, mode='valid')
            x_range = np.arange(len(smoothed_mean))
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

def run_gamma_experiment(n_nodes=1000, episodes=10000, total_repeats=100, batch_size=20):
    print("\n" + "="*70)
    print(f"ЭКСПЕРИМЕНТ 1: Gamma (Batched)")
    print(f"Параметры: N={n_nodes}, b=3.0, Total Repeats={total_repeats}, Batch Size={batch_size}")
    print("="*70)
    
    gammas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    results = {g: [] for g in gammas}
    final_stats = {}
    
    # Количество задач (процессов) = total_repeats / batch_size
    n_tasks = (total_repeats + batch_size - 1) // batch_size
    
    tasks = []
    for g in gammas:
        for i in range(n_tasks):
            # Реальное количество повторений в этом батче (может быть меньше в последнем)
            current_batch_size = min(batch_size, total_repeats - i*batch_size)
            if current_batch_size <= 0: break
            
            seed = int(time.time()) + i*1000 + int(g*100)
            tasks.append((3.0, g, episodes, n_nodes, 1, current_batch_size, seed))
            
    start_total = time.time()
    
    # 40GB A100 allows large batches.
    # Если batch_size=20, то 20 графов * (1000*1000 float) ~ 20 * 4MB = 80MB adjacency.
    # Q-table: 20 * 1000 agents * 1000 states * 2 actions * 4 bytes = 160MB.
    # Итого ~250-300MB на батч.
    # Можно запускать 10-20 процессов параллельно.
    max_workers = 8
    
    print(f"Запуск {len(tasks)} задач (батчей) на {max_workers} процессах...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_batched_experiment_wrapper, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations (Gamma)"):
            try:
                b_val, gamma_val, history_arr, elapsed = future.result()
                results[gamma_val].append(history_arr)
            except Exception as e:
                print(f"Ошибка в процессе: {e}")
                import traceback
                traceback.print_exc()

    print(f"Общее время: {time.time() - start_total:.2f} сек")
    
    os.makedirs("results/N_anchors/gamma_exp_stat", exist_ok=True)
    
    # Stats
    for g, history_list in results.items():
        if not history_list: continue
        full_hist = np.vstack(history_list) # (Total_repeats, episodes)
        finals = full_hist[:, -1]
        final_stats[g] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals))
        }
        
    with open("results/N_anchors/gamma_exp_stat/gamma_stats.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        
    plot_dynamics_with_std(results, 'gamma', "results/N_anchors/gamma_exp_stat/gamma_dynamics_std.png", f"Gamma Exp (N={n_nodes})")

def run_b_experiment(n_nodes=1000, episodes=10000, total_repeats=100, batch_size=20):
    print("\n" + "="*70)
    print(f"ЭКСПЕРИМЕНТ 2: B (Batched)")
    print(f"Параметры: N={n_nodes}, gamma=0.9, Total Repeats={total_repeats}, Batch Size={batch_size}")
    print("="*70)
    
    bs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {b_: [] for b_ in bs}
    final_stats = {}
    
    n_tasks = (total_repeats + batch_size - 1) // batch_size
    
    tasks = []
    for b_val in bs:
        for i in range(n_tasks):
            current_batch_size = min(batch_size, total_repeats - i*batch_size)
            if current_batch_size <= 0: break
            
            seed = int(time.time()) + i*2000 + int(b_val*100)
            tasks.append((b_val, 0.9, episodes, n_nodes, 1, current_batch_size, seed))
            
    start_total = time.time()
    max_workers = 8
    
    print(f"Запуск {len(tasks)} задач (батчей) на {max_workers} процессах...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_batched_experiment_wrapper, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations (B)"):
            try:
                b_res, gamma_res, history_arr, elapsed = future.result()
                results[b_res].append(history_arr)
            except Exception as e:
                print(f"Ошибка в процессе: {e}")

    print(f"Общее время: {time.time() - start_total:.2f} сек")
    
    os.makedirs("results/N_anchors/b_exp_stat", exist_ok=True)
    
    for b_val, history_list in results.items():
        if not history_list: continue
        full_hist = np.vstack(history_list)
        finals = full_hist[:, -1]
        final_stats[b_val] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals))
        }
        
    with open("results/N_anchors/b_exp_stat/b_stats.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        
    plot_dynamics_with_std(results, 'b', "results/N_anchors/b_exp_stat/b_dynamics_std.png", f"B Exp (N={n_nodes})")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    gpu_config.print_info()
    
    # Параметры для максимальной загрузки
    # n_nodes=1000 - агентов
    # total_repeats=100 - всего экспериментов
    # batch_size=25 - экспериментов в одном процессе (на одной VRAM)
    # max_workers=8 - процессов параллельно
    # Итого: 8 процессов * 25 симуляций = 200 симуляций одновременно? Нет.
    # Мы разбиваем 100 повторений на куски по 25. Это 4 задачи на параметр.
    # Если параметров 6 (гамма), то всего 24 задачи.
    # 8 воркеров будут брать их по очереди.
    
    # Чтобы еще сильнее загрузить, можно увеличить Batch Size до 50 или 100, 
    # если памяти хватает.
    # При N=1000, 100 симуляций занимают:
    # Adj: 100 * 4MB = 400MB
    # Q: 100 * 1000 * 1000 * 2 * 4 = 800MB
    # Misc: ~200MB. Итого ~1.5 GB на процесс.
    # 8 процессов * 1.5GB = 12 GB.
    # Можно смело ставить batch_size=100 (весь эксперимент в одном батче) и запускать параллельно разные гаммы.
    
    run_gamma_experiment(n_nodes=1000, episodes=10000, total_repeats=100, batch_size=50) # 2 задачи на параметр
    run_b_experiment(n_nodes=1000, episodes=10000, total_repeats=100, batch_size=50)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
