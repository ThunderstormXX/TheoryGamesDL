import sys
import os
import time
import torch
import numpy as np

# Добавляем текущую директорию в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_structure import SmallWorldGraph
from gpu_version.gpu_learner import GPUQLearner
from gpu_version.gpu_reward_model import GPUPPReward
from gpu_version.gpu_game_launcher import GPUMonteKarloPairGame
from gpu_version.gpu_utils import gpu_config

def run_experiment():
    print("="*70)
    print("Запуск эксперимента: Small World, k_anchors=1, episodes=10000 (GPU)")
    print("="*70)
    
    # Вывод информации о GPU
    gpu_config.print_info()
    
    # Параметры графа Small World
    N_NODES = 100  # Количество агентов (можете изменить)
    K_NEIGHBORS = 4 # Каждый узел соединен с k ближайшими соседями
    P_REWIRING = 0.1 # Вероятность переподключения
    
    print(f"\nСоздание графа Small World (N={N_NODES}, k={K_NEIGHBORS}, p={P_REWIRING})...")
    graph = SmallWorldGraph(n=N_NODES, k=K_NEIGHBORS, p=P_REWIRING)
    
    # Параметры обучения
    EPISODES = 10000
    K_ANCHORS = 1
    
    # Инициализация агентов (GPUQLearner)
    # max_states должно быть больше максимальной степени узла в графе
    # В Small World максимальная степень обычно не сильно превышает K_NEIGHBORS, 
    # но для безопасности можно поставить N_NODES + 1
    print("Инициализация агентов...")
    learners = [GPUQLearner(
        action_space_size=2,
        learning_rate=0.2,
        discount_factor=0.9,
        strategy='boltzmann',
        max_states=N_NODES + 1 
    ) for _ in range(N_NODES)]
    
    # Инициализация модели награды
    reward_model = GPUPPReward(b=3.0, c=1.0)
    
    # Инициализация игры
    print(f"Инициализация игры (k_anchors={K_ANCHORS})...")
    game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=K_ANCHORS)
    
    # Запуск симуляции
    print(f"\nЗапуск симуляции на {EPISODES} эпизодов...")
    
    # Синхронизация CUDA для точного замера времени
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    # Прогресс-бар (простой)
    for episode in range(EPISODES):
        game.round()
        
        if (episode + 1) % 1000 == 0:
            coop_rate = float(game.strategies.mean().item())
            print(f"Эпизод {episode + 1}/{EPISODES} | Доля кооперации: {coop_rate:.4f}")
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    elapsed_time = time.time() - start_time
    
    # Результаты
    final_coop_rate = float(game.strategies.mean().item())
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    print("="*70)
    print(f"Финальная доля кооперации: {final_coop_rate:.4f}")
    print(f"Затраченное время:         {elapsed_time:.4f} сек")
    print(f"Производительность:        {(N_NODES * EPISODES / elapsed_time):.0f} шагов агентов/сек")
    
    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_experiment()
