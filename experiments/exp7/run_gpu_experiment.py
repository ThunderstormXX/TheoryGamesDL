import sys
import os
import time
import torch
import numpy as np
import json

# Добавляем текущую директорию в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_structure import SmallWorldGraph
from gpu_version.gpu_learner import GPUQLearner
from gpu_version.gpu_reward_model import GPUPPReward
from gpu_version.gpu_game_launcher import GPUMonteKarloPairGame
from gpu_version.gpu_utils import gpu_config

def run_single_simulation(b, gamma, episodes=10000, n_nodes=100, k_neighbors=1, p_rewiring=0.1, k_anchors=1):
    """Запускает одну симуляцию с заданными параметрами и возвращает финальную долю кооперации."""
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
    
    for _ in range(episodes):
        game.round()
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    elapsed_time = time.time() - start_time
    final_coop_rate = float(game.strategies.mean().item())
    
    return final_coop_rate, elapsed_time

def run_gamma_experiment():
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: Зависимость от gamma (discount factor)")
    print("Параметры: Small World, k_anchors=1, episodes=10000, b=3.0")
    print("="*70)
    
    gammas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    results = {}
    
    for gamma in gammas:
        print(f"Запуск для gamma = {gamma}...")
        coop_rate, elapsed = run_single_simulation(b=3.0, gamma=gamma)
        print(f"  -> Доля кооперации: {coop_rate:.4f} (Время: {elapsed:.2f} сек)")
        results[gamma] = coop_rate
        
    # Сохранение результатов
    os.makedirs("results/N_anchors/gamma_exp", exist_ok=True)
    with open("results/N_anchors/gamma_exp/gamma_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Результаты сохранены в results/N_anchors/gamma_exp/gamma_results.json")

def run_b_experiment():
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: Зависимость от b (reward parameter)")
    print("Параметры: Small World, k_anchors=1, episodes=10000, gamma=0.9")
    print("="*70)
    
    bs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {}
    
    for b in bs:
        print(f"Запуск для b = {b}...")
        coop_rate, elapsed = run_single_simulation(b=b, gamma=0.9)
        print(f"  -> Доля кооперации: {coop_rate:.4f} (Время: {elapsed:.2f} сек)")
        results[b] = coop_rate
        
    # Сохранение результатов
    os.makedirs("results/N_anchors/b_exp", exist_ok=True)
    with open("results/N_anchors/b_exp/b_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Результаты сохранены в results/N_anchors/b_exp/b_results.json")

if __name__ == "__main__":
    gpu_config.print_info()
    
    # Запуск обоих экспериментов
    run_gamma_experiment()
    run_b_experiment()
    
    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
