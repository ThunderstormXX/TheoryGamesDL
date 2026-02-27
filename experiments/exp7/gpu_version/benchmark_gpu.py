"""
Benchmark script comparing CPU vs GPU implementations
"""
import numpy as np
import time
import sys
import os

# Add to path
sys.path.append(os.getcwd())

from graph_structure import StarGraph, SmallWorldGraph
from gpu_utils import gpu_config

def benchmark_cpu_implementation(n_nodes=50, n_episodes=100, n_experiments=5):
    """Benchmark original CPU implementation"""
    from learner import QLearner
    from reward_model import PPReward
    from game_launcher import MonteKarloPairGame
    
    print(f"\n{'='*60}")
    print("CPU IMPLEMENTATION BENCHMARK")
    print(f"{'='*60}")
    print(f"Config: {n_nodes} nodes, {n_episodes} episodes, {n_experiments} experiments")
    
    times = []
    
    for exp in range(n_experiments):
        graph = StarGraph(n_nodes)
        learners = [QLearner(action_space_size=2, learning_rate=0.2, 
                             discount_factor=0.9, strategy='boltzmann') 
                   for _ in range(n_nodes)]
        reward_model = PPReward(b=3.0, c=1.0)
        game = MonteKarloPairGame(graph, learners, reward_model, k_anchors=1)
        
        start = time.time()
        for _ in range(n_episodes):
            game.round()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"Experiment {exp+1}: {elapsed:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAverage: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Throughput: {(n_nodes * n_episodes / avg_time):.0f} agent-steps/sec")
    
    return avg_time

def benchmark_gpu_implementation(n_nodes=50, n_episodes=100, n_experiments=5):
    """Benchmark GPU-optimized implementation"""
    from gpu_learner import GPUQLearner
    from gpu_reward_model import GPUPPReward
    from gpu_game_launcher import GPUMonteKarloPairGame
    
    print(f"\n{'='*60}")
    print("GPU IMPLEMENTATION BENCHMARK")
    print(f"{'='*60}")
    print(f"Config: {n_nodes} nodes, {n_episodes} episodes, {n_experiments} experiments")
    print(f"Device: {gpu_config.device}")
    
    times = []
    
    # Warmup
    print("Warming up GPU...")
    graph = StarGraph(n_nodes)
    learners = [GPUQLearner(action_space_size=2, learning_rate=0.2, 
                            discount_factor=0.9, strategy='boltzmann', max_states=n_nodes+1)
               for _ in range(n_nodes)]
    reward_model = GPUPPReward(b=3.0, c=1.0)
    game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=1)
    
    for _ in range(10):
        game.round()
    
    for exp in range(n_experiments):
        graph = StarGraph(n_nodes)
        learners = [GPUQLearner(action_space_size=2, learning_rate=0.2, 
                                discount_factor=0.9, strategy='boltzmann', max_states=n_nodes+1)
                   for _ in range(n_nodes)]
        reward_model = GPUPPReward(b=3.0, c=1.0)
        game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=1)
        
        start = time.time()
        for _ in range(n_episodes):
            game.round()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"Experiment {exp+1}: {elapsed:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAverage: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Throughput: {(n_nodes * n_episodes / avg_time):.0f} agent-steps/sec")
    
    return avg_time

def run_benchmarks():
    """Run comprehensive benchmarks"""
    gpu_config.print_info()
    
    # Small scale test
    print("\n" + "="*60)
    print("SMALL SCALE (50 nodes)")
    print("="*60)
    cpu_time_small = benchmark_cpu_implementation(n_nodes=50, n_episodes=100, n_experiments=3)
    gpu_time_small = benchmark_gpu_implementation(n_nodes=50, n_episodes=100, n_experiments=3)
    
    speedup_small = cpu_time_small / gpu_time_small
    print(f"\n✓ GPU Speedup: {speedup_small:.2f}x")
    
    # Medium scale test
    print("\n" + "="*60)
    print("MEDIUM SCALE (200 nodes)")
    print("="*60)
    cpu_time_med = benchmark_cpu_implementation(n_nodes=200, n_episodes=100, n_experiments=2)
    gpu_time_med = benchmark_gpu_implementation(n_nodes=200, n_episodes=100, n_experiments=2)
    
    speedup_med = cpu_time_med / gpu_time_med
    print(f"\n✓ GPU Speedup: {speedup_med:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Small scale (50 nodes):   CPU {cpu_time_small:.4f}s → GPU {gpu_time_small:.4f}s ({speedup_small:.2f}x)")
    print(f"Medium scale (200 nodes): CPU {cpu_time_med:.4f}s → GPU {gpu_time_med:.4f}s ({speedup_med:.2f}x)")
    print(f"Average speedup: {(speedup_small + speedup_med) / 2:.2f}x")

if __name__ == "__main__":
    run_benchmarks()
