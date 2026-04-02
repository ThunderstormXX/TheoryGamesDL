"""
GPU Optimization Migration Example

This script demonstrates how to migrate from CPU-based to GPU-based implementation
with minimal code changes.
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.getcwd())

def example_cpu_based():
    """Original CPU-based implementation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Original CPU-based Implementation")
    print("="*70)
    
    from graph_structure import StarGraph
    from learner import QLearner
    from reward_model import PPReward
    from game_launcher import MonteKarloPairGame
    
    # Setup
    N_NODES = 100
    graph = StarGraph(N_NODES)
    learners = [QLearner(learning_rate=0.2, discount_factor=0.9, strategy='boltzmann') 
               for _ in range(N_NODES)]
    reward_model = PPReward(b=3.0, c=1.0)
    game = MonteKarloPairGame(graph, learners, reward_model, k_anchors=1)
    
    # Run simulation
    start = time.time()
    for episode in range(100):
        game.round()
    elapsed = time.time() - start
    
    print(f"Configuration:")
    print(f"  - Nodes: {N_NODES}")
    print(f"  - Episodes: 100")
    print(f"  - Device: CPU")
    print(f"\nResults:")
    print(f"  - Final cooperation rate: {np.mean(game.strategies):.4f}")
    print(f"  - Time: {elapsed:.4f}s")
    print(f"  - Throughput: {(N_NODES * 100 / elapsed):.0f} agent-steps/sec")


def example_gpu_based():
    """New GPU-based implementation with minimal changes"""
    print("\n" + "="*70)
    print("EXAMPLE 2: GPU-based Implementation (Fast!)")
    print("="*70)
    
    from graph_structure import StarGraph
    from gpu_learner import GPUQLearner  # ← CHANGED
    from gpu_reward_model import GPUPPReward  # ← CHANGED
    from gpu_game_launcher import GPUMonteKarloPairGame  # ← CHANGED
    from gpu_utils import gpu_config
    
    # Show GPU info
    gpu_config.print_info()
    
    # Setup (code is almost identical!)
    N_NODES = 100
    graph = StarGraph(N_NODES)
    learners = [GPUQLearner(  # ← CHANGED (added max_states)
        learning_rate=0.2,
        discount_factor=0.9,
        strategy='boltzmann',
        max_states=N_NODES+1  # ← NEW PARAMETER
    ) for _ in range(N_NODES)]
    reward_model = GPUPPReward(b=3.0, c=1.0)  # ← CHANGED
    game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=1)  # ← CHANGED
    
    # Run simulation
    import torch
    torch.cuda.synchronize()
    start = time.time()
    for episode in range(100):
        game.round()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nConfiguration:")
    print(f"  - Nodes: {N_NODES}")
    print(f"  - Episodes: 100")
    print(f"  - Device: GPU")
    print(f"\nResults:")
    print(f"  - Final cooperation rate: {float(game.strategies.mean().item()):.4f}")
    print(f"  - Time: {elapsed:.4f}s")
    print(f"  - Throughput: {(N_NODES * 100 / elapsed):.0f} agent-steps/sec")
    
    # Cleanup
    torch.cuda.empty_cache()


def comparison_benchmark():
    """Run both versions and compare performance"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    import torch
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("GPU not available! Skipping performance comparison.")
        return
    
    from graph_structure import StarGraph
    from learner import QLearner
    from gpu_learner import GPUQLearner
    from reward_model import PPReward
    from gpu_reward_model import GPUPPReward
    from game_launcher import MonteKarloPairGame
    from gpu_game_launcher import GPUMonteKarloPairGame
    
    configs = [
        {"n_nodes": 50, "episodes": 100},
        {"n_nodes": 200, "episodes": 100},
    ]
    
    results = []
    
    for config in configs:
        n_nodes = config['n_nodes']
        episodes = config['episodes']
        
        print(f"\nBenchmarking with {n_nodes} nodes, {episodes} episodes:")
        print("-" * 70)
        
        # CPU version
        graph_cpu = StarGraph(n_nodes)
        learners_cpu = [QLearner(learning_rate=0.2, discount_factor=0.9, strategy='boltzmann')
                       for _ in range(n_nodes)]
        reward_model_cpu = PPReward(b=3.0, c=1.0)
        game_cpu = MonteKarloPairGame(graph_cpu, learners_cpu, reward_model_cpu, k_anchors=1)
        
        start = time.time()
        for _ in range(episodes):
            game_cpu.round()
        cpu_time = time.time() - start
        
        # GPU version
        graph_gpu = StarGraph(n_nodes)
        learners_gpu = [GPUQLearner(learning_rate=0.2, discount_factor=0.9, strategy='boltzmann',
                                   max_states=n_nodes+1)
                       for _ in range(n_nodes)]
        reward_model_gpu = GPUPPReward(b=3.0, c=1.0)
        game_gpu = GPUMonteKarloPairGame(graph_gpu, learners_gpu, reward_model_gpu, k_anchors=1)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(episodes):
            game_gpu.round()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"  CPU:  {cpu_time:.4f}s ({(n_nodes*episodes/cpu_time):.0f} agent-steps/sec)")
        print(f"  GPU:  {gpu_time:.4f}s ({(n_nodes*episodes/gpu_time):.0f} agent-steps/sec)")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'nodes': n_nodes,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Nodes':<12} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['nodes']:<12} {r['cpu_time']:<15.4f}s {r['gpu_time']:<15.4f}s {r['speedup']:<12.2f}x")
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    torch.cuda.empty_cache()


def large_scale_experiment():
    """Example of large-scale experiment only possible with GPU"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Large-Scale Experiment (GPU only)")
    print("="*70)
    
    import torch
    
    if not torch.cuda.is_available():
        print("GPU not available! Skipping large-scale experiment.")
        return
    
    from graph_structure import SmallWorldGraph
    from gpu_learner import GPUQLearner
    from gpu_reward_model import GPUPPReward
    from gpu_game_launcher import GPUMonteKarloPairGame
    
    print("\nRunning 500-node experiment:")
    print("-" * 70)
    
    N_NODES = 500
    EPISODES = 1000
    B_VALUES = [2, 3, 4, 5]
    
    results = {}
    
    for b in B_VALUES:
        print(f"\nTesting b = {b}...")
        graph = SmallWorldGraph(N_NODES, k=4, p=0.1)
        
        learners = [
            GPUQLearner(
                learning_rate=0.2,
                discount_factor=0.9,
                strategy='boltzmann',
                temperature=1.0,
                max_states=N_NODES+1
            )
            for _ in range(N_NODES)
        ]
        
        reward_model = GPUPPReward(b=b, c=1.0)
        game = GPUMonteKarloPairGame(graph, learners, reward_model, k_anchors=1)
        
        cooperation_rates = []
        
        torch.cuda.synchronize()
        for episode in range(EPISODES):
            game.round()
            coop_rate = float(game.strategies.mean().item())
            cooperation_rates.append(coop_rate)
            
            if (episode + 1) % 200 == 0:
                print(f"  Episode {episode+1}/{EPISODES}: coop_rate = {coop_rate:.4f}")
        
        results[b] = cooperation_rates
        torch.cuda.synchronize()
    
    print("\n" + "-"*70)
    print("Final Results:")
    for b in B_VALUES:
        final_coop = np.mean(results[b][-100:])  # Average of last 100 episodes
        print(f"  b = {b}: final cooperation rate = {final_coop:.4f}")
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU OPTIMIZATION EXAMPLES")
    print("="*70)
    
    # Run examples
    example_cpu_based()
    example_gpu_based()
    comparison_benchmark()
    large_scale_experiment()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
