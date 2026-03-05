import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from experiments.exp8.gpu_version.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.gpu_utils import gpu_config

def main():
    # Parameters
    BATCH_SIZE = 100
    N_AGENTS = 50
    N_ROUNDS = 1000
    K_NEIGHBORS = 4
    
    graph_params = {
        'k': K_NEIGHBORS,
        'p': 0.1 # Small world rewiring probability
    }
    
    learner_params = {
        'learning_rate': 0.01,
        'discount_factor': 0.95,
        'exploration_rate': 0.1,
        'strategy': 'epsilon_greedy',
        'max_states': K_NEIGHBORS + 1  # State is number of cooperating neighbors (0 to k)
    }
    
    reward_params = {
        'b': 1.5,
        'c': 0.5
    }
    
    # Initialize game
    print(f"Initializing game with {BATCH_SIZE} simulations of {N_AGENTS} agents each...")
    game = BatchedGPUMonteKarloPairGame(
        BATCH_SIZE, N_AGENTS, 
        graph_params, learner_params, reward_params
    )
    
    # Run simulation
    print("Running simulation...")
    history_coop = []
    
    for i in range(N_ROUNDS):
        metrics = game.round()
        if i % 100 == 0:
            print(f"Round {i}: Mean Cooperation = {metrics['mean_cooperation']:.4f}, Mean Reward = {metrics['mean_reward']:.4f}")
        history_coop.append(metrics['mean_cooperation'])
        
    # Plot cooperation rate
    plt.figure(figsize=(10, 6))
    plt.plot(history_coop)
    plt.title('Mean Cooperation Rate over Time')
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.grid(True)
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/cooperation_rate.png')
    print("Saved cooperation rate plot to ../results/cooperation_rate.png")
    
    # Plot Q-function for a few agents
    print("Plotting Q-functions...")
    # Select a random agent from a random batch to visualize
    # Batch 0, Agent 0
    fig = game.plot_q_function(0, 0) # This returns plt but current plt context logic in batched_gpu needs to be cleaner.
    # Actually, plot_q_function calls plt.figure(), so it creates a new figure.
    # But it returns plt module which is weird.
    # Let's trust it works or fix it.
    # It calls plt.figure() inside, so we just need to save.
    
    plt.savefig('../results/q_function_agent_0_0.png')
    print("Saved Q-function plot for Agent 0 in Batch 0 to ../results/q_function_agent_0_0.png")

    # Batch 0, Agent 1
    # We need to close previous figure or handle multiple figures
    plt.close() # Close previous
    
    # Let's call plot_q_function again
    # But wait, `plot_q_function` in `batched_gpu.py` does `plt.figure(...)`
    # So `plt.savefig` will save the current figure.
    
    # Let's try plotting for an agent in a HIGH cooperation batch vs LOW cooperation batch?
    # Or just random.
    
    # Agent 1, Batch 0
    game.plot_q_function(0, 1)
    plt.savefig('../results/q_function_agent_0_1.png')
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
