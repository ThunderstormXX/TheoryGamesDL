
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
from tqdm import tqdm

# Ensure current directory is in path
sys.path.append(os.getcwd())

from graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from learner import QLearner
from reward_model import PPReward
from game_launcher import MonteKarloPairGame

def run_experiment_star_graph():
    print("Running Star Graph Experiment...")
    # Configuration
    N_NODES = 100
    B_VALUES = list(range(1, 10))  # Benefit values to sweep
    C = 1.0
    K_ANCHORS = 1
    N_EPISODES = 1000
    N_EXPS = 100  # Reduced for quicker execution, adjust as needed

    # Initialization
    graph = StarGraph(N_NODES)
    
    # Store results
    # Shape: (len(B), N_EXPS, N_EPISODES)
    all_cooperation_rates = np.zeros((len(B_VALUES), N_EXPS, N_EPISODES))

    for j, b in enumerate(B_VALUES):
        print(f"  Testing b={b}...")
        current_reward_model = PPReward(b=b, c=C)
        
        for i in tqdm(range(N_EXPS)):
            # Initialize learners for each experiment
            current_learners = [
                QLearner(
                    action_space_size=2, 
                    learning_rate=0.2, 
                    discount_factor=0.99, 
                    strategy='boltzmann',
                    temperature=1.0 # Default value, wasn't specified in first cell of sample_exp for star graph
                ) for _ in range(N_NODES)
            ]
            
            # Initialize new game instance
            game = MonteKarloPairGame(graph, current_learners, current_reward_model, k_anchors=K_ANCHORS)
            
            episode_coop_rates = []
            for episode in range(N_EPISODES):
                game.round()
                coop_rate = np.mean(game.strategies)
                episode_coop_rates.append(coop_rate)
            
            all_cooperation_rates[j, i, :] = episode_coop_rates

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate mean and std over experiments
    mean_coop_rates = np.mean(all_cooperation_rates, axis=1) # Shape: (len(B), N_EPISODES)
    std_coop_rates = np.std(all_cooperation_rates, axis=1)
    
    # Plot 1: Over Time with Variance
    for j, b in enumerate(B_VALUES):
        plt.plot(mean_coop_rates[j], label=f'b={b}')
        plt.fill_between(range(N_EPISODES), 
                         mean_coop_rates[j] - std_coop_rates[j], 
                         mean_coop_rates[j] + std_coop_rates[j], 
                         alpha=0.2)
    
    plt.title(f'Cooperation Rate over Time (Star Graph N={N_NODES})')
    plt.xlabel('Episode')
    plt.ylabel('Cooperation Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/star_graph_over_time.png')
    plt.close()

    # Plot 2: Average Final Cooperation Rate vs B
    final_avg_coop = np.mean(mean_coop_rates[:, -100:], axis=1) # Average of last 100 episodes
    plt.figure(figsize=(10, 6))
    plt.plot(B_VALUES, final_avg_coop, marker='o')
    plt.title(f'Final Cooperation Rate vs Benefit b (Star Graph N={N_NODES})')
    plt.xlabel('Benefit b')
    plt.ylabel('Average Final Cooperation Rate')
    plt.grid(True)
    plt.savefig('results/star_graph_vs_b.png')
    plt.close()


from joblib import Parallel, delayed
import math

def run_single_experiment(b, c, n_nodes, n_episodes, k_anchors, temp, graph):
    reward_model = PPReward(b=b, c=c)
    learners = [
        QLearner(
            action_space_size=2, 
            learning_rate=0.2, 
            discount_factor=0.9, 
            strategy='boltzmann',
            temperature=temp 
        ) for _ in range(n_nodes)
    ]
    
    game = MonteKarloPairGame(graph, learners, reward_model, k_anchors=k_anchors)
    
    episode_coop_rates = []
    for _ in range(n_episodes):
        game.round()
        episode_coop_rates.append(np.mean(game.strategies))
        
    return episode_coop_rates

def run_experiment_small_world():
    print("Running Small World Graph Experiment...")
    # Configuration
    N_NODES = 100
    B_VALUES = list(3 + np.arange(0, 4, 1))  # Finer sweep around critical region
    # TEMP_LIST = [0.5, 1.0, 2.0]
    TEMP_LIST = [1.0]
    C = 1.0
    K_ANCHORS = 1
    N_EPISODES = 10000
    N_EXPS = 100 

    # Retrieval of CPU count for parallel processing
    n_jobs = -1 
    
    # Initialization
    graph = SmallWorldGraph(N_NODES, k=4, p=0.1)
    # Get neighbors from graph
    neibhours = graph.get_neibhours()
    count_neibhours = [len(neibs) for neibs in neibhours.values()]
    
    for temp in TEMP_LIST:
        # Calculate theoretical rate for current temperature
        # p_i = 1 / (1 + exp(c * k_i / T))
        theor_coop_rate = sum(1.0 / (1.0 + np.exp(C * k / temp)) for k in count_neibhours) / N_NODES
        print(f"Theoretical Cooperation Rate for Small World Graph (T={temp}): {theor_coop_rate:.4f}")
        
        print(f"Running with temperature={temp}...")
        
        all_cooperation_rates = np.zeros((len(B_VALUES), N_EXPS, N_EPISODES))
        
        for j, b in enumerate(B_VALUES):
            # Execute experiments in parallel
            # We use tqdm here to show progress across the N_EXPS experiments
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_experiment)(
                    b, C, N_NODES, N_EPISODES, K_ANCHORS, temp, graph
                ) for _ in tqdm(range(N_EXPS), desc=f"Simulating b={b}")
            )
            
            # results is a list of lists (experiments x episodes)
            all_cooperation_rates[j] = np.array(results)

        # Visualization
        plt.figure(figsize=(10, 6))

        for j, b in enumerate(B_VALUES):
            mean_trace = np.mean(all_cooperation_rates[j], axis=0)
            std_trace = np.std(all_cooperation_rates[j], axis=0)
            
            plt.plot(mean_trace, label=f'b={b}')
            plt.fill_between(range(N_EPISODES), mean_trace - std_trace, mean_trace + std_trace, alpha=0.2)

        plt.title(f'Cooperation Rate over Time (Small World Graph N={N_NODES}, T={temp})')
        neibhours = graph.get_neibhours()
        count_neibhours = [len(neibs) for neibs in neibhours.values()]
        theor_coop_rate = sum(math.exp(-C * k / temp) for k in count_neibhours) / N_NODES
        plt.axhline(y=theor_coop_rate, color='r', linestyle='--', label=f'Theoretical Cooperation Rate = {theor_coop_rate:.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Cooperation Rate')
        plt.legend()
        plt.grid(True)
        filename = f'results/small_world_over_time_temp_{temp}_many_exps.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")


if __name__ == "__main__":
    # run_experiment_star_graph()
    run_experiment_small_world()
