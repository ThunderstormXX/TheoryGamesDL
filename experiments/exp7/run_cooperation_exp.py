
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
    N_EPISODES = 15000
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

def run_single_experiment(b, c, n_nodes, gamma, n_episodes, k_anchors, temp, graph):
    reward_model = PPReward(b=b, c=c)
    learners = [
        QLearner(
            action_space_size=2, 
            learning_rate=0.2, 
            discount_factor=gamma, 
            strategy='boltzmann',
            temperature=temp 
        ) for _ in range(n_nodes)
    ]
    
    game = MonteKarloPairGame(graph, learners, reward_model, k_anchors=k_anchors)
    
    episode_coop_rates = []
    episode_pairwise_coop = []
    for _ in range(n_episodes):
        game.round()
        episode_coop_rates.append(np.mean(game.strategies))
        episode_pairwise_coop.append(game.get_pairwise_cooperation())
        
        
    return episode_coop_rates, episode_pairwise_coop

def run_experiment_small_world_diff_b(gamma = 0):
    print(f"Running Small World Graph Experiment with fixed gamma = {gamma}...")
    # Configuration
    N_NODES = 50
    B_VALUES = list(3 + np.arange(0, 4, 1))  # Finer sweep around critical region
    # TEMP_LIST = [0.5, 1.0, 2.0]
    TEMP_LIST = [1.0]
    C = 1.0
    K_ANCHORS = N_NODES
    N_EPISODES = 1000
    N_EXPS = 100
    GAMMA = gamma
    # Retrieval of CPU count for parallel processing
    n_jobs = -1 
    
    # Ensure correct directory exists
    os.makedirs('results/N_anchors/b_exp', exist_ok=True)
    
    for temp in TEMP_LIST:
        # Calculate theoretical rate for current temperature
        # p_i = 1 / (1 + exp(c * k_i / T))
        theor_coop_rate = sum(1.0 / (1.0 + np.exp(C * k / temp)) for k in count_neibhours) / N_NODES
        print(f"Theoretical Cooperation Rate for Small World Graph (T={temp}): {theor_coop_rate:.4f}")
        
        print(f"Running with temperature={temp}...")
        
        all_cooperation_rates = np.zeros((len(B_VALUES), N_EXPS, N_EPISODES))
        all_pairwise_coop = np.zeros((len(B_VALUES), N_EXPS, N_EPISODES))
        for j, b in enumerate(B_VALUES):
            # Execute experiments in parallel
            # We use tqdm here to show progress across the N_EXPS experiments
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_experiment)(
                    b, C, N_NODES, GAMMA, N_EPISODES, K_ANCHORS, temp, graph
                ) for _ in tqdm(range(N_EXPS), desc=f"Simulating b={b}")
            )
            
            # Unzip the results: parallel_results is a list of tuples (coop_rate, pairwise_coop)
            results, pairwise_coop = zip(*parallel_results)
            
            # results is a list of lists (experiments x episodes)
            all_cooperation_rates[j] = np.array(results)
            all_pairwise_coop[j] = np.array(pairwise_coop)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
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
        filename = f'results/N_anchors/b_exp/small_world_over_time_temp_{temp}_gamma_{GAMMA}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

        plt.subplot(1, 2, 2)
        
        # Calculate final means for all b values
        final_coop_vs_b = []
        final_pairwise_vs_b = []
        
        for j, b in enumerate(B_VALUES):
            # Average over experiments AND over the last 100 episodes to get a single scalar per b
            # all_cooperation_rates[j] shape: (N_EXPS, N_EPISODES)
            mean_scalar_coop = np.mean(all_cooperation_rates[j][:, -100:])
            mean_scalar_pairwise = np.mean(all_pairwise_coop[j][:, -100:])
            
            final_coop_vs_b.append(mean_scalar_coop)
            final_pairwise_vs_b.append(mean_scalar_pairwise)

        # Plot Dependence on B
        plt.plot(B_VALUES, final_coop_vs_b, marker='o', label=f'T={temp}')
        plt.plot(B_VALUES, final_pairwise_vs_b, marker='s', label=f'T={temp} (Pairwise)')
        
        plt.title(f'Final Cooperation Rate vs Benefit b (Small World Graph N={N_NODES}, T={temp})')
        plt.xlabel('Benefit b')
        plt.ylabel('Average Final Cooperation Rate')
        plt.legend()
        plt.grid(True)
        filename = f'results/N_anchors/b_exp/small_world_vs_b_temp_{temp}_gamma_{GAMMA}.png'

def run_experiment_small_world_diff_gamma(b = 3):
    print(f"Running Small World Graph Experiment with fixed b = {b}...")
    # Configuration
    N_NODES = 50
    # TEMP_LIST = [0.5, 1.0, 2.0]
    TEMP_LIST = [1.0]
    C = 1.0
    K_ANCHORS = N_NODES
    N_EPISODES = 5000
    N_EXPS = 100
    GAMMA_LIST = list(np.arange(0, 1, 0.2))  # Sweep gamma from 0 to 1
    
    # Retrieval of CPU count for parallel processing
    n_jobs = -1 
    
    # Initialization
    graph = SmallWorldGraph(N_NODES, k=4, p=0.1)
    
    # Ensure correct directory exists
    os.makedirs('results/N_anchors/gamma_exp', exist_ok=True)

    for temp in TEMP_LIST:
        print(f"Running with temperature={temp}...")
        count_neibhours = [len(neibs) for neibs in graph.get_neibhours().values()]
        theor_coop_rate = sum(1.0 / (1.0 + np.exp(C * k / temp)) for k in count_neibhours) / N_NODES
        print(f"Theoretical Cooperation Rate for Small World Graph (T={temp}): {theor_coop_rate:.4f}")

        all_cooperation_rates = np.zeros((len(GAMMA_LIST), N_EXPS, N_EPISODES))
        all_pairwise_coop = np.zeros((len(GAMMA_LIST), N_EXPS, N_EPISODES))
        
        for j, gamma in enumerate(GAMMA_LIST):
            # Execute experiments in parallel with different Gamma
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_experiment)(
                    b, C, N_NODES, gamma, N_EPISODES, K_ANCHORS, temp, graph
                ) for _ in tqdm(range(N_EXPS), desc=f"Simulating gamma={gamma:.1f}")
            )
            
            # Unzip the results
            results, pairwise_coop = zip(*parallel_results)
            
            all_cooperation_rates[j] = np.array(results)
            all_pairwise_coop[j] = np.array(pairwise_coop)

        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Dynamics over time for different Gammas
        plt.subplot(1, 2, 1)
        for j, gamma in enumerate(GAMMA_LIST):
            mean_trace = np.mean(all_cooperation_rates[j], axis=0)
            std_trace = np.std(all_cooperation_rates[j], axis=0)
            
            plt.plot(mean_trace, label=f'gamma={gamma:.1f}')
            plt.fill_between(range(N_EPISODES), mean_trace - std_trace, mean_trace + std_trace, alpha=0.1)

        plt.axhline(y=theor_coop_rate, color='r', linestyle='--', label=f'Theoretical Cooperation Rate = {theor_coop_rate:.2f}')
        plt.title(f'Coop Rate (Small World, b={b}, T={temp})')
        plt.xlabel('Episode')
        plt.ylabel('Cooperation Rate')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Final Cooperation Rate vs Gamma
        plt.subplot(1, 2, 2)
        
        final_coop_vs_gamma = []
        final_pairwise_vs_gamma = []
        
        for j, gamma in enumerate(GAMMA_LIST):
            mean_scalar_coop = np.mean(all_cooperation_rates[j][:, -100:])
            mean_scalar_pairwise = np.mean(all_pairwise_coop[j][:, -100:])
            
            final_coop_vs_gamma.append(mean_scalar_coop)
            final_pairwise_vs_gamma.append(mean_scalar_pairwise)

        plt.plot(GAMMA_LIST, final_coop_vs_gamma, marker='o', label=f'T={temp}')
        plt.plot(GAMMA_LIST, final_pairwise_vs_gamma, marker='s', label=f'T={temp} (Pairwise)')
        
        plt.title(f'Final Coop Rate vs Gamma (b={b})')
        plt.xlabel('Discount Factor (gamma)')
        plt.ylabel('Average Final Cooperation Rate')
        plt.legend()
        plt.grid(True)
        
        # Save plot specifically in gamma_exp folder
        filename = f'results/N_anchors/gamma_exp/small_world_vs_gamma_b_{b}_temp_{temp}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('results/N_anchors/b_exp', exist_ok=True)
    os.makedirs('results/N_anchors/gamma_exp', exist_ok=True)
    
    # Run Experiments
    # Note: run_experiment_small_world_diff_b should also be updated to save to b_exp folder 
    # but I will just run the new gamma experiment here or both if needed.
    
    # run_experiment_small_world_diff_b(gamma = 0) # Assumes this function saves to the right place or needs edit
    run_experiment_small_world_diff_gamma(b = 3)