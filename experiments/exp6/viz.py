import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np

def ensure_results_dir(results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

def plot_metrics(history_mean_opinion, history_q_diff, results_dir="results"):
    ensure_results_dir(results_dir)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_mean_opinion)
    plt.title("Opinion Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Mean Opinion (0 to 1)")
    plt.ylim(0, 1)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_q_diff)
    plt.title("Q-Value Divergence (Confidence)")
    plt.xlabel("Episode")
    plt.ylabel("Mean |Q(1) - Q(0)|")
    plt.grid(True)
    
    save_path = os.path.join(results_dir, "training_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics saved to {save_path}")

def plot_distribution(current_opinions, results_dir="results"):
    ensure_results_dir(results_dir)
    plt.figure(figsize=(6, 4))
    plt.hist(current_opinions, bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8)
    plt.title("Final Opinion Distribution")
    plt.xticks([0, 1])
    
    save_path = os.path.join(results_dir, "final_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Distribution saved to {save_path}")

def save_animation(opinions_history, graph_obj, n, leaders=None, results_dir="results"):
    ensure_results_dir(results_dir)
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        neighbors = graph_obj.get_adjacency_list(i)
        for neighbor in neighbors:
            if i < neighbor:
                G.add_edge(i, neighbor)

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        current_ops = opinions_history[frame]
        
        node_colors = ['red' if op == 1 else 'blue' for op in current_ops]
        
        if leaders:
            node_sizes = [300 if i in leaders else 100 for i in range(n)]
            # Gold border for leaders, white-ish for others
            node_edge_colors = ['gold' if i in leaders else 'gray' for i in range(n)]
            line_widths = [2.5 if i in leaders else 1.0 for i in range(n)]
        else:
            node_sizes = 100
            node_edge_colors = 'gray'
            line_widths = 1.0

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="gray")
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                               node_color=node_colors, 
                               node_size=node_sizes, 
                               edgecolors=node_edge_colors, 
                               linewidths=line_widths)
                               
        ax.set_title(f"Episode {frame}")
        ax.axis('off')

    ani = FuncAnimation(fig, update, frames=len(opinions_history), interval=50, repeat=True)
    
    save_path = os.path.join(results_dir, "evolution.gif")
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()
    print(f"Animation saved to {save_path}")