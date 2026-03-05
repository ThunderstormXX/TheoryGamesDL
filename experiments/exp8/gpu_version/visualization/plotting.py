import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cooperation_with_std(history_data, labels, title="Cooperation Rate", save_path="results/plot.png", 
                               theory_values=None, experiment_info=None):
    """
    Plots mean cooperation rates with standard deviation shading.
    """
    plt.figure(figsize=(12, 7))
    
    rounds = np.arange(history_data[0].shape[0])
    
    for history, label in zip(history_data, labels):
        mean_coop = np.mean(history, axis=1)
        std_coop = np.std(history, axis=1)
        
        plt.plot(rounds, mean_coop, label=f"Mean {label}", linewidth=2)
        plt.fill_between(rounds, mean_coop - std_coop, mean_coop + std_coop, alpha=0.15)
        
    # Add theoretical horizontal lines
    if theory_values:
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(theory_values)))
        for i, (label, val) in enumerate(theory_values.items()):
            plt.axhline(y=val, color=colors[i], linestyle='--', alpha=0.8, 
                        label=f"Theory {label}: {val:.3f}")
        
    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.title(title, fontsize=14)
    
    # Add box with experiment info
    if experiment_info:
        info_text = "Experiment Parameters:\n" + "-"*20 + "\n"
        info_text += "\n".join([f"{k}: {v}" for k, v in experiment_info.items()])
        
        # Change position to upper left or lower right with better styling
        plt.text(0.98, 0.05, info_text, transform=plt.gca().transAxes, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='black'),
                 fontsize=10, family='monospace')
        
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    abs_save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
    plt.savefig(abs_save_path, dpi=150)
    print(f"Plot saved to: {abs_save_path}")
    plt.close()

def plot_q_table(q_table, k, agent_idx=0, batch_idx=0, save_path="results/q_table.png"):
    """
    Plots Q-values for a single agent.
    """
    q_np = q_table[batch_idx, agent_idx].cpu().detach().numpy()
    valid_states = k + 1
    q_subset = q_np[:valid_states, :]
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_subset[:, 0], label='Action 0 (Cooperate)', marker='o')
    plt.plot(q_subset[:, 1], label='Action 1 (Defect)', marker='x')
    plt.xlabel('State (Number of Cooperating Neighbors)')
    plt.ylabel('Q-Value')
    plt.title(f'Q-Function for Agent {agent_idx} (Batch {batch_idx})')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
