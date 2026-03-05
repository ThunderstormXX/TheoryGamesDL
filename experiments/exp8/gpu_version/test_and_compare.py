import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from experiments.exp8.gpu_version.batched_gpu import BatchedGPUMonteKarloPairGame
from experiments.exp8.gpu_version.graph_structure import SmallWorldGraph
from experiments.exp8.gpu_version.reward_models import RewardManager

def test_reward_models():
    print("Testing Reward Models...")
    device = torch.device('cpu')
    b, c = 2.0, 1.0
    
    # Setup simple manual test case
    # 3 nodes: 0-1, 1-2
    adj = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.float32)
    degrees = adj.sum(dim=2) # [[1, 2, 1]]
    cooperators = torch.tensor([[1, 0, 1]], dtype=torch.float32) # Node 0 and 2 cooperate
    
    # pf: b*sum(neigh_coop) - c*xi
    # Node 0: b*(0) - c*1 = -1
    # Node 1: b*(1+1) - c*0 = 4
    # Node 2: b*(0) - c*1 = -1
    rm_pf = RewardManager('pf', b, c)
    r_pf = rm_pf.calculate_rewards(cooperators, adj, degrees)
    expected_pf = torch.tensor([[-1.0, 4.0, -1.0]])
    assert torch.allclose(r_pf, expected_pf), f"PF failed: {r_pf}"
    
    # ff: b*sum(neigh_coop/k_j) - c*xi
    # Node 0: b*(0) - c*1 = -1
    # Node 1: b*(1/1 + 1/1) - c*0 = 4
    # Node 2: b*(0) - c*1 = -1
    rm_ff = RewardManager('ff', b, c)
    r_ff = rm_ff.calculate_rewards(cooperators, adj, degrees)
    assert torch.allclose(r_ff, expected_pf), "FF simple case failed"

    print("Reward models tests passed!")

def run_experiment(name, state_type, n_rounds=1000):
    print(f"\nRunning Experiment: {name}")
    
    batch_size = 64
    n_agents = 50
    k_neighbors = 4
    
    graph_params = {'k': k_neighbors, 'p': 0.1}
    
    # Stateless means max_states = 1 (all agents see same state 0)
    max_states = (k_neighbors + 1) if state_type == 'neighbor_coop' else 1
    
    learner_params = {
        'learning_rate': 0.05,
        'discount_factor': 0.9,
        'exploration_rate': 0.1,
        'strategy': 'softmax',
        'temperature': 0.5,
        'max_states': max_states
    }
    
    reward_params = {'b': 1.5, 'c': 0.5}
    
    game = BatchedGPUMonteKarloPairGame(
        batch_size, n_agents, 
        graph_params, learner_params, reward_params,
        reward_type='pf'
    )
    
    # If stateless, we must force states to be 0
    if state_type == 'stateless':
        game.current_states = torch.zeros_like(game.current_states)
        # We need to monkeypatch _get_states or handle it in round
        def stateless_get_states(actions):
            return torch.zeros((batch_size, n_agents), device=game.device, dtype=torch.long)
        game._get_states = stateless_get_states
        game.current_states = game._get_states(game.current_actions)

    history = []
    for i in range(n_rounds):
        metrics = game.round()
        history.append(metrics['mean_cooperation'])
        if i % 200 == 0:
            print(f"Round {i}: Coop = {metrics['mean_cooperation']:.4f}")
            
    final_coop = np.mean(history[-100:])
    print(f"Final Average Cooperation ({name}): {final_coop:.4f}")
    return history

if __name__ == "__main__":
    test_reward_models()
    
    h_stateless = run_experiment("Stateless Q-Learning", "stateless")
    h_stateful = run_experiment("Stateful (Neighbor Coop) Q-Learning", "neighbor_coop")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(h_stateless, label='Stateless')
    plt.plot(h_stateful, label='Stateful (Neighbors)')
    plt.title('Comparison: Stateless vs Stateful Q-Learning')
    plt.xlabel('Round')
    plt.ylabel('Mean Cooperation')
    plt.legend()
    plt.grid(True)
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/comparison_stateless_stateful.png')
    print("\nComparison plot saved to ../results/comparison_stateless_stateful.png")
