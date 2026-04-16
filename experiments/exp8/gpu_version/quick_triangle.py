import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
os.environ['TRAP_DEVICE'] = 'cpu'
DEVICE = torch.device('cpu')

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
from experiments.exp8.gpu_version.core.graph_structure import TriangleGraph

def run_sim(beta, gamma, iterations):
    graph = TriangleGraph(DEVICE)
    adj_t = graph.generate_adjacency_matrix()
    degrees = adj_t.sum(dim=1)
    
    batch_size = 5
    num_nodes = 3
    states = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=DEVICE)
    temp = 1.0 / beta
    
    learner = BatchedGPUQLearner(
        batch_size=batch_size, n_agents=num_nodes, action_space_size=2,
        learning_rate=0.02, discount_factor=gamma, exploration_rate=0.0,
        strategy='boltzmann', temperature=temp, max_states=1,
    )
    reward_manager = BonusRewardManager(reward_type='pp', b=3.0, c=1.0, bonus=1.0)
    
    adj_batched = adj_t.unsqueeze(0).expand(batch_size, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        for t in range(1, iterations + 1):
            actions = learner.get_actions(states)
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batched, deg_batched)
            learner.update(states, actions, rewards, states)
            
    q_now = learner.q_table[:, :, 0, :].cpu().numpy()
    probs = torch.softmax(torch.tensor(q_now) / temp, dim=-1).numpy()[..., 1]
    print(f"Beta={beta}, Gamma={gamma}, Iterations={iterations}")
    for b in range(batch_size):
        p = probs[b]
        stuck_p = [val for val in p if val < 0.05]
        print(f"  Rep {b}: P(C) = {p}, Stuck Cunt: {len(stuck_p)}")
        
if __name__ == '__main__':
    run_sim(0.5, 0.95, 1_000_000)
    run_sim(1.0, 0.95, 1_000_000)
    run_sim(2.0, 0.95, 1_000_000)
