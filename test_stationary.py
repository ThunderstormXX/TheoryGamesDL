
import torch
import numpy as np

def test_stationary_convergence():
    # Parameters
    alpha = 0.05
    gamma = 0.9
    c = 1.0
    t_temp = 1.0
    n_steps = 10000
    
    # Q-values for one agent, 2 actions
    # Stateless: Q is just [q_c, q_d]
    q = torch.zeros(2)
    
    # Fixed rewards environment
    # Let's say neighbor contribution is constant B=2.0
    # R(C) = 2.0 - c = 1.0
    # R(D) = 2.0 = 2.0
    # Actual difference is -1.0
    # But let's add NOISE to rewards to simulate neighbor stochasticity
    # Neighbor is Bernoulli(0.5). b=3, k=4.
    # neighbor_sum ~ Binomial(4, 0.5). Mean=2.
    # R_pool = 1.5 * neighbor_sum.
    
    b = 1.5
    k = 4
    monitor_diffs = []
    
    for _ in range(n_steps):
        # 1. Select action
        # Softmax
        probs = torch.softmax(q / t_temp, dim=0)
        action = torch.multinomial(probs, 1).item() # 0=C, 1=D
        
        # 2. Generate Reward
        # Neighbors are random
        n_coops = np.random.binomial(k, 0.3) # Assume neighbors coop rate 0.3
        pool = b * n_coops
        
        if action == 0: # C
            reward = pool - c
        else: # D
            reward = pool
            
        # 3. Update
        # Target = R + gamma * max(Q)
        target = reward + gamma * q.max().item()
        
        q[action] = (1 - alpha) * q[action] + alpha * target
        
        monitor_diffs.append(q[0].item() - q[1].item())
        
    print(f"Final Q: {q}")
    print(f"Final Diff: {q[0] - q[1]}")
    print(f"Average Diff (last 1000): {np.mean(monitor_diffs[-1000:])}")
    
    # Calculate prob
    diff = q[0] - q[1]
    prob = 1.0 / (1.0 + np.exp(-diff / t_temp))
    print(f"Prob from Final Diff: {prob}")
    
    # Calculate average prob over run (simulating population average)
    # We can analytically compute expected sigmoid of diff if we know dist of diff
    
if __name__ == "__main__":
    test_stationary_convergence()
