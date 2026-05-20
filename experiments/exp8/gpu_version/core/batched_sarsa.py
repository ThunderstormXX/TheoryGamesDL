"""
Batched SARSA Learner for multiple simulations running in parallel.

Key difference from Q-Learning:
  Q-Learning target: r + γ * max_a' Q(s', a')       [off-policy]
  SARSA target:      r + γ * Q(s', a')               [on-policy]
  where a' is the action actually taken in state s'.
"""

import torch
from experiments.exp8.gpu_version.utils.gpu_utils import gpu_config


class BatchedGPUSARSALearner:
    """
    Batched SARSA Learner.  API mirrors BatchedGPUQLearner except that
    `update()` requires an additional `next_actions` argument.
    """

    def __init__(self, batch_size, n_agents, action_space_size=2,
                 learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.1, strategy='epsilon_greedy',
                 temperature=1.0, max_states=1):

        self.batch_size = batch_size
        self.n_agents = n_agents
        self.total_agents = batch_size * n_agents
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.strategy = strategy
        self.temp = temperature
        self.device = gpu_config.device
        self.max_states = max_states

        self._batch_agent_linear_idx = torch.arange(
            self.total_agents, device=self.device)

        # Q-table shape: (batch_size, n_agents, max_states, action_space_size)
        self.q_table = torch.zeros(
            (batch_size, n_agents, max_states, action_space_size),
            device=self.device, dtype=torch.float32,
        )

    # ── action selection (identical to Q-learner) ──────────────────────
    def get_actions(self, states):
        """
        states: (batch_size, n_agents) tensor of state indices.
        Returns: (batch_size, n_agents) tensor of actions.
        """
        if self.max_states == 1:
            gathered_q = self.q_table[:, :, 0, :].reshape(
                -1, self.action_space_size)
        else:
            flat_states = states.view(-1)
            flat_q = self.q_table.view(
                -1, self.max_states, self.action_space_size)
            gathered_q = flat_q[self._batch_agent_linear_idx, flat_states]

        if self.strategy == 'epsilon_greedy':
            random_probs = torch.rand(
                self.total_agents, device=self.device)
            random_actions = torch.randint(
                0, self.action_space_size,
                (self.total_agents,), device=self.device)
            greedy_actions = torch.argmax(gathered_q, dim=1)
            actions = torch.where(
                random_probs < self.epsilon, random_actions, greedy_actions)

        elif self.strategy in ('softmax', 'boltzmann'):
            probabilities = torch.softmax(gathered_q / self.temp, dim=1)
            if self.action_space_size == 2:
                p1 = probabilities[:, 1]
                actions = (torch.rand_like(p1) < p1).long()
            else:
                actions = torch.multinomial(probabilities, 1).squeeze()
        else:
            actions = torch.argmax(gathered_q, dim=1)

        return actions.view(self.batch_size, self.n_agents)

    # ── SARSA update ───────────────────────────────────────────────────
    def update(self, states, actions, rewards, next_states, next_actions,
               mask=None):
        """
        On-policy SARSA update:
            Q(s, a) ← (1-α)·Q(s,a) + α·[r + γ·Q(s', a')]
        """
        current_lr = self.lr

        # ── Fast path: stateless (max_states == 1), no mask ──
        if self.max_states == 1 and mask is None:
            flat_actions = actions.view(-1)
            flat_rewards = rewards.view(-1)
            flat_next_actions = next_actions.view(-1)

            q = self.q_table[:, :, 0, :]                       # (B, N, A)
            q_flat = q.view(-1, self.action_space_size)         # (B*N, A)

            row_idx = self._batch_agent_linear_idx
            next_q = q_flat[row_idx, flat_next_actions]         # SARSA: Q(s',a')
            target = flat_rewards + self.gamma * next_q

            current_q = q_flat[row_idx, flat_actions]
            new_q_val = (1.0 - current_lr) * current_q + current_lr * target
            q_flat[row_idx, flat_actions] = new_q_val
            return

        # ── General path: stateful ──
        flat_states = states.view(-1)
        flat_actions = actions.view(-1)
        flat_rewards = rewards.view(-1)
        flat_next_states = next_states.view(-1)
        flat_next_actions = next_actions.view(-1)

        batch_agent_idx = self._batch_agent_linear_idx

        # Look up Q(s', a') for SARSA
        flat_q_data = self.q_table.view(-1)
        next_indices = (batch_agent_idx
                        * (self.max_states * self.action_space_size)
                        + flat_next_states * self.action_space_size
                        + flat_next_actions)
        next_q = flat_q_data[next_indices]                      # SARSA
        target = flat_rewards + self.gamma * next_q

        cur_indices = (batch_agent_idx
                       * (self.max_states * self.action_space_size)
                       + flat_states * self.action_space_size
                       + flat_actions)
        current_q = flat_q_data[cur_indices]
        new_q_val = (1.0 - current_lr) * current_q + current_lr * target

        if mask is not None:
            flat_mask = mask.view(-1)
            cur_indices = cur_indices[flat_mask]
            new_q_val = new_q_val[flat_mask]

        flat_q_data.scatter_(0, cur_indices, new_q_val)

    def get_q_table(self):
        return self.q_table
