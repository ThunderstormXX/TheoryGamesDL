import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Simple Bots
# -------------------------

class SimpleBot:
    """Боты с фиксированной стратегией: cooperate | betray | random"""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.history: List[float] = []

    def choose_action(self) -> float:
        if self.strategy == "cooperate":
            action = 1.0
        elif self.strategy == "betray":
            action = 0.0
        elif self.strategy == "random":
            action = np.random.uniform(0, 1)
        else:
            action = np.random.uniform(0, 1)

        self.history.append(action)
        return action

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.data = []

    def store(self, s, a, r, s2, d):
        if self.size < self.max_size:
            self.data.append((s, a, r, s2, d))
            self.size += 1
        else:
            self.data[self.ptr] = (s, a, r, s2, d)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, s2, d = zip(*batch)
        return s, a, r, s2, d

# -------------------------
# Actor
# -------------------------
class Actor(nn.Module):
    def __init__(self, state_dim=1, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)  # [-1,1]
        action_scaled = (action + 1)/2  # [0,1]
        # log_prob с поправкой tanh
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action_scaled, log_prob

# -------------------------
# Critic
# -------------------------
class Critic(nn.Module):
    def __init__(self, state_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# -------------------------
# SACBot
# -------------------------
class SACBot:
    def __init__(self, alpha=0.01, gamma=0.6, tau=0.005, batch_size=64, state_dim=1, device=torch.device("cpu")):
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim).to(self.device)
        self.critic1 = Critic(state_dim).to(self.device)
        self.critic2 = Critic(state_dim).to(self.device)
        self.target1 = Critic(state_dim).to(self.device)
        self.target2 = Critic(state_dim).to(self.device)

        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=0.0003)

        self.buffer = ReplayBuffer()
        self.history = []

    def choose_action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            a, _ = self.actor.sample(state)
        return a.squeeze(-1)

    def store_transition(self, s, a, r, s2, d):
        if a.ndim == 1:
            a = a.unsqueeze(1)
        self.buffer.store(s.cpu(), a.cpu(), float(r), s2.cpu(), d)
        self.history.append(float(r))

    def update(self):
        if self.buffer.size < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.cat(s).to(self.device)
        a = torch.cat(a).to(self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.cat(s2).to(self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_a, logp = self.actor.sample(s2)
            q1_t = self.target1(s2, next_a)
            q2_t = self.target2(s2, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp
            y = r + self.gamma * (1 - d) * q_t

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_c1 = F.mse_loss(q1, y)
        loss_c2 = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        loss_c1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss_c2.backward()
        self.critic2_opt.step()

        # ---------------- Actor update ----------------
        a_new, logp_new = self.actor.sample(s)
        q1_new = self.critic1(s, a_new)
        q2_new = self.critic2(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        loss_actor = (self.alpha * logp_new - q_new).mean()

        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        # ---------------- Soft update targets ----------------
        with torch.no_grad():
            for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# -------------------------
# SoftmaxSarsaAgent (из exp4 адаптирован для непрерывного интервала [0,1])
# -------------------------
class SoftmaxSarsaAgent:
    """SARSA агент с softmax-политикой над дискретными действиями.

    Оригинал из exp4 (`SoftmaxSARSAAgent`). Здесь:
    - Дискретизируем интервал действий [0,1] на num_actions точек.
    - choose_action возвращает НЕ индекс, а непосредственно значение действия в [0,1].
    - Храним Q в матрице (1, num_actions) как в исходном коде.
    - Обновление: Q[a] += alpha * (r + gamma * Q[a'] - Q[a]) (классический on-policy SARSA).
    """

    def __init__(self,
                 num_actions: int = 11,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 beta: float = 5.0,
                 seed: Optional[int] = None,
                 init_mode: str = "uniform",
                 init_action: Optional[int] = None,
                 init_epsilon: float = 1e-3):
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        # Единственное состояние (как в оригинале) -> Q shape (1, num_actions)
        self.Q = np.zeros((1, num_actions), dtype=float)
        self.state = 0
        self.last_action_idx: Optional[int] = None

        # Инициализация около дельта-распределения (init_mode == 'delta')
        if init_mode.lower() == "delta":
            A = self.num_actions
            target = 0 if init_action is None else int(init_action)
            target = max(0, min(A - 1, target))
            eps = float(init_epsilon)
            eps = max(0.0, min(0.5, eps))
            if A == 1:
                p0 = np.array([1.0], dtype=float)
            else:
                p0 = np.full(A, eps / (A - 1), dtype=float)
                p0[target] = 1.0 - eps
            logits = (1.0 / max(self.beta, 1e-8)) * np.log(p0 + 1e-12)
            self.Q[self.state, :] = logits

    # ----- Политика -----
    def _policy_probs(self) -> np.ndarray:
        logits = self.beta * self.Q[self.state]
        m = np.max(logits)
        exps = np.exp(logits - m)
        probs = exps / np.sum(exps)
        return probs

    # ----- Выбор дискретного индекса действия -----
    def _choose_action_index(self) -> int:
        probs = self._policy_probs()
        a_idx = int(self.rng.choice(self.num_actions, p=probs))
        return a_idx

    # ----- Отображение индекса в непрерывное действие -----
    def _idx_to_action_value(self, idx: int) -> float:
        if self.num_actions == 1:
            return 0.0
        return idx / (self.num_actions - 1)

    def start_episode(self) -> float:
        """Выбор первого действия (необходим до обновлений)."""
        a_idx = self._choose_action_index()
        self.last_action_idx = a_idx
        return self._idx_to_action_value(a_idx)

    def choose_action(self) -> float:
        """Выбор следующего действия на шаге (для совместимости с SimpleBot)."""
        # Если эпизод не стартовал явно, стартуем.
        if self.last_action_idx is None:
            return self.start_episode()
        a_idx = self._choose_action_index()
        self.last_action_idx = a_idx
        return self._idx_to_action_value(a_idx)

    def step(self, reward: float, next_action_value: Optional[float] = None) -> Tuple[float, float]:
        """SARSA обновление: используем текущее действие и следующее.

        Параметр next_action_value можно передать (уже как float в [0,1]); если None — выберем сами.
        Возвращает (предыдущее_действие, награда).
        """
        if self.last_action_idx is None:
            raise AssertionError("Call start_episode() or choose_action() before step().")
        a_idx = self.last_action_idx
        s = self.state
        r = float(reward)
        # Выбор следующего действия (индекса) если не задано значение
        if next_action_value is None:
            next_idx = self._choose_action_index()
        else:
            # Проецируем непрерывное значение обратно к ближайшему индексу
            next_idx = int(round(next_action_value * (self.num_actions - 1)))
            next_idx = max(0, min(self.num_actions - 1, next_idx))
        td_target = r + self.gamma * self.Q[s, next_idx]
        td_error = td_target - self.Q[s, a_idx]
        self.Q[s, a_idx] += self.alpha * td_error
        # фиксируем новое действие как текущее
        self.last_action_idx = next_idx
        return self._idx_to_action_value(a_idx), r

    def get_action_probs(self) -> np.ndarray:
        return self._policy_probs()

    def reset(self):
        self.last_action_idx = None

