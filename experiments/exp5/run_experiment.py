# run_experiment.py

import matplotlib.pyplot as plt
from bots import SACBot, SimpleBot, SoftmaxSarsaAgent  # type: ignore
from typing import List, Tuple
import torch
import numpy as np
import random
from tqdm import tqdm # type: ignore
import os
import csv

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS доступен! Устройство:", device)
else:
    device = torch.device("cpu")
    print("MPS не доступен, используем CPU.")


class ContinuousPDEnv:
    def __init__(self, r=3, p=5, q=0, s=1):
        self.r = r
        self.p = p
        self.q = q
        self.s = s
        self.state = np.zeros(2)  # прошлые действия агентов

    def reset(self):
        self.state = np.zeros(2)
        return self.state.copy()

    def step(self, a1, a2):
        a = np.clip(a1, 0, 1)
        b = np.clip(a2, 0, 1)
        r1 = self.r * a * b + self.p * (1 - a) * b + self.q * a * (1 - b) + self.s * (1 - a) * (1 - b)
        r2 = self.r * a * b + self.p * a * (1 - b) + self.q * (1 - a) * b + self.s * (1 - a) * (1 - b)
        self.state = np.array([a, b])
        done = False
        return self.state.copy(), (r1, r2), done


# -------------------------
# Функция для проведения матча между двумя ботами
# -------------------------
def arena(bot1, bot2, episodes: int = 200, max_steps: int = 10) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Универсальная арена между двумя ботами.

    Поддерживаемые типы: SimpleBot, SACBot, SoftmaxSarsaAgent.
    Возвращает списки: награды1, награды2, средняя_кооперация1, средняя_кооперация2.
    """
    env = ContinuousPDEnv()
    agent1_rewards: List[float] = []
    agent2_rewards: List[float] = []
    agent1_actions: List[float] = []
    agent2_actions: List[float] = []

    for ep in tqdm(range(episodes), desc="Episodes"):
        state = env.reset()
        # reset для SARSA
        if isinstance(bot1, SoftmaxSarsaAgent):
            bot1.reset()
            bot1.start_episode()
        if isinstance(bot2, SoftmaxSarsaAgent):
            bot2.reset()
            bot2.start_episode()

        ep_reward1 = 0.0
        ep_reward2 = 0.0
        actions1: List[float] = []
        actions2: List[float] = []

        for t in range(max_steps):
            # --- Выбор действий ---
            if isinstance(bot1, SACBot):
                s1 = torch.tensor([state[1]], dtype=torch.float32).unsqueeze(0).to(device)
                a1 = float(bot1.choose_action(s1).detach().cpu().numpy()[0])
            else:
                a1 = float(bot1.choose_action())

            if isinstance(bot2, SACBot):
                s2 = torch.tensor([state[0]], dtype=torch.float32).unsqueeze(0).to(device)
                a2 = float(bot2.choose_action(s2).detach().cpu().numpy()[0])
            else:
                a2 = float(bot2.choose_action())

            # --- Шаг среды ---
            next_state, (r1, r2), done = env.step(a1, a2)

            # --- Обновления обучения ---
            if isinstance(bot1, SACBot):
                bot1.store_transition(s1, torch.tensor([a1], dtype=torch.float32), r1,
                                      torch.tensor([next_state[1]], dtype=torch.float32).unsqueeze(0), done)
                bot1.update()
            elif isinstance(bot1, SoftmaxSarsaAgent):
                bot1.step(r1)

            if isinstance(bot2, SACBot):
                bot2.store_transition(s2, torch.tensor([a2], dtype=torch.float32), r2,
                                      torch.tensor([next_state[0]], dtype=torch.float32).unsqueeze(0), done)
                bot2.update()
            elif isinstance(bot2, SoftmaxSarsaAgent):
                bot2.step(r2)

            # --- Статистика ---
            ep_reward1 += r1
            ep_reward2 += r2
            actions1.append(a1)
            actions2.append(a2)
            state = next_state

        agent1_rewards.append(ep_reward1)
        agent2_rewards.append(ep_reward2)
        agent1_actions.append(float(np.mean(actions1)))
        agent2_actions.append(float(np.mean(actions2)))

    return agent1_rewards, agent2_rewards, agent1_actions, agent2_actions

# -------------------------
# Функция для построения графиков
# -------------------------
def plot_arena_results(rewards1, rewards2, actions1, actions2, title1="Bot1", title2="Bot2", save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(rewards1) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards1, label=f"{title1} Reward")
    plt.plot(episodes, rewards2, label=f"{title2} Reward")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.title(f"Rewards: {title1} vs {title2}")
    plt.legend()
    fname_rewards = os.path.join(save_dir, f"rewards_{title1}_vs_{title2}.png").replace(" ", "_")
    plt.savefig(fname_rewards, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, actions1, label=f"{title1} Cooperation")
    plt.plot(episodes, actions2, label=f"{title2} Cooperation")
    plt.xlabel("Эпизод")
    plt.ylabel("Среднее действие (кооперация)")
    plt.title(f"Cooperation: {title1} vs {title2}")
    plt.legend()
    fname_actions = os.path.join(save_dir, f"cooperation_{title1}_vs_{title2}.png").replace(" ", "_")
    plt.savefig(fname_actions, dpi=150)
    plt.close()

    return fname_rewards, fname_actions

# -------------------------
# Параметры эксперимента (увеличенный размер игры)
# -------------------------
EPISODES = 1000
STEPS = 50
RESULTS_DIR = "results"

# -------------------------
# Список ботов
# -------------------------
bots = [
    ("Simple_cooperate", SimpleBot("cooperate")),
    ("SAC", SACBot(alpha=0.2, device=device)),
    ("SARSA", SoftmaxSarsaAgent(num_actions=21, alpha=0.05, gamma=0.95, beta=6.0))
]

# -------------------------
# Генерация всех уникальных пар (без повторов и зеркал)
# -------------------------
pairs = []
for i in range(len(bots)):
    for j in range(i + 1, len(bots)):
        pairs.append((bots[i], bots[j]))

# -------------------------
# Сводная таблица результатов
# -------------------------
summary_rows = ["bot1,bot2,mean_reward1,mean_reward2,mean_coop1,mean_coop2"]

print(f"Запуск {len(pairs)} матчей...")
for (name1, b1), (name2, b2) in pairs:
    print(f"MATCH: {name1} vs {name2}")
    r1, r2, a1, a2 = arena(b1, b2, episodes=EPISODES, max_steps=STEPS)
    plot_arena_results(r1, r2, a1, a2, title1=name1, title2=name2, save_dir=RESULTS_DIR)
    summary_rows.append(
        f"{name1},{name2},{np.mean(r1):.4f},{np.mean(r2):.4f},{np.mean(a1):.4f},{np.mean(a2):.4f}"
    )

# -------------------------
# Сохранение summary.csv
# -------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
summary_path = os.path.join(RESULTS_DIR, "summary.csv")
with open(summary_path, "w", newline="") as f:
    for row in summary_rows:
        f.write(row + "\n")
print(f"Сводная таблица сохранена: {summary_path}")

