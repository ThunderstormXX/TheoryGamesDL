# run_experiment.py

import matplotlib.pyplot as plt
from bots import SACBot, SimpleBot, SoftmaxSarsaAgent, MASACSystem  # type: ignore
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
    def __init__(self, r=3, p=4, q=0, s=1):
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
def arena(bot1, bot2, episodes: int = 200, max_steps: int = 10) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Универсальная арена между двумя ботами.

    FIX: Исправлен порядок формирования переходов для SACBot — теперь next_state берётся ПОСЛЕ обновления истории противника.
    Поддерживаемые типы: SimpleBot, SACBot, SoftmaxSarsaAgent.
    Возвращает списки: награды1, награды2, средняя_кооперация1, средняя_кооперация2.
    """
    env = ContinuousPDEnv()
    agent1_rewards_sum: List[float] = []
    agent2_rewards_sum: List[float] = []
    agent1_actions_mean: List[float] = []
    agent2_actions_mean: List[float] = []
    agent1_rewards_avg: List[float] = []  # средняя награда за шаг
    agent2_rewards_avg: List[float] = []
    actions_all1: List[float] = []  # все действия первого
    actions_all2: List[float] = []  # все действия второго

    for ep in tqdm(range(episodes), desc="Episodes"):
        env.reset()
        # reset для SARSA
        if isinstance(bot1, SoftmaxSarsaAgent):
            bot1.reset(); bot1.start_episode()
        if isinstance(bot2, SoftmaxSarsaAgent):
            bot2.reset(); bot2.start_episode()

        ep_reward1 = 0.0
        ep_reward2 = 0.0
        actions1_episode: List[float] = []
        actions2_episode: List[float] = []

        for _ in range(max_steps):
            # --- текущее состояние для SAC ---
            if isinstance(bot1, SACBot):
                s1 = bot1.get_state()  # shape (1, history_window)
            else:
                s1 = None
            if isinstance(bot2, SACBot):
                s2 = bot2.get_state()
            else:
                s2 = None

            # --- выбор действий ---
            if isinstance(bot1, SACBot):
                a1 = float(bot1.choose_action().detach().cpu().item())
            elif isinstance(bot1, SoftmaxSarsaAgent):
                # Используем ТЕКУЩЕЕ действие (выбранное в start_episode или предыдущем step), не выбираем новое до награды
                a1 = float(bot1.get_current_action_value())
            else:  # SimpleBot
                a1 = float(bot1.choose_action())

            if isinstance(bot2, SACBot):
                a2 = float(bot2.choose_action().detach().cpu().item())
            elif isinstance(bot2, SoftmaxSarsaAgent):
                a2 = float(bot2.get_current_action_value())
            else:
                a2 = float(bot2.choose_action())

            # --- шаг среды ---
            _, (r1, r2), done = env.step(a1, a2)

            # --- обновление истории противника (теперь влияет на next_state) ---
            if isinstance(bot1, SACBot):
                bot1.update_history(a2)
                s1_next = bot1.get_state()
            else:
                s1_next = None
            if isinstance(bot2, SACBot):
                bot2.update_history(a1)
                s2_next = bot2.get_state()
            else:
                s2_next = None

            # --- обучение ---
            if isinstance(bot1, SACBot):
                bot1.store_transition(s1, torch.tensor([a1], dtype=torch.float32), r1, s1_next, done)
                bot1.update()
            elif isinstance(bot1, SoftmaxSarsaAgent):
                bot1.step(r1)

            if isinstance(bot2, SACBot):
                bot2.store_transition(s2, torch.tensor([a2], dtype=torch.float32), r2, s2_next, done)
                bot2.update()
            elif isinstance(bot2, SoftmaxSarsaAgent):
                bot2.step(r2)

            ep_reward1 += r1
            ep_reward2 += r2
            actions1_episode.append(a1)
            actions2_episode.append(a2)
            actions_all1.append(a1)
            actions_all2.append(a2)

        agent1_rewards_sum.append(ep_reward1)
        agent2_rewards_sum.append(ep_reward2)
        agent1_actions_mean.append(float(np.mean(actions1_episode)))
        agent2_actions_mean.append(float(np.mean(actions2_episode)))
        agent1_rewards_avg.append(ep_reward1 / max_steps)
        agent2_rewards_avg.append(ep_reward2 / max_steps)

    return (agent1_rewards_sum, agent2_rewards_sum,
            agent1_actions_mean, agent2_actions_mean,
            agent1_rewards_avg, agent2_rewards_avg,
            actions_all1, actions_all2)

# -------------------------
# Функция для построения графиков
# -------------------------
def plot_arena_results(rewards_sum1, rewards_sum2, actions_mean1, actions_mean2,
                       rewards_avg1, rewards_avg2, title1="Bot1", title2="Bot2",
                       save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(rewards_sum1) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards_sum1, label=f"{title1} SumReward")
    plt.plot(episodes, rewards_sum2, label=f"{title2} SumReward")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.title(f"Rewards: {title1} vs {title2}")
    plt.legend()
    fname_rewards = os.path.join(save_dir, f"sum_rewards_{title1}_vs_{title2}.png").replace(" ", "_")
    plt.savefig(fname_rewards, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, actions_mean1, label=f"{title1} Cooperation")
    plt.plot(episodes, actions_mean2, label=f"{title2} Cooperation")
    plt.xlabel("Эпизод")
    plt.ylabel("Среднее действие (кооперация)")
    plt.title(f"Cooperation: {title1} vs {title2}")
    plt.legend()
    fname_actions = os.path.join(save_dir, f"cooperation_{title1}_vs_{title2}.png").replace(" ", "_")
    plt.savefig(fname_actions, dpi=150)
    plt.close()

    # Средняя награда за шаг
    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards_avg1, label=f"{title1} AvgRewardPerStep")
    plt.plot(episodes, rewards_avg2, label=f"{title2} AvgRewardPerStep")
    plt.xlabel("Эпизод")
    plt.ylabel("Средняя награда за шаг")
    plt.title(f"Avg reward per step: {title1} vs {title2}")
    plt.legend()
    fname_avg = os.path.join(save_dir, f"avg_step_reward_{title1}_vs_{title2}.png").replace(" ", "_")
    plt.savefig(fname_avg, dpi=150)
    plt.close()

    return fname_rewards, fname_actions, fname_avg


def plot_policy_distribution(bot, actions_all: List[float], title: str, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    # Гистограмма действий
    plt.figure(figsize=(8, 4))
    plt.hist(actions_all, bins=20, range=(0, 1), color='steelblue', alpha=0.8)
    plt.xlabel("Действие")
    plt.ylabel("Частота")
    plt.title(f"Action distribution: {title}")
    fname_hist = os.path.join(save_dir, f"action_dist_{title}.png").replace(" ", "_")
    plt.savefig(fname_hist, dpi=150)
    plt.close()

    # Для SoftmaxSarsaAgent дополнительно распределение политики
    fname_policy = None
    if isinstance(bot, SoftmaxSarsaAgent):
        probs = bot.get_action_probs()
        plt.figure(figsize=(8, 4))
        plt.bar(np.linspace(0, 1, len(probs)), probs, width=1/len(probs)*0.9)
        plt.xlabel("Действие (дискретизация)")
        plt.ylabel("pi(a)")
        plt.title(f"Final policy probs: {title}")
        fname_policy = os.path.join(save_dir, f"policy_probs_{title}.png").replace(" ", "_")
        plt.savefig(fname_policy, dpi=150)
        plt.close()
    return fname_hist, fname_policy

# -------------------------
# Параметры эксперимента (увеличенный размер игры)
# -------------------------
EPISODES = 500
STEPS = 100
RESULTS_DIR = "results"

# -------------------------
# Список ботов
# -------------------------
bots = [
    ("Simple_cooperate", SimpleBot("cooperate")),
    ("SAC", SACBot(alpha=1.2, device=device)),
    ("SARSA", SoftmaxSarsaAgent(num_actions=101, alpha=0.01, gamma=0.8, beta=0.8))
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
summary_rows = ["bot1,bot2,mean_reward1,mean_reward2,mean_coop1,mean_coop2,mean_avg_step_reward1,mean_avg_step_reward2"]

print(f"Запуск {len(pairs)} матчей...")
for (name1, b1), (name2, b2) in pairs:
    print(f"MATCH: {name1} vs {name2}")
    (r1_sum, r2_sum,
     a1_mean, a2_mean,
     r1_avg, r2_avg,
     all_a1, all_a2) = arena(b1, b2, episodes=EPISODES, max_steps=STEPS)
    match_dir = os.path.join(RESULTS_DIR, f"{name1}_vs_{name2}".replace(" ", "_"))
    plot_arena_results(r1_sum, r2_sum, a1_mean, a2_mean, r1_avg, r2_avg, title1=name1, title2=name2, save_dir=match_dir)
    # распределения политик
    plot_policy_distribution(b1, all_a1, name1, save_dir=match_dir)
    plot_policy_distribution(b2, all_a2, name2, save_dir=match_dir)
    summary_rows.append(
        f"{name1},{name2},{np.mean(r1_sum):.4f},{np.mean(r2_sum):.4f},{np.mean(a1_mean):.4f},{np.mean(a2_mean):.4f},{np.mean(r1_avg):.4f},{np.mean(r2_avg):.4f}"
    )

# -------------------------
# Multi-Agent SAC Match (двухагентный централизованный)
# -------------------------
def run_masac_match(system: MASACSystem, episodes: int, max_steps: int) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    env = ContinuousPDEnv()
    r1_sum_list: List[float] = []
    r2_sum_list: List[float] = []
    a1_mean_list: List[float] = []
    a2_mean_list: List[float] = []
    r1_avg_list: List[float] = []
    r2_avg_list: List[float] = []
    all_actions1: List[float] = []
    all_actions2: List[float] = []

    for ep in tqdm(range(episodes), desc="MASAC Episodes"):
        system.reset()
        env.reset()
        ep_r1 = 0.0
        ep_r2 = 0.0
        actions_ep_1: List[float] = []
        actions_ep_2: List[float] = []

        for _ in range(max_steps):
            # states для каждого агента
            states = [agent.get_state() for agent in system.agents]
            actions_tensors = [agent.choose_action() for agent in system.agents]
            a1 = float(actions_tensors[0].detach().cpu().item())
            a2 = float(actions_tensors[1].detach().cpu().item())
            _, (r1, r2), done = env.step(a1, a2)
            # обновить истории (каждый видит действие другого)
            system.update_histories([a1, a2])
            next_states = [agent.get_state() for agent in system.agents]
            # opponent actions для хранения (каждому передаём действие другого)
            opp_actions = [a2, a1]
            system.store_transitions(states,
                                     [torch.tensor([a1], dtype=torch.float32, device=device), torch.tensor([a2], dtype=torch.float32, device=device)],
                                     [r1, r2],
                                     next_states,
                                     [done, done],
                                     opponent_actions=opp_actions)
            system.update_all()

            ep_r1 += r1; ep_r2 += r2
            actions_ep_1.append(a1); actions_ep_2.append(a2)
            all_actions1.append(a1); all_actions2.append(a2)
            if done:
                break

        r1_sum_list.append(ep_r1)
        r2_sum_list.append(ep_r2)
        a1_mean_list.append(float(np.mean(actions_ep_1)))
        a2_mean_list.append(float(np.mean(actions_ep_2)))
        r1_avg_list.append(ep_r1 / max_steps)
        r2_avg_list.append(ep_r2 / max_steps)

    return (r1_sum_list, r2_sum_list,
            a1_mean_list, a2_mean_list,
            r1_avg_list, r2_avg_list,
            all_actions1, all_actions2)

print("Запуск Multi-Agent SAC матча...")
system_masac = MASACSystem(num_agents=2, alpha=0.2, gamma=0.99, tau=0.005, batch_size=64, device=device, history_window=20)
(masac_r1_sum, masac_r2_sum,
 masac_a1_mean, masac_a2_mean,
 masac_r1_avg, masac_r2_avg,
 masac_actions_all1, masac_actions_all2) = run_masac_match(system_masac, episodes=EPISODES, max_steps=STEPS)
masac_dir = os.path.join(RESULTS_DIR, "MASAC_Agent1_vs_MASAC_Agent2")
plot_arena_results(masac_r1_sum, masac_r2_sum, masac_a1_mean, masac_a2_mean, masac_r1_avg, masac_r2_avg,
                   title1="MASAC_Agent1", title2="MASAC_Agent2", save_dir=masac_dir)
plot_policy_distribution(system_masac.agents[0], masac_actions_all1, "MASAC_Agent1", save_dir=masac_dir)
plot_policy_distribution(system_masac.agents[1], masac_actions_all2, "MASAC_Agent2", save_dir=masac_dir)
summary_rows.append(
    f"MASAC_Agent1,MASAC_Agent2,{np.mean(masac_r1_sum):.4f},{np.mean(masac_r2_sum):.4f},{np.mean(masac_a1_mean):.4f},{np.mean(masac_a2_mean):.4f},{np.mean(masac_r1_avg):.4f},{np.mean(masac_r2_avg):.4f}"
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

# Диагностика: статистика Q для SARSA (проверка обновления)
for name, bot in bots:
    if isinstance(bot, SoftmaxSarsaAgent):
        q_vals = bot.Q[0]
        print(f"SARSA Q stats после обучения: min={q_vals.min():.4f} max={q_vals.max():.4f} mean={q_vals.mean():.4f}")

