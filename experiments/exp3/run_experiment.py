"""
Run key experiments for exp3:
1) SmartAgent vs always-betray bot (2 players, PD)
2) Two SmartAgents (2 players, PD)
3a) Three players — SmartAgent vs two betrayers (custom 3-player game)
3b) Three players — SmartAgent vs betrayer and cooperator (custom 3-player game)
4) Three SmartAgents (custom 3-player game)
"""

import random
from typing import List

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # fallback if tqdm is not available
    def tqdm(x):
        return x

from .bots import SmartAgent, SimpleBot
from .environment import GameFactory
from .viz_utils import plot_rewards_and_coop, smooth


def exp1_smart_vs_betrayer(rounds: int = 1000):
    print("\n=== Эксперимент 1: Умный агент против всегда предающего (2 игрока) ===")
    smart_agent = SmartAgent("Умник")
    opponent = SimpleBot("betray")
    pd_game = GameFactory.create_generalized_prisoners_dilemma(2)

    rewards: List[float] = []
    actions: List[int] = []
    for i in tqdm(range(rounds)):
        a = smart_agent.choose_action()
        b = opponent.choose_action()
        r1 = float(pd_game.get_payoff([a, b]))  # выплата игрока 0
        smart_agent.learn(a, r1)
        rewards.append(r1)
        actions.append(a)
        if i % 100 == 0 and i > 0:
            smart_agent.exploration = max(0.01, smart_agent.exploration * 0.9)

    smart_agent.print_stats()
    plot_rewards_and_coop(rewards, actions, title_prefix=smart_agent.name)


def exp2_two_smart_agents(rounds: int = 2000):
    print("\n=== Эксперимент 2: Два умных агента (2 игрока) ===")
    agent1 = SmartAgent("Алиса")
    agent2 = SmartAgent("Боб")
    pd_game = GameFactory.create_generalized_prisoners_dilemma(2)

    rewards1: List[float] = []
    rewards2: List[float] = []
    actions1: List[int] = []
    actions2: List[int] = []

    for i in tqdm(range(rounds)):
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        r1 = float(pd_game.get_payoff([a1, a2]))
        r2 = float(pd_game.get_payoff([a2, a1]))  # выплата игрока 1 как вращение профиля
        agent1.learn(a1, r1)
        agent2.learn(a2, r2)
        rewards1.append(r1); rewards2.append(r2)
        actions1.append(a1); actions2.append(a2)
        if i % 200 == 0 and i > 0:
            agent1.exploration = max(0.01, agent1.exploration * 0.9)
            agent2.exploration = max(0.01, agent2.exploration * 0.9)

    agent1.print_stats(); agent2.print_stats()

    # Небольшая визуализация
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    window = 50
    axes[0, 0].plot(smooth(rewards1, window), color='blue'); axes[0, 0].set_title('Награды Алисы')
    axes[0, 1].plot(smooth(rewards2, window), color='red');  axes[0, 1].set_title('Награды Боба')
    coop1 = [1 - np.mean(actions1[i:i + window]) for i in range(max(1, len(actions1) - window))]
    coop2 = [1 - np.mean(actions2[i:i + window]) for i in range(max(1, len(actions2) - window))]
    axes[1, 0].plot(coop1, color='blue'); axes[1, 0].set_title('Сотрудничество Алисы'); axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(coop2, color='red');  axes[1, 1].set_title('Сотрудничество Боба');  axes[1, 1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()


def exp3a_three_players_smart_vs_two_betrayers(rounds: int = 1000):
    print("\n=== Эксперимент 3a: Три игрока — умник против двух предателей ===")
    smart_agent = SmartAgent("Умник")
    opponent_1 = SimpleBot("betray")
    opponent_2 = SimpleBot("betray")
    tri_game = GameFactory.create_custom_three_player_game()

    rewards: List[float] = []
    actions: List[int] = []
    for i in tqdm(range(rounds)):
        a = smart_agent.choose_action()
        b = opponent_1.choose_action()
        c = opponent_2.choose_action()
        r1 = float(tri_game.get_payoff([a, b, c]))
        smart_agent.learn(a, r1)
        rewards.append(r1); actions.append(a)
        if i % 100 == 0 and i > 0:
            smart_agent.exploration = max(0.01, smart_agent.exploration * 0.9)

    smart_agent.print_stats()
    plot_rewards_and_coop(rewards, actions, title_prefix=smart_agent.name)


def exp3b_three_players_mixed(rounds: int = 1000):
    print("\n=== Эксперимент 3b: Три игрока — умник против предателя и кооператора ===")
    smart_agent = SmartAgent("Умник")
    opponent_1 = SimpleBot("betray")
    opponent_2 = SimpleBot("cooperate")
    tri_game = GameFactory.create_custom_three_player_game()

    rewards: List[float] = []
    actions: List[int] = []
    for i in tqdm(range(rounds)):
        a = smart_agent.choose_action()
        b = opponent_1.choose_action()
        c = opponent_2.choose_action()
        r1 = float(tri_game.get_payoff([a, b, c]))
        smart_agent.learn(a, r1)
        rewards.append(r1); actions.append(a)
        if i % 100 == 0 and i > 0:
            smart_agent.exploration = max(0.01, smart_agent.exploration * 0.9)

    smart_agent.print_stats()
    plot_rewards_and_coop(rewards, actions, title_prefix=smart_agent.name)


def exp4_three_smart_agents(rounds: int = 2000):
    print("\n=== Эксперимент 4: Три умных агента ===")
    agent1 = SmartAgent("Алиса")
    agent2 = SmartAgent("Боб")
    agent3 = SmartAgent("Чарли")
    tri_game = GameFactory.create_custom_three_player_game()

    rewards1: List[float] = []
    rewards2: List[float] = []
    rewards3: List[float] = []
    actions1: List[int] = []
    actions2: List[int] = []
    actions3: List[int] = []

    for i in tqdm(range(rounds)):
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        a3 = agent3.choose_action()
        r1 = float(tri_game.get_payoff([a1, a2, a3]))
        r2 = float(tri_game.get_payoff([a2, a3, a1]))
        r3 = float(tri_game.get_payoff([a3, a1, a2]))
        agent1.learn(a1, r1)
        agent2.learn(a2, r2)
        agent3.learn(a3, r3)
        rewards1.append(r1); rewards2.append(r2); rewards3.append(r3)
        actions1.append(a1); actions2.append(a2); actions3.append(a3)
        if i % 200 == 0 and i > 0:
            agent1.exploration = max(0.01, agent1.exploration * 0.9)
            agent2.exploration = max(0.01, agent2.exploration * 0.9)
            agent3.exploration = max(0.01, agent3.exploration * 0.9)

    agent1.print_stats(); agent2.print_stats(); agent3.print_stats()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    window = 50
    axes[0, 0].plot(smooth(rewards1, window), color='blue');  axes[0, 0].set_title('Награды Алисы')
    axes[1, 0].plot(smooth(rewards2, window), color='red');   axes[1, 0].set_title('Награды Боба')
    axes[2, 0].plot(smooth(rewards3, window), color='green'); axes[2, 0].set_title('Награды Чарли')
    coop1 = [1 - np.mean(actions1[i:i + window]) for i in range(max(1, len(actions1) - window))]
    coop2 = [1 - np.mean(actions2[i:i + window]) for i in range(max(1, len(actions2) - window))]
    coop3 = [1 - np.mean(actions3[i:i + window]) for i in range(max(1, len(actions3) - window))]
    axes[0, 1].plot(coop1, color='blue');  axes[0, 1].set_title('Сотрудничество Алисы');  axes[0, 1].set_ylim(0, 1)
    axes[1, 1].plot(coop2, color='red');   axes[1, 1].set_title('Сотрудничество Боба');   axes[1, 1].set_ylim(0, 1)
    axes[2, 1].plot(coop3, color='green'); axes[2, 1].set_title('Сотрудничество Чарли'); axes[2, 1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # Беглый прогон (можно закомментировать ненужные)
    exp1_smart_vs_betrayer()
    exp2_two_smart_agents()
    exp3a_three_players_smart_vs_two_betrayers()
    exp3b_three_players_mixed()
    exp4_three_smart_agents()
