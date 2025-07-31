import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from alpharank_agent import AlphaRankAgent, compute_alpharank, compute_payoff_matrix

class AlphaRankSimulation:
    def __init__(self, n_agents=2, game_payoffs=[3, 1, 0, 4], lr=0.001):
        self.n_agents = n_agents
        self.game_payoffs = game_payoffs
        self.agents = [AlphaRankAgent(i, lr=lr) for i in range(n_agents)]
        self.history = {
            'rewards': [],
            'alpharank_scores': [],
            'strategies': [],
            'actions': []
        }
        
    def step(self, episode):
        """Один шаг симуляции"""
        # Получаем действия всех агентов
        states = [np.random.rand(4) for _ in range(self.n_agents)]  # Случайные состояния
        actions = []
        
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i])
            actions.append(action)
        
        # Вычисляем матрицу выплат
        payoff_matrix = compute_payoff_matrix(self.agents, self.game_payoffs)
        
        # Вычисляем AlphaRank
        alpharank_scores = compute_alpharank(payoff_matrix)
        
        # Назначаем rewards на основе AlphaRank
        rewards = []
        for i, agent in enumerate(self.agents):
            # Reward = доля агента в AlphaRank
            agent_score = alpharank_scores[i].sum()
            rewards.append(agent_score)
        
        # Обучаем агентов
        next_states = [np.random.rand(4) for _ in range(self.n_agents)]
        for i, agent in enumerate(self.agents):
            try:
                agent.remember(states[i], actions[i], rewards[i], next_states[i], False)
                if len(agent.memory) > 32:
                    agent.replay(32)
            except Exception as e:
                print(f"Ошибка обучения агента {i}: {e}")
        
        # Сохраняем историю
        self.history['rewards'].append(rewards)
        self.history['alpharank_scores'].append(alpharank_scores.tolist())
        self.history['strategies'].append([agent.get_strategy().tolist() for agent in self.agents])
        self.history['actions'].append(actions)
        
        # Обновляем историю стратегий агентов
        for agent in self.agents:
            agent.update_strategy_history()
    
    def run(self, episodes=1000):
        """Запуск симуляции"""
        print(f"Запуск AlphaRank симуляции на {episodes} эпизодов...")
        
        for episode in range(episodes):
            self.step(episode)
            
            if episode % 100 == 0:
                avg_reward = np.mean([r[0] for r in self.history['rewards'][-100:]])
                print(f"Эпизод {episode}, средний reward: {avg_reward:.4f}")
        
        print("Симуляция завершена!")
    
    def save_results(self, filename=None):
        """Сохранение результатов"""
        # Определяем папку эксперимента
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(logs_dir, f"alpharank_results_{timestamp}.json")
        else:
            filename = os.path.join(logs_dir, filename)
        
        # Конвертируем numpy типы в обычные Python типы
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            return obj
        
        results = {
            'config': {
                'n_agents': int(self.n_agents),
                'game_payoffs': [float(x) for x in self.game_payoffs],
                'episodes': len(self.history['rewards'])
            },
            'history': convert_numpy(self.history)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Результаты сохранены в {filename}")
        return filename
    
    def plot_results(self):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График rewards
        rewards_array = np.array(self.history['rewards'])
        for i in range(self.n_agents):
            axes[0, 0].plot(rewards_array[:, i], label=f'Agent {i}')
        axes[0, 0].set_title('AlphaRank Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        
        # График стратегий (вероятность кооперации)
        strategies_array = np.array(self.history['strategies'])
        for i in range(self.n_agents):
            coop_probs = strategies_array[:, i, 0]  # Вероятность кооперации
            axes[0, 1].plot(coop_probs, label=f'Agent {i}')
        axes[0, 1].set_title('Cooperation Probability')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('P(Cooperate)')
        axes[0, 1].legend()
        
        # График AlphaRank scores
        alpharank_array = np.array(self.history['alpharank_scores'])
        for i in range(self.n_agents):
            scores = alpharank_array[:, i, :].sum(axis=1)
            axes[1, 0].plot(scores, label=f'Agent {i}')
        axes[1, 0].set_title('AlphaRank Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        
        # Гистограмма финальных стратегий
        final_strategies = strategies_array[-1]
        x = np.arange(self.n_agents)
        width = 0.35
        axes[1, 1].bar(x - width/2, final_strategies[:, 0], width, label='Cooperate')
        axes[1, 1].bar(x + width/2, final_strategies[:, 1], width, label='Defect')
        axes[1, 1].set_title('Final Strategies')
        axes[1, 1].set_xlabel('Agent')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_xticks(x)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Определяем папку эксперимента
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(script_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        viz_path = os.path.join(viz_dir, 'alpharank_results.png')
        plt.savefig(viz_path)
        print(f"График сохранен в {viz_path}")
        plt.close()

if __name__ == "__main__":
    # Запуск эксперимента
    sim = AlphaRankSimulation(n_agents=2, game_payoffs=[3, 1, 0, 4])
    sim.run(episodes=500)
    sim.save_results()
    sim.plot_results()