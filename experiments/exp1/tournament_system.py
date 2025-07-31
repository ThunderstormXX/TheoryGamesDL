import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from simple_agent import SimpleQLearningAgent
import json
import os
from datetime import datetime

class TournamentSystem:
    def __init__(self, n_agents=6, game_payoffs=[3, 1, 0, 4], games_per_pair=100):
        self.n_agents = n_agents
        self.game_payoffs = game_payoffs
        self.games_per_pair = games_per_pair
        
        # Создаем агентов
        self.agents = [SimpleQLearningAgent(i, lr=0.1, epsilon=0.1) for i in range(n_agents)]
        
        # История
        self.history = {
            'payoff_matrices': [],
            'transition_matrices': [],
            'stationary_distributions': [],
            'mean_rewards': [],
            'round_results': []
        }
        
    def play_game(self, agent1, agent2, episodes=50):
        """Играет серию игр между двумя агентами"""
        rewards1, rewards2 = [], []
        
        for episode in range(episodes):
            # Случайные состояния
            state1 = np.random.rand(4)
            state2 = np.random.rand(4)
            
            # Действия агентов
            action1 = agent1.act(state1)
            action2 = agent2.act(state2)
            
            # Вычисляем награды на основе матрицы выплат
            reward1, reward2 = self.compute_rewards(action1, action2)
            
            rewards1.append(reward1)
            rewards2.append(reward2)
            
            # Обучение агентов
            next_state1 = np.random.rand(4)
            next_state2 = np.random.rand(4)
            
            agent1.remember(state1, action1, reward1, next_state1, False)
            agent2.remember(state2, action2, reward2, next_state2, False)
            
            if len(agent1.memory) > 32:
                agent1.replay(16)
            if len(agent2.memory) > 32:
                agent2.replay(16)
        
        return np.mean(rewards1), np.mean(rewards2)
    
    def compute_rewards(self, action1, action2):
        """Вычисляет награды на основе действий"""
        # game_payoffs = [CC, DD, DC, CD]
        if action1 == 0 and action2 == 0:  # CC
            return self.game_payoffs[0], self.game_payoffs[0]
        elif action1 == 1 and action2 == 1:  # DD
            return self.game_payoffs[1], self.game_payoffs[1]
        elif action1 == 1 and action2 == 0:  # DC
            return self.game_payoffs[2], self.game_payoffs[3]
        else:  # CD
            return self.game_payoffs[3], self.game_payoffs[2]
    
    def run_tournament_round(self):
        """Проводит один раунд турнира между всеми парами агентов"""
        payoff_matrix = np.zeros((self.n_agents, self.n_agents))
        round_results = {}
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    # Играем между агентами i и j
                    reward_i, reward_j = self.play_game(
                        self.agents[i], self.agents[j], self.games_per_pair
                    )
                    payoff_matrix[i, j] = reward_i
                    round_results[f"{i}_vs_{j}"] = {
                        'agent_i_reward': float(reward_i),
                        'agent_j_reward': float(reward_j)
                    }
                else:
                    # Самоигра - средняя награда
                    payoff_matrix[i, i] = np.mean(self.game_payoffs[:2])  # Среднее от CC и DD
        
        return payoff_matrix, round_results
    
    def compute_transition_matrix(self, payoff_matrix, temperature=1.0):
        """Вычисляет матрицу переходов на основе выплат"""
        n = self.n_agents
        transition_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Вероятность перехода от i к j основана на разности выплат
                    payoff_diff = payoff_matrix[j, i] - payoff_matrix[i, i]
                    transition_matrix[i, j] = np.exp(payoff_diff / temperature)
                else:
                    # Вероятность остаться в том же состоянии
                    transition_matrix[i, i] = 1.0
        
        # Нормализация строк
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def compute_stationary_distribution(self, transition_matrix):
        """Вычисляет стационарное распределение"""
        try:
            eigenvalues, eigenvectors = eig(transition_matrix.T)
            # Находим собственный вектор с собственным значением 1
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = np.abs(stationary_dist)
            
            # Нормализация
            if stationary_dist.sum() > 0:
                stationary_dist = stationary_dist / stationary_dist.sum()
            else:
                stationary_dist = np.ones(self.n_agents) / self.n_agents
                
        except Exception as e:
            print(f"Ошибка вычисления стационарного распределения: {e}")
            stationary_dist = np.ones(self.n_agents) / self.n_agents
        
        return stationary_dist
    
    def run_evolution(self, rounds=20):
        """Запускает эволюцию системы на несколько раундов"""
        print(f"Запуск эволюции на {rounds} раундов с {self.n_agents} агентами...")
        
        for round_num in range(rounds):
            print(f"Раунд {round_num + 1}/{rounds}")
            
            # Проводим турнир
            payoff_matrix, round_results = self.run_tournament_round()
            
            # Вычисляем матрицу переходов
            transition_matrix = self.compute_transition_matrix(payoff_matrix)
            
            # Вычисляем стационарное распределение
            stationary_dist = self.compute_stationary_distribution(transition_matrix)
            
            # Средние награды агентов
            mean_rewards = np.mean(payoff_matrix, axis=1)
            
            # Сохраняем историю
            self.history['payoff_matrices'].append(payoff_matrix.tolist())
            self.history['transition_matrices'].append(transition_matrix.tolist())
            self.history['stationary_distributions'].append(stationary_dist.tolist())
            self.history['mean_rewards'].append(mean_rewards.tolist())
            self.history['round_results'].append(round_results)
            
            # Выводим текущие результаты
            print(f"  Стационарное распределение: {stationary_dist}")
            print(f"  Средние награды: {mean_rewards}")
            print(f"  Корреляция: {np.corrcoef(stationary_dist, mean_rewards)[0,1]:.3f}")
        
        print("Эволюция завершена!")
    
    def plot_evolution(self, experiment_name="tournament"):
        """Визуализирует эволюцию системы"""
        rounds = len(self.history['stationary_distributions'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Эволюция стационарного распределения
        stationary_data = np.array(self.history['stationary_distributions'])
        for agent_id in range(self.n_agents):
            axes[0, 0].plot(stationary_data[:, agent_id], 
                           label=f'Agent {agent_id}', marker='o')
        axes[0, 0].set_title('Эволюция стационарного распределения')
        axes[0, 0].set_xlabel('Раунд')
        axes[0, 0].set_ylabel('Вероятность в стационарном распределении')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Эволюция средних наград
        rewards_data = np.array(self.history['mean_rewards'])
        for agent_id in range(self.n_agents):
            axes[0, 1].plot(rewards_data[:, agent_id], 
                           label=f'Agent {agent_id}', marker='s')
        axes[0, 1].set_title('Эволюция средних наград')
        axes[0, 1].set_xlabel('Раунд')
        axes[0, 1].set_ylabel('Средняя награда')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Корреляция между стационарным распределением и наградами
        correlations = []
        for round_num in range(rounds):
            stat_dist = stationary_data[round_num]
            rewards = rewards_data[round_num]
            corr = np.corrcoef(stat_dist, rewards)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        axes[1, 0].plot(correlations, marker='d', color='red', linewidth=2)
        axes[1, 0].set_title('Корреляция: Стационарное распределение vs Награды')
        axes[1, 0].set_xlabel('Раунд')
        axes[1, 0].set_ylabel('Коэффициент корреляции')
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Финальное сравнение
        final_stat = stationary_data[-1]
        final_rewards = rewards_data[-1]
        
        x = np.arange(self.n_agents)
        width = 0.35
        
        axes[1, 1].bar(x - width/2, final_stat, width, 
                      label='Стационарное распределение', alpha=0.7)
        
        # Нормализуем награды для сравнения
        normalized_rewards = final_rewards / np.sum(final_rewards)
        axes[1, 1].bar(x + width/2, normalized_rewards, width, 
                      label='Нормализованные награды', alpha=0.7)
        
        axes[1, 1].set_title('Финальное сравнение')
        axes[1, 1].set_xlabel('Агент')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Agent {i}' for i in range(self.n_agents)])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Определяем папку эксперимента
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(script_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(viz_dir, f'{experiment_name}_evolution_{timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {viz_path}")
        plt.close()
        return viz_path
    
    def save_results(self, filename=None):
        """Сохраняет результаты"""
        # Определяем папку эксперимента
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(logs_dir, f"tournament_results_{timestamp}.json")
        else:
            filename = os.path.join(logs_dir, filename)
        
        results = {
            'config': {
                'n_agents': self.n_agents,
                'game_payoffs': self.game_payoffs,
                'games_per_pair': self.games_per_pair,
                'rounds': len(self.history['stationary_distributions'])
            },
            'history': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Результаты сохранены в {filename}")
        return filename