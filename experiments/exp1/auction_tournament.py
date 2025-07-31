import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from auction_agent import AuctionAgent, compute_auction_payoffs, create_auction_agents
import json
import os
from datetime import datetime

class AuctionTournament:
    """Турнирная система для двустороннего аукциона с AlphaRank"""
    
    def __init__(self, n_buyers=3, n_sellers=3, n_actions=11, auctions_per_pair=100):
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.n_actions = n_actions
        self.auctions_per_pair = auctions_per_pair
        
        # Создаем агентов
        self.agents = create_auction_agents(n_buyers, n_sellers, n_actions)
        self.n_agents = len(self.agents)
        
        # Разделяем агентов по типам
        self.buyers = [agent for agent in self.agents if agent.agent_type == 'buyer']
        self.sellers = [agent for agent in self.agents if agent.agent_type == 'seller']
        
        # История
        self.history = {
            'payoff_matrices': [],
            'transition_matrices': [],
            'stationary_distributions': [],
            'mean_rewards': [],
            'round_results': [],
            'trade_statistics': []
        }
        
    def run_auction_pair(self, buyer, seller, episodes=50):
        """Проводит серию аукционов между покупателем и продавцом"""
        buyer_rewards = []
        seller_rewards = []
        trades_completed = 0
        
        for episode in range(episodes):
            # Получаем ставки
            buyer_action, buyer_log_prob = buyer.sample_action()
            seller_action, seller_log_prob = seller.sample_action()
            
            # Вычисляем выплаты
            buyer_reward, seller_reward = compute_auction_payoffs(
                buyer_action, seller_action, buyer.value_or_cost, seller.value_or_cost
            )
            
            buyer_rewards.append(buyer_reward)
            seller_rewards.append(seller_reward)
            
            if buyer_reward > 0 or seller_reward > 0:
                trades_completed += 1
            
            # Немедленное обучение
            buyer.learn_from_action(buyer_log_prob, buyer_reward)
            seller.learn_from_action(seller_log_prob, seller_reward)
            
            # Сохраняем для статистики
            buyer.remember(1.0, buyer_action, buyer_reward, 1.0, False)
            seller.remember(1.0, seller_action, seller_reward, 1.0, False)
        
        avg_buyer_reward = np.mean(buyer_rewards)
        avg_seller_reward = np.mean(seller_rewards)
        trade_rate = trades_completed / episodes
        
        return avg_buyer_reward, avg_seller_reward, trade_rate
    
    def run_tournament_round(self):
        """Проводит один раунд турнира между всеми парами покупатель-продавец"""
        payoff_matrix = np.zeros((self.n_agents, self.n_agents))
        round_results = {}
        trade_stats = {'total_trades': 0, 'total_pairs': 0, 'trade_rates': []}
        
        # Проводим аукционы между всеми парами покупатель-продавец
        for buyer in self.buyers:
            for seller in self.sellers:
                buyer_reward, seller_reward, trade_rate = self.run_auction_pair(
                    buyer, seller, self.auctions_per_pair
                )
                
                # Заполняем матрицу выплат
                payoff_matrix[buyer.agent_id, seller.agent_id] = buyer_reward
                payoff_matrix[seller.agent_id, buyer.agent_id] = seller_reward
                
                # Сохраняем результаты
                pair_key = f"buyer_{buyer.agent_id}_seller_{seller.agent_id}"
                round_results[pair_key] = {
                    'buyer_reward': float(buyer_reward),
                    'seller_reward': float(seller_reward),
                    'trade_rate': float(trade_rate)
                }
                
                trade_stats['total_pairs'] += 1
                trade_stats['trade_rates'].append(trade_rate)
                if trade_rate > 0:
                    trade_stats['total_trades'] += 1
        
        # Заполняем диагональ средними значениями
        for i, agent in enumerate(self.agents):
            if agent.agent_type == 'buyer':
                payoff_matrix[i, i] = np.mean([payoff_matrix[i, j] for j in range(self.n_agents) if i != j and payoff_matrix[i, j] != 0])
            else:
                payoff_matrix[i, i] = np.mean([payoff_matrix[i, j] for j in range(self.n_agents) if i != j and payoff_matrix[i, j] != 0])
            
            # Если нет данных, используем 0
            if np.isnan(payoff_matrix[i, i]):
                payoff_matrix[i, i] = 0
        
        trade_stats['avg_trade_rate'] = np.mean(trade_stats['trade_rates'])
        
        return payoff_matrix, round_results, trade_stats
    
    def compute_transition_matrix(self, payoff_matrix, temperature=1.0):
        """Вычисляет матрицу переходов на основе выплат"""
        n = self.n_agents
        transition_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Вероятность перехода от i к j основана на разности выплат
                    payoff_diff = payoff_matrix[j, :].mean() - payoff_matrix[i, :].mean()
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
        print(f"Запуск эволюции аукционной системы на {rounds} раундов...")
        print(f"Покупатели: {len(self.buyers)}, Продавцы: {len(self.sellers)}")
        
        for round_num in range(rounds):
            print(f"Раунд {round_num + 1}/{rounds}")
            
            # Проводим турнир
            payoff_matrix, round_results, trade_stats = self.run_tournament_round()
            
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
            self.history['trade_statistics'].append(trade_stats)
            
            # Обновляем историю стратегий агентов
            for agent in self.agents:
                agent.update_strategy_history()
            
            # Выводим текущие результаты
            print(f"  Средняя частота сделок: {trade_stats['avg_trade_rate']:.3f}")
            print(f"  Стационарное распределение: {stationary_dist}")
            print(f"  Средние награды: {mean_rewards}")
            
            # Корреляция
            if len(stationary_dist) > 1 and len(mean_rewards) > 1:
                corr = np.corrcoef(stationary_dist, mean_rewards)[0,1]
                if not np.isnan(corr):
                    print(f"  Корреляция: {corr:.3f}")
        
        print("Эволюция завершена!")
    
    def plot_evolution(self, experiment_name="auction"):
        """Визуализирует эволюцию системы"""
        rounds = len(self.history['stationary_distributions'])
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Цветовая палитра для всех агентов
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.agents)))
        
        # 1. Эволюция стационарного распределения
        stationary_data = np.array(self.history['stationary_distributions'])
        for agent_id, agent in enumerate(self.agents):
            axes[0, 0].plot(stationary_data[:, agent_id], 
                           label=str(agent), marker='o', color=colors[agent_id], alpha=0.8)
        axes[0, 0].set_title('Эволюция AlphaRank (стационарное распределение)')
        axes[0, 0].set_xlabel('Раунд')
        axes[0, 0].set_ylabel('AlphaRank скор')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Эволюция средних наград (выгодность)
        rewards_data = np.array(self.history['mean_rewards'])
        for agent_id, agent in enumerate(self.agents):
            axes[0, 1].plot(rewards_data[:, agent_id], 
                           label=str(agent), marker='s', color=colors[agent_id], alpha=0.8)
        axes[0, 1].set_title('Эволюция средней выгодности')
        axes[0, 1].set_xlabel('Раунд')
        axes[0, 1].set_ylabel('Средняя выгодность')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Частота сделок и анализ победителей
        trade_rates = [stats['avg_trade_rate'] for stats in self.history['trade_statistics']]
        axes[0, 2].plot(trade_rates, marker='d', color='green', linewidth=2, label='Частота сделок')
        
        # Добавляем анализ победителей по раундам
        winner_rates = []
        for round_num in range(rounds):
            round_rewards = rewards_data[round_num]
            positive_rewards = np.sum(round_rewards > 0) / len(round_rewards)
            winner_rates.append(positive_rewards)
        
        axes[0, 2].plot(winner_rates, marker='o', color='orange', linewidth=2, alpha=0.7, label='Доля победителей')
        axes[0, 2].set_title('Эффективность системы')
        axes[0, 2].set_xlabel('Раунд')
        axes[0, 2].set_ylabel('Доля')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Корреляция между AlphaRank и выгодностью
        correlations = []
        for round_num in range(rounds):
            stat_dist = stationary_data[round_num]
            rewards = rewards_data[round_num]
            corr = np.corrcoef(stat_dist, rewards)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        axes[1, 0].plot(correlations, marker='d', color='purple', linewidth=2)
        axes[1, 0].set_title('Корреляция: AlphaRank vs Выгодность')
        axes[1, 0].set_xlabel('Раунд')
        axes[1, 0].set_ylabel('Корреляция')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_ylim(-1, 1)
        
        # 5. Финальные стратегии всех агентов
        final_strategies = []
        agent_labels = []
        for agent in self.agents:
            if agent.strategy_history:
                final_strategies.append(agent.strategy_history[-1])
                agent_labels.append(str(agent))
        
        if final_strategies:
            final_strategies = np.array(final_strategies)
            im1 = axes[1, 1].imshow(final_strategies, cmap='viridis', aspect='auto')
            axes[1, 1].set_title('Финальные стратегии всех агентов')
            axes[1, 1].set_xlabel('Ставка (0-10)')
            axes[1, 1].set_ylabel('Агент')
            axes[1, 1].set_yticks(range(len(agent_labels)))
            axes[1, 1].set_yticklabels(agent_labels, fontsize=8)
            axes[1, 1].set_xticks(range(0, self.n_actions, 2))
            plt.colorbar(im1, ax=axes[1, 1], label='Вероятность')
        
        # 6. Анализ победителей: финальное сравнение
        final_rewards = rewards_data[-1]
        final_alpharank = stationary_data[-1]
        
        # Сортируем по AlphaRank
        sorted_indices = np.argsort(final_alpharank)[::-1]
        
        x_pos = np.arange(len(self.agents))
        bars1 = axes[1, 2].bar(x_pos - 0.2, final_alpharank[sorted_indices], 0.4, 
                              label='AlphaRank скор', alpha=0.7, color='skyblue')
        
        # Нормализуем выгодность для сравнения
        normalized_rewards = final_rewards[sorted_indices]
        max_reward = np.max(np.abs(normalized_rewards)) if np.max(np.abs(normalized_rewards)) > 0 else 1
        normalized_rewards = normalized_rewards / max_reward * np.max(final_alpharank)
        
        bars2 = axes[1, 2].bar(x_pos + 0.2, normalized_rewards, 0.4, 
                              label='Норм. выгодность', alpha=0.7, color='lightcoral')
        
        axes[1, 2].set_title('Финальное сравнение (по AlphaRank)')
        axes[1, 2].set_xlabel('Агент (отсортировано по AlphaRank)')
        axes[1, 2].set_ylabel('Нормализованное значение')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels([str(self.agents[i]) for i in sorted_indices], 
                                  rotation=45, ha='right', fontsize=8)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Добавляем общий заголовок
        fig.suptitle(f'Анализ аукционной системы: {experiment_name}', 
                    fontsize=16, y=0.98)
        
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
            filename = os.path.join(logs_dir, f"auction_results_{timestamp}.json")
        else:
            filename = os.path.join(logs_dir, filename)
        
        # Информация об агентах
        agents_info = []
        for agent in self.agents:
            agents_info.append({
                'id': agent.agent_id,
                'type': agent.agent_type,
                'value_or_cost': agent.value_or_cost
            })
        
        results = {
            'config': {
                'n_buyers': self.n_buyers,
                'n_sellers': self.n_sellers,
                'n_actions': self.n_actions,
                'auctions_per_pair': self.auctions_per_pair,
                'rounds': len(self.history['stationary_distributions'])
            },
            'agents': agents_info,
            'history': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Результаты сохранены в {filename}")
        return filename