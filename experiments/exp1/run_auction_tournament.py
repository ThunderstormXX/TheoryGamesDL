#!/usr/bin/env python3
"""
Эксперимент 1: Турнирная система двустороннего аукциона с AlphaRank анализом

Обучает нейросетевых агентов в двустороннем аукционе с дискретными ставками,
вычисляет матрицы переходов и анализирует эволюцию стационарного распределения.
"""

import sys
import os

# Добавляем путь к папке эксперимента
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from auction_tournament import AuctionTournament
import numpy as np

def run_small_auction():
    """Малый аукцион: 2 покупателя, 2 продавца"""
    print("=== Малый аукцион: 2x2 ===")
    
    tournament = AuctionTournament(
        n_buyers=2,
        n_sellers=2,
        n_actions=11,  # Ставки от 0 до 10
        auctions_per_pair=100
    )
    
    tournament.run_evolution(rounds=15)
    tournament.plot_evolution("small_auction")
    filename = tournament.save_results("small_auction_results.json")
    
    return tournament, filename

def run_medium_auction():
    """Средний аукцион: 3 покупателя, 3 продавца"""
    print("\n=== Средний аукцион: 3x3 ===")
    
    tournament = AuctionTournament(
        n_buyers=3,
        n_sellers=3,
        n_actions=11,
        auctions_per_pair=80
    )
    
    tournament.run_evolution(rounds=20)
    tournament.plot_evolution("medium_auction")
    filename = tournament.save_results("medium_auction_results.json")
    
    return tournament, filename

def run_large_auction():
    """Большой аукцион: 4 покупателя, 4 продавца"""
    print("\n=== Большой аукцион: 4x4 ===")
    
    tournament = AuctionTournament(
        n_buyers=4,
        n_sellers=4,
        n_actions=11,
        auctions_per_pair=60
    )
    
    tournament.run_evolution(rounds=25)
    tournament.plot_evolution("large_auction")
    filename = tournament.save_results("large_auction_results.json")
    
    return tournament, filename

def analyze_auction_results(tournament):
    """Анализирует результаты аукционного турнира"""
    print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
    
    # Финальные данные
    final_stationary = np.array(tournament.history['stationary_distributions'][-1])
    final_rewards = np.array(tournament.history['mean_rewards'][-1])
    final_trade_stats = tournament.history['trade_statistics'][-1]
    
    print(f"Финальная частота сделок: {final_trade_stats['avg_trade_rate']:.3f}")
    
    print(f"\nФинальное стационарное распределение:")
    for i, agent in enumerate(tournament.agents):
        print(f"  {agent}: {final_stationary[i]:.4f}")
    
    print(f"\nФинальные средние награды:")
    for i, agent in enumerate(tournament.agents):
        print(f"  {agent}: {final_rewards[i]:.4f}")
    
    # Корреляция
    correlation = np.corrcoef(final_stationary, final_rewards)[0, 1]
    if not np.isnan(correlation):
        print(f"\nКорреляция между стационарным распределением и наградами: {correlation:.4f}")
    
    # Анализ по типам агентов
    buyer_indices = [i for i, agent in enumerate(tournament.agents) if agent.agent_type == 'buyer']
    seller_indices = [i for i, agent in enumerate(tournament.agents) if agent.agent_type == 'seller']
    
    buyer_stationary = final_stationary[buyer_indices]
    seller_stationary = final_stationary[seller_indices]
    buyer_rewards = final_rewards[buyer_indices]
    seller_rewards = final_rewards[seller_indices]
    
    print(f"\nАнализ покупателей:")
    print(f"  Средняя доля в стационарном распределении: {buyer_stationary.mean():.4f}")
    print(f"  Средняя награда: {buyer_rewards.mean():.4f}")
    print(f"  Лучший покупатель: {tournament.buyers[np.argmax(buyer_rewards)]}")
    
    print(f"\nАнализ продавцов:")
    print(f"  Средняя доля в стационарном распределении: {seller_stationary.mean():.4f}")
    print(f"  Средняя награда: {seller_rewards.mean():.4f}")
    print(f"  Лучший продавец: {tournament.sellers[np.argmax(seller_rewards)]}")
    
    # Анализ победителей и проигравших
    print(f"\nАнализ победителей и проигравших:")
    
    # Сортируем по AlphaRank
    alpharank_ranking = np.argsort(final_stationary)[::-1]
    reward_ranking = np.argsort(final_rewards)[::-1]
    
    print(f"  Топ-3 по AlphaRank:")
    for i in range(min(3, len(alpharank_ranking))):
        agent_idx = alpharank_ranking[i]
        agent = tournament.agents[agent_idx]
        print(f"    {i+1}. {agent}: AlphaRank={final_stationary[agent_idx]:.4f}, Выгодность={final_rewards[agent_idx]:.4f}")
    
    print(f"  Топ-3 по выгодности:")
    for i in range(min(3, len(reward_ranking))):
        agent_idx = reward_ranking[i]
        agent = tournament.agents[agent_idx]
        print(f"    {i+1}. {agent}: Выгодность={final_rewards[agent_idx]:.4f}, AlphaRank={final_stationary[agent_idx]:.4f}")
    
    # Анализ стратегий
    print(f"\nАнализ стратегий:")
    for agent in tournament.agents:
        if agent.strategy_history:
            final_strategy = agent.strategy_history[-1]
            preferred_bid = np.argmax(final_strategy)
            strategy_entropy = -np.sum(final_strategy * np.log(final_strategy + 1e-10))
            print(f"  {agent}: предпочитаемая ставка={preferred_bid}, prob={final_strategy[preferred_bid]:.3f}, энтропия={strategy_entropy:.3f}")
    
    # Анализ симметричности задачи
    buyer_rewards = [final_rewards[i] for i, agent in enumerate(tournament.agents) if agent.agent_type == 'buyer']
    seller_rewards = [final_rewards[i] for i, agent in enumerate(tournament.agents) if agent.agent_type == 'seller']
    
    print(f"\nАнализ асимметричности:")
    print(f"  Средняя выгодность покупателей: {np.mean(buyer_rewards):.4f}")
    print(f"  Средняя выгодность продавцов: {np.mean(seller_rewards):.4f}")
    print(f"  Разность: {np.mean(buyer_rewards) - np.mean(seller_rewards):.4f}")
    
    return {
        'final_stationary': final_stationary.tolist(),
        'final_rewards': final_rewards.tolist(),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'trade_rate': final_trade_stats['avg_trade_rate'],
        'buyer_avg_reward': float(buyer_rewards.mean()),
        'seller_avg_reward': float(seller_rewards.mean()),
        'alpharank_ranking': alpharank_ranking.tolist(),
        'reward_ranking': reward_ranking.tolist(),
        'asymmetry': float(np.mean(buyer_rewards) - np.mean(seller_rewards))
    }

def main():
    """Главная функция"""
    print("Запуск турнирной системы двустороннего аукциона с AlphaRank анализом...")
    
    experiments = [
        ("Small Auction", run_small_auction),
        ("Medium Auction", run_medium_auction),
        ("Large Auction", run_large_auction)
    ]
    
    results = {}
    
    for name, experiment_func in experiments:
        print(f"\n{'='*60}")
        print(f"Запуск эксперимента: {name}")
        print(f"{'='*60}")
        
        try:
            tournament, filename = experiment_func()
            analysis = analyze_auction_results(tournament)
            
            results[name] = {
                'tournament': tournament,
                'filename': filename,
                'analysis': analysis
            }
            
            print(f"✓ Эксперимент {name} завершен успешно")
            
        except Exception as e:
            print(f"✗ Ошибка в эксперименте {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}
    
    # Общая сводка
    print(f"\n{'='*60}")
    print("ОБЩАЯ СВОДКА")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"\n{name}: ОШИБКА - {result['error']}")
        else:
            analysis = result['analysis']
            print(f"\n{name}:")
            print(f"  Файл: {result['filename']}")
            print(f"  Частота сделок: {analysis['trade_rate']:.3f}")
            print(f"  Корреляция: {analysis['correlation']:.4f}")
            print(f"  Средняя награда покупателей: {analysis['buyer_avg_reward']:.4f}")
            print(f"  Средняя награда продавцов: {analysis['seller_avg_reward']:.4f}")

if __name__ == "__main__":
    main()