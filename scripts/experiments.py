import numpy as np
import matplotlib.pyplot as plt
from game_theory import prisoners_dilemma, coordination_game, matching_pennies
from models import train_players, evaluate_equilibrium

def run_experiment(game_name: str, game, n_episodes: int = 1000):
    """Run single experiment on a game"""
    print(f"\n=== {game_name} ===")
    
    # Train players
    player1, player2 = train_players(game, n_episodes)
    
    # Evaluate results
    results = evaluate_equilibrium(game, player1, player2)
    
    print(f"Final strategies:")
    print(f"Player 1: {results['strategy_p1']}")
    print(f"Player 2: {results['strategy_p2']}")
    print(f"Payoffs: P1={results['payoff_p1']:.3f}, P2={results['payoff_p2']:.3f}")
    print(f"Nash Equilibrium: {results['is_nash']}")
    
    return results

def run_all_experiments():
    """Run experiments on all game setups"""
    games = {
        "Prisoner's Dilemma": prisoners_dilemma(),
        "Coordination Game": coordination_game(),
        "Matching Pennies": matching_pennies()
    }
    
    results = {}
    for name, game in games.items():
        results[name] = run_experiment(name, game)
    
    return results

def compare_models(game, model_configs: list, n_episodes: int = 1000):
    """Compare different model architectures"""
    from models import GamePlayer
    
    results = []
    
    for i, config in enumerate(model_configs):
        print(f"\nTesting config {i+1}: {config}")
        
        n_actions = game.payoff_p1.shape[0]
        player1 = GamePlayer(n_actions, config)
        player2 = GamePlayer(n_actions, config)
        
        # Simple training loop
        s1 = np.random.dirichlet(np.ones(n_actions))
        s2 = np.random.dirichlet(np.ones(n_actions))
        
        payoffs_history = []
        
        for episode in range(n_episodes):
            s1 = player1.get_strategy(s2)
            s2 = player2.get_strategy(s1)
            
            payoff1, payoff2 = game.get_payoffs((s1, s2))
            payoffs_history.append((payoff1, payoff2))
            
            player1.update(s2, payoff1)
            player2.update(s1, payoff2)
        
        final_result = evaluate_equilibrium(game, player1, player2)
        final_result['payoffs_history'] = payoffs_history
        results.append(final_result)
    
    return results

if __name__ == "__main__":
    # Run all experiments
    all_results = run_all_experiments()
    
    # Compare different architectures on Prisoner's Dilemma
    print("\n=== Model Architecture Comparison ===")
    configs = [[32], [64, 32], [128, 64, 32]]
    pd_game = prisoners_dilemma()
    model_results = compare_models(pd_game, configs, 500)
    
    for i, result in enumerate(model_results):
        print(f"Config {configs[i]}: Nash={result['is_nash']}, "
              f"Payoffs=({result['payoff_p1']:.3f}, {result['payoff_p2']:.3f})")