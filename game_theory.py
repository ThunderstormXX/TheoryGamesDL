import numpy as np
from typing import Tuple, List, Optional

class Game:
    """Base class for game theory setups"""
    
    def __init__(self, payoff_matrix: np.ndarray):
        self.payoff_matrix = payoff_matrix
        self.n_players = len(payoff_matrix.shape) // 2
        
    def get_payoffs(self, strategies: np.ndarray) -> np.ndarray:
        """Calculate payoffs for given mixed strategies"""
        raise NotImplementedError

class TwoPlayerGame(Game):
    """Two-player normal form game"""
    
    def __init__(self, payoff_matrix: Tuple[np.ndarray, np.ndarray]):
        self.payoff_p1, self.payoff_p2 = payoff_matrix
        
    def get_payoffs(self, strategies: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        s1, s2 = strategies
        payoff1 = s1.T @ self.payoff_p1 @ s2
        payoff2 = s1.T @ self.payoff_p2 @ s2
        return float(payoff1), float(payoff2)
    
    def best_response(self, player: int, opponent_strategy: np.ndarray) -> np.ndarray:
        """Calculate best response for a player"""
        if player == 1:
            utilities = self.payoff_p1 @ opponent_strategy
        else:
            utilities = self.payoff_p2.T @ opponent_strategy
        
        best_action = np.argmax(utilities)
        response = np.zeros(len(utilities))
        response[best_action] = 1.0
        return response

def prisoners_dilemma() -> TwoPlayerGame:
    """Classic Prisoner's Dilemma"""
    payoff_p1 = np.array([[3, 0], [5, 1]])
    payoff_p2 = np.array([[3, 5], [0, 1]])
    return TwoPlayerGame((payoff_p1, payoff_p2))

def coordination_game() -> TwoPlayerGame:
    """Coordination game with multiple equilibria"""
    payoff_p1 = np.array([[2, 0], [0, 1]])
    payoff_p2 = np.array([[2, 0], [0, 1]])
    return TwoPlayerGame((payoff_p1, payoff_p2))

def matching_pennies() -> TwoPlayerGame:
    """Zero-sum matching pennies game"""
    payoff_p1 = np.array([[1, -1], [-1, 1]])
    payoff_p2 = np.array([[-1, 1], [1, -1]])
    return TwoPlayerGame((payoff_p1, payoff_p2))

class MatrixGame:
    """General matrix game with custom payoff matrices"""
    
    def __init__(self, payoff_p1: np.ndarray, payoff_p2: np.ndarray):
        self.payoff_p1 = payoff_p1
        self.payoff_p2 = payoff_p2
        self.n_actions_p1, self.n_actions_p2 = payoff_p1.shape
    
    def get_payoffs(self, action_p1: int, action_p2: int) -> Tuple[float, float]:
        """Get deterministic payoffs for pure actions"""
        return float(self.payoff_p1[action_p1, action_p2]), float(self.payoff_p2[action_p1, action_p2])
    
    def get_expected_payoffs(self, prob_p1: np.ndarray, prob_p2: np.ndarray) -> Tuple[float, float]:
        """Get expected payoffs for mixed strategies"""
        payoff1 = prob_p1.T @ self.payoff_p1 @ prob_p2
        payoff2 = prob_p1.T @ self.payoff_p2 @ prob_p2
        return float(payoff1), float(payoff2)

def is_nash_equilibrium(game: TwoPlayerGame, strategies: Tuple[np.ndarray, np.ndarray], 
                       tolerance: float = 1e-6) -> bool:
    """Check if strategy profile is Nash equilibrium"""
    s1, s2 = strategies
    
    # Check if each strategy is best response to the other
    br1 = game.best_response(1, s2)
    br2 = game.best_response(2, s1)
    
    # For mixed strategies, check if support is optimal
    current_payoff1 = s1.T @ game.payoff_p1 @ s2
    current_payoff2 = s1.T @ game.payoff_p2 @ s2
    
    br_payoff1 = br1.T @ game.payoff_p1 @ s2
    br_payoff2 = s1.T @ game.payoff_p2 @ br2
    
    return (abs(br_payoff1 - current_payoff1) < tolerance and 
            abs(br_payoff2 - current_payoff2) < tolerance)