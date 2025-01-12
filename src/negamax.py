import math
from typing import Callable, Optional, Tuple
from utac.utils import move_to_index

class Negamax:
    def __init__(self, evaluator: Callable, max_depth: int = 4):
        self.evaluator = evaluator
        self.max_depth = max_depth

    def choose_action(self, state, depth: Optional[int] = None) -> int:
        """Returns the best move index according to negamax search with alpha-beta pruning"""
        depth = depth or self.max_depth
        best_value = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        for move in state.get_legal_actions():
            next_state = state.clone()
            next_state.step(move)
            value = -self._negamax(next_state, depth - 1, -beta, -alpha)
            
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)

        return move_to_index(best_move)

    def _negamax(self, state, depth: int, alpha: float, beta: float) -> float:
        """Recursive negamax implementation with alpha-beta pruning"""
        if depth == 0 or state.is_terminal():
            return self.evaluator(state.get_features())

        value = -math.inf
        for move in state.get_legal_actions():
            next_state = state.clone()
            next_state.step(move)
            value = max(value, -self._negamax(next_state, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff

        return value
