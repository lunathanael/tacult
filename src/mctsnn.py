import math
import random
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
import numpy as np
import logging


EPS = 1e-8
log = logging.getLogger(__name__)

from utac.utils import move_to_index

class MCTSNNNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNNNode] = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

class MCTSNN:
    def __init__(self, network, exploration_weight=1.414, selection_method: Literal["argmax", "sample", "random"]="argmax"):
        self.network = network
        self.exploration_weight = exploration_weight
        self.selection_method = selection_method
        self.root = None

    def choose_action(self, state, num_simulations=1000, temperature=1.0):
        self.root = MCTSNNNode(state)
        
        # Get initial policy from neural network
        obs = torch.FloatTensor(state._get_obs()).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.network(obs)
        policy = policy.squeeze(0).numpy()
        
        # Set priors for root node's children
        legal_moves = state.get_legal_actions()
        legal_move_indices = [move_to_index(move) for move in legal_moves]
        policy_legal = policy[legal_move_indices]
        sum_policy = policy_legal.sum()
        
        if sum_policy > 0:
            policy_legal = policy_legal / sum_policy
        else:
            log.warning("All moves were masked")
            policy_legal = np.ones_like(policy_legal) / len(policy_legal)
        
        for move, prior in zip(legal_moves, policy_legal):
            new_state = state.clone()
            new_state.step(move)
            child = MCTSNNNode(new_state, parent=self.root, action=move_to_index(move))
            child.prior = prior
            self.root.children.append(child)
        
        for _ in range(num_simulations):
            node = self._select(self.root)
            value = self._evaluate(node.state)
            self._backpropagate(node, value)
        
        visits = np.array([child.visits for child in self.root.children])
        if temperature == 0:
            best_actions = np.argwhere(visits == np.max(visits)).flatten()
            chosen_idx = np.random.choice(best_actions)
            return self.root.children[chosen_idx].action
        
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        chosen_idx = np.random.choice(len(self.root.children), p=probs)
        return self.root.children[chosen_idx].action

    def _select(self, node: MCTSNNNode) -> MCTSNNNode:
        while not node.state.is_terminal():
            if len(node.children) == 0:
                return self._expand(node)
            node = self._best_child(node, self.exploration_weight)
        return node

    def _expand(self, node: MCTSNNNode) -> MCTSNNNode:
        # Get policy from neural network
        obs = torch.FloatTensor(node.state._get_obs()).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.network(obs)
        policy = policy.squeeze(0).numpy()
        
        # Create children for legal moves
        legal_moves = node.state.get_legal_actions()
        legal_move_indices = [move_to_index(move) for move in legal_moves]
        policy_legal = policy[legal_move_indices]
        policy_legal = policy_legal / policy_legal.sum()  # Renormalize
        
        for move, prior in zip(legal_moves, policy_legal):
            if not any(child.action == move_to_index(move) for child in node.children):
                new_state = node.state.clone()
                new_state.step(move)
                child = MCTSNNNode(new_state, parent=node, action=move_to_index(move))
                child.prior = prior
                node.children.append(child)
        
        return random.choice(node.children)

    def _evaluate(self, state) -> float:
        if state.is_terminal():
            return state.get_reward(state.current_player)
        obs = torch.FloatTensor(state._get_obs()).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(obs)
        return value.item()

    def _backpropagate(self, node: MCTSNNNode, value: float):
        while node is not None:
            node.visits += 1
            node.value += value
            value = -value  # Flip value for opponent
            node = node.parent

    def _best_child(self, node: MCTSNNNode, exploration_weight: float) -> MCTSNNNode:
        if not node.children:
            raise ValueError("No children to choose from")

        def ucb_score(n: MCTSNNNode) -> float:
            if n.visits == 0:
                return exploration_weight * n.prior * math.sqrt(node.visits + 1e-8)
            
            q_value = n.value / n.visits
            u_value = exploration_weight * n.prior * math.sqrt(node.visits) / (1 + n.visits)
            return q_value + u_value

        return max(node.children, key=ucb_score)

    def get_root_evaluation(self) -> Tuple[dict, List[dict]]:
        """Returns the evaluation of the root state and its actions"""
        if self.root is None:
            raise ValueError("No search has been performed yet")
        
        state_eval = {
            "state": self.root.state.clone(),
            "visits": self.root.visits,
            "value": self.root.value,
            "state_value_mean": self.root.value / max(1, self.root.visits),
        }
        
        if not self.root.children:
            raise ValueError("Root has no children")
        if self.root.visits == 0:
            raise ValueError("Root was not visited")
            
        action_evals = []
        for child in self.root.children:
            action_eval = {
                "action": child.action,
                "visits": child.visits,
                "prior": child.prior,
                "value": -child.value,
                "state_value_mean": -child.value / max(1, child.visits),
            }
            action_evals.append(action_eval)
            
        return state_eval, action_evals
