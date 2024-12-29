import math
import random
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
import numpy as np

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

    def choose_action(self, state, num_simulations=1000):
        root = MCTSNNNode(state)
        
        # Get initial policy from neural network
        obs = torch.FloatTensor(state._get_obs()).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.network(obs)
        policy = policy.squeeze(0).numpy()
        
        # Set priors for root node's children
        legal_moves = state.get_legal_actions()
        legal_move_indices = [move_to_index(move) for move in legal_moves]
        policy_legal = policy[legal_move_indices]
        policy_legal = policy_legal / policy_legal.sum()  # Renormalize
        
        # Create children with priors
        for move, prior in zip(legal_moves, policy_legal):
            new_state = state.clone()
            new_state.step(move)
            child = MCTSNNNode(new_state, parent=root, action=move_to_index(move))
            child.prior = prior
            root.children.append(child)
        
        # Run simulations
        for _ in range(num_simulations):
            node = self._select(root)
            value = self._evaluate(node.state)
            self._backpropagate(node, value)
        
        # Select action based on specified method
        if self.selection_method == "argmax":
            return self._best_child(root, exploration_weight=0).action
        elif self.selection_method == "sample":
            visits = np.array([child.visits for child in root.children])
            probs = visits / visits.sum()
            chosen_idx = np.random.choice(len(root.children), p=probs)
            return root.children[chosen_idx].action
        else:  # random
            return random.choice(root.children).action

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
                return float('inf')
            
            q_value = n.value / n.visits
            u_value = exploration_weight * n.prior * math.sqrt(node.visits) / (1 + n.visits)
            return q_value + u_value

        return max(node.children, key=ucb_score)
