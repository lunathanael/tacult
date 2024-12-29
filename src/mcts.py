import math
import random
from typing import List, Optional, Tuple

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

class MCTS:
    def __init__(self, exploration_weight=1.414):
        self.exploration_weight = exploration_weight

    def choose_action(self, state, num_simulations=1000):
        root = MCTSNode(state)
        
        for _ in range(num_simulations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        # Choose the action with the highest number of visits
        print(self._best_child(root, exploration_weight=0).value)
        return self._best_child(root, exploration_weight=0).action

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if len(node.children) == 0:
                return self._expand(node)
            node = self._best_child(node, self.exploration_weight)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        actions = node.state.get_legal_actions()
        for action in actions:
            if not any(child.action == action for child in node.children):
                new_state = node.state.clone()
                new_state.step(action)
                new_node = MCTSNode(new_state, parent=node, action=action)
                node.children.append(new_node)
                return new_node
        return node

    def _simulate(self, state) -> float:
        state = state.clone()
        current_player = state.current_player
        while not state.is_terminal():
            action = random.choice(state.get_legal_actions())
            state.step(action)
        return state.get_reward(current_player)

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_child(self, node: MCTSNode, exploration_weight: float) -> MCTSNode:
        if not node.children:
            raise ValueError("No children to choose from")

        def ucb_score(n: MCTSNode) -> float:
            if n.visits == 0:
                return float('inf')
            
            exploitation = n.value / n.visits
            exploration = exploration_weight * math.sqrt(math.log(node.visits) / n.visits)
            return exploitation + exploration

        return max(node.children, key=ucb_score)
