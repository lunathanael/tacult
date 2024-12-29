import torch
import torch.optim as optim
from collections import deque
import numpy as np
from typing import List, Tuple, Dict
import random
import pickle

from src.network import Network
from src.mctsnn import MCTSNN
from src.utacenv.envs.utacenv import UtacEnv


def random_orientation_inplace(data: tuple) -> tuple:
    state, policy, value = data
    if random.randint(0, 1):
        state = np.flip(state, 1)
        policy = np.flip(policy, 0)
        value = np.flip(value, 0)
    if random.randint(0, 1):
        state = np.flip(state, 2) 
        policy = np.flip(policy, 1)
        value = np.flip(value, 1)
    if random.randint(0, 1):
        state = np.rollaxis(state, 2, 1)
        policy = np.rollaxis(policy, 1, 0)
        value = np.rollaxis(value, 1, 0)
    return state, policy, value


class Trainer:
    def __init__(
        self,
        network: Network,
        num_games: int = 10_000,
        num_simulations: int = 80,
        batch_size: int = 128,
        num_epochs: int = 4,
        buffer_size: int = 10_000,
        warmup_buffer_size: int = 1_000,
        lr_schedule: Dict[int, float] = {
            0: 3e-5,
            1000: 5e-5,
            2000: 1e-4,
            3000: 2e-4,
            50000: 1e-4,
            85000: 3e-5,
        }
    ):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr_schedule[0])
        self.buffer = deque(maxlen=buffer_size)
        self.warmup_buffer_size = warmup_buffer_size
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def self_play(self) -> List[Tuple]:
        game = UtacEnv()
        game.reset()
        
        mcts = MCTSNN(
            self.network, 
            selection_method="sample"
        )
        game_history = []

        while not game.is_terminal():
            current_player = game.current_player
            
            # Run MCTS and get action probabilities
            action = mcts.choose_action(game, num_simulations=self.num_simulations)
            
            # Store state, probabilities, and current player
            game_history.append((
                game._get_obs(),
                action,
                current_player
            ))
            
            # Make the move
            game.apply_move_index(action)
        
        # Get game outcome
        final_reward = game.get_reward(1)  # From perspective of player 1
        
        # Convert game history to training data
        training_data = []
        for state, action, player in game_history:
            # Adjust reward based on player perspective
            reward = final_reward if player == 1 else -final_reward
            
            # Create policy target (one-hot)
            policy_target = np.zeros(81)
            policy_target[action] = 1
            
            training_data.append((state, policy_target, reward))
            
        return training_data

    def train_epoch(self):
        self.network.train()
        
        # Sample batch from replay buffer
        if len(self.buffer) < self.batch_size:
            return
            
        batch = random.sample(self.buffer, self.batch_size)
        batch = [random_orientation_inplace(data) for data in batch]
        states, policies, values = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states))
        policy_batch = torch.FloatTensor(np.array(policies))
        value_batch = torch.FloatTensor(np.array(values)).unsqueeze(1)
        
        # Forward pass
        policy_pred, value_pred = self.network(state_batch)
        
        # Calculate loss
        loss, policy_loss, value_loss = self.network.loss_function(
            policy_pred, value_pred, policy_batch, value_batch
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()

    def train(self):
        steps = 0

        for iteration in range(self.num_games):
            
            # Self-play phase
            training_data = self.self_play()
            self.buffer.extend(training_data)
            
            # Training phase
            if len(self.buffer) >= self.warmup_buffer_size:
                print(f"Training... (lr={self.optimizer.param_groups[0]['lr']:.6f})")

                print(f"Training...")
                for _ in range(self.num_epochs):
                    loss, policy_loss, value_loss = self.train_epoch()
                    
                print(f"Game {iteration + 1}/{self.num_games}")
                print(f"Loss: {loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
                print(f"Buffer size: {len(self.buffer)}")

                steps += 1
                
                if steps in self.lr_schedule:
                    self.optimizer.param_groups[0]['lr'] = self.lr_schedule[steps]
                
            # Save model periodically
            if (iteration + 1) % 100 == 0:
                print(f"Saving model... at {iteration + 1}")
                torch.save(self.network.state_dict(), f"model_checkpoint_{iteration+1}.pt")
                print(f"Saving buffer...")
                with open(f"buffer_{iteration+1}.pkl", "wb") as f:
                    pickle.dump(self.buffer, f)

if __name__ == "__main__":
    network = Network()
    trainer = Trainer(network, num_simulations=10)
    trainer.train()
