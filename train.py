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


def get_all_symmetries(data: tuple) -> List[tuple]:
    state, policy, value = data
    symmetries = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            for transpose in [False, True]:
                current_state = state.copy()
                _current_policy = policy.copy()
                current_policy = _current_policy.reshape(9, 9)
                
                if flip_h:
                    current_state = np.flip(current_state, 1)
                    current_policy = np.flip(current_policy, 0)
                if flip_v:
                    current_state = np.flip(current_state, 2)
                    current_policy = np.flip(current_policy, 1)
                if transpose:
                    current_state = np.rollaxis(current_state, 2, 1)
                    current_policy = np.rollaxis(current_policy, 1, 0)
                

                symmetries.append((current_state, current_policy.flatten(), value))
    
    return symmetries


class Trainer:
    def __init__(
        self,
        network: Network,
        num_simulations: int = 80,
        num_games: int = 50_000,
        model_name: str = "model",
        batch_size: int = 128,
        num_epochs: int = 4,
        buffer_size: int = 1_000_000,
        warmup_buffer_size: int = 10_000,
        save_buffer: bool = False,
        lr_schedule: Dict[int, float] = {
            0: 1e-4,
            2_500: 5e-4,
            10_000: 2e-4,
            25_000: 1e-4,
            40_000: 5e-5,
        }
    ):  
        self.network = network
        self.model_name = model_name
        self.lr_schedule = lr_schedule
        self.optimizer = optim.Adam(network.parameters(), lr=lr_schedule[0])
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.warmup_buffer_size = warmup_buffer_size
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_buffer = save_buffer

        self.iterations = 0
        self.steps = 0

    def self_play(self) -> List[Tuple]:
        game = UtacEnv()
        game.reset()

        
        temperature = 1.0 if self.iterations < self.num_games * 0.75 else 0.1
        
        mcts = MCTSNN(
            self.network, 
            selection_method="sample",
            temperature=temperature
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
        final_reward = game.get_reward(1)

        training_data = []
        for state, action, player in game_history:
            reward = final_reward if player == 1 else -final_reward
            
            policy_target = np.zeros(81)
            policy_target[action] = 1
            
            # training_data.append((state, policy_target, reward))
            symmetries = get_all_symmetries((state, policy_target, reward))
            training_data.extend(symmetries)
            
        return training_data

    def train_epoch(self):
        self.network.train()
        
        if len(self.buffer) < self.batch_size:
            return
            
        batch = random.sample(self.buffer, self.batch_size)
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
        save_points = {int(self.num_games * (1.05 ** i - 1) / (1.05 ** 50 - 1)) for i in range(50)}
        while self.iterations < self.num_games:
            self.iterations += 1

            training_data = self.self_play()
            self.buffer.extend(training_data)
            
            # Training phase
            if len(self.buffer) >= self.warmup_buffer_size:
                print(f"Training... (lr={self.optimizer.param_groups[0]['lr']:.6f})")
                for _ in range(self.num_epochs):
                    loss, policy_loss, value_loss = self.train_epoch()
                    
                print(f"Game {self.iterations}/{self.num_games}, Step {self.steps}, Buffer size: {len(self.buffer)}")
                print(f"Loss: {loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")

                self.steps += 1
                
                if self.steps in self.lr_schedule:
                    self.optimizer.param_groups[0]['lr'] = self.lr_schedule[self.steps]
                
            if self.iterations in save_points:
                print(f"Saving model... at {self.iterations}")
                self.save_trainer(self.iterations, self.steps)

        print(f"Training complete. Saving model...")
        self.save_trainer(self.iterations, self.steps)

    def save_trainer(self, iterations: int, steps: int):
        buffer = deque(maxlen=self.buffer_size)
        if self.save_buffer:
            buffer = self.buffer
        
        trainer_state = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'buffer': buffer,
            'buffer_size': self.buffer_size,
            'num_games': self.num_games,
            'num_simulations': self.num_simulations,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'warmup_buffer_size': self.warmup_buffer_size,
            'lr_schedule': self.lr_schedule,
            'model_name': self.model_name,
            'save_buffer': self.save_buffer,
            'iterations': iterations,
            'steps': steps,
        }
        
        with open(f"{self.model_name}_{iterations}.{self.num_games}.pkl", "wb") as f:
            pickle.dump(trainer_state, f)

        print(f"Saved model to {self.model_name}_{iterations}/{self.num_games}.pkl")

    @staticmethod
    def load_trainer(path: str) -> "Trainer":
        with open(path, "rb") as f:
            trainer_state = pickle.load(f)
        network = Network()
        network.load_state_dict(trainer_state['network_state'])
        optimizer = optim.Adam(network.parameters(), lr=trainer_state['lr_schedule'][0])
        optimizer.load_state_dict(trainer_state['optimizer_state'])

        kwargs = {
            "network": network,
            "buffer_size": trainer_state['buffer_size'],
            "warmup_buffer_size": trainer_state['warmup_buffer_size'],
            "num_games": trainer_state['num_games'],
            "num_simulations": trainer_state['num_simulations'],
            "batch_size": trainer_state['batch_size'],
            "num_epochs": trainer_state['num_epochs'],
            "save_buffer": trainer_state['save_buffer'],
            "lr_schedule": trainer_state['lr_schedule'],
            "model_name": trainer_state['model_name'],
        }

        trainer = Trainer(**kwargs)
        trainer.network = network
        trainer.optimizer = optimizer
        trainer.buffer = trainer_state['buffer']
        trainer.steps = trainer_state['steps']
        trainer.iterations = trainer_state['iterations']

        return trainer

if __name__ == "__main__":
    network = Network()
    trainer = Trainer(network, num_simulations=25, model_name="model_822eff0")
    trainer.train()
