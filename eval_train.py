import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from src import UtacEnv

class EvaluatorTrainer:
    def __init__(self, evaluator, learning_rate=1e-4):
        self.evaluator = evaluator
        self.optimizer = optim.Adam(evaluator.parameters(), lr=learning_rate)
        
    def generate_training_positions(self, num_positions=1000):
        positions = []
        values = []
        
        for _ in range(num_positions):
            game = UtacEnv()
            depth = random.randint(0, 20)
            
            for _ in range(depth):
                if game.is_terminal():
                    break
                action = random.choice(game.get_legal_actions())
                game.step(action)
            
            value = self.get_ground_truth(game, depth=6)
            positions.append(game._get_obs())
            values.append(value)
            
        return positions, values
    
    def train_batch(self, positions, values):
        self.evaluator.train()
        self.optimizer.zero_grad()
        
        position_tensor = torch.FloatTensor(np.array(positions))
        value_tensor = torch.FloatTensor(values).unsqueeze(1)
        
        predictions = self.evaluator(position_tensor)
        loss = F.mse_loss(predictions, value_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()