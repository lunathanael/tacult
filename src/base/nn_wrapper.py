import os
import time
import numpy as np
import sys
sys.path.append('..')
from .nn import NeuralNet
import torch

from tqdm import tqdm

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

# args = dotdict({
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': False,
#     'num_channels': 512,
# })

class NNetWrapper(NeuralNet):
    def __init__(self, network, args):
        self.nnet = network
        self.board_x, self.board_y, self.board_z = 9, 9, 2
        self.action_size = 81
        self.args = args
        # Create optimizer once
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Convert data to PyTorch tensors
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = torch.FloatTensor(np.array(input_boards))
        target_pis = torch.FloatTensor(np.array(target_pis))
        target_vs = torch.FloatTensor(np.array(target_vs))

        if self.args.cuda:
            input_boards = input_boards.cuda()
            target_pis = target_pis.cuda()
            target_vs = target_vs.cuda()

        # Create data loader
        dataset = torch.utils.data.TensorDataset(input_boards, target_pis, target_vs)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=self.args.shuffle_data
        )

        # Training loop
        self.nnet.train()
        for epoch in range(self.args.epochs):
            total_pi_loss = 0
            total_v_loss = 0

            # Create progress bar for fixed number of steps
            pbar = tqdm(range(self.args.steps_per_epoch), desc=f'Epoch {epoch+1}/{self.args.epochs}')
            for step in pbar:
                # Get next batch from dataloader
                try:
                    batch_boards, batch_pis, batch_vs = next(dataloader_iter)
                except (StopIteration, NameError):
                    dataloader_iter = iter(dataloader)
                    batch_boards, batch_pis, batch_vs = next(dataloader_iter)

                # Forward pass
                self.optimizer.zero_grad()
                pi, v = self.nnet(batch_boards)
                
                # Calculate losses
                pi_loss = -torch.sum(batch_pis * torch.log(pi + 1e-8)) / batch_pis.size()[0]
                v_loss = torch.mean((batch_vs - v.squeeze()) ** 2)
                total_loss = pi_loss + v_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Track metrics
                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()

                # Update progress bar with current losses
                pbar.set_postfix({
                    'pi_loss': f'{pi_loss.item():.4f}',
                    'v_loss': f'{v_loss.item():.4f}'
                })

            # Print epoch results
            avg_pi_loss = total_pi_loss / self.args.steps_per_epoch
            avg_v_loss = total_v_loss / self.args.steps_per_epoch
            print(f'Epoch {epoch+1}/{self.args.epochs}')
            print(f'Steps: {self.args.steps_per_epoch}, Average Loss - Policy: {avg_pi_loss:.4f}, Value: {avg_v_loss:.4f}')

    def predict(self, obs):
        """
        obs: np array with board
        """
        self.nnet.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Convert numpy array to tensor
            obs = torch.FloatTensor(obs)
            if self.args.cuda:
                obs = obs.cuda()
            
            # Get predictions
            pi, v = self.nnet(obs)
            
            # Convert to numpy
            return pi.cpu().numpy(), v.cpu().numpy()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")
        
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_mode(self):
        """Set the network to training mode"""
        self.nnet.train()

    def eval_mode(self):
        """Set the network to evaluation mode"""
        self.nnet.eval()