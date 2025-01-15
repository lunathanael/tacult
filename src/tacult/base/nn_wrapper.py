import os
import numpy as np
import torch
from tqdm import tqdm

from .nn import NeuralNet
from .utils import get_device

import warnings

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""


class NNetWrapper(NeuralNet):
    def __init__(self, network, args):
        self.nnet = network
        self.board_x, self.board_y, self.board_z = 9, 9, 2
        self.action_size = 81
        self.args = args
        self.device = get_device(args.cuda)
        # Create optimizer once
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Convert data to PyTorch tensors
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = torch.FloatTensor(np.array(input_boards)).to(
            self.device
        )
        target_pis = torch.FloatTensor(np.array(target_pis)).to(
            self.device
        )
        target_vs = torch.FloatTensor(np.array(target_vs)).to(
            self.device
        )

        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            input_boards, target_pis, target_vs
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=self.args.shuffle_data,
            pin_memory=True,
        )

        # Training loop
        self.nnet.train()
        for epoch in range(self.args.epochs):
            total_pi_loss = 0
            total_v_loss = 0

            # Create progress bar for fixed number of steps
            desc = f'Epoch {epoch+1}/{self.args.epochs}'
            pbar = tqdm(
                range(self.args.steps_per_epoch),
                desc=desc
            )
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
                pi_loss = -torch.sum(
                    batch_pis * torch.log(pi + 1e-8)
                ) / batch_pis.size()[0]
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
            print(
                'Steps: {}, Average Loss - Policy: {:.4f}, Value: {:.4f}'.format(
                    self.args.steps_per_epoch, avg_pi_loss, avg_v_loss
                )
            )

    def predict(self, obs):
        """
        obs: np array with board
        """
        self.nnet.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Convert numpy array to tensor
            obs = torch.FloatTensor(obs).to(self.device)
            
            # Get predictions
            pi, v = self.nnet(obs)
            
            # Convert to numpy
            return pi.cpu().numpy(), v.cpu().numpy()

    def save_checkpoint(
        self, folder: str = 'checkpoint', filename: str = 'checkpoint.pt'
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! "
                f"Making directory {folder}"
            )
            os.mkdir(folder)
        
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(
        self, folder: str = 'checkpoint', filename: str = 'checkpoint.pt'
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")
        
        checkpoint = torch.load(filepath, map_location=self.device)

        invalid_keys = []
        for key in list(checkpoint['state_dict'].keys()):
            if key not in self.nnet.state_dict().keys():
                invalid_keys.append(key)
                del checkpoint['state_dict'][key]

        if invalid_keys:
            warnings.warn(f"Invalid keys in checkpoint: {invalid_keys}")

        self.nnet.load_state_dict(checkpoint['state_dict'])

        if self.args.ignore_optimizer:
            warnings.warn("Ignoring optimizer state dict")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_mode(self):
        """Set the network to training mode"""
        self.nnet.train()

    def eval_mode(self):
        """Set the network to evaluation mode"""
        self.nnet.eval()