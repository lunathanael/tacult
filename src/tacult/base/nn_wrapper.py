import os
import numpy as np
import torch
from tqdm import tqdm
import dill

from .nn import NeuralNet
from .utils import get_device

import logging

log = logging.getLogger(__name__)

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
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)
        
        try:
            T_max = args.epochs * args.steps_per_epoch * args.numIters / args.num_warm_restarts
        except KeyError:
            try:
                T_max = args.steps_per_warm_restart
            except KeyError:
                T_max = args.epochs * args.steps_per_epoch * args.numIters

        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=T_max
        )

        log.info(f"Network {network.__class__.__name__} initialized on device {self.device}")

    def training_step(self, boards, pis, vs):
        self.optimizer.zero_grad()
        out_pi, out_v = self.nnet(boards)
        pi_loss = -torch.sum(pis * torch.log(out_pi + 1e-8)) / pis.size()[0]
        v_loss = torch.mean((vs - out_v.squeeze()) ** 2)
        total_loss = pi_loss + v_loss
        total_loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return pi_loss.item(), v_loss.item()

    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Convert data to PyTorch tensors
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = torch.stack(input_boards)
        target_pis = torch.stack(target_pis) 
        target_vs = torch.tensor(target_vs)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            input_boards, target_pis, target_vs
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle_data,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
        )
        dataloader_iter = iter(dataloader)

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
                try:
                    batch_boards, batch_pis, batch_vs = next(dataloader_iter)
                except (StopIteration, NameError):
                    dataloader_iter = iter(dataloader)
                    batch_boards, batch_pis, batch_vs = next(dataloader_iter)

                pi_loss, v_loss = self.training_step(batch_boards, batch_pis, batch_vs)
                
                # Optionally log the learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Track metrics
                total_pi_loss += pi_loss
                total_v_loss += v_loss

                pbar.set_postfix({
                    'pi_loss': f'{pi_loss:.4f}',
                    'v_loss': f'{v_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
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
            obs = obs.to(self.device)
            
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
        
        classpath = os.path.join(folder, 'network_class.pkl')
        module_path = os.path.join(folder, 'network_module.pkl')
        if not os.path.exists(classpath) or not os.path.exists(module_path):
            dill.dump_module(filename=module_path, module=dill.detect.getmodule(self.nnet.__class__))
            with open(classpath, 'wb') as f:
                dill.dump({
                    'module': self.nnet.__class__.__module__,
                    'class': self.nnet.__class__.__name__,
                    'args': self.args,
                }, f)

        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(
        self, folder: str = 'checkpoint', filename: str = 'checkpoint.pt'
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        state_dict = checkpoint['state_dict']
        policy_value_net_keys = set(self.nnet.state_dict().keys())

        optimizer_deleted = False
        for key in list(state_dict.keys()):
            if key not in policy_value_net_keys:
                del state_dict[key]
                if not optimizer_deleted:
                    del checkpoint['optimizer']
                    optimizer_deleted = True

        self.nnet.load_state_dict(state_dict)
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            log.warning("No optimizer state dict found in checkpoint")

        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            log.warning("No scheduler state dict found in checkpoint")

    def train_mode(self):
        """Set the network to training mode"""
        self.nnet.train()

    def eval_mode(self):
        """Set the network to evaluation mode"""
        self.nnet.eval()

def load_network_class(folder: str):
    classpath = os.path.join(folder, 'network_class.pkl')
    module_path = os.path.join(folder, 'network_module.pkl')

    with open(classpath, 'rb') as f:
        state_dict = dill.load(f)

    module_name = state_dict['module']
    class_name = state_dict['class']
    args = state_dict['args']

    module = dill.load_module(module_path, module_name)
    nnet = getattr(module, class_name)

    def _NWrap(nnet):
        class NWrap(NNetWrapper):
            def __init__(self, args):
                super().__init__(nnet(args), args)
        return NWrap

    return _NWrap(nnet)(args)

def load_network(folder: str, filename: str):
    cls = load_network_class(folder)
    cls.load_checkpoint(folder, filename)

    log.info(f'Loaded network from {folder}/{filename}')
    log.info('Network architecture:')
    log.info(cls.nnet)
    return cls
