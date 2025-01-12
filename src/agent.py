# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .utils import layer_init


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(
                nn.Linear(obs_dim, 512)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, action_dim), std=0.005)
        # self.actor2 = layer_init(nn.Linear(512, 2), std=0.01)
        self.critic = layer_init(nn.Linear(1024, 1), std=1)

    def get_value(self, x):
        return self.get_action_and_value(x, None)[3]
    
    def get_action(self, x, info=None):
        return self.get_action_and_value(x, None)[0]

    def get_action_and_value(self, _x, action=None):
        x = self.network(_x)
        logits = self.actor(x)
        
        action_mask = _x[:, 0:1, :, :]
        # Reshape to match logits shape (batch x 81)
        action_mask = action_mask.reshape(action_mask.shape[0], -1)
        # Apply mask by setting logits of invalid actions to large negative value
        logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e8).to(logits.device))

        probs = Categorical(logits=logits)
        # probs2 = torch.distributions.Categorical(logits=logits2)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, obs_dim, action_dim):
        agent = Agent(obs_dim, action_dim)
        agent.load_state_dict(torch.load(path))
        return agent
