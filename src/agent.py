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
                nn.Linear(obs_dim, 256)
            ),  # First layer expanded to handle 52 inputs
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_dim), std=0.005)
        # self.actor2 = layer_init(nn.Linear(512, 2), std=0.01)
        self.critic = layer_init(nn.Linear(512 + action_dim, 1), std=1)

    def get_value(self, x):
        return self.get_action_and_value(x, None)[3]
    
    def get_action(self, x):
        return self.get_action_and_value(x, None)[0]

    def get_action_and_value(self, x, action=None):
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # probs2 = torch.distributions.Categorical(logits=logits2)
        if action is None:
            action = probs.sample()

        action_encoded = torch.zeros_like(logits)
        action_encoded[torch.arange(len(action)), action] = 1
        x = torch.cat([x, action_encoded], dim=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, obs_dim, action_dim):
        agent = Agent(obs_dim, action_dim)
        agent.load_state_dict(torch.load(path))
        return agent
