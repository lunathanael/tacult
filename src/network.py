import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class Network(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=128):
        super().__init__()
        
        # Input layer
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 9 * 9)  # Output for each position
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, x):
        # x shape: (batch_size, 3, 9, 9)
        x = self.conv_input(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy output (action probabilities)
        policy = self.policy_head(x)
        policy = F.softmax(policy, dim=1)  # Shape: (batch_size, 81)
        
        # Value output (state evaluation)
        value = self.value_head(x)  # Shape: (batch_size, 1)
        
        return policy, value

    def loss_function(self, policy_output, value_output, policy_target, value_target):
        """
        Calculate the combined loss for policy and value outputs
        
        Args:
            policy_output: predicted policy (batch_size, 81)
            value_output: predicted value (batch_size, 1)
            policy_target: target policy (batch_size, 81)
            value_target: target value (batch_size, 1)
        """
        # Policy loss - cross entropy
        policy_loss = -torch.sum(policy_target * torch.log(policy_output + 1e-8)) / policy_output.shape[0]
        
        # Value loss - MSE
        value_loss = F.mse_loss(value_output, value_target)
        
        # Combined loss (you can adjust these weights)
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss


class UTTTEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature Transformer
        self.feature_network = nn.Sequential(
            # Input: 6 x 9 x 9
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 32 x 9 x 9
        )
        
        # Accumulator Network
        self.accumulator = nn.Sequential(
            nn.Linear(32 * 81, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        
        # Output Network
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Transform features
        features = self.feature_network(x)
        features = features.view(-1, 32 * 81)
        
        # Accumulate
        accumulated = self.accumulator(features)
        
        # Final evaluation
        return self.output(accumulated)