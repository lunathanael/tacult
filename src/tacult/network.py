import sys
from tacult.base.utils import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')


def UtacNNet(
    onnx_export: bool = False,
    **kwargs
) -> nn.Module:
    net = _UtacNNet(**kwargs)
    if not onnx_export:
        net.compile()
    else:
        net.eval()
    return net


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class _UtacNNet(nn.Module):
    def __init__(
            self,
            cuda: bool = True,
            dropout: float = 0.3,
            channels: int = 16,
            num_residual_blocks: int = 4,
            embedding_size: int = 256,
    ):
        super(_UtacNNet, self).__init__()
        
        self.board_x, self.board_y, self.board_z = 9, 9, 4
        self.action_size = 81
        self.device = get_device(cuda)

        # Neural Net layers
        self.conv_in = nn.Conv2d(self.board_z, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        
        res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        self.resnet = nn.Sequential(*res_blocks)
        
        # Policy and value heads
        conv_output_size = self.board_x * self.board_y * channels
        
        self.fc1 = nn.Linear(conv_output_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        
        self.pi = nn.Linear(embedding_size, self.action_size)
        self.v = nn.Linear(embedding_size, 1)
        
        self.dropout = dropout
        
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        
        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # Residual blocks
        for block in self.resnet:
            x = block(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layers
        pi = F.softmax(self.pi(x), dim=1)
        v = torch.tanh(self.v(x))
        
        return pi, v
