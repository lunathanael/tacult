import sys
from tacult.base.utils import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')


def UtacNNet(
    cuda: bool = True, dropout: float = 0.3, onnx_export: bool = False
) -> nn.Module:
    if onnx_export:
        net = _OnnxExportUtacNNet(cuda)
    else:
        net = _UtacNNet(cuda, dropout)
        net.compile()
    return net


class _UtacNNet(nn.Module):
    def __init__(self, cuda: bool = True, dropout: float = 0.3):
        super(_UtacNNet, self).__init__()
        
        self.board_x, self.board_y, self.board_z = 9, 9, 4
        self.action_size = 81
        self.device = get_device(cuda)

        # Neural Net layers
        self.conv1 = nn.Conv2d(self.board_z, 256, padding=1, stride=1, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, padding=0, stride=1, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 512, padding=1, stride=1, kernel_size=3)
        self.conv4 = nn.Conv2d(512, 512, padding=0, stride=1, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        # Calculate size after convolutions
        conv_output_size = self.board_x * self.board_y * 512

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)
        
        self.dropout = dropout

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layers
        pi = F.softmax(self.pi(x), dim=1)
        v = torch.tanh(self.v(x))

        return pi, v


class _OnnxExportUtacNNet(nn.Module):
    def __init__(self, cuda: bool = True):
        super(_OnnxExportUtacNNet, self).__init__()
        
        self.board_x, self.board_y, self.board_z = 9, 9, 4
        self.action_size = 81
        self.device = get_device(cuda)

        # Neural Net layers
        self.conv1 = nn.Conv2d(self.board_z, 256, padding=1, stride=1, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, padding=0, stride=1, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 512, padding=1, stride=1, kernel_size=3)
        self.conv4 = nn.Conv2d(512, 512, padding=0, stride=1, kernel_size=1)

        conv_output_size = self.board_x * self.board_y * 512

        self.fc1 = nn.Linear(conv_output_size, 512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))

        # Output layers
        pi = F.softmax(self.pi(x), dim=1)
        v = torch.tanh(self.v(x))

        return pi, v