import sys
import torch

import torch.nn as nn
import torch.nn.functional as F


sys.path.append('..')


def UtacNNet(cuda: bool=True, dropout: float=0.3, onnx_export: bool=False) -> nn.Module:
    if onnx_export:
        return _OnnxExportUtacNNet(cuda)
    else:
        return _UtacNNet(cuda, dropout)

class _UtacNNet(nn.Module):
    def __init__(self, cuda: bool=True, dropout: float=0.3):
        super(_UtacNNet, self).__init__()
        
        self.board_x, self.board_y, self.board_z = 9, 9, 2
        self.action_size = 81

        # Neural Net layers
        self.conv1 = nn.Conv2d(self.board_z, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=0)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        # Calculate size after convolutions
        conv_output_size = ((self.board_x - 2) * (self.board_y - 2) * 512)

        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)
        
        self.dropout = dropout

        if cuda:
            self.cuda()

    def forward(self, x):
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
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layers
        pi = F.softmax(self.pi(x), dim=1)
        v = torch.tanh(self.v(x))

        return pi, v
    

class _OnnxExportUtacNNet(nn.Module):
    def __init__(self, cuda: bool=True):
        super(_OnnxExportUtacNNet, self).__init__()
        
        self.board_x, self.board_y, self.board_z = 9, 9, 2
        self.action_size = 81

        # Neural Net layers
        self.conv1 = nn.Conv2d(self.board_z, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=0)

        # Calculate size after convolutions
        conv_output_size = ((self.board_x - 2) * (self.board_y - 2) * 512)

        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)

        if cuda:
            self.cuda()

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layers
        pi = F.softmax(self.pi(x), dim=1)
        v = torch.tanh(self.v(x))

        return pi, v