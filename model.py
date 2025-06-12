import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

import data_setup as ds
import chess_funcs as cf

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_conv = nn.Linear(256 + 8, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)

        self.y_from = nn.Linear(128, 8)
        self.x_from = nn.Linear(128, 8)
        self.y_to = nn.Linear(128, 8)
        self.x_to = nn.Linear(128, 8)

    def forward(self, x, return_logits=False):
        x = x.float()
        
        board = x[:, 1:65].view(-1, 1, 8, 8)
        aux = torch.cat([x[:, 0:1], x[:, 65:]], dim=1)

        conv_out = F.relu(self.conv1(board))
        conv_out = self.pool(conv_out)
        conv_out = conv_out.view(-1, 16 * 4 * 4)

        x = torch.cat([conv_out, aux], dim=1)
        x = F.relu(self.fc_conv(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        y_from_logits = self.y_from(x)
        x_from_logits = self.x_from(x)
        y_to_logits = self.y_to(x)
        x_to_logits = self.x_to(x)

        if return_logits:
            return y_from_logits, x_from_logits, y_to_logits, x_to_logits
        else:
            y_from = torch.argmax(y_from_logits, dim=1)
            x_from = torch.argmax(x_from_logits, dim=1)
            y_to = torch.argmax(y_to_logits, dim=1)
            x_to = torch.argmax(x_to_logits, dim=1)
            return torch.stack([y_from, x_from, y_to, x_to], dim=1)
