import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

import data_setup as ds
import chess_funcs as cf

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.fc1 = nn.Linear(72, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)


        self.y_from = nn.Linear(128, 8)
        self.x_from = nn.Linear(128, 8)
        self.y_to = nn.Linear(128, 8)
        self.x_to = nn.Linear(128, 8)
    
    def forward(self, x, return_logits=False):
            x = x.float()
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))

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