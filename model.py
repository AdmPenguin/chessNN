import torch
import torch.nn as nn
import torch.nn.functional as functional

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.fc1 = nn.Linear(72, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 4)
    
    def forward(self, x):
        x = x.float()
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))

        out = self.output(x)
        
        return out

def main():
    # Main trains the algorithm to output valid moves randomly
    ...

if __name__ == "__main__":
    main()