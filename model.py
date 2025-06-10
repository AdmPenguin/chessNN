import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

import data_setup as ds
import chess_funcs as cf

LEARNING_RATE = 0.005
BATCH_SIZE = 50
OUTPUT = "models/valid.pth"

LOAD = True
LOAD_FROM = "models/valid.pth"

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.fc1 = nn.Linear(72, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)


        self.y_from = nn.Linear(128, 8)
        self.x_from = nn.Linear(128, 8)
        self.y_to = nn.Linear(128, 8)
        self.x_to = nn.Linear(128, 8)
    
    def forward(self, x, return_logits=False):
            x = x.float()
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))

            y_from_logits = self.y_from(x)
            x_from_logits = self.x_from(x)
            y_to_logits = self.y_to(x)
            x_to_logits = self.x_to(x)

            if return_logits:
                return y_from_logits, x_from_logits, y_to_logits, x_to_logits
            else:
                # Output class predictions: shape [batch_size, 4]
                y_from = torch.argmax(y_from_logits, dim=1)
                x_from = torch.argmax(x_from_logits, dim=1)
                y_to = torch.argmax(y_to_logits, dim=1)
                x_to = torch.argmax(x_to_logits, dim=1)
                return torch.stack([y_from, x_from, y_to, x_to], dim=1)


def discretize_move(move_tensor):
    return torch.clamp(torch.round(move_tensor), 0, 7).int()


def train_model(model, dataloader, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    for epoch in range(epochs):
        total_valid = 0
        total_predicted = 0
        total_loss = 0

        for batch in dataloader:
            boards = batch[0].to(device)  # shape: (batch_size, 72)

            optimizer.zero_grad()

            outputs = model(boards)  # shape: (batch_size, 4)

            # Add noise to encourage exploration
            noise_scale = max(0.1, 1.0 - epoch / epochs)  # decay noise
            noisy_outputs = outputs + torch.randn_like(outputs) * noise_scale

            # Discretize
            pred_moves = discretize_move(noisy_outputs)

            # Validate moves
            rewards = []
            for i in range(len(boards)):
                board_np = boards[i].detach().cpu().numpy()
                move_np = pred_moves[i].tolist()
                try:
                    valid = cf.checkValid(board_np, move_np)[0]
                except:
                    valid = False
                rewards.append(1.0 if valid else 0.0)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

            # Compute loss
            penalty = ((1 - rewards.unsqueeze(1)) * outputs.pow(2)).mean()
            reward_bonus = rewards.mean() * 0.5

            # Add entropy-based exploration encouragement
            entropy = -outputs.std(dim=0).mean()
            entropy_weight = 0.01

            loss = penalty - reward_bonus + entropy_weight * entropy
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            total_valid += rewards.sum().item()
            total_predicted += len(rewards)

        # Logging
        valid_rate = total_valid / total_predicted if total_predicted > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Valid Move Rate: {valid_rate:.2%} | Loss: {total_loss:.4f}")
        if epoch % 10000 == 0:
            print("Model saved to {}".format(OUTPUT))
            torch.save(model.state_dict(), OUTPUT)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Main trains the algorithm to output valid moves randomly
    sample_boards = ds.sample_games(ds.FILE, 500, 100)
    sample_boards = torch.tensor(sample_boards, dtype = torch.float32)
    
    dataset = TensorDataset(sample_boards)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = ChessMovePredictor().to(device)
    if LOAD:
        model.load_state_dict(torch.load(LOAD_FROM, weights_only = True, map_location = device))
        model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_model(model, dataloader, optimizer, 1000000)

    print("Model saved to {}".format(OUTPUT))
    torch.save(model.state_dict(), OUTPUT)

if __name__ == "__main__":
    main()