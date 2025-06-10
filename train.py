import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import model as m
import chess_funcs as cf
import data_setup as ds

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
INVALID_PENALTY = 10

USE_PLAYER_DATASET = False
PLAYER_DATASET = "datasets/stella.csv"

OUTPUT = "models/lichess.pth"
LOAD_FROM = "models/lichess.pth"
LOAD = True

AUTOSAVE_INTERVAL = 100
LOG_INTERVAL = 10
LOG_FILE = "logs/lichess.csv"

def load_data(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            arr1_str, arr2_str = row
            arr1 = np.array(list(map(int, arr1_str.split())))
            arr2 = np.array(list(map(int, arr2_str.split())))
            data.append([arr1, arr2])
    return data

# Loss
def compute_loss(outputs, targets, criterion):
    y_from_logits, x_from_logits, y_to_logits, x_to_logits = outputs
    y_from_target, x_from_target, y_to_target, x_to_target = targets.T

    return (
        criterion(y_from_logits, y_from_target) +
        criterion(x_from_logits, x_from_target) +
        criterion(y_to_logits, y_to_target) +
        criterion(x_to_logits, x_to_target)
    )

# Accuracy
def compute_accuracy(outputs, targets):
    preds = [torch.argmax(o, dim=1) for o in outputs]
    correct = sum((p == targets[:, i]).sum().item() for i, p in enumerate(preds))
    return correct / (targets.shape[0] * 4)

# Training loop
def train_model(model, dataloader, optimizer, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_moves = 0
        valid_moves_count = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb, return_logits=True)
            loss = compute_loss(outputs, yb, criterion)

            # Compute predicted moves
            preds = torch.stack([torch.argmax(o, dim=1) for o in outputs], dim=1)  # shape: [batch_size, 4]

            # Penalty for invalid moves
            batch_penalty = 0.0
            for i in range(xb.size(0)):
                board = xb[i].cpu().numpy()
                move = [int(min(max(x, 0), 7)) for x in preds[i].cpu().tolist()]
                total_moves += 1
                if cf.checkValid(board, move):
                    valid_moves_count += 1
                else:
                    batch_penalty += INVALID_PENALTY

            penalty = batch_penalty / xb.size(0)
            loss += penalty

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, yb)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        valid_rate = valid_moves_count / total_moves if total_moves > 0 else 0.0

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc*100:.2f}%, Valid Moves = {valid_rate*100:.2f}%")

        if epoch % LOG_INTERVAL == 0:
            log.append([epoch, avg_loss, avg_acc, valid_rate])

        if epoch % AUTOSAVE_INTERVAL == 0:
            print(f"Final model saved to {OUTPUT}")
            torch.save(model.state_dict(), OUTPUT)
    
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Accuracy", "ValidRate"])  # Header
        for row in log:
            writer.writerow(row)

    print(f"Training log saved to {LOG_FILE}")

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = m.ChessMovePredictor().to(device)

    if LOAD:
        model.load_state_dict(torch.load(LOAD_FROM, map_location=device))
        model.eval()

    if USE_PLAYER_DATASET:
        data = load_data(PLAYER_DATASET)
        inputs = torch.tensor([pair[0] for pair in data], dtype=torch.float32)
        outputs = torch.tensor([pair[1] for pair in data], dtype=torch.long)

        dataset = TensorDataset(inputs, outputs)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        data = ds.load_training_pairs_from_games(ds.FILE, ds.convertModuletoArray, 25000)
        inputs = torch.tensor([pair[0] for pair in data], dtype=torch.float32)
        outputs = torch.tensor([pair[1] for pair in data], dtype=torch.long)

        dataset = TensorDataset(inputs, outputs)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, dataloader, optimizer)

    print(f"Final model saved to {OUTPUT}")
    torch.save(model.state_dict(), OUTPUT)

if __name__ == "__main__":
    main()
