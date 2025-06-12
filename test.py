import model as m
import chess_funcs as cf
import data_setup as ds

import torch

MODEL = "models/lichess.pth"

def get_model_move(model, board_tensor):
    with torch.no_grad():
        board_tensor = board_tensor.unsqueeze(0)  # Add batch dim
        outputs = model(board_tensor, return_logits=True)
        predicted = [torch.argmax(o, dim=1).item() for o in outputs]
    return predicted

# Allows you to play against the model
def game():
    # Load board
    board = cf.STANDARD_BOARD.copy()

    # Load model
    model = m.ChessMovePredictor()
    model.load_state_dict(torch.load(MODEL, weights_only=True, map_location="cpu"))
    model.eval()

    player = input("Play as white or black (enter in lowercase): ").strip()
    if player not in ["white", "black"]:
        print("Error, invalid input. Quitting...")
        return

    player_color = 1 if player == "white" else -1

    cf.printBoard(board)

    while True:
        if cf.isInCheck(board):
            checkmateCheck = cf.isCheckmate(board)
            if checkmateCheck[0]:
                print("Black wins" if checkmateCheck[0] == 1 else "White wins")
                break

        if board[0] == player_color:
            # Player's turn
            move = input("Enter move (e.g., '2 1 4 1' or 'resign'): ").strip().split()
            if move == ["resign"]:
                winner = "Black" if player_color == 1 else "White"
                print(f"{['White', 'Black'][player_color == 1]} has resigned. {winner} wins.")
                break
            if len(move) != 4:
                print("Invalid input. Enter four numbers (1-8) or 'resign'.")
                continue
            try:
                move = [int(x) - 1 for x in move]
            except ValueError:
                print("Invalid input, use numbers between 1 and 8.")
                continue

            valid, result = cf.checkValid(board, move, True)
            if isinstance(result, list) and len(result) == 1:
                print("Black wins" if result[0] == 1 else "White wins")
                break
            if valid:
                board = result
                cf.printBoard(board)
            else:
                print("Invalid move, try again.")
        else:
            # Model's turn
            print("Model is thinking...")
            board_tensor = torch.tensor(board, dtype=torch.float32)
            move = get_model_move(model, board_tensor)
            print(f"Model move: {[x+1 for x in move]}")

            valid, result = cf.checkValid(board, move, True)
            if isinstance(result, list) and len(result) == 1:
                print("Black wins" if result[0] == 1 else "White wins")
                break
            if valid:
                board = result
                cf.printBoard(board)
            else:
                print("Model made an invalid move. You win!")
                break

# Variant where the model has no validity check
def game_nocheck():
    # Load board
    board = cf.STANDARD_BOARD.copy()

    # Load model
    model = m.ChessMovePredictor()
    model.load_state_dict(torch.load(MODEL, weights_only=True, map_location="cpu"))
    model.eval()

    player = input("Play as white or black (enter in lowercase): ").strip()
    if player not in ["white", "black"]:
        print("Error, invalid input. Quitting...")
        return

    player_color = 1 if player == "white" else -1

    cf.printBoard(board)

    while True:
        # Checkmate check
        if cf.isInCheck(board):
            checkmateCheck = cf.isCheckmate(board)
            if checkmateCheck[0]:
                print("Black wins" if checkmateCheck[0] == 1 else "White wins")
                break

        if board[0] == player_color:
            # Player's turn
            move = input("Enter move (e.g., '2 1 4 1' or 'resign'): ").strip().split()
            if move == ["resign"]:
                winner = "Black" if player_color == 1 else "White"
                print(f"{['White', 'Black'][player_color == 1]} has resigned. {winner} wins.")
                break
            if len(move) != 4:
                print("Invalid input. Enter four numbers (1-8) or 'resign'.")
                continue
            try:
                move = [int(x) - 1 for x in move]
            except ValueError:
                print("Invalid input, use numbers between 1 and 8.")
                continue

            valid, result = cf.checkValid(board, move, True)
            if isinstance(result, list) and len(result) == 1:
                print("Black wins" if result[0] == 1 else "White wins")
                break
            if valid:
                board = result
                cf.printBoard(board)
            else:
                print("Invalid move, try again.")
        else:
            # Model's turn — no validity check
            print("Model is thinking...")
            board_tensor = torch.tensor(board, dtype=torch.float32)
            move = get_model_move(model, board_tensor)
            print(f"Model move: {[x+1 for x in move]}")

            # Directly apply move
            from_idx = 8 * move[0] + move[1] + 1
            to_idx = 8 * move[2] + move[3] + 1
            board[to_idx] = board[from_idx]
            board[from_idx] = 0
            board[0] *= -1  # Switch turn

            cf.printBoard(board)

def randomMove():
    # Load model
    model = m.ChessMovePredictor()
    model.load_state_dict(torch.load(MODEL, weights_only=True, map_location="cpu"))
    model.eval()

    testData = ds.sample_games(ds.FILE, 50, 1)

    for board in testData:
        cf.printBoard(board)
        board_tensor = torch.tensor(board, dtype=torch.float32)
        move = get_model_move(model, board_tensor)
        print(move)
        try:
            valid = cf.checkValid(board, move)[0]
            if valid:
                cf.printBoard(board)
            else:
                print("Invalid move.")
        except:
            print("Invalid move.")

if __name__ == "__main__":
    mode = input("Mode: ")
    if mode == "1":
        game()
    elif mode == "2":
        game_nocheck()
    elif mode == "3":
        randomMove()