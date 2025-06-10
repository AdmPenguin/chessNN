import chess.pgn
import random

import chess_funcs as cf

FILE = "lichess_db_standard_rated_2017-02.pgn"

def sample_games(pgn_path, g = 1000, n = 250):
    sampled_games = []
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        for i in range(g):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) == 0:
                continue
            sampled_games.append(moves)
    
    if n > len(sampled_games):
        n = len(sampled_games)

    samples = []
    indicies = random.sample(range(len(sampled_games)), n)

    for i in indicies:
        moves = sampled_games[i]
        max_move = len(moves)

        if len(moves) == 0:
            continue
        j = random.randint(1, max_move)

        board = chess.Board()
        for move in moves[:i]:
            board.push(move)
        
        samples.append(convertModuletoArray(board))

    return samples

def convertModuletoArray(board):
    # Converts a chess module board into an array following chess_funcs definition
    arr = [0] * 72

    # Turn
    arr[0] = 1 if board.turn == chess.WHITE else -1

    # Board layout
    piece_map = {
        chess.PAWN: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                val = -val
            arr[square + 1] = val  # +1 because index 0 is turn

    # Last double pawn move (index 65)
    ep_square = board.ep_square
    if ep_square is not None:
        file_index = chess.square_file(ep_square) + 1  # file a=1, ..., h=8
        arr[65] = file_index
    else:
        arr[65] = 0

    # Castling moved flags (index 66–71)
    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]

    # Flips board to correct orientation
    tempArr = arr.copy()
    for i in range(1, 65):
        arr[i] = tempArr[((i - 1) ^ 56) + 1]
    
    arr[66] = int(not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE))
    arr[67] = int(not board.has_queenside_castling_rights(chess.WHITE))
    arr[68] = int(not board.has_kingside_castling_rights(chess.WHITE))
    arr[69] = int(not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK))
    arr[70] = int(not board.has_queenside_castling_rights(chess.BLACK))
    arr[71] = int(not board.has_kingside_castling_rights(chess.BLACK))

    return arr

def square_to_coords(square):
    row = 7 - (square // 8)
    col = square % 8
    return [row, col]

def move_to_label(move):
    from_sq = move.from_square
    to_sq = move.to_square
    y_from, x_from = square_to_coords(from_sq)
    y_to, x_to = square_to_coords(to_sq)
    return [y_from, x_from, y_to, x_to]

def load_training_pairs_from_games(pgn_file_path, convertModuletoArray, n = 100):
    training_pairs = []

    with open(pgn_file_path) as pgn_file:
        for _ in range(n):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            moves = list(game.mainline_moves())
            if len(moves) < 2:
                continue

            # Choose a random valid move index
            move_index = random.randint(1, len(moves) - 1)

            board = game.board()
            for i in range(move_index):
                board.push(moves[i])

            input_array = convertModuletoArray(board)
            label = move_to_label(moves[move_index])

            training_pairs.append((input_array, label))

    return training_pairs

def main():
    testData = load_training_pairs_from_games(FILE, convertModuletoArray, n = 10)
    print(testData)



if __name__ == "__main__":
    main()