# Helper functions related to chess
# When run allows the user to play chess

import numpy as np

# A board is a 72 number array
# First value is turn (1 for white, -1 for black)
# Pieces hold the following numerical value (negative for black)

# Pawn = 1
# Bishop = 2
# Knight = 3
# Rook = 4
# Queen = 5
# King = 6

# idx 65 represents if the last move was a double pawn move (number = column)
# idx 66 - 71 represent if the king or rooks have moved (order is king, l rook, r rook, white, black)

# Moves are a 4 number array, with the first two being the idx of the piece
# being moved, and the next two being the square to move it to

STANDARD_BOARD = [1, 
                  -4, -3, -2, -5, -6, -2, -3, -4, 
                  -1, -1, -1, -1, -1, -1, -1, -1,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1,
                  4, 3, 2, 5, 6, 2, 3, 4,
                  0,
                  1, 1, 1, 1, 1, 1
                  ]

def getAllValidMoves(board):
    moves = []
    pieces = []
    turn = np.sign(board[0])

    # Find all pieces that can move
    for i in range(1, 65):
        if np.sign(board[i]) == turn:
            pieces.append(i)

    for piece in pieces:
        y, x = (piece - 1) // 8, (piece - 1) % 8  # row (y), col (x)

        piece_type = np.abs(board[piece])

        if piece_type == 1:
            # Pawn
            direction = -1 if turn == 1 else 1  # White moves up (-1), black down (+1)
            start_row = 6 if turn == 1 else 1

            # One step forward
            if 0 <= y + direction <= 7:
                if checkValid(board, [y, x, y + direction, x])[0]:
                    moves.append([y, x, y + direction, x])

            # Two steps forward from start row
            if y == start_row and 0 <= y + 2 * direction <= 7:
                if checkValid(board, [y, x, y + 2 * direction, x])[0]:
                    moves.append([y, x, y + 2 * direction, x])

            # Captures
            for dx in [-1, 1]:
                nx = x + dx
                ny = y + direction
                if 0 <= nx <= 7 and 0 <= ny <= 7:
                    if checkValid(board, [y, x, ny, nx])[0]:
                        moves.append([y, x, ny, nx])

        elif piece_type == 2:
            # Bishop
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dy, dx in directions:
                currY, currX = y + dy, x + dx
                while 0 <= currX <= 7 and 0 <= currY <= 7:
                    if checkValid(board, [y, x, currY, currX])[0]:
                        moves.append([y, x, currY, currX])
                        if board[(currY)*8 + currX + 1] != 0:
                            break  # stop if capturing
                    else:
                        break
                    currY += dy
                    currX += dx

        elif piece_type == 3:
            # Knight moves
            knight_moves = [
                (y - 2, x - 1), (y - 2, x + 1),
                (y + 2, x - 1), (y + 2, x + 1),
                (y - 1, x - 2), (y - 1, x + 2),
                (y + 1, x - 2), (y + 1, x + 2)
            ]
            for ny, nx in knight_moves:
                if 0 <= nx <= 7 and 0 <= ny <= 7:
                    if checkValid(board, [y, x, ny, nx])[0]:
                        moves.append([y, x, ny, nx])

        elif piece_type == 4:
            # Rook
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dy, dx in directions:
                currY, currX = y + dy, x + dx
                while 0 <= currX <= 7 and 0 <= currY <= 7:
                    if checkValid(board, [y, x, currY, currX])[0]:
                        moves.append([y, x, currY, currX])
                        if board[(currY)*8 + currX + 1] != 0:
                            break
                    else:
                        break
                    currY += dy
                    currX += dx

        elif piece_type == 5:
            # Queen (combines bishop + rook)
            directions = [
                (-1, -1), (-1, 1), (1, -1), (1, 1),
                (-1, 0), (1, 0), (0, -1), (0, 1)
            ]
            for dy, dx in directions:
                currY, currX = y + dy, x + dx
                while 0 <= currX <= 7 and 0 <= currY <= 7:
                    if checkValid(board, [y, x, currY, currX])[0]:
                        moves.append([y, x, currY, currX])
                        if board[(currY)*8 + currX + 1] != 0:
                            break
                    else:
                        break
                    currY += dy
                    currX += dx

        elif piece_type == 6:
            # King
            king_moves = [
                (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                (y, x - 1),                 (y, x + 1),
                (y + 1, x - 1), (y + 1, x), (y + 1, x + 1),
            ]
            for ny, nx in king_moves:
                if 0 <= nx <= 7 and 0 <= ny <= 7:
                    if checkValid(board, [y, x, ny, nx])[0]:
                        moves.append([y, x, ny, nx])

            # Castling (assuming board[66..71] are castling flags)
            if turn == 1:
                # White castling flags at 66, 67, 68
                if board[66] == 1 and board[67] == 1:
                    if checkValid(board, [y, x, y, x - 2])[0]:
                        moves.append([y, x, y, x - 2])
                if board[66] == 1 and board[68] == 1:
                    if checkValid(board, [y, x, y, x + 2])[0]:
                        moves.append([y, x, y, x + 2])
            else:
                # Black castling flags at 69, 70, 71
                if board[69] == 1 and board[70] == 1:
                    if checkValid(board, [y, x, y, x - 2])[0]:
                        moves.append([y, x, y, x - 2])
                if board[69] == 1 and board[71] == 1:
                    if checkValid(board, [y, x, y, x + 2])[0]:
                        moves.append([y, x, y, x + 2])

    return moves

def checkValid(board, move, doPrint=False):
    # Checks if move is valid given board
    # Returns a tuple of (validity_bool, resulting_board)
    # Returns original board if move invalid

    # Out of bounds check (coordinates must be between 0 and 7 inclusive)
    for coord in move:
        if coord < 0 or coord > 7:
            if doPrint:
                print(f"Move out of bounds: {coord}")
            return False, board

    newBoard = board.copy()

    pieceToMove = board[(move[0] * 8) + move[1] + 1]
    # Rule 1: Piece belongs to current turn's side
    if np.sign(pieceToMove) != np.sign(board[0]) or pieceToMove == 0:
        if doPrint:
            print("Not your piece or empty square.")
        return False, board

    # Absolute value for piece type
    absPiece = abs(pieceToMove)
    xDist = move[3] - move[1]
    yDist = move[2] - move[0]

    # Helper to get landing square piece
    landingSquare = board[move[2] * 8 + move[3] + 1]

    if absPiece == 1:
        direction = board[0]  # 1 for white moving up (y-1), -1 for black moving down (y+1)
        start_row = 6 if direction == 1 else 1
        promotion_row = 0 if direction == 1 else 7

        # Forward move by 1
        if xDist == 0 and yDist == -direction:
            if landingSquare != 0:
                if doPrint:
                    print("Pawn blocked in front.")
                return False, board

        # Forward move by 2 from start position
        elif xDist == 0 and yDist == -2 * direction:
            if move[0] != start_row:
                if doPrint:
                    print("Pawn two-step move not from start row.")
                return False, board
            intermediate_square = board[int((move[0] - direction) * 8 + move[1] + 1)]
            if intermediate_square != 0 or landingSquare != 0:
                if doPrint:
                    print("Pawn two-step move blocked.")
                return False, board
            newBoard[65] = move[1]  # record column for double pawn move

        # Capture move (diagonal by 1)
        elif abs(xDist) == 1 and yDist == -direction:
            if landingSquare == 0:
                # En passant capture
                ep_row = 3 if direction == 1 else 4
                if move[0] != ep_row or move[1] != board[65]:
                    if doPrint:
                        print("Invalid en passant.")
                    return False, board
                # Remove the pawn being captured en passant
                captured_pawn_index = (move[0]) * 8 + move[3] + 1
                newBoard[captured_pawn_index] = 0
            else:
                # Normal capture
                if np.sign(landingSquare) == direction:
                    if doPrint:
                        print("Pawn cannot capture own piece.")
                    return False, board

        else:
            if doPrint:
                print("Invalid pawn move.")
            return False, board

        # Promotion
        if move[2] == promotion_row:
            newBoard[move[2] * 8 + move[3] + 1] = 5 * direction  # promote to queen


    elif absPiece == 2:
        # Bishop move: diagonal with clear path
        if abs(xDist) != abs(yDist) or xDist == 0:
            if doPrint:
                print("Bishop move not diagonal.")
            return False, board
        for i in range(1, abs(xDist)):
            idx = ((move[0] + i * np.sign(yDist)) * 8) + (move[1] + i * np.sign(xDist)) + 1
            if board[idx] != 0:
                if doPrint:
                    print("Bishop path blocked.")
                return False, board
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            if doPrint:
                print("Bishop can't capture friendly piece.")
            return False, board

    elif absPiece == 3:
        # Knight move: L-shape
        if not ((abs(xDist) == 2 and abs(yDist) == 1) or (abs(xDist) == 1 and abs(yDist) == 2)):
            if doPrint:
                print("Invalid knight move.")
            return False, board
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            if doPrint:
                print("Knight can't capture friendly piece.")
            return False, board

    elif absPiece == 4:
        # Rook move: horizontal or vertical with clear path
        if xDist != 0 and yDist != 0:
            if doPrint:
                print("Rook move not straight.")
            return False, board

        step_x = np.sign(xDist)
        step_y = np.sign(yDist)
        distance = abs(xDist) if xDist != 0 else abs(yDist)

        for i in range(1, distance):
            idx = ((move[0] + i * step_y) * 8) + (move[1] + i * step_x) + 1
            if board[idx] != 0:
                if doPrint:
                    print("Rook path blocked.")
                return False, board
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            if doPrint:
                print("Rook can't capture friendly piece.")
            return False, board

        # Update rook movement flags for castling
        if board[0] == 1:
            if move[0] == 7 and move[1] == 0:
                newBoard[67] = 0  # white queenside rook moved
            elif move[0] == 7 and move[1] == 7:
                newBoard[68] = 0  # white kingside rook moved
        else:
            if move[0] == 0 and move[1] == 0:
                newBoard[70] = 0  # black queenside rook moved
            elif move[0] == 0 and move[1] == 7:
                newBoard[71] = 0  # black kingside rook moved

    elif absPiece == 5:
        # Queen move: diagonal or straight with clear path
        if abs(xDist) == abs(yDist) and xDist != 0:
            # Diagonal move
            for i in range(1, abs(xDist)):
                idx = ((move[0] + i * np.sign(yDist)) * 8) + (move[1] + i * np.sign(xDist)) + 1
                if board[idx] != 0:
                    if doPrint:
                        print("Queen path blocked diagonally.")
                    return False, board
        elif (xDist == 0) != (yDist == 0):
            # Straight move
            step_x = np.sign(xDist)
            step_y = np.sign(yDist)
            distance = abs(xDist) if xDist != 0 else abs(yDist)
            for i in range(1, distance):
                idx = ((move[0] + i * step_y) * 8) + (move[1] + i * step_x) + 1
                if board[idx] != 0:
                    if doPrint:
                        print("Queen path blocked straight.")
                    return False, board
        else:
            if doPrint:
                print("Invalid queen move.")
            return False, board

        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            if doPrint:
                print("Queen can't capture friendly piece.")
            return False, board

    elif absPiece == 6:
        # King move: one step in any direction or castling
        if max(abs(xDist), abs(yDist)) == 1:
            # Normal king move
            if landingSquare != 0 and np.sign(landingSquare) == board[0]:
                if doPrint:
                    print("King can't capture friendly piece.")
                return False, board
            # Mark king moved
            if board[0] == 1:
                newBoard[66] = 0
            else:
                newBoard[69] = 0

        elif yDist == 0 and abs(xDist) == 2:
            # Castling
            if board[0] == 1:
                # White castling
                if newBoard[66] == 0:
                    if doPrint:
                        print("King has already moved, no castling.")
                    return False, board
                if xDist == -2:
                    # Queenside castling
                    if newBoard[67] == 0:
                        if doPrint:
                            print("Queenside rook moved, no castling.")
                        return False, board
                    if board[58] != 0 or board[59] != 0 or board[60] != 0:
                        if doPrint:
                            print("Pieces in the way for queenside castling.")
                        return False, board
                    # Move pieces for castling
                    newBoard[57] = 0
                    newBoard[59] = 6
                    newBoard[60] = 4
                    newBoard[67] = 0
                    newBoard[66] = 0
                elif xDist == 2:
                    # Kingside castling
                    if newBoard[68] == 0:
                        if doPrint:
                            print("Kingside rook moved, no castling.")
                        return False, board
                    if board[62] != 0 or board[63] != 0:
                        if doPrint:
                            print("Pieces in the way for kingside castling.")
                        return False, board
                    newBoard[64] = 0
                    newBoard[63] = 6
                    newBoard[62] = 4
                    newBoard[68] = 0
                    newBoard[66] = 0
                else:
                    if doPrint:
                        print("Invalid castling destination.")
                    return False, board
            else:
                # Black castling
                if newBoard[69] == 0:
                    if doPrint:
                        print("King has already moved, no castling.")
                    return False, board
                if xDist == -2:
                    # Queenside castling
                    if newBoard[70] == 0:
                        if doPrint:
                            print("Queenside rook moved, no castling.")
                        return False, board
                    if board[2] != 0 or board[3] != 0 or board[4] != 0:
                        if doPrint:
                            print("Pieces in the way for queenside castling.")
                        return False, board
                    newBoard[1] = 0
                    newBoard[3] = -6
                    newBoard[4] = -4
                    newBoard[70] = 0
                    newBoard[69] = 0
                elif xDist == 2:
                    # Kingside castling
                    if newBoard[71] == 0:
                        if doPrint:
                            print("Kingside rook moved, no castling.")
                        return False, board
                    if board[6] != 0 or board[7] != 0:
                        if doPrint:
                            print("Pieces in the way for kingside castling.")
                        return False, board
                    newBoard[8] = 0
                    newBoard[7] = -6
                    newBoard[6] = -4
                    newBoard[71] = 0
                    newBoard[69] = 0
                else:
                    if doPrint:
                        print("Invalid castling destination.")
                    return False, board
        else:
            if doPrint:
                print("Invalid king move.")
            return False, board
    else:
        if doPrint:
            print("Unknown piece type.")
        return False, board

    # Move piece on new board
    newBoard[move[0] * 8 + move[1] + 1] = 0
    newBoard[move[2] * 8 + move[3] + 1] = pieceToMove
    # Flip turn
    newBoard[0] = -newBoard[0]

    # Rule 3: Move should not leave player in check
    testBoard = newBoard.copy()
    testBoard[0] = -testBoard[0]  # flip back turn to test opponent's check
    if isInCheck(testBoard):
        if doPrint:
            print("Cannot end move in check.")
        return False, board

    return True, newBoard

def isCheckmate(board):
    # Runs a check to see if a move is checkmate
    # Runs through all valid moves to see if still in check
    valid = getAllValidMoves(board)
    if len(valid) == 0:
        return True, board[0]
    return False, board[0]

def isInCheck(board):
    king_pos = 0

    # Find king position of current side (board[0] == 1 for white, -1 for black)
    king_piece = 6 * board[0]  # 6 for white king, -6 for black king
    for i in range(1, 65):
        if board[i] == king_piece:
            king_pos = i
            break
    if king_pos == 0:
        # King not found - abnormal
        return False

    enemy = -board[0]
    # Convert pos to (row, col)
    row = (king_pos - 1) // 8
    col = (king_pos - 1) % 8

    # Directions for rook/queen (straight lines)
    directions_straight = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Directions for bishop/queen (diagonals)
    directions_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Check straight lines for rook or queen attacks
    for dr, dc in directions_straight:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            pos = r * 8 + c + 1
            piece = board[pos]
            if piece == 0:
                r += dr
                c += dc
                continue
            if piece == 4 * enemy or piece == 5 * enemy:  # Rook or Queen
                return True
            break

    # Check diagonal lines for bishop or queen attacks
    for dr, dc in directions_diag:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            pos = r * 8 + c + 1
            piece = board[pos]
            if piece == 0:
                r += dr
                c += dc
                continue
            if piece == 2 * enemy or piece == 5 * enemy:  # Bishop or Queen
                return True
            break

    # Check pawn attacks (enemy pawns threaten king)
    if board[0] == 1:
        # White to move => enemy is black pawns (-1)
        # Black pawns attack diagonally down (+1 row), cols ±1
        pawn_row = row + 1
        if pawn_row < 8:
            for dc in [-1, 1]:
                c = col + dc
                if 0 <= c < 8:
                    pos = pawn_row * 8 + c + 1
                    if board[pos] == -1:
                        return True
    else:
        # Black to move => enemy is white pawns (+1)
        # White pawns attack diagonally up (-1 row), cols ±1
        pawn_row = row - 1
        if pawn_row >= 0:
            for dc in [-1, 1]:
                c = col + dc
                if 0 <= c < 8:
                    pos = pawn_row * 8 + c + 1
                    if board[pos] == 1:
                        return True

    # Knight moves relative to king
    knight_moves = [
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2), (1, 2),
        (2, -1), (2, 1)
    ]
    for dr, dc in knight_moves:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            pos = r * 8 + c + 1
            if board[pos] == 3 * enemy:  # Knight
                return True

    # King moves (one square in any direction)
    king_moves = directions_straight + directions_diag
    for dr, dc in king_moves:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            pos = r * 8 + c + 1
            if board[pos] == 6 * enemy:  # Enemy king
                return True

    return False

def printBoard(board):
    # Takes 65 array board and prints it out
    # Uppercase = white, lowercase = black
    print("   1  2  3  4  5  6  7  8")
    for i in range(8):
        print("  " + "---" * 8)
        print(i + 1, end=' ')
        for square in range(8):
            piece = " "
            curr = board[(i * 8) + square + 1]
            if curr != 0:
                if curr == 1:
                    piece = "P"
                if curr == 2:
                    piece = "B"
                if curr == 3:
                    piece = "N"
                if curr == 4:
                    piece = "R"
                if curr == 5:
                    piece = "Q"
                if curr == 6:
                    piece = "K"
                if curr == -1:
                    piece = "p"
                if curr == -2:
                    piece = "b"
                if curr == -3:
                    piece = "n"
                if curr == -4:
                    piece = "r"
                if curr == -5:
                    piece = "q"
                if curr == -6:
                    piece = "k"
            print("|{}|".format(piece), end="")
        print(" {}".format(i + 1))
    print("  " + "---" * 8)
    print("   1  2  3  4  5  6  7  8")

    if board[0] == 1:
        print("White's Turn")
    else:
        print("Black's Turn")

def main():
    board = STANDARD_BOARD.copy()
    printBoard(board)

    while(1):
        if isInCheck(board):
            checkmateCheck = isCheckmate(board)
            if(checkmateCheck[0]):
                if(checkmateCheck[0] == 1):
                    print("Black wins")
                else:
                    print("White wins")
                break

        move = input("Enter move: ").split(" ")
        if move == ["resign"]:
            if board[0] == 1:
                print("White has resigned. Black wins.")
            elif board[0] == -1:
                print("Black has resigned. White wins.")
            break
        for elem in range(len(move)):
            move[elem] = int(move[elem]) - 1

        validation = checkValid(board, move, True)
        if len(validation[1]) == 1:
            if (validation[1][0] == 1):
                print("Black wins")
            else:
                print("White wins")
            break
        if validation[0]:
            board = validation[1]
            printBoard(board)
        else:
            print("Invalid move, try again")

if __name__ == '__main__':
    main()