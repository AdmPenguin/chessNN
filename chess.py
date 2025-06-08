# Helper functions related to chess
# When run allows the user to play chess

# TODO: 
# - Fix castling
# - Finish isInCheck

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
    # Given a board, returns all valid moves which can be performed by the current player
    moves = []
    pieces = []
    turn = np.sign(board[0])

    # Find all pieces that can move
    for i in range(1, 65):
        if np.sign(board[i]) == turn:
            pieces.append(i)
    
    for piece in pieces:
        x, y = piece // 8, (piece - 1) % 8
        if np.abs(board[piece]) == 1:
            # pawn
            # white
            if board[0] == 1:
                if y > 0:
                    if checkValid(board, [y, x, y - 1, x])[0]:
                        moves.append([y, x, y - 1, x])
                    if x > 0:
                        if checkValid(board, [y, x, y - 1, x - 1])[0]:
                            moves.append([y, x, y - 1, x - 1])
                    if x < 7:
                        if checkValid(board, [y, x, y - 1, x + 1])[0]:
                            moves.append([y, x, y - 1, x + 1])
                if y > 1:
                    if checkValid(board[y, x, y - 2, x])[0]:
                        moves.append([y, x, y - 2, x])
            # black
            else:
                if y < 7:
                    if checkValid(board, [y, x, y + 1, x])[0]:
                        moves.append([y, x, y + 1, x])
                    if x > 0:
                        if checkValid(board, [y, x, y + 1, x - 1])[0]:
                            moves.append([y, x, y + 1, x - 1])
                    if x < 7:
                        if checkValid(board, [y, x, y + 1, x + 1])[0]:
                            moves.append([y, x, y + 1, x + 1])
                if y < 6:
                    if checkValid(board[y, x, y + 2, x])[0]:
                        moves.append([y, x, y + 2, x])
        elif np.abs(board[piece]) == 2:
            # bishop
            x, y = piece // 8, (piece - 1) % 8
            # ul
            currX, currY = x - 1, y - 1
            while(0 < currX < 7 and 0 < currY < 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY -= 1
                currX -= 1
            # ur
            currX, currY = x + 1, y - 1
            while(0 < currX < 7 and 0 < currY < 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY -= 1
                currX += 1
            # dl
            currX, currY = x - 1, y + 1
            while(0 < currX < 7 and 0 < currY < 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY += 1
                currX -= 1
            # dr
            currX, currY = x + 1, y + 1
            while(0 < currX < 7 and 0 < currY < 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY += 1
                currX += 1
        elif np.abs(board[piece]) == 3:
            # knight
            x, y = piece // 8, (piece - 1) % 8
            if checkValid(board, [y, x, y - 2, x - 1])[0]:
                moves.append(board, [y, x, y - 2, x - 1])
            if checkValid(board, [y, x, y + 2, x - 1])[0]:
                moves.append(board, [y, x, y + 2, x - 1])
            if checkValid(board, [y, x, y - 2, x + 1])[0]:
                moves.append(board, [y, x, y - 2, x + 1])
            if checkValid(board, [y, x, y + 2, x + 1])[0]:
                moves.append(board, [y, x, y + 2, x + 1])

            if checkValid(board, [y, x, y - 1, x - 2])[0]:
                moves.append(board, [y, x, y - 1, x - 2])
            if checkValid(board, [y, x, y + 1, x - 2])[0]:
                moves.append(board, [y, x, y + 1, x - 2])
            if checkValid(board, [y, x, y - 1, x + 2])[0]:
                moves.append(board, [y, x, y - 1, x + 2])
            if checkValid(board, [y, x, y + 1, x + 2])[0]:
                moves.append(board, [y, x, y + 1, x + 2])
        elif np.abs(board[piece]) == 4:
            # rook
            x, y = piece // 8, (piece - 1) % 8
            # u
            currY = y - 1
            while(currY >= 0):
                if checkValid(board, [y, x, currY, x])[0]:
                    moves.append([y, x, currY, x])
                    currY -= 1
            # d
            currY = y + 1
            while(currY <= 7):
                if checkValid(board, [y, x, currY, x])[0]:
                    moves.append([y, x, currY, x])
                    currY += 1
            # l
            currX = x - 1
            while(currX >= 0):
                if checkValid(board, [y, x, y, currX])[0]:
                    moves.append([y, x, y, currX])
                    currX -= 1
            # r
            currX = x + 1
            while(currX >= 0):
                if checkValid(board, [y, x, y, currX])[0]:
                    moves.append([y, x, y, currX])
                    currX += 1
        elif np.abs(board[piece]) == 5:
            # queen
            x, y = piece // 8, (piece - 1) % 8
            # ul
            currX, currY = x - 1, y - 1
            while(0 <= currX <= 7 and 0 <= currY <= 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY -= 1
                currX -= 1
            # ur
            currX, currY = x + 1, y - 1
            while(0 <= currX <= 7 and 0 <= currY <= 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY -= 1
                currX += 1
            # dl
            currX, currY = x - 1, y + 1
            while(0 <= currX <= 7 and 0 <= currY <= 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY += 1
                currX -= 1
            # dr
            currX, currY = x + 1, y + 1
            while(0 <= currX <= 7 and 0 <= currY <= 7):
                if checkValid(board, [y, x, currY, currX])[0]:
                    moves.append([y, x, currY, currX])
                currY += 1
                currX += 1
            # u
            currY = y - 1
            while(currY >= 0):
                if checkValid(board, [y, x, currY, x])[0]:
                    moves.append([y, x, currY, x])
                    currY -= 1
            # d
            currY = y + 1
            while(currY <= 7):
                if checkValid(board, [y, x, currY, x])[0]:
                    moves.append([y, x, currY, x])
                    currY += 1
            # l
            currX = x - 1
            while(currX >= 0):
                if checkValid(board, [y, x, y, currX])[0]:
                    moves.append([y, x, y, currX])
                    currX -= 1
            # r
            currX = x + 1
            while(currX >= 0):
                if checkValid(board, [y, x, y, currX])[0]:
                    moves.append([y, x, y, currX])
                    currX += 1
        elif np.abs(board[piece]) == 6:
            # king
            x, y = piece // 8, (piece - 1) % 8
            if x > 0:
                if checkValid(board[y, x, y, x - 1])[0]:
                    moves.append([y, x, y, x - 1])
                if y > 0:
                    if checkValid(board[y, x, y - 1, x - 1])[0]:
                        moves.append([y, x, y - 1, x - 1])
                if y < 7:
                    if checkValid(board[y, x, y + 1, x - 1])[0]:
                        moves.append([y, x, y + 1, x - 1])
            if x < 0:
                if checkValid(board[y, x, y, x + 1])[0]:
                    moves.append([y, x, y, x + 1])
                if y > 0:
                    if checkValid(board[y, x, y - 1, x + 1])[0]:
                        moves.append([y, x, y - 1, x + 1])
                if y < 7:
                    if checkValid(board[y, x, y + 1, x + 1])[0]:
                        moves.append([y, x, y + 1, x + 1])
            if y > 0:
                if checkValid(board[y, x, y - 1, x])[0]:
                    moves.append([y, x, y - 1, x])
            if y < 7:
                if checkValid(board[y, x, y + 1, x])[0]:
                    moves.append([y, x, y + 1, x])

            # white
            if board[0] == 1:
                if board[66] == 1:
                    if board[67] == 1:
                        if checkValid(board, [y, x, y, x - 2])[0]:
                            moves.append([y, x, y, x - 2])
                    if board[68] == 1:
                        if checkValid(board, [y, x, y, x + 2])[0]:
                            moves.append([y, x, y, x + 2])
            # black
            else:
                if board[69] == 1:
                    if board[70] == 1:
                        if checkValid(board, [y, x, y, x - 2])[0]:
                            moves.append([y, x, y, x - 2])
                    if board[71] == 1:
                        if checkValid(board, [y, x, y, x + 2])[0]:
                            moves.append([y, x, y, x + 2])
    
    return moves

def checkValid(board, move):
    # Checks if move is valid given board

    # Returns a tuple of whether the move is valid
    # and the resulant board (returns original if false)

    # Out of bounds check
    for coord in move:
        if 0 > coord < 7:
            return False, board

    # Checkmate check
    if isInCheck(board):
        if isCheckmate(board):
            return False, [board[0]]

    newBoard = board

    pieceToMove = board[(move[0] * 8) + move[1] + 1]
    # Rule 1: Piece is current turn's side
    if np.sign(pieceToMove) != np.sign(board[0]):
        print("Not your piece.")
        return False, board

    # Rule 2: Move follows piece's move
    # Rule 2.5: Piece doesn't run into another piece / friendly capture
    pieceToMove = np.abs(pieceToMove)
    if pieceToMove == 1:
        # pawn
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        # pawn movement check
        # regular move
        if np.abs(yDist) == 1 and np.abs(xDist) == 0:
            if np.sign(yDist) == board[0]:
                print("Invalid pawn move.")
                return False, board
            elif board[move[2] * 8 + move[3] + 1] != 0:
                print("Pawn hit another piece.")
                return False, board
        # capture
        elif np.abs(yDist) == 1 and np.abs(xDist) == 1:
            if np.sign(yDist) == board[0]:
                print("Invalid pawn move.")
                return False, board
            elif move[2] * 8 + move[3] + 1 == 0:
                # White's en passant check
                if board[0] == 1:
                    if move[3] != board[65] or move[2] != 4:
                        print("Pawn cannot move diagonally without capture")
                        return False, board
                # Black's en passant check
                elif board[0] == -1:
                    if move[3] != board[65] or move[2] != 5:
                        print("Pawn cannot move diagonally without capture")
                        return False, board
            elif board[move[0] * 8 + (np.sign(xDist) + move[1]) + 1] == board[0]:
                print("Piece can't capture friendly piece.")
                return False, board
        # two jump
        elif np.abs(yDist) == 2 and np.abs(xDist) == 0:
            if board[0] == 1 and move[0] != 6:
                print("Invalid pawn move.")
                return False, board
            elif board[0] == -1 and move[0] != 1:
                print("Invalid pawn move.")
                return False, board
            newBoard[65] = move[3]
        else:
            print("Invalid pawn move.")
            return False, board
        # promotion
        if (move[2] == 1 and board[0] == 1) or (move[2] == 8 and board[0] == -1):
            pieceToMove = 5

    elif pieceToMove == 2:
        # bishop
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        # diagonal check
        if np.abs(xDist) != np.abs(yDist):
            print("Bishop move not diagonal.")
            return False, board
        
        # clear path check
        for i in range(1, np.abs(xDist)):
            currSquare = (((np.sign(yDist) * i) + move[0]) * 8) + ((np.sign(xDist) * i) + move[1]) + 1
            if board[currSquare] != 0:
                print("Bishop hits another piece.")
                return False, board
        # doesn't capture friendly check
        landingSquare = board[move[2] * 8 + move[3] + 1]
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            print("Piece can't capture friendly piece.")
            return False, board
    elif pieceToMove == 3:
        # knight
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        if np.abs(xDist) == 2 and np.abs(yDist) == 1:
            landingSquare = board[move[2] * 8 + move[3] + 1]
            if landingSquare != 0 and np.sign(landingSquare) == board[0]:
                print("Piece can't capture friendly piece.")
                return False, board
        elif np.abs(xDist) == 1 and np.abs(yDist) == 2:
            landingSquare = board[move[2] * 8 + move[3] + 1]
            if landingSquare != 0 and np.sign(landingSquare) == board[0]:
                print("Piece can't capture friendly piece.")
                return False, board
        else:
            print("Invalid knight move.")
            return False, board

    elif pieceToMove == 4:
        # rook
        xDist, yDist = move[3] - move[1], move[2] - move[0]

        # horizontal check
        if (xDist == 0) == (yDist == 0):
            print("Rook move not horizontal.")
            return False, board
        
        # cleaer path check
        if yDist != 0:
            for i in range(1, np.abs(yDist)):
                currSquare = ((np.sign(yDist) * i + move[0]) * 8) + move[1] + 1
                if board[currSquare] != 0:
                    print("Rook hits another piece.")
                    return False, board
        elif xDist != 0:
            for i in range(1, np.abs(xDist)):
                currSquare = (move[0] * 8) + ((np.sign(xDist) * i) + move[1]) + 1
                if board[currSquare] != 0:
                    print("Rook hits another piece.")
                    return False, board
        # doesn't capture friendly check
        landingSquare = board[move[2] * 8 + move[3] + 1]
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            print("Piece can't capture friendly piece.")
            return False, board
        # set movement flags
        if move[0] == 8 and move[1] == 1:
            newBoard[67] = 0
        elif move[0] == 8 and move[1] == 8:
            newBoard[68] = 0
        elif move[0] == 1 and move[1] == 1:
            newBoard[70] = 0
        elif move[0] == 1 and move[1] == 8:
            newBoard[71] = 0
    elif pieceToMove == 5:
        # queen
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        if np.abs(xDist) == np.abs(yDist):
            # diagonal move
            for i in range(1, np.abs(xDist)):
                currSquare = (((np.sign(yDist) * i) + move[0]) * 8) + ((np.sign(xDist) * i) + move[1]) + 1
                if board[currSquare] != 0:
                    print("Queen hits another piece.")
                    return False, board
        elif (xDist == 0) != (yDist == 0):
            # horizontal move
            if yDist != 0:
                for i in range(1, np.abs(yDist)):
                    currSquare = ((np.sign(yDist) * i + move[0]) * 8) + move[1] + 1
                    if board[currSquare] != 0:
                        print("Queen hits another piece.")
                        return False, board
            elif xDist != 0:
                for i in range(1, np.abs(xDist)):
                    currSquare = (move[0] * 8) + ((np.sign(xDist) * i) + move[1]) + 1
                    if board[currSquare] != 0:
                        print("Queen hits another piece.")
                        return False, board
        else:
            print("Invalid Queen move.")
            return False, board

        # doesn't capture friendly check
        landingSquare = board[move[2] * 8 + move[3] + 1]
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            print("Piece can't capture friendly piece.")
            return False, board
    elif pieceToMove == 6:
        # king
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        # one square check
        if np.abs(xDist) != 1 and np.abs(yDist) != 1:
            # doesn't capture friendly check
            landingSquare = board[move[2] * 8 + move[3] + 1]
            if landingSquare != 0 and np.sign(landingSquare) == board[0]:
                print("Piece can't capture friendly piece.")
                return False, board
        # castling check
        elif move[3] == 3 or move[3] == 7:
            # white castling
            if (move[2]) == 8 and board[1] == 1:
                if board[66] == 0:
                    print("Invalid castling, King already moved.")
                    return False, board
                elif move[3] == 2: 
                    if board[67] == 0:
                        print("Invalid castling, Rook already moved.")
                        return False, board
                    else:
                        if board[58] == 0 and board[59] == 0 and board[60] == 0:
                            newBoard[57] = 0
                            newBoard[60] = 4
                        else:
                            print("Invalid castling, pieces in way.")
                            return False, board
                elif move[3] == 6:
                    if board[68] == 0:
                        print("Invalid castling, Rook already moved.")
                        return False, board
                    else:
                        if board[62] == 0 and board[63] == 0:
                            print(board[64])
                            newBoard[64] = 0
                            newBoard[62] = 4
                        else:
                            print("Invalid castling, pieces in way.")
                            return False, board
                else:
                    print("Invalid King move.")
                    return False, board
                
            # black castling
            if (move[2]) == 1 and board[1] == -1:
                if board[69] == 0:
                    print("Invalid castling, King already moved.")
                    return False, board
                elif move[3] == 2: 
                    if board[70] == 0:
                        print("Invalid castling, Rook already moved.")
                        return False, board
                    else:
                        if board[2] == 0 and board[3] == 0 and board[4] == 0:
                            newBoard[1] = 0
                            newBoard[4] = -4
                        else:
                            print("Invalid castling, pieces in way.")
                            return False, board
                elif move[3] == 6:
                    if board[71] == 0:
                        print("Invalid castling, Rook already moved.")
                        return False, board
                    else:
                        if board[6] == 0 and board[7] == 0:
                            newBoard[8] = 0
                            newBoard[6] = -4
                        else:
                            print("Invalid castling, pieces in way.")
                            return False, board
                else:
                    print("Invalid King move.")
                    return False, board
        else:
            print("Invalid king move.")
            return False, board
        
        if board[1] == 1:
            newBoard[66] = 0
        else:
            newBoard[69] = 0
        
    newBoard[move[0] * 8 + move[1] + 1] = 0
    newBoard[move[2] * 8 + move[3] + 1] = pieceToMove * newBoard[0]
    newBoard[0] = -newBoard[0]

    # Rule 3: Doesn't lead to check / stays in check
    if isInCheck(newBoard):
        print("Cannot end move in check")
        return False, board

    return True, newBoard

def isCheckmate(board):
    # Runs a check to see if a move is checkmate
    # Runs through all valid moves to see if still in check
    valid = getAllValidMoves(board)
    if len(valid) == 0:
        return True
    return False

def isInCheck(board):
    # Checks board state to see if either side is in check
    king = 0

    # find square king is on
    if board[0] == 1:
        for i in range(1, 65):
            if board[i] == 6:
                king = i
                break
    else:
        for i in range(1, 65):
            if board[i] == -6:
                king = i
                break
    
    enemy = -board[0]

    x, y = king // 8, (king - 1) % 8
    # u
    currY = y - 1
    while currY >= 0:
        pos = currY * 8 + x + 1
        if board[pos] == 0:
            currY -= 1
            continue
        elif board[pos] == 4 * enemy or 5 * enemy:
            return True
        else:
            break
    # d
    currY = y + 1
    while currY <= 7:
        pos = currY * 8 + x + 1
        if board[pos] == 0:
            currY += 1
            continue
        elif board[pos] == 4 * enemy or 5 * enemy:
            return True
        else:
            break
    # l
    currX = x + 1
    while currX >= 0:
        pos = y * 8 + currX + 1
        if board[pos] == 0:
            currX -= 1
            continue
        elif board[pos] == 4 * enemy or 5 * enemy:
            return True
        else:
            break
    # r
    currX = x + 1
    while currX >= 0:
        pos = y * 8 + currX + 1
        if board[pos] == 0:
            currX -= 1
            continue
        elif board[pos] == 4 * enemy or 5 * enemy:
            return True
        else:
            break
    # ul
    currY, currX = y - 1, x - 1
    while currX >= 0 and currY >= 0:
        pos = currY * 8 + currX + 1
        if board[pos] == 0:
            currX -= 1
            currY -= 1
            continue
        elif board[pos] == 2 * enemy or 5 * enemy:
            return True
        else:
            break
    # ur
    currY, currX = y - 1, x + 1
    while currX <= 7 and currY >= 0:
        pos = currY * 8 + currX + 1
        if board[pos] == 0:
            currX += 1
            currY -= 1
            continue
        elif board[pos] == 2 * enemy or 5 * enemy:
            return True
        else:
            break
    # dl
    currY, currX = y + 1, x - 1
    while currX >= 0 and currY <= 7:
        pos = currY * 8 + currX + 1
        if board[pos] == 0:
            currX -= 1
            currY += 1
            continue
        elif board[pos] == 2 * enemy or 5 * enemy:
            return True
        else:
            break
    # dr
    currY, currX = y + 1, x + 1
    while currX <= 7 and currY <= 7:
        pos = currY * 8 + currX + 1
        if board[pos] == 0:
            currX += 1
            currY += 1
            continue
        elif board[pos] == 2 * enemy or 5 * enemy:
            return True
        else:
            break
    
    # pawns
    if board[0] == 1:
        ...
    else:
        ...

    # knights

    # kings

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
    board = STANDARD_BOARD
    printBoard(board)

    while(1):
        move = input("Enter move: ").split(" ")
        if move == ["resign"]:
            if board[0] == 1:
                print("White has resigned. Black wins.")
            elif board[0] == -1:
                print("Black has resigned. White wins.")
            break
        for elem in range(len(move)):
            move[elem] = int(move[elem]) - 1

        validation = checkValid(board, move)
        if len(validation[1]) == 1:
            if (validation[1][0] == 1):
                print("")
            break
        if validation[0]:
            board = validation[1]
            printBoard(board)
            print(board[65:])
        else:
            print("Invalid move, try again")

if __name__ == '__main__':
    main()