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

def getAllValid(board):
    # Given a board, returns all valid moves which can be performed by the current player
    moves = []
    pieces = []
    turn = np.sign(board[0])

    # Find all pieces that can move
    for i in range(1, 65):
        if np.sign(board[i]) == turn:
            pieces.append(i)
    
    return moves

def checkValid(board, move):
    # Checks if move is valid given board

    # Returns a tuple of whether the move is valid
    # and the resulant board (returns original if false)

    pieceToMove = board[(move[0] * 8) + move[1] + 1]
    # Rule 1: Piece is current turn's side
    if np.sign(pieceToMove) != np.sign(board[0]):
        return False, board

    # Rule 2: Move follows piece's move
    # Rule 2.5: Piece doesn't run into another piece / friendly capture
    # TODO: Finish
    # En Passant
    # Castling
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
            # TODO: Add en passant
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
        else:
            print("Invalid pawn move.")
            return False, board

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
        elif np.abs(xDist) == 1 and np.abs(yDist) == 1:
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
        elif (xDist == 0) == (yDist == 0):
            # horizontal move
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
        # TODO: Castling
        xDist, yDist = move[3] - move[1], move[2] - move[0]
        # one square check
        if np.abs(xDist) != 1 and np.abs(yDist) != 1:
            print("Invalid king move.")
            return False, board
        
        # doesn't capture friendly check
        landingSquare = board[move[2] * 8 + move[3] + 1]
        if landingSquare != 0 and np.sign(landingSquare) == board[0]:
            print("Piece can't capture friendly piece.")
            return False, board
        

    # Rule 3: Doesn't lead to check / stays in check
    # TODO: Finish

    newBoard = board
    newBoard[move[0] * 8 + move[1] + 1] = 0
    newBoard[move[2] * 8 + move[3] + 1] = pieceToMove * newBoard[0]
    newBoard[0] = -newBoard[0]

    return True, newBoard

def isCheckmate(board):
    # Runs a check to see if a move is checkmate
    # Runs through all valid moves to see if still in check
    # TODO: Implement function
    if isInCheck(board)[0]:
        ...
        return True
    elif isInCheck(board)[1]:
        ...
        return True
    return False

def isInCheck(board):
    # Checks board state to see if either side is in check
    # Returns a tuple of values based on that (white in check, black in check)
    # TODO: Implement
    whiteKing = 0
    blackKing = 0
    
    # find square king is on
    for i in range(1, 65):
        if board[i] == 6:
            whiteKing = i
        elif board[i] == -6:
            blackKing = i
    
    return False, False

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
        if validation[0]:
            board = validation[1]
            printBoard(board)
        else:
            print("Invalid move, try again")

if __name__ == '__main__':
    main()