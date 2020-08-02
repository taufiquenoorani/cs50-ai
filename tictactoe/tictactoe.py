"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # If board is empty, X gets the first move
    if board == initial_state():
        return X

    # Count number of X's and O's on board
    numX = 0
    numO = 0
    for row in board:
        numX += row.count(X)
        numO += row.count(O)

    # Next player is the one with the fewest moves currently
    if numX > numO:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    # (row, cell) is legal move if its position on the board is currently empty
    legal_actions = set()
    for row in range(3):
        for cell in range(3):
            if board[row][cell] == EMPTY:
                legal_actions.add((row, cell))

    return legal_actions
    

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # Raise exception if index out of range or cell is already taken
    if action[0] not in range(0, 3) or action[1] not in range(0, 3) or board[action[0]][action[1]] is not EMPTY:
        raise Exception("Invalid move")

    # Let player whose turn it is to make their move on the board
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    # Evaluate board for each player
    for mark in [X, O]:
        
        # Check for 3 moves in a row horizontally
        for row in range(0, 3):
            if all(board[row][col]==mark for col in range(0, 3)):
                return mark
        
        # Check for 3 moves in a row vertically
        for col in range(0, 3):
            if all(board[row][col]==mark for row in range(0, 3)):
                return mark

        # Check for 3 moves in a row diagonally
        diagonals = [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
        for diagonal in diagonals:
            if all(board[row][col]==mark for (row, col) in diagonal):
                return mark

    # Game is a tie or still in progress
    return None

        
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    
    # Terminal if there is a winner
    if winner(board) is not None:
        return True
    
    # Terminal if all cells have been filled
    all_moves = [cell for row in board for cell in row]
    if not any(move==EMPTY for move in all_moves):
        return True

    # Otherwise game in progress
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    if terminal(board):
        return None

    if player(board) == X:
        best_v = -math.inf
        for move in actions(board):
            max_v = min_value(result(board, move))
            if max_v > best_v:
                best_v = max_v
                best_move = move
    
    elif player(board) == O:
        best_v = math.inf
        for move in actions(board):
            min_v = max_value(result(board, move)) 
            if min_v < best_v:
                best_v = min_v
                best_move = move
    return best_move 


def min_value(board):
    """
    Returns the minimum utility of the current board.
    """

    if terminal(board):
        return utility(board)
    
    v = math.inf
    for move in actions(board):
        v = min(v, max_value(result(board, move)))
    return v


def max_value(board):
    """
    Returns the maximum utility of the current board.
    """

    if terminal(board):
        return utility(board)

    v = -math.inf
    for move in actions(board):
        v = max(v, min_value(result(board, move)))
    return v
