""" Implementation of a Tic-Tac-Toe game """

from typing import List


def get_actions(board: List[int]) -> List[int]:
    """ Collect a list of all possible actions for the given state.
    
    Args:
        board: The state to evaluate.
    
    Returns:
        A list of all possible actions for the given state.
    """
    
    return [i for i, x in enumerate(board) if x == 0]


def get_next_state(board: List[int], action: int, current_player: int) -> List[int]:
    """ Applies the given action to the given state and returns the resulting state.
    
    Args:
        board: The state to apply the action to.
        action: The action to apply.
        current_player: The player to move next.

    Returns:
        The resulting state.
    """
    
    board = board.copy()
    board[action] = current_player
    return board


def get_winner(board: List[int], size: int) -> int:
    """ Get the current status of the game. 
    
    Args:
        board: The state to evaluate.
        size: The size of the given board. e.g 3 for a 3x3 board.

    Returns:
        The current status of the game. 
         0 if the game is not over, 1 if player 1 wins,
         2 if player 2 wins, and 3 if the game is a draw.
    """

    for player in [1, 2]:

        # Check rows
        for row in range(size):
            if all([board[row * size + col] == player for col in range(size)]):
                return player

        # Check columns
        for col in range(size):
            if all([board[row * size + col] == player for row in range(size)]):
                return player

        # Check diagonal
        if all([board[i * size + i] == player for i in range(size)]):
            return player

        # Check other diagonal
        if all([board[i * size + 2 - i] == player for i in range(size)]):
            return player

    # Check if the board is full, if so, its a draw
    if 0 not in board:
        return 3

    # Game is not over yet
    return 0

def pretty_format_board(board: List[int], size: int) -> str:
    """ Format a given board into a pretty string multi-line string.
    
    Args:
        board: The state to evaluate.
        size: The size of the given board. e.g 3 for a 3x3 board.
    
    Returns:
        A string representation of the given board.
    """

    char_map = {0: '-', 1: 'X', 2: 'O'}
    return '\n'.join(' '.join(char_map[board[i]] for i in range(row * size, (row + 1) * size)) for row in range(size))

