""" Implementation of the minimax algorithm (and a random agent) for Tic-Tac-Toe. """

from tictactoe import get_winner, get_actions, get_next_state
from typing import List, Tuple, Union
import numpy as np


def evaluate_state(board: List[int], associate: int, size: int) -> int:
    """ Returns the value of the given state.

    Args:
        board: The state to evaluate.
        associate: The player associated with the agent.
        size: The size of the given board. e.g 3 for a 3x3 board.

    Returns:
        The value of the given state. 1 if the agent wins, -1 if the agent loses, and 0 if the game is a draw or not yet finished.
    """

    if get_winner(board, size) == associate:
        return 1
    elif get_winner(board, size) == associate % 2 + 1:
        return -1
    else:
        return 0


def minimax(board: List[int], current_player: int, associate: int, size: int) -> Tuple[Union[int, None], int]:
    """ Implements the minimax algorithm. No alpha-beta pruning is used nor is there any depth limit.

    Args:
        board: The state to evaluate.
        current_player: The player to move next.
        associate: The player associated with the agent.
        size: The size of the given board. e.g 3 for a 3x3 board.

    Returns:
        A tuple of (action, value) where action is the best action and value is the value of that action.
    """


    if tuple(board) in minimax.cache:
        return minimax.cache[tuple(board)]

    # If the game is over, return the value of the state
    if get_winner(board, size) > 0:
        return None, evaluate_state(board, associate, size)

    # Get all possible actions the current player can make aka. all empty cells
    actions = get_actions(board)

    best_action, best_value = None, -np.inf if current_player == associate else np.inf

    # Loop over all possible actions
    for action in actions:

        next_state = get_next_state(board, action, current_player)

        # Get the best value attainable from the 'next_state' state
        _, value = minimax(next_state, current_player % 2 + 1, associate, size)

        # Maximize the value if the current player is the associate
        if current_player == associate:
            if value > best_value:
                best_action, best_value = action, value

        # Minimize the value if the current player is not the associate
        else:
            if value < best_value:
                best_action, best_value = action, value

    minimax.cache[tuple(board)] = best_action, best_value
    return best_action, best_value

# Initialize the cache, this speeds up the algorithm by a lot, no seriously, a lot.
minimax.cache = {}

def random(board: List[int], associate: int, size: int) -> int:
    """ Implements a random agent. Used for testing purposes.
    
    Args:
        board: The state to evaluate.
        associate: The player associated with the agent.
        size: The size of the given board. e.g 3 for a 3x3 board.
    
    Returns:
        A random action.
    """

    if get_winner(board, size) > 0:
        return None, evaluate_state(board, associate, size)

    actions = get_actions(board)
    best_action = np.random.choice(actions)

    return best_action, 0
