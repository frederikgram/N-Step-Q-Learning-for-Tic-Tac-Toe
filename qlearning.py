""" Implementation of Q-Learning to play Tic-Tac-Toe. """

from tictactoe import get_winner, get_actions, get_next_state, pretty_format_board
from minimax import minimax, random

from typing import Dict, Tuple, List, Any, Union
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Default values for the game, the agent always goes first.
ASSOCIATE = 1
OPPONENT = 2

def take_step(qtable: Dict[Tuple[int, Tuple[int, ...]], float], board: List[int], player: int, size: int,  epsilon: int, opponent: str) -> Tuple[List[int], int]:
    """ Take a step in the game using either the Q-Learning algorithm or minimax based on the given player """

    # Get all possible actions
    actions = get_actions(board)

    # The agent is taking a step
    if player == ASSOCIATE:

        # Exploit the Q-Table
        if np.random.uniform(0, 1) > epsilon:
            action = max(actions, key=lambda a: qtable[(a, tuple(board))])
        
        # Explore outside of the Q-Table
        else:
            action = np.random.choice(actions)

        board = get_next_state(board, action, ASSOCIATE)
        return board, action

    # Otherwise, the opponent is taking a step
    if opponent == "minimax":
        action, _ = minimax(board, OPPONENT, OPPONENT, size) 
    else:
        action, _ = random(board, OPPONENT, size)

    # Sometimes the opponent will not have any actions available, in which case the game is a draw
    if action != None:
        board = get_next_state(board, action, OPPONENT)

    return board, action

def train(episodes: int, epsilon: float, gamma: float, alpha: float, size: int, qtable: Dict[Tuple[int, Tuple[int, ...]], float], n: int, debug: bool, lerp: bool, opponent: str)  -> Tuple[Dict[Tuple[int, Tuple[int, ...]], float], Dict[str, Any]]:
    """ Train the Q-Learning algorithm.
    
    Args:
        episodes: The number of episodes to train for.
        epsilon: The epsilon value for the epsilon-greedy algorithm.
        gamma: The discount factor.
        alpha: The learning rate.
        size: The size of the board. e.g 3 for a 3x3 board.
        qtable: The Q-Table to train.
        n: The number of episodes to average over when calculating the average reward.
        debug: Whether or not to print the average reward every n episodes.
        lerp: Whether or not to linearly interpolate the epsilon value.
        opponent: The opponent to use. Either "minimax" or "random".

    Returns:
        The trained Q-Table and a dictionary containing the average reward and epsilon values. (qtable, stats)
    """

    metrics = {
        "episodes": [],
        "epsilon": epsilon,
        "gamma": gamma,
        "alpha": alpha,
        "size": size,
        "n-steps": n,
        "lerp": lerp,
        "total_games": 0,
        "total_draws": 0,
        "total_losses": 0,
        "associate": ASSOCIATE,
    }


    # Each episode is a single tic-tac-toe game
    for episode in range(episodes):

        lerped_epsilon = np.round( epsilon * (1 - (episode / (episodes ))), decimals = 3) if lerp else epsilon 

        board = [0] * (size * size)

        metrics["episodes"].append({
            "episode": episode,
            "epsilon": lerped_epsilon,
            "cumulative_reward": metrics["episodes"][-1]["cumulative_reward"] if len(metrics["episodes"]) > 0 else 0
        })

        # Continue to take n-steps at the time, until the game is over
        while (result := get_winner(board, size)) == 0:

            history: List[Tuple[int, Tuple[int, ...]]] = list()
            last_board: Union[List[int], None] = None

            # Take n-steps
            for _ in range(n):  

                # Agent's move
                last_board, (board, action) = board, take_step(qtable, board, ASSOCIATE, size, lerped_epsilon, opponent)
                history.append((action, tuple(last_board)))

                # Opponent's move
                last_board, (board, action) = board, take_step(qtable, board, OPPONENT, size, 1, opponent)
                history.append((action, tuple(last_board)))

                # Break if the game finished before n steps were taken
                if (result := get_winner(board, size)) != 0:
                    break
    
            # Victory
            if result == ASSOCIATE:
                reward = 1
                
            # Loss
            elif result == OPPONENT:
                reward = -1
                
            # Draw
            elif result == 3:
                reward = 0.5
                
            # Game is still in progress
            else:
                reward = 0

            metrics["episodes"][-1]["cumulative_reward"] += reward
            best_value = -np.inf

            # Iterate through the move history in reverse order skipping the opponent's moves
            for enum, (action, state) in enumerate(history[::2][::-1]):
                
                # Get the current Q-value
                current_q = qtable[(action, tuple(state))]

                # Overwrite the Q-value for the latest move in the history
                if enum == 0:
                    qtable[(action, tuple(state))] = reward

                # Calculate the new Q-value using the Bellman equation with the discount factor
                else:
                    qtable[(action, tuple(state))] = current_q + alpha * ((gamma ** n) * best_value - current_q)

                # Track the best value
                best_value = max(best_value, qtable[(action, tuple(state))])


        metrics["total_games"] += 1

        if result == OPPONENT:
            metrics["total_losses"] += 1
        else:
            metrics["total_draws"] += 1

        metrics["episodes"][-1].update({
            "total_games": metrics["total_games"],
            "total_draws":  metrics["total_draws"],
            "total_losses": metrics["total_losses"],
            "loss_rate": np.round(metrics["total_losses"] / metrics["total_games"], decimals = 3),
            "draw_rate": np.round(metrics["total_draws"] / metrics["total_games"], decimals  = 3),
        })

    return qtable, metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Q-Learning agent to play Tic-Tac-Toe.")
    
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes to train for.")

    parser.add_argument("--epsilons", type=float, required=True, nargs="+", help="Epsilon value(s) to train for.")
    parser.add_argument("--nsteps", type=int, required=True, nargs="+", help="Number of steps to look ahead increasing from 1 to n.")

    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value(s) to train for.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha value(s) to train for.")

    parser.add_argument("--repetitions", type=int, default=10, help="Number of times to repeat each combination of hyperparameters.")
    parser.add_argument("--opponent", type=str, default="minimax", help="The opponent to play against, either 'random' or 'minimax'.")
    parser.add_argument("--size", type=int, default=3, help="The size of the board, default is 3x3.")

    parser.add_argument("--debug", action="store_true", default=False, help="Print the board after each step.")
    parser.add_argument("--nolerp", action="store_false", default=False, help="Don't Linearly interpolate epsilon over the number of episodes.")

    parser.add_argument("--output", type=str, default=".", help="Path ")
    parser.add_argument("--save-models", default=False, action="store_true", help="Save the models after training.")
    parser.add_argument("--save-metrics", default=False, action="store_true", help="Save the metrics after training.")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot the metrics after training.")

    args = parser.parse_args()

    # Sanity check
    if args.opponent not in ["random", "minimax"]:
        raise ValueError(f"Invalid opponent: {args.opponent}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Setup Matplotlib figures
    if args.plot:
        fig, axs = plt.subplots(
            len(args.nsteps),
            len(args.epsilons),
            constrained_layout=True,
            figsize=(10, 4), 
            num=f"Tic-Tac-Toe Q-Learning ({args.size}x{args.size}) | Episodes: {args.episodes} | Gamma: {args.gamma} | Alpha: {args.alpha} | N-Steps: {args.nsteps} | Epsilon(s): {args.epsilons} | Opponent: {args.opponent}"
        )

        gridspec = axs[0, 0].get_subplotspec().get_gridspec()

        # Clear gridspec for outer axes
        for i in range(0, len(args.epsilons)):
            for ax in axs[0:, i]:
                ax.remove()

    # Enumerate all combinations of hyperparameters
    for i, step in enumerate(args.nsteps):
        
            for j, epsilon in enumerate(args.epsilons):

                if args.plot:
                    training_history = list()

                for k in range(args.repetitions):

                    qtable, metrics = train(
                        qtable=defaultdict(float),
                        episodes=args.episodes,
                        epsilon=epsilon,
                        alpha=args.alpha,
                        gamma=args.gamma,
                        n=step,
                        size=args.size,
                        debug=args.debug,
                        lerp=args.nolerp == False,
                        opponent=args.opponent
                    )

                    metrics["episodes"].insert(0, {
                        "episode": -1,
                        "cumulative_reward": 0,
                        "draw_rate": 0,
                        "loss_rate": 1 
                    })

                    if args.save_models:
                        with open(os.path.join(args.output, f"qtable_{i}_{j}_{k}.pkl"), "wb") as f:
                            pickle.dump(qtable, f)
                        
                    if args.save_metrics:
                        with open(os.path.join(args.output, f"metrics_{i}_{j}_{k}.pkl"), "wb") as f:
                            pickle.dump(metrics, f)
                    
                    if args.plot:
                        training_history.append(metrics)

                    if args.debug:
                        print(f"Finished training with epsilon={epsilon}, n={step}, repetition={k}.")

                if args.plot:

                    # Create average metrics from the repetition history
                    average_metrics = {"episodes": []}
                    for episode in range(args.episodes + 1):
                        average_metrics["episodes"].append({
                            "cumulative_reward": np.mean(
                                [metrics["episodes"][episode]["cumulative_reward"] for metrics in training_history]
                            ),
                            "draw_rate": np.mean(
                                [metrics["episodes"][episode]["draw_rate"] for metrics in training_history]
                            ),
                            "loss_rate": np.mean(
                                [metrics["episodes"][episode]["loss_rate"] for metrics in training_history]
                            ),
                            "episode": episode,
                    })

                    # Create subfigure to contain plots for cumulative reward and loss-rate
                    subfig = fig.add_subfigure(gridspec[i, j])
                    subfig.set_facecolor('0.9')
                    subfig.suptitle(f"n = {step}, ε = {epsilon}")
                    axs = subfig.subplots(3, 1, sharex=True)

                    # Plot the loss rate
                    axs[0].plot([episode["episode"] for episode in average_metrics["episodes"]], [episode["loss_rate"] for episode in average_metrics["episodes"]], label="Loss rate", color="red")
                    axs[0].legend()


                    axs[0].set_yticklabels(["{:.0%}".format(x) for x in axs[0].get_yticks()])
                    
                    # Plot the cumulative reward
                    axs[1].plot([episode["episode"] for episode in average_metrics["episodes"]], [episode["cumulative_reward"] for episode in average_metrics["episodes"]], label="Cumulative Reward", color="orange")
                    axs[1].legend()

                    # Plot the learning curve
                    axs[2].plot([episode["episode"] for episode in average_metrics["episodes"]], [max(0, episode["cumulative_reward"] * (-1)) / (enum + 1)  for enum, episode in enumerate(metrics["episodes"])], label="Learning Curve", color="green")
                    axs[2].legend()
                    axs[2].set_yticks([0, 1])
                    axs[2].set_xlabel("Episode(s)")


                    # Calculate Standard Deviation and mean of cumulative reward for each episode
                    cumulative_reward_std = [np.std([metrics["episodes"][episode]["cumulative_reward"] for metrics in training_history]) for episode in range(args.episodes + 1)]
                    cumulative_reward_mean = [np.mean([metrics["episodes"][episode]["cumulative_reward"] for metrics in training_history]) for episode in range(args.episodes + 1)]

                    # Plot the standard deviation of the cumulative reward as a shaded area
                    axs[1].fill_between([episode["episode"] for episode in average_metrics["episodes"]], [cumulative_reward_mean[episode] - cumulative_reward_std[episode] for episode in range(args.episodes + 1)], [cumulative_reward_mean[episode] + cumulative_reward_std[episode] for episode in range(args.episodes + 1)], alpha=0.2, color="orange")


    # Print Hyperparameters
    logging.info(f"Hyperparameters: {args}")

    if args.save_models or args.save_metrics:
        logging.info(f"Outputs saved to path '{args.output}' are formatted as 'qtable_{i}_{j}_{k}.pkl' and 'metrics_{i}_{j}_{k}.pkl' where i is the index of the nstep, j is the index of the epsilon and k is the repetition.")

    if args.plot:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(25, 12) # set figure's size manually to your full screen (32x18)
        plt.show()
