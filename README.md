# Tic-Tac-Toe N-Step Q-Learning

    $ python3 qlearning.py --episodes 80 --epsilons 0.1 0.2 0.3 0.4 0.5 --nsteps 1 2 3 --opponent minimax --debug --plot

![Metrics](./tictactoe-qlearning-metrics.png "Example output")

## Usage

### Requirements
    $ pip install -r requirements.txt

### Select Hyperparameters
- gamma: Discount factor
- epsilon: Exploration factor
- alpha: Learning rate
- nsteps: Number of steps

note: both epsilon and nsteps can be formatted as lists eg. `--epsilons 0.1 0.2 0.3 --nsteps 1 2 3`

### Arguments

```
usage: qlearning.py [-h] --episodes EPISODES --epsilons EPSILONS [EPSILONS ...] --nsteps NSTEPS [NSTEPS ...]
                    [--gamma GAMMA] [--alpha ALPHA] [--repetitions REPETITIONS] [--opponent OPPONENT] [--size SIZE]
                    [--debug] [--nolerp] [--output OUTPUT] [--save-models] [--save-metrics] [--plot]

Train a Q-Learning agent to play Tic-Tac-Toe.

optional arguments:
  -h, --help            show this help message and exit
  --episodes EPISODES   Number of episodes to train for.
  --epsilons EPSILONS [EPSILONS ...]
                        Epsilon value(s) to train for.
  --nsteps NSTEPS [NSTEPS ...]
                        Number of steps to look ahead increasing from 1 to n.
  --gamma GAMMA         Gamma value(s) to train for.
  --alpha ALPHA         Alpha value(s) to train for.
  --repetitions REPETITIONS
                        Number of times to repeat each combination of hyperparameters.
  --opponent OPPONENT   The opponent to play against, either 'random' or 'minimax'.
  --size SIZE           The size of the board, default is 3x3.
  --debug               Print the board after each step.
  --nolerp              Don't Linearly interpolate epsilon over the number of episodes.
  --output OUTPUT       Path
  --save-models         Save the models after training.
  --save-metrics        Save the metrics after training.
  --plot                Plot the metrics after training.
```

### Run
    $ python3 qlearning.py --episodes 80 --epsilons 0.1 0.2 0.3 0.4 0.5 --nsteps 1 2 3 --debug --plot

### Conditional Outputs
If `--output` is not specified, the output will be saved in the `.` directory.


If `--save-models` is specified, the q-tables will be saved as pickle files to the output directory.


If `--save-metrics` is specified, the metrics will be saved as pickle file representing a dictionary to the output directory.

If `--plot` is specified, the metrics will be plotted and shown using matplotlib, from which you can save the plot manually.
