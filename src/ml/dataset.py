"""
Feature engineering and dataset generation for PopOut.

Board → feature vector
----------------------
The board is a 6×7 grid where each cell is 0 (empty), 1 (Player 1), or
2 (Player 2).  We flatten it row-by-row into a 42-element integer vector
and append the current player (1 or 2) as a 43rd feature.

    features = [r0c0, r0c1, …, r5c6, current_player]   (43 values)

Move → integer label
--------------------
    drop col  →  col          (0 – 6)
    pop  col  →  col + 7      (7 – 13)
    draw      →  14
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.popout_board import PopOutBoard


# ── Feature / label encoding ──────────────────────────────────────────────────

def board_to_features(board: "PopOutBoard") -> list[int]:
    """Return the 43-element integer feature vector for *board*."""
    return board.board.flatten().tolist() + [board.current_player]


def move_to_label(move: tuple) -> int:
    """Encode a move tuple as an integer label (0–14)."""
    move_type, col = move
    if move_type == 'drop':
        return col
    if move_type == 'pop':
        return col + 7
    return 14  # draw


def label_to_move(label: int) -> tuple:
    """Decode an integer label back to a move tuple."""
    if label <= 6:
        return ('drop', label)
    if label <= 13:
        return ('pop', label - 7)
    return ('draw', None)


FEATURE_NAMES: list[str] = (
    [f'r{r}c{c}' for r in range(6) for c in range(7)]
    + ['current_player']
)

N_LABELS: int = 15  # 0–14


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    n_games: int = 300,
    mcts_iterations: int = 300,
    rollout_strategy: str = 'random',
    verbose: bool = True,
) -> tuple[list[list[int]], list[int]]:
    """Generate a labelled dataset by playing MCTS self-play games.

    At each turn the MCTS agent selects a move — that move is the label.
    The resulting dataset can be used to train a decision tree that
    approximates the MCTS policy.

    Parameters
    ----------
    n_games:
        Number of complete games to play.
    mcts_iterations:
        MCTS iterations per move (higher = stronger labels, slower).
    rollout_strategy:
        ``'random'`` or ``'heuristic'``.
    verbose:
        Print progress every 50 games.

    Returns
    -------
    (X, y)
        X — list of 43-element feature vectors
        y — list of integer labels (0–14)
    """
    # Import here to avoid circular imports at module load time
    from src.game.popout_board import PopOutBoard
    from src.ai.agents import make_mcts_agent

    agent = make_mcts_agent(
        iterations=mcts_iterations,
        rollout_strategy=rollout_strategy,
    )

    X: list[list[int]] = []
    y: list[int] = []

    t0 = time.time()

    for game_i in range(n_games):
        if verbose and (game_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f'  Game {game_i + 1:4d}/{n_games}  |  '
                f'{len(X):6d} samples  |  {elapsed:.0f}s elapsed'
            )

        board = PopOutBoard()
        while not board.is_game_over:
            features = board_to_features(board)
            move     = agent(board)
            X.append(features)
            y.append(move_to_label(move))
            board.apply_move(move)

    if verbose:
        print(f'  Done. {len(X)} samples from {n_games} games '
              f'in {time.time() - t0:.1f}s')

    return X, y


# ── Persistent storage ────────────────────────────────────────────────────────

# Repo-level data/ directory (works regardless of working directory)
DATA_DIR: Path = Path(__file__).resolve().parents[2] / 'data'

POPOUT_DATASET_PATH: Path = DATA_DIR / 'popout_mcts.csv'
IRIS_DATASET_PATH:   Path = DATA_DIR / 'iris.csv'


def save_dataset(
    X: list[list[int]],
    y: list[int],
    path: str | Path | None = None,
) -> None:
    """Save a (X, y) dataset to a CSV file.

    Parameters
    ----------
    X : list of feature vectors
    y : list of labels
    path : output CSV path (default: ``data/popout_mcts.csv``)
    """
    path = Path(path) if path else POPOUT_DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    header = FEATURE_NAMES + ['label']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for features, label in zip(X, y):
            writer.writerow(features + [label])

    print(f'Saved {len(X)} samples → {path}')


def load_dataset(
    path: str | Path | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Load a (X, y) dataset from a CSV file.

    Parameters
    ----------
    path : input CSV path (default: ``data/popout_mcts.csv``)

    Returns
    -------
    (X, y)
        X — list of 43-element integer feature vectors
        y — list of integer labels (0–14)
    """
    path = Path(path) if path else POPOUT_DATASET_PATH

    X: list[list[int]] = []
    y: list[int] = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            values = list(map(int, row))
            X.append(values[:-1])
            y.append(values[-1])

    print(f'Loaded {len(X)} samples ← {path}')
    return X, y
