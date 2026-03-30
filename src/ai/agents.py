"""
Agent factories for PopOut.

An agent is any callable with the signature::

    agent(board: PopOutBoard) -> tuple

that returns a legal move for the current player.

Available agents
----------------
random_agent        — picks a random legal move (baseline)
make_mcts_agent()   — creates a MCTS agent with configurable parameters
make_dt_agent()     — wraps a trained DecisionTreeID3 as a game agent
"""

from __future__ import annotations

import math
import random
from typing import Callable, TYPE_CHECKING

from src.game.popout_board import PopOutBoard
from src.ai.mcts import mcts_search, RolloutStrategy

if TYPE_CHECKING:
    from src.ml.decision_tree import DecisionTreeID3

Agent = Callable[[PopOutBoard], tuple]


# ── Random baseline ───────────────────────────────────────────────────────────

def random_agent(board: PopOutBoard) -> tuple:
    """Pick a random legal move, avoiding draws unless forced."""
    moves = board.get_legal_moves()
    non_draw = [m for m in moves if m[0] != 'draw']
    return random.choice(non_draw if non_draw else moves)


# ── MCTS agent factory ────────────────────────────────────────────────────────

def make_mcts_agent(
    iterations: int = 1000,
    c: float = math.sqrt(2),
    expand_k: int = 1,
    rollout_strategy: RolloutStrategy = 'random',
    max_time: float | None = None,
    early_stop_threshold: float = 1.0,
) -> Agent:
    """Return a MCTS agent callable.

    Parameters
    ----------
    iterations:
        Maximum number of MCTS iterations per move decision.
    c:
        UCT exploration constant (default √2).
    expand_k:
        Number of children to expand per iteration (1 = standard MCTS).
    rollout_strategy:
        ``'random'`` or ``'heuristic'``.
    max_time:
        Optional per-move wall-clock time limit in seconds.
    early_stop_threshold:
        Win-rate threshold (0–1) for early termination. ``1.0`` disables it.

    Returns
    -------
    agent : callable
        A function ``agent(board) -> move``.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    if expand_k < 1:
        raise ValueError(f"expand_k must be >= 1, got {expand_k}")
    if rollout_strategy not in ('random', 'heuristic', 'greedy'):
        raise ValueError(f"Unknown rollout strategy: {rollout_strategy!r}")

    def agent(board: PopOutBoard) -> tuple:
        move, _ = mcts_search(
            board,
            iterations=iterations,
            c=c,
            expand_k=expand_k,
            rollout_strategy=rollout_strategy,
            max_time=max_time,
            early_stop_threshold=early_stop_threshold,
        )
        return move

    agent.__name__ = (
        f'MCTS(n={iterations}, c={c:.2f}, k={expand_k}, {rollout_strategy})'
    )
    return agent


# ── Decision Tree agent factory ───────────────────────────────────────────────

def make_dt_agent(
    dt_model: "DecisionTreeID3",
    fallback: Agent | None = None,
) -> Agent:
    """Wrap a trained :class:`~src.ml.decision_tree.DecisionTreeID3` as an agent.

    The DT predicts a move from the board features. If the predicted move is
    not legal (possible for board states not seen during training), the agent
    falls back to *fallback* (default: :func:`random_agent`).

    Parameters
    ----------
    dt_model:
        A fitted DecisionTreeID3 instance.
    fallback:
        Agent used when the DT predicts an illegal move.
        Defaults to :func:`random_agent`.

    Returns
    -------
    agent : callable
    """
    from src.ml.dataset import board_to_features, label_to_move

    _fallback = fallback if fallback is not None else random_agent

    def agent(board: PopOutBoard) -> tuple:
        features   = board_to_features(board)
        pred_label = dt_model.predict_one(features)
        pred_move  = label_to_move(pred_label)

        if pred_move in board.get_legal_moves():
            return pred_move
        return _fallback(board)

    agent.__name__ = 'DTAgent'
    return agent
