"""
Evaluation utilities: tournaments, metrics, train/test splits.

All implementations are from scratch — no scikit-learn.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.popout_board import PopOutBoard

Agent = Callable[["PopOutBoard"], tuple]


# ── Data splitting ────────────────────────────────────────────────────────────

def train_test_split(
    X: list,
    y: list,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[list, list, list, list]:
    """Split dataset into train and test sets.

    Parameters
    ----------
    X, y:
        Feature matrix and labels.
    test_ratio:
        Fraction of data to hold out for testing.
    seed:
        Random seed for reproducibility.
    stratify:
        If ``True``, maintain class proportions across splits.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = random.Random(seed)

    if stratify:
        class_idx: dict[Any, list[int]] = defaultdict(list)
        for i, label in enumerate(y):
            class_idx[label].append(i)

        train_indices, test_indices = [], []
        for indices in class_idx.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            split = int(len(shuffled) * (1 - test_ratio))
            train_indices.extend(shuffled[:split])
            test_indices.extend(shuffled[split:])
    else:
        indices = list(range(len(y)))
        rng.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        train_indices = indices[:split]
        test_indices  = indices[split:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test  = [X[i] for i in test_indices]
    y_test  = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test


# ── Classification metrics ────────────────────────────────────────────────────

def confusion_matrix(
    y_true: list,
    y_pred: list,
    classes: list | None = None,
) -> tuple[list[list[int]], list]:
    """Compute a confusion matrix.

    Returns
    -------
    (matrix, classes)
        matrix — 2-D list [true_class][predicted_class]
        classes — sorted list of class labels
    """
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred), key=str)

    idx = {c: i for i, c in enumerate(classes)}
    n   = len(classes)
    cm  = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    return cm, classes


def classification_report(
    y_true: list,
    y_pred: list,
    classes: list | None = None,
    class_names: list[str] | None = None,
) -> str:
    """Return a human-readable classification report (precision, recall, F1)."""
    cm, cls = confusion_matrix(y_true, y_pred, classes)
    n = len(cls)

    lines = []
    header = f"{'Class':>20s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}"
    lines.append(header)
    lines.append('-' * len(header))

    total_support = 0
    macro_p = macro_r = macro_f = 0.0

    for i, c in enumerate(cls):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        label = (class_names[i] if class_names and i < len(class_names)
                 else str(c))
        lines.append(
            f'{label:>20s}  {precision:>10.3f}  {recall:>8.3f}  '
            f'{f1:>8.3f}  {support:>8d}'
        )
        macro_p += precision
        macro_r += recall
        macro_f += f1
        total_support += support

    lines.append('-' * len(header))
    acc = sum(cm[i][i] for i in range(n)) / total_support if total_support else 0
    lines.append(f'{"Accuracy":>20s}  {"":>10s}  {"":>8s}  {acc:>8.3f}  {total_support:>8d}')
    lines.append(
        f'{"Macro avg":>20s}  {macro_p/n:>10.3f}  {macro_r/n:>8.3f}  '
        f'{macro_f/n:>8.3f}  {total_support:>8d}'
    )

    return '\n'.join(lines)


# ── Game tournament ───────────────────────────────────────────────────────────

def tournament(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int = 20,
    verbose: bool = False,
) -> dict:
    """Play *n_games* between two agents, alternating first-move advantage.

    Parameters
    ----------
    agent_a, agent_b:
        Callable agents ``f(board) -> move``.
    n_games:
        Total games played (half with A as P1, half with B as P1).
    verbose:
        Print a progress bar.

    Returns
    -------
    dict with keys: wins_a, wins_b, draws, win_rate_a
    """
    from src.game.popout_board import PopOutBoard

    wins_a = wins_b = draws = 0

    for i in range(n_games):
        if verbose:
            print(f'\r  Game {i+1}/{n_games}', end='', flush=True)

        flipped = (i % 2 == 1)
        p1, p2  = (agent_b, agent_a) if flipped else (agent_a, agent_b)

        board = PopOutBoard()
        while not board.is_game_over:
            agent = p1 if board.current_player == 1 else p2
            board.apply_move(agent(board))

        winner = board.winner
        if winner == 0:
            draws += 1
        elif (winner == 1 and not flipped) or (winner == 2 and flipped):
            wins_a += 1
        else:
            wins_b += 1

    if verbose:
        print()

    return {
        'wins_a':    wins_a,
        'wins_b':    wins_b,
        'draws':     draws,
        'win_rate_a': wins_a / n_games,
    }
