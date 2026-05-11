"""
generate_extra_data.py
======================
Generates additional training games and appends them to data/popout_mcts.csv.

Strategy distribution (500 total new games):
  - heuristic : 300 games  (strongest labels; most valuable for the tree)
  - greedy    : 150 games  (fast, positionally aware)
  - random    :  50 games  (variety / exploration noise)

Progress is saved to a temporary CSV every CHECKPOINT_EVERY games so you can
monitor progress and resume from a partial run if needed.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR          = Path('data')
MAIN_CSV          = DATA_DIR / 'popout_mcts.csv'
TEMP_CSV          = DATA_DIR / 'popout_mcts_extra_temp.csv'

MCTS_ITERATIONS   = 300          # strength of the "teacher" MCTS agent
CHECKPOINT_EVERY  = 25           # save a temp CSV snapshot every N games

# (strategy, n_games) — order matters: strongest first for best progress feedback
BATCHES: list[tuple[str, int]] = [
    ('heuristic', 300),
    ('greedy',    150),
    ('random',     50),
]

# ── Feature names (must match dataset.py) ─────────────────────────────────────

FEATURE_NAMES = (
    [f'r{r}c{c}' for r in range(6) for c in range(7)]
    + ['current_player']
)
HEADER = FEATURE_NAMES + ['label']

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[list[int]], append: bool = False) -> None:
    mode = 'a' if append else 'w'
    with open(path, mode, newline='') as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)


def _count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, 'r') as f:
        return sum(1 for _ in f) - 1   # minus header


# ── Core generation ───────────────────────────────────────────────────────────

def generate_batch(
    strategy: str,
    n_games: int,
    mcts_iterations: int,
) -> list[list[int]]:
    """Play *n_games* of MCTS self-play and return flat CSV rows (features + label)."""
    from src.game.popout_board import PopOutBoard
    from src.ai.agents import make_mcts_agent
    from src.ml.dataset import board_to_features, move_to_label

    agent = make_mcts_agent(iterations=mcts_iterations, rollout_strategy=strategy)
    rows: list[list[int]] = []

    t0 = time.time()

    for game_i in range(n_games):
        board = PopOutBoard()
        while not board.is_game_over:
            features  = board_to_features(board)
            move      = agent(board)
            label     = move_to_label(move)
            rows.append(features + [label])
            board.apply_move(move)

        # ── Checkpoint ──────────────────────────────────────────────────────
        if (game_i + 1) % CHECKPOINT_EVERY == 0 or (game_i + 1) == n_games:
            elapsed = time.time() - t0
            samples_so_far = len(rows)
            print(
                f'  [{strategy:>10s}]  game {game_i+1:>4d}/{n_games}'
                f'  |  {samples_so_far:>6d} samples'
                f'  |  {elapsed:>6.0f}s elapsed'
            )
            # Save / overwrite the temp checkpoint
            _write_csv(TEMP_CSV, rows, append=False)

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    existing = _count_existing_rows(MAIN_CSV)
    print(f"Current dataset: {existing:,} samples in {MAIN_CSV}")
    print()

    all_new_rows: list[list[int]] = []
    total_games = sum(n for _, n in BATCHES)
    total_done  = 0

    for strategy, n_games in BATCHES:
        print(f"{'='*60}")
        print(f"Batch: {strategy.upper()}  ({n_games} games, MCTS n={MCTS_ITERATIONS})")
        print(f"{'='*60}")

        batch_rows = generate_batch(strategy, n_games, MCTS_ITERATIONS)
        all_new_rows.extend(batch_rows)
        total_done += n_games

        print(
            f"  → Batch complete: {len(batch_rows):,} new samples"
            f"  (total new so far: {len(all_new_rows):,})"
        )
        print()

    # ── Append all new rows to the main CSV ─────────────────────────────────
    # If main CSV doesn't exist yet, write with header; otherwise append without.
    append_mode = MAIN_CSV.exists()
    _write_csv(MAIN_CSV, all_new_rows, append=append_mode)

    final_count = _count_existing_rows(MAIN_CSV)
    print(f"{'='*60}")
    print(f"DONE!")
    print(f"  New samples added : {len(all_new_rows):,}")
    print(f"  Total dataset size: {final_count:,} samples")
    print(f"  Saved to          : {MAIN_CSV}")

    # Clean up temp file
    if TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print(f"  Temp file removed : {TEMP_CSV.name}")


if __name__ == '__main__':
    main()
