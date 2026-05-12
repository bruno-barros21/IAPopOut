"""
generate_extra_data.py
======================
Generates additional MCTS self-play games and appends them to
data/popout_mcts.csv (keeps the existing dataset as requested by the professor).

Adapted from Rita's approach:
  - Time-limited MCTS per move (3 s early game, 1.5 s late game)
  - Partial CSV save every CHECKPOINT_EVERY games
  - 1000 total games across three rollout strategies (heuristic-heavy)

Strategy split (1000 games):
  - heuristic : 600 games  (2-ply look-ahead; strongest labels)
  - greedy    : 300 games  (1-ply heuristic + positional scoring)
  - random    : 100 games  (diversity / exploration noise)

Estimated total runtime  (wall-clock):
  Each game ≈ 10 moves × 3 s  +  25 moves × 1.5 s  ≈  67 s
  1000 games × 67 s ≈  18–20 hours  (leave overnight / across a weekend)
  Checkpoint every 10 games so progress is never lost.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR          = Path('data')
MAIN_CSV          = DATA_DIR / 'popout_mcts.csv'
TEMP_CSV          = DATA_DIR / 'mcts_dataset_parcial.csv'   # matches Rita's naming

# Time budget per move (seconds) -- mirrors Rita's time_limit logic
TIME_EARLY_GAME   = 5.0    # moves 1–10: more time for critical opening decisions
TIME_LATE_GAME    = 2.5    # moves 11+:  faster, position is more deterministic
EARLY_GAME_MOVES  = 10     # threshold between "early" and "late" game

CHECKPOINT_EVERY  = 10     # save partial CSV every N games (matches Rita)

# Strategy batches -- heuristic-heavy per professor guidance
BATCHES: list[tuple[str, int]] = [
    ('heuristic', 300),
    ('greedy',    150),
    ('random',     50),
]

# ── Feature / label helpers (must match dataset.py) ──────────────────────────

FEATURE_NAMES = [f'r{r}c{c}' for r in range(6) for c in range(7)] + ['current_player']
HEADER        = FEATURE_NAMES + ['move']   # 'move' to mirror Rita's column name

def _board_to_features(board) -> list[int]:
    return board.board.flatten().tolist() + [board.current_player]

def _move_to_label(move: tuple) -> int:
    move_type, col = move
    if move_type == 'drop': return col
    if move_type == 'pop':  return col + 7
    return 14  # draw

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_dataset(dataset: list[tuple[list[int], int]], path: Path, append: bool = False) -> None:
    """Save (or append) dataset rows to *path*."""
    mode = 'a' if append else 'w'
    with open(path, mode, newline='') as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(HEADER)
        for features, label in dataset:
            writer.writerow(features + [label])


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, 'r') as f:
        return sum(1 for _ in f) - 1  # minus header


# ── Core game loop -- adapted from Rita's generate_mcts_dataset ────────────────

def generate_mcts_dataset(
    strategy:  str,
    num_games: int,
) -> list[tuple[list[int], int]]:
    """Play *num_games* of MCTS self-play with *strategy* rollouts.

    Uses 3 s per move for the first EARLY_GAME_MOVES moves, then 1.5 s
    (mirrors Rita's time_limit logic).

    Returns
    -------
    list of (board_features, move_label) pairs
    """
    from src.game.popout_board import PopOutBoard
    from src.ai.agents import make_mcts_agent

    # Two agents -- same strategy, different time budgets
    agent_early = make_mcts_agent(
        iterations=10_000,
        rollout_strategy=strategy,
        max_time=TIME_EARLY_GAME,
        early_stop_threshold=0.95,
    )
    agent_late = make_mcts_agent(
        iterations=10_000,
        rollout_strategy=strategy,
        max_time=TIME_LATE_GAME,
        early_stop_threshold=0.95,
    )

    dataset: list[tuple[list[int], int]] = []

    for game_idx in range(num_games):
        print(f"\n[GAME] [{strategy:>10s}]  Jogo {game_idx + 1} de {num_games}")
        board      = PopOutBoard()
        move_count = 0

        while not board.is_game_over:
            legal_moves = board.get_legal_moves()
            non_draw    = [m for m in legal_moves if m[0] != 'draw']

            # Choose agent based on game phase (mirrors Rita's time_limit switch)
            agent = agent_early if move_count < EARLY_GAME_MOVES else agent_late
            move  = agent(board)

            # Safety: fall back to random if agent returns an illegal move
            if move not in legal_moves:
                print(f"  [!] Movimento ilegal sugerido: {move} -- usando fallback aleatório")
                move = random.choice(non_draw if non_draw else legal_moves)

            # Record state BEFORE applying the move (mirrors Rita's approach)
            features = _board_to_features(board)
            label    = _move_to_label(move)
            dataset.append((features, label))

            board.apply_move(move)
            move_count += 1

        print(f"  [OK]  Jogo {game_idx + 1} concluido -- {move_count} jogadas  |  "
              f"dataset total: {len(dataset)} exemplos")

        # ── Partial save every CHECKPOINT_EVERY games (Rita's approach) ─────
        if (game_idx + 1) % CHECKPOINT_EVERY == 0:
            _save_dataset(dataset, TEMP_CSV, append=False)
            print(f"  [SAVE] Dataset parcial salvo: {len(dataset)} exemplos -> {TEMP_CSV.name}")

    return dataset


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    existing = _count_rows(MAIN_CSV)
    print(f"Dataset existente: {existing:,} amostras em {MAIN_CSV}")
    print(f"Serão adicionados jogos de {sum(n for _, n in BATCHES)} jogos novos.\n")

    all_new: list[tuple[list[int], int]] = []
    t_total = time.time()

    for strategy, num_games in BATCHES:
        print(f"\n{'='*60}")
        print(f"BATCH: {strategy.upper()}  ({num_games} jogos)")
        print(f"  Tempo por jogada: {TIME_EARLY_GAME}s (early) / {TIME_LATE_GAME}s (late)")
        print(f"{'='*60}")

        t0    = time.time()
        batch = generate_mcts_dataset(strategy=strategy, num_games=num_games)
        elapsed = time.time() - t0

        all_new.extend(batch)
        print(f"\n-> Batch '{strategy}' completo: {len(batch):,} amostras  |  {elapsed/3600:.1f}h")

    # ── Append to main CSV ───────────────────────────────────────────────────
    append_mode = MAIN_CSV.exists()
    _save_dataset(all_new, MAIN_CSV, append=append_mode)

    final_count = _count_rows(MAIN_CSV)
    total_elapsed = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"[DONE] Dataset final salvo com {len(all_new):,} novos exemplos!")
    print(f"   Total no CSV      : {final_count:,} amostras")
    print(f"   Tempo total       : {total_elapsed/3600:.1f} horas")
    print(f"   Ficheiro          : {MAIN_CSV}")

    # Remove temp file
    if TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print(f"   Temp removido     : {TEMP_CSV.name}")


if __name__ == '__main__':
    main()
