"""
Dataset generation: 50 random + 150 greedy + 250 heuristic games.
Uses 30 iters/move for all strategies (calibrated for ~2h total).

INCREMENTAL SAVING: saves after each strategy block so progress is
preserved if the process is interrupted (shutdown, sleep, etc.).
On resume, already-completed blocks are detected and skipped.
"""
import sys, time, csv
sys.path.insert(0, '.')
from pathlib import Path
from collections import Counter
from src.ml.dataset import generate_dataset, save_dataset, FEATURE_NAMES, POPOUT_DATASET_PATH

# ── Config ────────────────────────────────────────────────────────────────────
CONFIGS = [
    ('random',     50,  30),   # ~1 min
    ('greedy',    150,  30),   # ~53 min
    ('heuristic', 250,  30),   # ~60-90 min (estimate)
]

# Checkpoint dir: one CSV per completed block
CHECKPOINT_DIR = Path('data/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def checkpoint_path(strat: str) -> Path:
    return CHECKPOINT_DIR / f'block_{strat}.csv'

def load_checkpoint(path: Path):
    """Load a previously saved block CSV, returns (X, y) or ([], [])."""
    if not path.exists():
        return [], []
    X, y = [], []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vals = list(map(int, row))
            X.append(vals[:-1])
            y.append(vals[-1])
    print(f'  [RESUME] Loaded {len(X)} samples from {path.name}')
    return X, y

def save_checkpoint(X, y, path: Path):
    header = FEATURE_NAMES + ['label']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for feats, label in zip(X, y):
            writer.writerow(feats + [label])
    print(f'  [CHECKPOINT] Saved {len(X)} samples -> {path.name}')

# ── Main generation ───────────────────────────────────────────────────────────
X_all, y_all = [], []
t_total = time.time()

for strat, ngames, niters in CONFIGS:
    ckpt = checkpoint_path(strat)

    # Skip if already done
    X_block, y_block = load_checkpoint(ckpt)
    if X_block:
        print(f'SKIP {strat} ({len(X_block)} samples already on disk)')
        X_all.extend(X_block)
        y_all.extend(y_block)
        continue

    # Generate
    t1 = time.time()
    print(f'\n=== Strategy: {strat} | {ngames} games | {niters} iters/move ===')
    X_block, y_block = generate_dataset(
        n_games=ngames,
        mcts_iterations=niters,
        rollout_strategy=strat,
        verbose=True,
    )
    elapsed_block = time.time() - t1
    print(f'  Block done in {elapsed_block/60:.1f} min | {len(X_block)} samples')

    # Save checkpoint immediately
    save_checkpoint(X_block, y_block, ckpt)

    X_all.extend(X_block)
    y_all.extend(y_block)
    total_elapsed = time.time() - t_total
    print(f'  Running total: {len(X_all)} samples | {total_elapsed/60:.1f} min elapsed')

# ── Merge & save final dataset ────────────────────────────────────────────────
print(f'\n=== ALL DONE: {len(X_all)} samples in {(time.time()-t_total)/60:.1f} min ===')
save_dataset(X_all, y_all)

# ── Label distribution ────────────────────────────────────────────────────────
print('\nLabel distribution:')
ct = Counter(y_all)
for lbl in sorted(ct):
    if lbl <= 6:
        move_str = f'drop col {lbl}'
    elif lbl <= 13:
        move_str = f'pop  col {lbl - 7}'
    else:
        move_str = 'draw'
    pct = 100 * ct[lbl] / len(y_all)
    print(f'  {lbl:2d} ({move_str:15s}): {ct[lbl]:5d}  ({pct:.1f}%)')

print('\nCheckpoints in data/checkpoints/ can be deleted after training is complete.')
