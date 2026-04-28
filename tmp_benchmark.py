"""Quick benchmark to calibrate iters per strategy."""
import sys, time
sys.path.insert(0, '.')
from src.ml.dataset import generate_dataset

N_TEST = 3  # games per test

for strat, iters in [('random', 30), ('greedy', 30), ('heuristic', 30)]:
    t = time.time()
    X, y = generate_dataset(n_games=N_TEST, mcts_iterations=iters,
                             rollout_strategy=strat, verbose=False)
    elapsed = time.time() - t
    pg = elapsed / N_TEST
    print(f"{strat:12s} @ {iters:3d} iters: {pg:5.1f}s/game "
          f"| 50g={50*pg/60:.0f}min 150g={150*pg/60:.0f}min 250g={250*pg/60:.0f}min")
