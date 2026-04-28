"""
Train and evaluate the Decision Tree on the PopOut MCTS dataset.

This script:
1. Loads the generated dataset (data/popout_mcts.csv).
2. Performs an 80/20 train/test split.
3. Trains a DecisionTreeID3 on the training set (with max_depth to prevent overfitting).
4. Evaluates accuracy on both train and test sets.
5. Serializes (pickles) the trained model to data/popout_tree.pkl for use in the game.
"""

from __future__ import annotations

import pickle
import random
import time
from pathlib import Path

from src.ml.dataset import load_dataset, FEATURE_NAMES, DATA_DIR
from src.ml.decision_tree import DecisionTreeID3

def train_and_save_model(
    max_depth: int = 15,
    min_samples: int = 2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> None:
    # ── Load and Split ────────────────────────────────────────────────────────
    print("Loading dataset from data/popout_mcts.csv...")
    try:
        X, y = load_dataset()
    except FileNotFoundError:
        print("Error: Dataset not found. Please run tmp_generate.py first.")
        return

    n_samples = len(X)
    if n_samples == 0:
        print("Error: Dataset is empty.")
        return

    # Shuffle instances
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)

    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]

    print(f"Dataset Split: {n_train} train | {n_test} test (Ratio: {1 - test_ratio:.1f}/{test_ratio:.1f})")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining DecisionTreeID3 (max_depth={max_depth}, min_samples={min_samples})...")
    print("This might take a few moments for large datasets.")
    
    t0 = time.time()
    
    # We do NOT let the auto-detector run for PopOut because these are discrete features 
    # (0,1,2), so categorical splitting is correct and much faster than continuous.
    tree = DecisionTreeID3(
        max_depth=max_depth,
        min_samples=min_samples,
        feature_names=FEATURE_NAMES,
        continuous_features=set()  # Force all features to be treated as categorical
    )
    
    tree.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.2f}s")
    print(f"Tree structure: {tree.n_nodes} nodes | Maximum depth reached: {tree.depth}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating model accuracy...")
    acc_train = tree.accuracy(X_train, y_train)
    acc_test  = tree.accuracy(X_test, y_test)
    
    print(f"Training Accuracy : {acc_train*100:.2f}%")
    print(f"Testing Accuracy  : {acc_test*100:.2f}%")
    
    if acc_train - acc_test > 0.15:
        print("Warning: The model might be overfitting. Consider reducing max_depth.")

    # ── Serialization ─────────────────────────────────────────────────────────
    out_path = DATA_DIR / 'popout_tree.pkl'
    print(f"\nSaving model to {out_path.name}...")
    with open(out_path, 'wb') as f:
        pickle.dump(tree, f)
        
    print("Success! You can now load this model in IAPopOut.ipynb using 'pickle.load()'.")

if __name__ == '__main__':
    # A max_depth of 15 is a reasonable starting point to avoid exploding the tree with 11k samples.
    # The user can adjust this if necessary.
    train_and_save_model(max_depth=15, test_ratio=0.2)
