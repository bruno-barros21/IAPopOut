"""
ID3 Decision Tree — implemented from scratch.

No scikit-learn (or equivalent) is used for training or prediction.
pandas / numpy are used only for data handling outside this module.

Supports
--------
- Categorical features  : split on each unique value
- Continuous features   : binary threshold split that maximises IG
- max_depth             : regularisation to prevent overfitting
- min_samples           : minimum samples required to split a node
- Readable tree output  : print_tree() and as_text()

Algorithm
---------
ID3 (Iterative Dichotomiser 3, Quinlan 1986):

    build(S, features):
        if all labels in S are the same → leaf(label)
        if features is empty or |S| < min_samples → leaf(majority)
        if depth >= max_depth → leaf(majority)

        for each feature f:
            compute Information Gain  IG(S, f)

        split on f* = argmax IG
        recurse on each partition

Information Gain
----------------
    IG(S, A) = H(S) − Σ_v |S_v|/|S| · H(S_v)

    H(S) = −Σ_c p_c · log2(p_c)

For continuous features a threshold t is chosen to maximise IG over all
candidate midpoints between consecutive distinct values.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ── Information-theoretic utilities ──────────────────────────────────────────

def _entropy(labels: list) -> float:
    """Shannon entropy of a label list (in bits)."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _information_gain(parent_labels: list, child_groups: list[list]) -> float:
    """IG = H(parent) − weighted_avg_H(children)."""
    n = len(parent_labels)
    if n == 0:
        return 0.0
    h_parent = _entropy(parent_labels)
    h_children = sum(
        (len(g) / n) * _entropy(g)
        for g in child_groups
        if len(g) > 0
    )
    return h_parent - h_children


def _best_threshold(
    col: list[float],
    labels: list,
) -> tuple[float, float]:
    """Find the binary threshold for a continuous column that maximises IG.

    Returns
    -------
    (best_gain, best_threshold)
    """
    unique_sorted = sorted(set(col))
    if len(unique_sorted) == 1:
        return 0.0, unique_sorted[0]

    # Candidate thresholds: midpoints between consecutive unique values
    thresholds = [
        (unique_sorted[i] + unique_sorted[i + 1]) / 2.0
        for i in range(len(unique_sorted) - 1)
    ]

    best_gain, best_t = -1.0, thresholds[0]
    for t in thresholds:
        left  = [labels[i] for i, v in enumerate(col) if v <= t]
        right = [labels[i] for i, v in enumerate(col) if v >  t]
        gain  = _information_gain(labels, [left, right])
        if gain > best_gain:
            best_gain, best_t = gain, t

    return best_gain, best_t


# ── Tree node ─────────────────────────────────────────────────────────────────

class _Node:
    """Internal node of the decision tree."""

    __slots__ = (
        'is_leaf', 'label',
        'feature_idx', 'feature_name',
        'threshold',          # float if continuous, else None
        'children',           # dict: value/key → _Node
    )

    def __init__(self) -> None:
        self.is_leaf:      bool         = False
        self.label:        Any          = None
        self.feature_idx:  int | None   = None
        self.feature_name: str | None   = None
        self.threshold:    float | None = None
        self.children:     dict         = {}

    @property
    def is_continuous(self) -> bool:
        return self.threshold is not None


# ── Decision Tree ─────────────────────────────────────────────────────────────

class DecisionTreeID3:
    """ID3 Decision Tree with support for continuous and categorical features.

    Parameters
    ----------
    max_depth:
        Maximum depth of the tree. ``None`` means unlimited.
    min_samples:
        Minimum number of samples required to split an internal node.
    feature_names:
        Optional list of human-readable feature names (for display).
    continuous_features:
        Set of feature indices to be treated as continuous.
        If ``None``, any column containing at least one ``float`` value is
        treated as continuous automatically.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples: int = 2,
        feature_names: list[str] | None = None,
        continuous_features: set[int] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_names = feature_names
        self.continuous_features: set[int] = (
            set() if continuous_features is None else set(continuous_features)
        )
        self._auto_detect_continuous = continuous_features is None

        self.root: _Node | None = None
        self.n_nodes: int = 0
        self.depth: int = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: list[list], y: list) -> "DecisionTreeID3":
        """Train the tree on dataset (X, y).

        Parameters
        ----------
        X:
            List of samples, each a list of feature values.
        y:
            List of labels (one per sample).

        Returns
        -------
        self
        """
        if not X:
            raise ValueError("X must not be empty.")

        n_features = len(X[0])
        feature_indices = list(range(n_features))

        if self._auto_detect_continuous:
            self.continuous_features = {
                fi for fi in feature_indices
                if any(isinstance(x[fi], float) for x in X)
            }

        self.n_nodes = 0
        self.depth = 0
        self.root = self._build(X, y, feature_indices, depth=0)
        return self

    def _majority(self, y: list) -> Any:
        return Counter(y).most_common(1)[0][0]

    def _build(
        self,
        X: list[list],
        y: list,
        features: list[int],
        depth: int,
    ) -> _Node:
        self.n_nodes += 1
        self.depth = max(self.depth, depth)
        node = _Node()

        # ── Base cases ────────────────────────────────────────────────────────
        if len(set(y)) == 1:
            node.is_leaf = True
            node.label = y[0]
            return node

        if not features or len(X) < self.min_samples:
            node.is_leaf = True
            node.label = self._majority(y)
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.label = self._majority(y)
            return node

        # ── Find best split ───────────────────────────────────────────────────
        best_gain     = -1.0
        best_fi:       int | None   = None
        best_threshold: float | None = None
        best_partitions: list | None = None  # list of (key, X_sub, y_sub)

        for fi in features:
            col = [x[fi] for x in X]

            if fi in self.continuous_features:
                gain, t = _best_threshold(col, y)
                if gain > best_gain:
                    best_gain      = gain
                    best_fi        = fi
                    best_threshold = t
                    best_partitions = [
                        ('left',  [X[i] for i, v in enumerate(col) if v <= t],
                                  [y[i] for i, v in enumerate(col) if v <= t]),
                        ('right', [X[i] for i, v in enumerate(col) if v >  t],
                                  [y[i] for i, v in enumerate(col) if v >  t]),
                    ]
            else:
                values = set(col)
                groups = [
                    [y[i] for i, v in enumerate(col) if v == val]
                    for val in values
                ]
                gain = _information_gain(y, groups)
                if gain > best_gain:
                    best_gain      = gain
                    best_fi        = fi
                    best_threshold = None
                    best_partitions = [
                        (val,
                         [X[i] for i, v in enumerate(col) if v == val],
                         [y[i] for i, v in enumerate(col) if v == val])
                        for val in values
                    ]

        if best_gain <= 0 or best_fi is None:
            node.is_leaf = True
            node.label = self._majority(y)
            return node

        # ── Build internal node ───────────────────────────────────────────────
        node.feature_idx  = best_fi
        node.feature_name = (
            self.feature_names[best_fi]
            if self.feature_names and best_fi < len(self.feature_names)
            else f'feature_{best_fi}'
        )
        node.threshold = best_threshold

        # Continuous features can be reused (different thresholds per branch).
        # Categorical features are consumed once.
        if node.is_continuous:
            next_features = features
        else:
            next_features = [f for f in features if f != best_fi]

        for key, X_sub, y_sub in best_partitions:
            if not y_sub:
                leaf = _Node()
                leaf.is_leaf = True
                leaf.label   = self._majority(y)
                node.children[key] = leaf
            else:
                node.children[key] = self._build(
                    X_sub, y_sub, next_features, depth + 1
                )

        return node

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_one(self, x: list) -> Any:
        """Predict the label for a single sample."""
        node = self.root
        while not node.is_leaf:
            val = x[node.feature_idx]
            if node.is_continuous:
                key = 'left' if val <= node.threshold else 'right'
            else:
                key = val
            child = node.children.get(key)
            if child is None:
                # Unseen value during inference — return majority at this node
                break
            node = child
        return node.label

    def predict(self, X: list[list]) -> list:
        """Predict labels for all samples in X."""
        return [self.predict_one(x) for x in X]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def accuracy(self, X: list[list], y: list) -> float:
        """Fraction of correctly classified samples."""
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    # ── Display ───────────────────────────────────────────────────────────────

    def print_tree(
        self,
        node: _Node | None = None,
        indent: int = 0,
        max_display_depth: int = 6,
    ) -> None:
        """Print the tree structure to stdout."""
        if node is None:
            node = self.root
        if node is None:
            print('(empty tree)')
            return

        pad = '  ' * indent

        if indent > max_display_depth:
            print(f'{pad}...')
            return

        if node.is_leaf:
            print(f'{pad}→ {node.label}')
            return

        if node.is_continuous:
            print(f'{pad}[{node.feature_name} ≤ {node.threshold:.4f}?]')
            for key, label in [('left', f'≤ {node.threshold:.4f}'),
                                ('right', f'> {node.threshold:.4f}')]:
                print(f'{pad}  {label}:')
                if key in node.children:
                    self.print_tree(node.children[key], indent + 2,
                                    max_display_depth)
        else:
            print(f'{pad}[{node.feature_name}?]')
            for val in sorted(node.children, key=str):
                print(f'{pad}  = {val}:')
                self.print_tree(node.children[val], indent + 2,
                                max_display_depth)

    def as_text(self, max_display_depth: int = 6) -> str:
        """Return the tree as an indented string."""
        import io, sys
        buf = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, buf
        try:
            self.print_tree(max_display_depth=max_display_depth)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()
