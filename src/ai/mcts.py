"""
Monte Carlo Tree Search (MCTS) with UCT for PopOut.

Algorithm
---------
Each iteration executes four phases:

1. **Selection**     — descend the tree from the root, always choosing the
                       child with the highest UCT score, until a node with
                       unexplored children is found (or a terminal node).

2. **Expansion**     — expand *k* children of the selected node by choosing
                       *k* untried moves at random (default k=1, standard
                       MCTS).

3. **Simulation**    — play a random (or heuristic) game from the new node
                       to a terminal state.

4. **Backpropagation** — propagate the result back up the tree, updating
                         visit count N and total reward Q of every ancestor.

UCT formula
-----------
    UCT(v) = Q(v)/N(v)  +  C * sqrt( ln(N(parent)) / N(v) )

    Q — total reward accumulated at node v
    N — number of visits
    C — exploration constant (default √2)

The final move is chosen by **robust selection**: the child with the highest
visit count N (most reliable estimate, robust to lucky outliers).
"""

from __future__ import annotations

import math
import random
from typing import Literal

from src.game.popout_board import PopOutBoard


RolloutStrategy = Literal['random', 'heuristic']


# ── Tree node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = (
        'board', 'parent', 'move',
        'children', 'untried_moves',
        'N', 'Q',
    )

    def __init__(
        self,
        board: PopOutBoard,
        parent: MCTSNode | None = None,
        move: tuple | None = None,
    ) -> None:
        self.board = board
        self.parent = parent
        self.move = move            # move that led to this node
        self.children: list[MCTSNode] = []
        self.untried_moves: list[tuple] = board.get_legal_moves()
        self.N: int = 0             # visit count
        self.Q: float = 0.0         # total reward (root player's perspective)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.board.is_game_over

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    # ── UCT ───────────────────────────────────────────────────────────────────

    def uct_score(self, c: float) -> float:
        """UCT value of this node (called from the parent)."""
        if self.N == 0:
            return math.inf
        exploitation = self.Q / self.N
        exploration  = c * math.sqrt(math.log(self.parent.N) / self.N)
        return exploitation + exploration

    def best_uct_child(self, c: float) -> MCTSNode:
        return max(self.children, key=lambda ch: ch.uct_score(c))

    def best_robust_child(self) -> MCTSNode:
        """Most-visited child — used for the final move selection."""
        return max(self.children, key=lambda ch: ch.N)

    # ── Win-rate helper ───────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        return self.Q / self.N if self.N else 0.0


# ── Rollout strategies ────────────────────────────────────────────────────────

def _rollout_random(board: PopOutBoard) -> int | None:
    """Play uniformly at random until a terminal state."""
    sim = board.copy()
    while not sim.is_game_over:
        moves = sim.get_legal_moves()
        if not moves:
            break
        non_draw = [m for m in moves if m[0] != 'draw']
        sim.apply_move(random.choice(non_draw if non_draw else moves))
    return sim.winner


def _rollout_heuristic(board: PopOutBoard) -> int | None:
    """Play with a simple one-ply heuristic during rollout.

    Priority:
    1. Take an immediate winning move.
    2. Block an immediate opponent win.
    3. Otherwise pick uniformly at random.
    """
    sim = board.copy()
    while not sim.is_game_over:
        moves = sim.get_legal_moves()
        if not moves:
            break

        player   = sim.current_player
        opponent = 3 - player
        chosen   = None

        non_draw = [m for m in moves if m[0] != 'draw']
        candidates = non_draw if non_draw else moves

        # 1. Immediate win
        for move in candidates:
            tmp = sim.copy()
            tmp.apply_move(move)
            if tmp.winner == player:
                chosen = move
                break

        # 2. Block opponent win
        if chosen is None:
            for move in candidates:
                tmp = sim.copy()
                tmp.apply_move(move)
                if tmp.winner == opponent:
                    chosen = move
                    break

        if chosen is None:
            chosen = random.choice(candidates)

        sim.apply_move(chosen)

    return sim.winner


_ROLLOUT_FN = {
    'random':    _rollout_random,
    'heuristic': _rollout_heuristic,
}


# ── Main search ───────────────────────────────────────────────────────────────

def mcts_search(
    board: PopOutBoard,
    iterations: int = 1000,
    c: float = math.sqrt(2),
    expand_k: int = 1,
    rollout_strategy: RolloutStrategy = 'random',
) -> tuple[tuple, MCTSNode]:
    """Run MCTS and return the best move together with the search tree root.

    Parameters
    ----------
    board:
        Current game state. Not modified.
    iterations:
        Number of MCTS iterations (more = stronger, slower).
    c:
        UCT exploration constant. The theoretical optimum is √2.
    expand_k:
        Number of children to expand per iteration.
        k=1 is standard MCTS; k>1 trades depth for breadth.
    rollout_strategy:
        ``'random'``    — uniform random rollouts.
        ``'heuristic'`` — one-ply look-ahead during rollout.

    Returns
    -------
    (best_move, root_node)
    """
    rollout_fn = _ROLLOUT_FN[rollout_strategy]
    root_player = board.current_player
    root = MCTSNode(board.copy())

    for _ in range(iterations):
        node = root

        # ── 1. Selection ──────────────────────────────────────────────────────
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_uct_child(c)

        # ── 2. Expansion ──────────────────────────────────────────────────────
        if not node.is_terminal and node.untried_moves:
            k = min(expand_k, len(node.untried_moves))
            for _ in range(k):
                if not node.untried_moves:
                    break
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child_board = node.board.copy()
                child_board.apply_move(move)
                child = MCTSNode(child_board, parent=node, move=move)
                node.children.append(child)
            node = node.children[-1]

        # ── 3. Simulation ─────────────────────────────────────────────────────
        result = rollout_fn(node.board)

        # ── 4. Backpropagation ────────────────────────────────────────────────
        current = node
        while current is not None:
            current.N += 1
            if result == root_player:
                current.Q += 1.0
            elif result == 0:       # draw
                current.Q += 0.5
            # loss → +0
            current = current.parent

    if not root.children:
        fallback = board.get_legal_moves()
        return (fallback[0] if fallback else ('draw', None)), root

    return root.best_robust_child().move, root
