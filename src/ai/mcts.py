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

Extensions beyond standard MCTS
--------------------------------
- **Early termination** — stop iterating when the best child reaches a
  dominant win rate (configurable threshold), saving computation without
  sacrificing move quality.
- **Time-limited search** — optional wall-clock time limit as an alternative
  (or complement) to a fixed iteration budget.
"""

from __future__ import annotations

import math
import random
import time as _time
from typing import Literal

from src.game.popout_board import PopOutBoard


RolloutStrategy = Literal['random', 'heuristic', 'greedy']


# ── Tree node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    """A single node in the MCTS search tree."""
    # Per-node fields
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
        self.Q: float = 0.0         # total reward (from this node's active player perspective)

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

    def __repr__(self) -> str:
        return (f"MCTSNode(move={self.move}, N={self.N}, "
                f"Q={self.Q:.1f}, wr={self.win_rate:.3f})")


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

        # 2. Block: avoid leaving opponent a winning reply
        if chosen is None:
            for move in candidates:
                tmp = sim.copy()
                tmp.apply_move(move)
                if tmp.is_game_over:
                    continue
                # Check if opponent can win on their next move
                opp_can_win = False
                for opp_move in tmp.get_legal_moves():
                    if opp_move[0] == 'draw':
                        continue
                    tmp2 = tmp.copy()
                    tmp2.apply_move(opp_move)
                    if tmp2.winner == opponent:
                        opp_can_win = True
                        break
                if not opp_can_win:
                    chosen = move
                    break

        if chosen is None:
            chosen = random.choice(candidates)

        sim.apply_move(chosen)

    return sim.winner


def _rollout_greedy(board: PopOutBoard) -> int | None:
    """Play using a greedy one-ply heuristic with positional scoring.

    Priority:
    1. Take an immediate winning move.
    2. Block an immediate opponent winning move (one-ply threat check
       on the current board).
    3. Pick the move with the highest positional score (center column
       preference, drops favoured over pops).

    Compared to ``heuristic``, this is faster because the blocking
    check examines opponent threats on the *current* board (one-ply)
    rather than checking opponent responses *after* each candidate
    move (two-ply).  The positional scoring adds domain knowledge
    without the cost of deeper search.
    """
    # Positional value per column: center columns create more connections
    _COL_VALUE = [0, 1, 2, 3, 2, 1, 0]

    sim = board.copy()
    while not sim.is_game_over:
        moves = sim.get_legal_moves()
        if not moves:
            break

        player   = sim.current_player
        opponent = 3 - player
        chosen   = None

        non_draw   = [m for m in moves if m[0] != 'draw']
        candidates = non_draw if non_draw else moves

        # 1. Immediate win
        for move in candidates:
            tmp = sim.copy()
            tmp.apply_move(move)
            if tmp.winner == player:
                chosen = move
                break

        # 2. Block: find opponent's immediate winning threats
        #    Temporarily give the opponent the turn and check which
        #    moves would let them win right now.  If we share a legal
        #    move with that threat (e.g. drop in the same column), block.
        if chosen is None:
            tmp_opp = sim.copy()
            tmp_opp.current_player = opponent
            for opp_move in tmp_opp.get_legal_moves():
                if opp_move[0] == 'draw':
                    continue
                tmp2 = tmp_opp.copy()
                tmp2.apply_move(opp_move)
                if tmp2.winner == opponent and opp_move in candidates:
                    chosen = opp_move
                    break

        # 3. Positional scoring: prefer center columns + drops over pops
        if chosen is None:
            def _score(m: tuple) -> int:
                move_type, col = m
                s = _COL_VALUE[col]
                if move_type == 'drop':
                    s += 4   # drops build connections more reliably
                return s
            chosen = max(candidates, key=_score)

        sim.apply_move(chosen)

    return sim.winner


_ROLLOUT_FN = {
    'random':    _rollout_random,
    'heuristic': _rollout_heuristic,
    'greedy':    _rollout_greedy,
}


# ── Main search ───────────────────────────────────────────────────────────────

def mcts_search(
    board: PopOutBoard,
    iterations: int = 1000,
    c: float = math.sqrt(2),
    expand_k: int = 1,
    rollout_strategy: RolloutStrategy = 'random',
    max_time: float | None = None,
    early_stop_threshold: float = 1.0,
    early_stop_min_visits: int = 50,
) -> tuple[tuple, MCTSNode]:
    """Run MCTS and return the best move together with the search tree root.

    Parameters
    ----------
    board:
        Current game state. Not modified.
    iterations:
        Maximum number of MCTS iterations (more = stronger, slower).
    c:
        UCT exploration constant. The theoretical optimum is √2.
    expand_k:
        Number of children to expand per iteration.
        k=1 is standard MCTS; k>1 trades depth for breadth.
    rollout_strategy:
        ``'random'``    — uniform random rollouts.
        ``'greedy'``    — one-ply win/block + positional scoring.
        ``'heuristic'`` — two-ply look-ahead during rollout.
    max_time:
        Optional wall-clock time limit in seconds.  The search stops when
        either *iterations* or *max_time* is reached (whichever first).
        ``None`` means no time limit.
    early_stop_threshold:
        Win-rate threshold for early termination (0.0–1.0).  If the best
        child reaches this win rate with at least *early_stop_min_visits*,
        the search stops early.  Default ``1.0`` disables early stopping.
    early_stop_min_visits:
        Minimum visit count before early stopping can trigger (prevents
        premature decisions based on noisy estimates).

    Returns
    -------
    (best_move, root_node)
    """
    rollout_fn = _ROLLOUT_FN[rollout_strategy]
    root = MCTSNode(board.copy())
    start = _time.monotonic()

    for i in range(iterations):
        node = root

        # ── 1. Selection ──────────────────────────────────────────────────
        # Each child stores Q from its OWN current_player's perspective.
        # Since child.current_player == opponent(node.current_player),
        # selecting max UCT naturally maximises wins for the selecting
        # player (adversarial inversion).
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_uct_child(c)

        # ── 2. Expansion ─────────────────────────────────────────────────
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
            # Rollout from the last expanded child.  The other k-1 children
            # (with N=0 → UCT=∞) will be prioritised by UCT selection in
            # subsequent iterations.
            node = node.children[-1]

        # ── 3. Simulation ────────────────────────────────────────────────
        result = rollout_fn(node.board)

        # ── 4. Backpropagation ───────────────────────────────────────────
        # Q at each node is stored from the perspective of the player
        # WHO IS ABOUT TO MOVE at that node:
        #   result == node.current_player  →  +1.0  (win)
        #   result == 0 or None            →  +0.5  (draw)
        #   otherwise                      →  +0.0  (loss)
        current = node
        while current is not None:
            current.N += 1
            node_player = current.board.current_player
            if result == node_player:
                current.Q += 1.0
            elif result == 0 or result is None:
                current.Q += 0.5
            current = current.parent

        # ── 5. Early stopping ────────────────────────────────────────────
        # If the most-visited child already has a dominant win rate,
        # additional iterations are unlikely to change the decision.
        if (early_stop_threshold < 1.0
                and root.children
                and i >= max(iterations // 4, 30)):
            best = root.best_robust_child()
            if (best.N >= early_stop_min_visits
                    and best.win_rate >= early_stop_threshold):
                break

        # ── Time limit ───────────────────────────────────────────────────
        if max_time is not None and (_time.monotonic() - start) >= max_time:
            break

    if not root.children:
        fallback = board.get_legal_moves()
        return (fallback[0] if fallback else ('draw', None)), root

    return root.best_robust_child().move, root
