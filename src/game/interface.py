"""
Game loop and human input handling for PopOut.

Supported game scenarios
------------------------
- Human vs Human
- Human vs Computer
- Computer vs Computer

Usage
-----
    from src.game import play_game
    from src.ai  import make_mcts_agent

    mcts = make_mcts_agent(iterations=1000)
    play_game('human', mcts)          # human is Player 1
    play_game(mcts, 'human')          # human is Player 2
    play_game('human', 'human')       # two humans
    play_game(mcts, mcts)             # two computers

Move input format (human)
-------------------------
    drop <col>   — drop a piece into column col  (e.g. 'drop 3')
    pop  <col>   — pop own piece from column col (e.g. 'pop 2')
    draw         — declare a draw (only when available)
"""

from __future__ import annotations
from typing import Callable, Literal
from IPython.display import clear_output

from .popout_board import PopOutBoard
from .display import display_board

Agent = Callable[[PopOutBoard], tuple] | Literal['human']

_PLAYER_LABELS = {1: 'X  (Player 1)', 2: 'O  (Player 2)'}


# ── Human input ───────────────────────────────────────────────────────────────

def parse_human_move(
    raw: str,
    board: PopOutBoard,
) -> tuple | None:
    """Parse and validate a human move string.

    Returns the move tuple on success, or ``None`` if the input is invalid.
    Error messages are printed to stdout.
    """
    parts = raw.strip().lower().split()
    legal = board.get_legal_moves()

    if not parts:
        print('  Please enter a move.')
        return None

    if parts[0] == 'draw':
        if ('draw', None) in legal:
            return ('draw', None)
        print('  Draw is not available right now.')
        return None

    if len(parts) != 2:
        print("  Format: 'drop <col>', 'pop <col>', or 'draw'")
        return None

    move_type = parts[0]
    if move_type not in ('drop', 'pop'):
        print("  Unknown move type. Use 'drop', 'pop', or 'draw'.")
        return None

    try:
        col = int(parts[1])
    except ValueError:
        print('  Column must be an integer between 0 and 6.')
        return None

    move = (move_type, col)
    if move not in legal:
        non_draw = [m for m in legal if m[0] != 'draw']
        print(f'  Illegal move {move}.')
        print(f'  Legal moves: {non_draw}')
        return None

    return move


# ── Game loop ─────────────────────────────────────────────────────────────────

def play_game(
    player1: Agent,
    player2: Agent,
    verbose: bool = True,
) -> int | None:
    """Run a full PopOut game.

    Parameters
    ----------
    player1, player2:
        Either the string ``'human'`` (interactive input) or any callable
        that accepts a :class:`PopOutBoard` and returns a legal move tuple.
    verbose:
        If ``True``, the board is printed after each move using
        :func:`~src.game.display.display_board`.

    Returns
    -------
    int | None
        ``0`` for draw, ``1`` / ``2`` for the winning player,
        or ``None`` if the game was aborted.
    """
    board = PopOutBoard()
    agents: dict[int, Agent] = {1: player1, 2: player2}

    while not board.is_game_over:
        if verbose:
            clear_output(wait=True)
            display_board(board)

        agent = agents[board.current_player]

        if agent == 'human':
            move = _get_human_move(board)
            if move is None:
                return None  # game aborted
        else:
            move = agent(board)
            if verbose:
                print(f'  {_PLAYER_LABELS[board.current_player]} plays: {move}')

        board.apply_move(move)

    if verbose:
        clear_output(wait=True)
        display_board(board)

    return board.winner


def _get_human_move(board: PopOutBoard) -> tuple | None:
    while True:
        try:
            raw = input(
                f"  {_PLAYER_LABELS[board.current_player]} "
                "— enter move (e.g. 'drop 3', 'pop 2', 'draw'): "
            )
        except (EOFError, KeyboardInterrupt):
            print('\n  Game aborted.')
            return None

        move = parse_human_move(raw, board)
        if move is not None:
            return move
