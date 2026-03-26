"""
Board rendering utilities for PopOut.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .popout_board import PopOutBoard

_SYMBOLS = {0: '.', 1: 'X', 2: 'O'}
_PLAYER_LABELS = {1: 'X  (Player 1)', 2: 'O  (Player 2)'}


def display_board(board: "PopOutBoard") -> None:
    """Print the board to stdout in a human-readable format.

    Example output::

        0 1 2 3 4 5 6
        -------------
      0|. . . . . . .|
      1|. . . . . . .|
      2|. . . . . . .|
      3|. . . . . . .|
      4|. . X O . . .|
      5|. . O X . . .|
        -------------
        Turn: X  (Player 1)
    """
    b = board.board
    rows, cols = b.shape

    header = '  ' + ' '.join(str(c) for c in range(cols))
    sep = '  ' + '-' * (cols * 2 - 1)

    print(header)
    print(sep)
    for r in range(rows):
        cells = ' '.join(_SYMBOLS[int(b[r, c])] for c in range(cols))
        print(f'{r} |{cells}|')
    print(sep)

    if board.is_game_over:
        if board.winner == 0:
            print('  Result : Draw')
        else:
            print(f'  Result : {_PLAYER_LABELS[board.winner]} wins!')
    else:
        print(f'  Turn   : {_PLAYER_LABELS[board.current_player]}')
    print()
