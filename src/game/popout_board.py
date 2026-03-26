"""
PopOut game engine.

PopOut is a Connect-4 variant where players can also remove ('pop') their
own pieces from the bottom of a column, shifting all pieces above down.

Special rules
-------------
1. Simultaneous win  : If a pop creates 4-in-a-row for both players, the
                       player who popped wins.
2. Full-board draw   : When the board is full the current player may declare
                       a draw instead of popping.
3. Repetition draw   : If the same board state occurs three times, either
                       player may declare a draw.

Move encoding
-------------
('drop', col)  — drop a piece at the top of column col
('pop',  col)  — pop own piece from the bottom of column col
('draw', None) — declare a draw (only when the rule applies)
"""

import numpy as np


class PopOutBoard:
    """Mutable game state for a PopOut match."""

    ROWS: int = 6
    COLS: int = 7

    # Cell values
    EMPTY:    int = 0
    PLAYER_1: int = 1
    PLAYER_2: int = 2

    def __init__(self) -> None:
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player: int = self.PLAYER_1
        self.is_game_over: bool = False
        # None  → game in progress
        # 0     → draw
        # 1 / 2 → winner
        self.winner: int | None = None
        self._state_history: dict[bytes, int] = {}
        self._record_state()

    # ── Copying ───────────────────────────────────────────────────────────────

    def copy(self) -> "PopOutBoard":
        """Return a deep copy of the board (needed by search algorithms)."""
        clone = PopOutBoard.__new__(PopOutBoard)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.is_game_over = self.is_game_over
        clone.winner = self.winner
        clone._state_history = self._state_history.copy()
        return clone

    # ── State history ─────────────────────────────────────────────────────────

    def _record_state(self) -> None:
        key = self.board.tobytes()
        self._state_history[key] = self._state_history.get(key, 0) + 1

    def _state_repeated(self) -> bool:
        return self._state_history.get(self.board.tobytes(), 0) >= 3

    # ── Legal moves ───────────────────────────────────────────────────────────

    def get_legal_moves(self) -> list[tuple]:
        """Return all legal moves for the current player."""
        if self.is_game_over:
            return []

        moves: list[tuple] = []
        board_full = True

        for col in range(self.COLS):
            if self.board[0, col] == self.EMPTY:
                moves.append(('drop', col))
                board_full = False
            if self.board[self.ROWS - 1, col] == self.current_player:
                moves.append(('pop', col))

        if board_full or self._state_repeated():
            moves.append(('draw', None))

        return moves

    # ── Move application ──────────────────────────────────────────────────────

    def apply_move(self, move: tuple) -> None:
        """Apply a move and update the game state in-place."""
        move_type, col = move

        if move_type == 'draw':
            self.is_game_over = True
            self.winner = 0
            return

        if move_type == 'drop':
            self._drop(col)
        elif move_type == 'pop':
            self._pop(col)
        else:
            raise ValueError(f"Unknown move type: {move_type!r}")

        self._check_winner()

        if not self.is_game_over:
            self.current_player = 3 - self.current_player  # 1→2, 2→1
            self._record_state()

    def _drop(self, col: int) -> None:
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row, col] == self.EMPTY:
                self.board[row, col] = self.current_player
                return

    def _pop(self, col: int) -> None:
        # Shift everything above the bottom piece down by one
        self.board[1:, col] = self.board[:-1, col]
        self.board[0, col] = self.EMPTY

    # ── Win detection ─────────────────────────────────────────────────────────

    def _check_winner(self) -> None:
        p1 = self._has_four(self.PLAYER_1)
        p2 = self._has_four(self.PLAYER_2)

        if p1 and p2:
            # Rule 1: simultaneous win goes to the player who just moved
            self.winner = self.current_player
            self.is_game_over = True
        elif p1:
            self.winner = self.PLAYER_1
            self.is_game_over = True
        elif p2:
            self.winner = self.PLAYER_2
            self.is_game_over = True

    def _has_four(self, player: int) -> bool:
        b = self.board
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if b[r, c] == b[r, c+1] == b[r, c+2] == b[r, c+3] == player:
                    return True
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if b[r, c] == b[r+1, c] == b[r+2, c] == b[r+3, c] == player:
                    return True
        # Diagonal ↘
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == b[r+3, c+3] == player:
                    return True
        # Diagonal ↗
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if b[r, c] == b[r-1, c+1] == b[r-2, c+2] == b[r-3, c+3] == player:
                    return True
        return False

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        rows = [''.join(symbols[self.board[r, c]] for c in range(self.COLS))
                for r in range(self.ROWS)]
        return '\n'.join(rows)
