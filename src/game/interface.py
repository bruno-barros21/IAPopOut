"""
Interactive game interface for PopOut using ipywidgets.

Supported game scenarios
------------------------
- Human vs Human      — both players use button controls
- Human vs Computer   — human uses buttons, computer thinks in background
- Computer vs Computer — automated loop with configurable move delay

Usage
-----
    from src.game import play_game
    from src.ai   import make_mcts_agent

    mcts = make_mcts_agent(iterations=1000)
    play_game('human', mcts)          # human is Player 1
    play_game(mcts, 'human')          # human is Player 2
    play_game('human', 'human')       # two humans
    play_game(mcts, mcts)             # two computers (watch with delay)
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Literal

import ipywidgets as widgets
from IPython.display import display as ipy_display, clear_output, HTML

from .popout_board import PopOutBoard
from .display import (
    render_board_html, display_board, display_board_text,
    find_last_move_cell, move_label,
    PLAYER_NAME, PLAYER_EMOJI,
)

Agent = Callable[[PopOutBoard], tuple] | Literal['human']

_CSS_CLASS = {1: 'red', 2: 'yellow'}


# ── Legacy text input (kept for compatibility) ───────────────────────────────

def parse_human_move(raw: str, board: PopOutBoard) -> tuple | None:
    """Parse and validate a human move string.

    Returns the move tuple on success, or ``None`` if the input is invalid.
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

# ── Thinking indicator ────────────────────────────────────────────────────────

_THINKING_HTML = """<div style="text-align:center;padding:10px 0;color:#94a3b8;
font-family:'Segoe UI',system-ui,sans-serif;font-size:14px">
<span style="display:inline-block;animation:thinkDot 1.4s infinite">{emoji}</span>
 {name} is thinking…
<style>@keyframes thinkDot{{0%,100%{{opacity:.3}}50%{{opacity:1}}}}</style>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive UI (games with at least one human)
# ══════════════════════════════════════════════════════════════════════════════

class PopOutGameUI:
    """Widget-based interactive game UI for Jupyter notebooks.

    Attributes
    ----------
    result : int | None
        ``None`` while the game is in progress;
        ``0`` for draw, ``1``/``2`` for the winning player.
    """

    def __init__(
        self,
        player1: Agent,
        player2: Agent,
        move_delay: float = 0.3,
    ) -> None:
        self.board = PopOutBoard()
        self.agents: dict[int, Agent] = {1: player1, 2: player2}
        self.move_delay = move_delay
        self.move_history: list[str] = []
        self.last_move_cell: tuple[int, int] | None = None
        self.animate_cell: tuple[int, int] | None = None
        self.result: int | None = None

        # Widgets
        self.board_output = widgets.Output()
        self.controls_output = widgets.Output()
        self.container = widgets.VBox([
            self.board_output,
            self.controls_output,
        ])

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Display the UI and kick off the first turn."""
        ipy_display(self.container)
        self._advance()

    # ── Internal flow ─────────────────────────────────────────────────────

    def _advance(self) -> None:
        """Process turns until a human needs to act."""
        if self.board.is_game_over:
            self._render()
            self._clear_controls()
            self.result = self.board.winner
            return

        agent = self.agents[self.board.current_player]

        if agent == 'human':
            self._render()
            self._show_controls()
        else:
            # Computer's turn
            player = self.board.current_player
            self._render()
            self._show_thinking(player)
            # Run in thread to keep UI responsive
            threading.Thread(target=self._computer_turn, daemon=True).start()

    def _computer_turn(self) -> None:
        """Compute and apply a computer move (runs in a background thread)."""
        player = self.board.current_player
        agent = self.agents[player]
        move = agent(self.board)
        time.sleep(self.move_delay)  # brief pause so user sees "thinking"
        self._apply_move(move, player)
        self._advance()

    # ── Move application ──────────────────────────────────────────────────

    def _apply_move(self, move: tuple, player: int) -> None:
        self.board.apply_move(move)
        self.move_history.append(move_label(player, move))
        cell = find_last_move_cell(self.board, move)
        self.last_move_cell = cell
        self.animate_cell = cell if move[0] == 'drop' else None

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self) -> None:
        with self.board_output:
            clear_output(wait=True)
            html = render_board_html(
                self.board,
                last_move_cell=self.last_move_cell,
                move_history=self.move_history,
                animate_cell=self.animate_cell,
            )
            ipy_display(HTML(html))

    def _show_thinking(self, player: int) -> None:
        with self.controls_output:
            clear_output(wait=True)
            ipy_display(HTML(_THINKING_HTML.format(
                emoji=PLAYER_EMOJI[player], name=PLAYER_NAME[player],
            )))

    def _clear_controls(self) -> None:
        with self.controls_output:
            clear_output()

    # ── Human controls ────────────────────────────────────────────────────

    def _show_controls(self) -> None:
        legal = self.board.get_legal_moves()
        player = self.board.current_player
        color = '#1e4a8a' if player == 1 else '#7a5c00'
        disabled_color = '#1a1a2e'

        # Drop buttons
        drop_btns = []
        for col in range(self.board.COLS):
            m = ('drop', col)
            btn = widgets.Button(
                description=f'↓ {col}',
                tooltip=f'Drop in column {col}',
                layout=widgets.Layout(width='64px', height='38px'),
                disabled=(m not in legal),
            )
            btn.style.button_color = color if m in legal else disabled_color
            btn.style.font_weight = 'bold'
            if m in legal:
                btn.on_click(lambda _, mv=m: self._on_move(mv))
            drop_btns.append(btn)

        # Pop buttons
        pop_btns = []
        pop_legal = [m for m in legal if m[0] == 'pop']
        if pop_legal:
            for col in range(self.board.COLS):
                m = ('pop', col)
                btn = widgets.Button(
                    description=f'↑ {col}',
                    tooltip=f'Pop from column {col}',
                    layout=widgets.Layout(width='64px', height='38px'),
                    disabled=(m not in legal),
                )
                btn.style.button_color = '#6b2f1a' if m in legal else disabled_color
                btn.style.font_weight = 'bold'
                if m in legal:
                    btn.on_click(lambda _, mv=m: self._on_move(mv))
                pop_btns.append(btn)

        # Draw button
        draw_btn = None
        if ('draw', None) in legal:
            draw_btn = widgets.Button(
                description='🤝 Declare Draw',
                layout=widgets.Layout(width='180px', height='38px'),
            )
            draw_btn.style.button_color = '#3b3b4f'
            draw_btn.on_click(lambda _: self._on_move(('draw', None)))

        # Assemble
        lbl_style = 'color:#9ca3af;font-size:13px;margin:6px 0 2px;font-family:sans-serif'
        rows = [
            widgets.HTML(f'<div style="{lbl_style}">Drop a disc into a column:</div>'),
            widgets.HBox(drop_btns, layout=widgets.Layout(gap='4px')),
        ]
        if pop_btns:
            rows.append(widgets.HTML(f'<div style="{lbl_style}">Pop your disc from the bottom:</div>'))
            rows.append(widgets.HBox(pop_btns, layout=widgets.Layout(gap='4px')))
        if draw_btn:
            rows.append(draw_btn)

        with self.controls_output:
            clear_output(wait=True)
            ipy_display(widgets.VBox(rows, layout=widgets.Layout(padding='4px 0')))

    def _on_move(self, move: tuple) -> None:
        """Handle a human move button click."""
        player = self.board.current_player
        self._apply_move(move, player)
        self._clear_controls()

        if self.board.is_game_over:
            self._render()
            self.result = self.board.winner
            return

        # Next turn
        self._advance()


# ══════════════════════════════════════════════════════════════════════════════
#  Computer-vs-Computer loop (blocking, with visual updates)
# ══════════════════════════════════════════════════════════════════════════════

def _play_computer_game(
    player1: Agent,
    player2: Agent,
    move_delay: float = 0.8,
    verbose: bool = True,
) -> int:
    """Run a computer-vs-computer game with live board updates."""
    board = PopOutBoard()
    agents: dict[int, Agent] = {1: player1, 2: player2}
    history: list[str] = []
    last_cell = None

    out = widgets.Output()
    ipy_display(out)

    # Show initial board
    with out:
        ipy_display(HTML(render_board_html(board)))

    while not board.is_game_over:
        player = board.current_player
        move = agents[player](board)
        board.apply_move(move)

        history.append(move_label(player, move))
        cell = find_last_move_cell(board, move)
        last_cell = cell
        anim = cell if move[0] == 'drop' else None

        with out:
            clear_output(wait=True)
            ipy_display(HTML(render_board_html(
                board,
                last_move_cell=last_cell,
                move_history=history,
                animate_cell=anim,
            )))

        if not board.is_game_over:
            time.sleep(move_delay)

    return board.winner


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def play_game(
    player1: Agent,
    player2: Agent,
    verbose: bool = True,
    move_delay: float = 0.8,
) -> "int | PopOutGameUI":
    """Run a full PopOut game with the visual interface.

    Parameters
    ----------
    player1, player2:
        Either the string ``'human'`` (interactive buttons) or any callable
        that accepts a :class:`PopOutBoard` and returns a legal move tuple.
    verbose:
        If ``True``, the board is displayed after each move.
    move_delay:
        Seconds between computer moves in computer-vs-computer mode (default
        0.8).  For human-vs-computer, a brief 0.3s "thinking" indicator is
        shown instead.

    Returns
    -------
    int | PopOutGameUI
        For **computer-vs-computer**: the winner (``0`` draw, ``1``/``2`` winner).
        For **human games**: a :class:`PopOutGameUI` instance whose ``.result``
        attribute is set once the game finishes (``None`` while in progress).
    """
    has_human = (player1 == 'human' or player2 == 'human')

    if has_human:
        ui = PopOutGameUI(player1, player2, move_delay=0.3)
        ui.start()
        return ui
    else:
        if verbose:
            return _play_computer_game(player1, player2, move_delay=move_delay)
        else:
            # Silent mode — no display, just return result
            board = PopOutBoard()
            agents = {1: player1, 2: player2}
            while not board.is_game_over:
                move = agents[board.current_player](board)
                board.apply_move(move)
            return board.winner
