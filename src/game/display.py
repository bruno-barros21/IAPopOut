"""
Board rendering utilities for PopOut.

Two rendering modes:

- **Rich HTML** (default): A visually styled board rendered in Jupyter
  notebooks via ``IPython.display.HTML``, featuring gradient colours,
  3D-shaded discs, drop animations, and last-move highlights.
- **Plain text**: A minimal ASCII representation kept for logging/debugging.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from IPython.display import HTML, display as ipy_display

if TYPE_CHECKING:
    from .popout_board import PopOutBoard

# ── Constants ────────────────────────────────────────────────────────────────

_SYMBOLS = {0: '.', 1: 'X', 2: 'O'}
_TEXT_LABELS = {1: 'X  (Player 1)', 2: 'O  (Player 2)'}
_CSS_CLASS = {0: 'empty', 1: 'red', 2: 'yellow'}
PLAYER_NAME = {1: 'Red', 2: 'Yellow'}
PLAYER_EMOJI = {1: '🔴', 2: '🟡'}


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_last_move_cell(board: "PopOutBoard", move: tuple) -> tuple[int, int] | None:
    """Return the (row, col) most affected by *move* (called after apply)."""
    move_type, col = move
    if col is None:
        return None
    if move_type == 'drop':
        for r in range(board.ROWS):
            if board.board[r, col] != 0:
                return (r, col)
    elif move_type == 'pop':
        return (board.ROWS - 1, col)
    return None


def move_label(player: int, move: tuple) -> str:
    """Compact human-readable move description."""
    mt, col = move
    arrow = {'drop': '↓', 'pop': '↑', 'draw': '🤝'}.get(mt, '?')
    return f"{PLAYER_EMOJI[player]}{arrow}{col if col is not None else ''}"


# ── Rich HTML rendering ─────────────────────────────────────────────────────

def render_board_html(
    board: "PopOutBoard",
    last_move_cell: tuple[int, int] | None = None,
    move_history: list[str] | None = None,
    animate_cell: tuple[int, int] | None = None,
) -> str:
    """Return a self-contained HTML string that renders the board.

    Parameters
    ----------
    board : PopOutBoard
    last_move_cell : (row, col) to highlight with a glow effect.
    move_history : compact move descriptions shown below the board.
    animate_cell : (row, col) that should play the drop-in animation.
    """
    uid = f"po{uuid.uuid4().hex[:8]}"
    b = board.board
    rows, cols = b.shape

    # --- Cells ---
    cells = []
    for r in range(rows):
        for c in range(cols):
            val = int(b[r, c])
            cls = _CSS_CLASS[val]
            extras = []
            if last_move_cell == (r, c) and val != 0:
                extras.append('last-move')
            if animate_cell == (r, c):
                extras.append('drop-anim')
            disc_cls = f"disc {cls} {' '.join(extras)}".strip()
            cells.append(f'<div class="cell"><div class="{disc_cls}"></div></div>')

    cells_html = '\n'.join(cells)
    col_labels = ''.join(f'<span class="col-lbl">{c}</span>' for c in range(cols))

    # --- Status ---
    if board.is_game_over:
        if board.winner == 0:
            status_html = '<span class="draw-text">🤝 Game Drawn</span>'
        else:
            w = board.winner
            status_html = (
                f'<span class="winner-text">'
                f'{PLAYER_EMOJI[w]} {PLAYER_NAME[w]} Wins!</span>'
            )
    else:
        p = board.current_player
        status_html = (
            f'<span class="turn-dot {_CSS_CLASS[p]}"></span>'
            f'<span class="turn-text">{PLAYER_NAME[p]}\'s Turn</span>'
        )

    # --- History ---
    history_html = ''
    if move_history:
        items = '  '.join(move_history[-14:])
        history_html = f'<div class="history">{items}</div>'

    return f"""<style>
#{uid}{{--cell:64px;--disc:56px;--gap:6px;--pad:14px;
font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
background:linear-gradient(145deg,#0f0f1a,#1a1a2e 50%,#16213e);
border-radius:20px;padding:28px 32px;display:inline-block;
box-shadow:0 12px 48px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.05);
color:#e0e0e0;user-select:none}}
#{uid} .hdr{{text-align:center;font-size:20px;font-weight:800;
letter-spacing:3px;text-transform:uppercase;margin-bottom:18px;
background:linear-gradient(90deg,#4a9eff,#a855f7,#ec4899);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
#{uid} .clbls{{display:grid;grid-template-columns:repeat(7,var(--cell));
gap:var(--gap);padding:0 var(--pad);margin-bottom:4px}}
#{uid} .col-lbl{{text-align:center;font-size:12px;font-weight:700;color:#4b5563;letter-spacing:1px}}
#{uid} .board{{display:grid;grid-template-columns:repeat(7,var(--cell));
grid-template-rows:repeat(6,var(--cell));gap:var(--gap);
background:linear-gradient(180deg,#1e4a8a,#143a6e 40%,#0d2b5e);
border-radius:14px;padding:var(--pad);
box-shadow:inset 0 2px 12px rgba(0,0,0,.4),0 6px 24px rgba(30,74,138,.25)}}
#{uid} .cell{{width:var(--cell);height:var(--cell);border-radius:50%;
background:#0a1628;display:flex;align-items:center;justify-content:center;
box-shadow:inset 0 3px 8px rgba(0,0,0,.6),inset 0 -1px 3px rgba(255,255,255,.05)}}
#{uid} .disc{{width:var(--disc);height:var(--disc);border-radius:50%}}
#{uid} .disc.red{{background:radial-gradient(circle at 38% 32%,#ff8a8a,#e63946 45%,#c1121f 80%,#9b0e1a);
box-shadow:0 3px 12px rgba(230,57,70,.6),inset 0 -3px 6px rgba(0,0,0,.25),inset 0 2px 4px rgba(255,255,255,.15)}}
#{uid} .disc.yellow{{background:radial-gradient(circle at 38% 32%,#fff176,#ffd700 45%,#e6a800 80%,#cc9400);
box-shadow:0 3px 12px rgba(255,215,0,.5),inset 0 -3px 6px rgba(0,0,0,.2),inset 0 2px 4px rgba(255,255,255,.2)}}
#{uid} .disc.last-move{{animation:{uid}G 1.5s ease-in-out infinite alternate}}
#{uid} .disc.drop-anim{{animation:{uid}D .45s cubic-bezier(.34,1.56,.64,1)}}
@keyframes {uid}G{{0%{{filter:brightness(1)}}100%{{filter:brightness(1.3)}}}}
@keyframes {uid}D{{0%{{transform:translateY(-350px);opacity:.3}}50%{{opacity:1}}75%{{transform:translateY(6px)}}100%{{transform:translateY(0);opacity:1}}}}
#{uid} .status{{margin-top:16px;padding:12px 20px;background:rgba(255,255,255,.04);
border-radius:12px;display:flex;align-items:center;justify-content:center;gap:10px;
font-size:15px;font-weight:500}}
#{uid} .turn-dot{{width:14px;height:14px;border-radius:50%;display:inline-block}}
#{uid} .turn-dot.red{{background:#e63946;box-shadow:0 0 10px rgba(230,57,70,.6)}}
#{uid} .turn-dot.yellow{{background:#ffd700;box-shadow:0 0 10px rgba(255,215,0,.6)}}
#{uid} .winner-text{{font-size:20px;font-weight:800;
background:linear-gradient(90deg,#4ade80,#22c55e);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
animation:{uid}W 1s ease-in-out infinite alternate}}
@keyframes {uid}W{{0%{{transform:scale(1)}}100%{{transform:scale(1.06)}}}}
#{uid} .draw-text{{font-size:18px;font-weight:600;color:#94a3b8}}
#{uid} .history{{margin-top:10px;font-size:12px;color:#6b7280;text-align:center;line-height:1.8}}
</style>
<div id="{uid}">
<div class="hdr">PopOut</div>
<div class="clbls">{col_labels}</div>
<div class="board">{cells_html}</div>
<div class="status">{status_html}</div>
{history_html}
</div>"""


def display_board(
    board: "PopOutBoard",
    last_move_cell: tuple[int, int] | None = None,
    move_history: list[str] | None = None,
    animate_cell: tuple[int, int] | None = None,
) -> None:
    """Render the board as rich HTML in the current notebook cell."""
    ipy_display(HTML(render_board_html(board, last_move_cell, move_history, animate_cell)))


# ── Plain text rendering (debug/logging) ────────────────────────────────────

def display_board_text(board: "PopOutBoard") -> None:
    """Print the board as plain ASCII."""
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
            print(f'  Result : {_TEXT_LABELS[board.winner]} wins!')
    else:
        print(f'  Turn   : {_TEXT_LABELS[board.current_player]}')
    print()
