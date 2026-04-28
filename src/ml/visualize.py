"""
Decision tree visualisation using matplotlib.

Works with any DecisionTreeID3 from src.ml.decision_tree.
No external libraries beyond matplotlib are required.

Usage
-----
    from src.ml.decision_tree import DecisionTreeID3
    from src.ml.visualize import plot_tree, save_tree_png

    tree = DecisionTreeID3().fit(X_train, y_train)
    plot_tree(tree, title="Iris Decision Tree")
    save_tree_png(tree, "outputs/iris_tree.png")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Colour palette — one per class label (auto-assigned)
_LEAF_COLOURS = [
    '#4CAF50',  # green
    '#2196F3',  # blue
    '#FF9800',  # orange
    '#9C27B0',  # purple
    '#F44336',  # red
    '#00BCD4',  # cyan
]
_INTERNAL_COLOUR = '#37474F'   # dark blue-grey
_EDGE_COLOUR     = '#90A4AE'
_TEXT_COLOUR     = '#FFFFFF'
_BG_COLOUR       = '#1E272E'


# ── Layout computation ────────────────────────────────────────────────────────

class _LayoutNode:
    """Stores position and rendering info for one node."""
    __slots__ = ('x', 'y', 'label', 'is_leaf', 'colour', 'edge_labels', 'children')

    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.label: str = ''
        self.is_leaf: bool = False
        self.colour: str = _INTERNAL_COLOUR
        self.edge_labels: dict[Any, str] = {}   # child_key -> edge label text
        self.children: list['_LayoutNode'] = []


def _build_layout(
    tree_node,
    depth: int,
    pos: list[float],
    layout_root: _LayoutNode,
    label_colour: dict,
    max_depth: int | None,
) -> _LayoutNode:
    """Recursively assign (x, y) positions using an in-order counter."""
    node = _LayoutNode()
    node.y = -depth

    if tree_node.is_leaf:
        node.is_leaf = True
        lbl = str(tree_node.label)
        # Shorten Iris-xxx → xxx
        short = lbl.replace('Iris-', '')
        node.label = short
        node.colour = label_colour.get(lbl, '#555555')
        node.x = pos[0]
        pos[0] += 1.0
        return node

    if max_depth is not None and depth >= max_depth:
        node.is_leaf = True
        node.label = '...'
        node.colour = '#555'
        node.x = pos[0]
        pos[0] += 1.0
        return node

    # Internal node label
    if tree_node.is_continuous:
        node.label = f'{tree_node.feature_name}\n<= {tree_node.threshold:.2f}'
    else:
        node.label = tree_node.feature_name

    # Process children
    children_keys = list(tree_node.children.keys())
    for key in children_keys:
        child_tree = tree_node.children[key]
        child_layout = _build_layout(
            child_tree, depth + 1, pos, layout_root, label_colour, max_depth
        )
        # Edge label
        if tree_node.is_continuous:
            edge_lbl = '<= {:.2f}'.format(tree_node.threshold) if key == 'left' else '> {:.2f}'.format(tree_node.threshold)
        else:
            edge_lbl = str(key)
        node.edge_labels[id(child_layout)] = edge_lbl
        node.children.append(child_layout)

    # Centre this node over its children
    if node.children:
        node.x = sum(c.x for c in node.children) / len(node.children)
    else:
        node.x = pos[0]
        pos[0] += 1.0

    return node


def _collect_all(node: _LayoutNode) -> list[_LayoutNode]:
    result = [node]
    for c in node.children:
        result.extend(_collect_all(c))
    return result


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_node(ax, node: _LayoutNode, node_w: float = 0.8, node_h: float = 0.45) -> None:
    x, y = node.x, node.y
    bbox = FancyBboxPatch(
        (x - node_w / 2, y - node_h / 2),
        node_w, node_h,
        boxstyle='round,pad=0.05',
        linewidth=1.2,
        edgecolor='#B0BEC5',
        facecolor=node.colour,
        zorder=3,
    )
    ax.add_patch(bbox)
    ax.text(
        x, y, node.label,
        ha='center', va='center',
        fontsize=7.5, color=_TEXT_COLOUR,
        fontweight='bold',
        wrap=True,
        zorder=4,
    )


def _draw_edges(ax, node: _LayoutNode, node_h: float = 0.45) -> None:
    for child in node.children:
        ax.plot(
            [node.x, child.x],
            [node.y - node_h / 2, child.y + node_h / 2],
            color=_EDGE_COLOUR, lw=1.2, zorder=1,
        )
        # Edge label midpoint
        mx = (node.x + child.x) / 2
        my = (node.y + child.y) / 2
        edge_lbl = node.edge_labels.get(id(child), '')
        ax.text(
            mx, my, edge_lbl,
            ha='center', va='center',
            fontsize=6.5, color='#CFD8DC',
            bbox=dict(facecolor=_BG_COLOUR, edgecolor='none', pad=1.5),
            zorder=2,
        )
        _draw_edges(ax, child, node_h)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_tree(
    tree,
    title: str = 'Decision Tree',
    max_display_depth: int | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a DecisionTreeID3 as a matplotlib figure.

    Parameters
    ----------
    tree:
        A fitted ``DecisionTreeID3`` instance.
    title:
        Figure title.
    max_display_depth:
        Maximum depth to render. Deeper nodes are collapsed to '...'.
    figsize:
        Override the auto-computed figure size as (width, height) in inches.
    show:
        If True, call ``plt.show()`` at the end.
    save_path:
        If given, save the figure to this path (e.g. 'outputs/tree.png').

    Returns
    -------
    matplotlib Figure
    """
    if tree.root is None:
        raise ValueError("Tree has not been fitted yet (root is None).")

    # Collect unique leaf labels for colour mapping
    all_node_data = []

    def _collect_labels(n):
        if n.is_leaf:
            all_node_data.append(str(n.label))
        else:
            for c in n.children.values():
                _collect_labels(c)

    _collect_labels(tree.root)
    unique_labels = sorted(set(all_node_data))
    label_colour = {
        lbl: _LEAF_COLOURS[i % len(_LEAF_COLOURS)]
        for i, lbl in enumerate(unique_labels)
    }

    # Build layout
    pos = [0.0]
    layout_root = _build_layout(
        tree.root, depth=0, pos=pos,
        layout_root=None, label_colour=label_colour,
        max_depth=max_display_depth,
    )

    all_nodes = _collect_all(layout_root)
    xs = [n.x for n in all_nodes]
    ys = [n.y for n in all_nodes]

    width  = max(xs) - min(xs) + 2.5
    height = max(ys) - min(ys) + 2.0

    if figsize is None:
        figsize = (max(8.0, width * 1.2), max(5.0, abs(height) * 1.4))

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG_COLOUR)
    ax.set_facecolor(_BG_COLOUR)
    ax.axis('off')

    # Draw edges first, then nodes (nodes on top)
    _draw_edges(ax, layout_root)
    for n in all_nodes:
        _draw_node(ax, n)

    # Legend
    legend_patches = [
        mpatches.Patch(color=label_colour[lbl], label=lbl.replace('Iris-', ''))
        for lbl in unique_labels
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=8,
        framealpha=0.3,
        labelcolor='white',
    )

    ax.set_title(title, color='white', fontsize=13, pad=10)
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=_BG_COLOUR)
        print(f'Tree saved -> {save_path}')

    if show:
        plt.show()

    return fig


def save_tree_png(tree, path: str, **kwargs) -> plt.Figure:
    """Convenience wrapper: save tree to PNG without showing it."""
    return plot_tree(tree, save_path=path, show=False, **kwargs)
