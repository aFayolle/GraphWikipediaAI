"""Two-column prerequisite view: what X needs (left) and what X enables (right).

The generic renderer was the wrong tool for this. A force layout scatters
prerequisites and unlockables together, and without arrowheads the picture
cannot answer the only question it exists to answer. Here columns encode hop
distance, colour encodes direction, and every edge keeps its arrowhead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from .render import BACKGROUND, FOREGROUND  # noqa: E402

log = logging.getLogger(__name__)

PREREQ_COLOR = "#4E79A7"    # blue: what the focus needs
ENABLES_COLOR = "#2CA02C"   # green: what the focus unlocks
FOCUS_COLOR = "#FFFFFF"


def render_prerequisite_view(
    dag: nx.DiGraph,
    focus: str,
    out_path: Path,
    radius: int = 2,
    figsize: tuple[float, float] = (16.0, 11.0),
    dpi: int = 110,
) -> Path:
    """Render the ego view of ``focus`` as prerequisites-in / enables-out."""
    needed = nx.single_source_shortest_path_length(dag.reverse(copy=False), focus)
    unlocks = nx.single_source_shortest_path_length(dag, focus)

    positions: dict[str, tuple[float, float]] = {}
    for sign, distances in ((-1, needed), (1, unlocks)):
        for distance in range(1, radius + 1):
            column = sorted(
                node
                for node, value in distances.items()
                if value == distance and node != focus
            )
            if not column:
                continue
            for index, node in enumerate(column):
                offset = index - (len(column) - 1) / 2
                positions[node] = (float(sign * distance), float(offset))

    positions[focus] = (0.0, 0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    drawn = 0
    for source, target in dag.edges():
        if source not in positions or target not in positions:
            continue
        # Blue when the edge feeds the focus, green when the focus enables it.
        colour = PREREQ_COLOR if source in needed else ENABLES_COLOR
        ax.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="-|>",
                mutation_scale=8,
                color=colour,
                linewidth=0.8,
                alpha=0.6,
                shrinkA=5,
                shrinkB=5,
                zorder=1,
            )
        )
        drawn += 1

    for node, (x, y) in positions.items():
        is_focus = node == focus
        colour = FOCUS_COLOR if is_focus else (PREREQ_COLOR if node in needed else ENABLES_COLOR)
        ax.scatter(
            [x], [y],
            s=260 if is_focus else 60,
            c=colour,
            edgecolors=BACKGROUND,
            linewidths=0.7,
            zorder=3,
        )
        label = node if len(node) <= 34 else node[:32] + "…"
        text = ax.annotate(
            label,
            (x, y),
            fontsize=8.0 if is_focus else 6.6,
            color=FOREGROUND,
            ha="center",
            va="bottom" if y >= 0 else "top",
            xytext=(0, 7 if y >= 0 else -7),
            textcoords="offset points",
            zorder=4,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.4, foreground=BACKGROUND)]
        )

    ax.set_title(
        f"{focus} — what it needs, what it enables",
        color=FOREGROUND,
        fontsize=14,
        pad=16,
        loc="left",
    )
    ax.annotate(
        f"{len(needed) - 1} prerequisites (left) · {len(unlocks) - 1} enabled (right)"
        f" · {drawn} edges · radius {radius}",
        xy=(0.0, 1.005),
        xycoords="axes fraction",
        color="#9aa4b2",
        fontsize=9,
        va="bottom",
        ha="left",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=PREREQ_COLOR, lw=2, label="needed before it"),
            Line2D([0], [0], color=ENABLES_COLOR, lw=2, label="enabled by it"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.005),
        ncols=4,
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=9,
    )
    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    log.info("render: %s (%d edges drawn)", out_path, drawn)
    return out_path
