"""Two-column prerequisite view: what X needs (left) and what X enables (right).

The generic renderer was the wrong tool for this. A force layout scatters
prerequisites and unlockables together, and without arrowheads the picture
cannot answer the only question it exists to answer. Here columns encode hop
distance, colour encodes direction, and every edge keeps its arrowhead.
"""

from __future__ import annotations

import logging
import math
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


# --------------------------------------------------------------------------- #
# Radial ego view
# --------------------------------------------------------------------------- #

def render_ego(
    dag: nx.DiGraph,
    focus: str,
    out_path: Path,
    radius: int = 2,
    figsize: tuple[float, float] = (14.0, 10.0),
    dpi: int = 110,
) -> Path:
    """Concentric rings: hop distance from ``focus`` decides the ring.

    Answers "which notions sit within two hops of Topology" in a form you can
    read at a glance, instead of a force layout that hides the distance.
    """
    distances = nx.single_source_shortest_path_length(
        dag.to_undirected(), focus, cutoff=radius
    )
    positions: dict[str, tuple[float, float]] = {focus: (0.0, 0.0)}
    by_distance: dict[int, list[str]] = {}
    for node, distance in distances.items():
        if node != focus:
            by_distance.setdefault(distance, []).append(node)

    for distance, nodes in sorted(by_distance.items()):
        nodes.sort()
        for index, node in enumerate(nodes):
            angle = 2 * math.pi * index / max(1, len(nodes))
            positions[node] = (distance * math.cos(angle), distance * math.sin(angle))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for source, target in dag.edges():
        if source in positions and target in positions:
            ax.plot(
                [positions[source][0], positions[target][0]],
                [positions[source][1], positions[target][1]],
                color="#8899aa",
                linewidth=0.5,
                alpha=0.35,
                zorder=1,
            )

    palette = {1: PREREQ_COLOR, 2: "#EDC948", 3: "#B07AA1"}
    for node, (x, y) in positions.items():
        if node == focus:
            colour, size = FOCUS_COLOR, 260
        else:
            distance = distances.get(node, 0)
            colour, size = palette.get(distance, "#76B7B2"), 70
        ax.scatter([x], [y], s=size, c=colour, edgecolors=BACKGROUND, linewidths=0.6, zorder=3)
        label = node if len(node) <= 30 else node[:28] + "…"
        text = ax.annotate(
            label, (x, y), fontsize=6.8, color=FOREGROUND, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points", zorder=4,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground=BACKGROUND)])

    ax.set_title(
        f"{focus} — everything within {radius} hops",
        color=FOREGROUND, fontsize=14, pad=16, loc="left",
    )
    counts = ", ".join(f"{d} hop(s): {len(n)}" for d, n in sorted(by_distance.items()))
    ax.annotate(
        counts or "no neighbours", xy=(0.0, 1.005), xycoords="axes fraction",
        color="#9aa4b2", fontsize=9, va="bottom", ha="left",
    )
    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    log.info("render: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Chain view
# --------------------------------------------------------------------------- #

def render_chain(
    chain: list[str],
    out_path: Path,
    kind: str = "directed",
    figsize: tuple[float, float] = (16.0, 3.4),
    dpi: int = 110,
) -> Path:
    """One row, left to right: the prerequisite chain as an ordered sequence."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for index in range(len(chain) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (index, 0), (index + 1, 0),
                arrowstyle="-|>", mutation_scale=18,
                color=PREREQ_COLOR if kind == "directed" else "#EDC948",
                linewidth=2.0, shrinkA=18, shrinkB=18, zorder=1,
            )
        )

    for index, node in enumerate(chain):
        ax.scatter([index], [0], s=180, c=FOCUS_COLOR if index in (0, len(chain) - 1) else PREREQ_COLOR,
                   edgecolors=BACKGROUND, linewidths=0.8, zorder=3)
        ax.annotate(
            node, (index, 0), fontsize=8.5, color=FOREGROUND, ha="center", va="bottom",
            xytext=(0, 12), textcoords="offset points", rotation=18, zorder=4,
        )
        ax.annotate(
            str(index + 1), (index, 0), fontsize=7, color="#9aa4b2", ha="center",
            va="top", xytext=(0, -10), textcoords="offset points", zorder=4,
        )

    ax.set_title(
        f"{chain[0]} → {chain[-1]}" if chain else "no path",
        color=FOREGROUND, fontsize=13, pad=18, loc="left",
    )
    ax.annotate(
        f"{len(chain)} steps · {kind} path",
        xy=(0.0, 1.01), xycoords="axes fraction", color="#9aa4b2", fontsize=9,
        va="bottom", ha="left",
    )
    ax.set_xlim(-0.6, max(1, len(chain)) - 0.4)
    ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    log.info("render: %s", out_path)
    return out_path
