"""Draw communication graphs and edge-weight heatmaps.

The heatmap and the circular graph drawer are **matplotlib-only** (a NumPy
circular layout, no NetworkX), so they work directly on a ``[N, N]`` weight
matrix loaded from ``edge_weights.npz``. ``draw_interaction_graph`` (NetworkX) is
kept for convenience when you already hold an :class:`InteractionGraph`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from communication.graph import InteractionGraph


def draw_interaction_graph(
    graph: InteractionGraph,
    ax: Any | None = None,
    layout: str = "spring",
    seed: int = 0,
):
    """Draw ``graph`` with edge widths proportional to weight.

    Returns the matplotlib ``Axes``. Requires matplotlib (``[viz]`` extra).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "draw_interaction_graph requires matplotlib. Install viz extras: "
            "`pip install -e .[viz]`."
        ) from exc
    import networkx as nx

    g = graph.graph
    layouts = {
        "spring": lambda: nx.spring_layout(g, seed=seed),
        "circular": lambda: nx.circular_layout(g),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(g),
    }
    pos = layouts.get(layout, layouts["spring"])()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    weights = [g[u][v].get("weight", 1.0) for u, v in g.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 4.0 * (w / max_w) for w in weights] if max_w > 0 else 1.0

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#dddddd", edgecolors="#333333")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.7, arrows=True)
    ax.set_axis_off()
    return ax


def plot_edge_weight_heatmap(
    matrix: np.ndarray,
    agent_ids: Sequence[str],
    ax: Any | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> tuple[Any, Any]:
    """Heatmap of a ``[N, N]`` edge-weight matrix (rows=receiver, cols=sender).

    Returns ``(ax, image)``; pass the image to ``fig.colorbar`` to add a colour
    bar. Share ``vmin``/``vmax`` across panels to compare snapshots on one scale.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.0))
    data = np.asarray(matrix, dtype=float)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(agent_ids)))
    ax.set_yticks(range(len(agent_ids)))
    ax.set_xticklabels(agent_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(agent_ids, fontsize=6)
    ax.set_xlabel("sender")
    ax.set_ylabel("receiver")
    if title:
        ax.set_title(title)
    return ax, image


def draw_communication_graph(
    matrix: np.ndarray,
    agent_ids: Sequence[str],
    ax: Any | None = None,
    threshold: float = 1e-3,
    max_weight: float | None = None,
    title: str | None = None,
) -> Any:
    """Draw a weighted communication graph on a circular layout (matplotlib only).

    Nodes are placed on a circle; each edge ``sender -> receiver`` with weight
    above ``threshold`` is drawn with width/opacity proportional to its weight.
    Node size and colour encode *out-strength* (how much each agent is listened
    to), so hubs are visible. Pass ``max_weight`` to keep edge widths comparable
    across snapshots.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 5.0))

    data = np.asarray(matrix, dtype=float)
    n = len(agent_ids)
    angles = 2.0 * np.pi * np.arange(n) / max(1, n) + np.pi / 2.0
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    off_diagonal = ~np.eye(n, dtype=bool)
    peak = max_weight if max_weight is not None else (data[off_diagonal].max() if n > 1 else 1.0)
    peak = peak if peak and peak > 0 else 1.0

    for i in range(n):          # receiver
        for j in range(n):      # sender
            if i == j:
                continue
            weight = data[i, j]
            if weight <= threshold:
                continue
            frac = min(1.0, weight / peak)
            ax.plot(
                [pos[j, 0], pos[i, 0]], [pos[j, 1], pos[i, 1]],
                color="#555555", linewidth=0.5 + 3.5 * frac, alpha=0.12 + 0.6 * frac,
                solid_capstyle="round", zorder=1,
            )

    out_strength = data.sum(axis=0)  # influence of each sender (column sum)
    peak_strength = out_strength.max() if out_strength.size and out_strength.max() > 0 else 1.0
    sizes = 220.0 + 900.0 * (out_strength / peak_strength)
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], s=sizes, c=out_strength, cmap="viridis",
        edgecolors="black", linewidths=1.0, zorder=2,
    )
    for k, agent in enumerate(agent_ids):
        ax.annotate(str(agent), pos[k], fontsize=7, ha="center", va="center", zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax, scatter


__all__ = [
    "draw_interaction_graph",
    "plot_edge_weight_heatmap",
    "draw_communication_graph",
]
