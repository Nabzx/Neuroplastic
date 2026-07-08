"""Draw interaction graphs, with edge width encoding plastic weight."""

from __future__ import annotations

from typing import Any

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


__all__ = ["draw_interaction_graph"]
