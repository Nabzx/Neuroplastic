"""Graph-theoretic measures of the interaction topology.

These operate directly on an :class:`~communication.graph.InteractionGraph` and
are **functional** -- they can be run today on any graph (e.g. a synthetic or
recorded topology). Each metric is registered under ``GRAPH_METRICS`` so the
evaluator can compute a config-selected subset by name.

Directed graphs are projected to undirected for measures that assume it
(clustering, path length, modularity); this is stated per-metric.
"""

from __future__ import annotations

import networkx as nx

from communication.graph import InteractionGraph
from core.registry import Registry

GRAPH_METRICS: Registry = Registry("graph_metric")


@GRAPH_METRICS.register("density")
def density(graph: InteractionGraph) -> float:
    """Fraction of possible directed edges that are present."""
    return float(nx.density(graph.graph))


@GRAPH_METRICS.register("global_clustering")
def global_clustering(graph: InteractionGraph) -> float:
    """Average clustering coefficient (undirected projection)."""
    g = graph.graph.to_undirected()
    if g.number_of_nodes() == 0:
        return float("nan")
    return float(nx.average_clustering(g))


@GRAPH_METRICS.register("characteristic_path_length")
def characteristic_path_length(graph: InteractionGraph) -> float:
    """Mean shortest-path length over the largest connected component.

    Falls back to the largest weakly-connected component when the graph is not
    connected, so the measure is always defined.
    """
    g = graph.graph.to_undirected()
    if g.number_of_nodes() < 2:
        return float("nan")
    if not nx.is_connected(g):
        largest = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest)
    if g.number_of_nodes() < 2:
        return float("nan")
    return float(nx.average_shortest_path_length(g))


@GRAPH_METRICS.register("degree_centralisation")
def degree_centralisation(graph: InteractionGraph) -> float:
    """Freeman degree centralisation of the (undirected) graph in ``[0, 1]``.

    ``0`` = every node equally connected; ``1`` = star topology.
    """
    g = graph.graph.to_undirected()
    n = g.number_of_nodes()
    if n < 3:
        return float("nan")
    degrees = [d for _, d in g.degree()]
    d_max = max(degrees)
    numerator = sum(d_max - d for d in degrees)
    denominator = (n - 1) * (n - 2)
    return float(numerator / denominator) if denominator else float("nan")


@GRAPH_METRICS.register("modularity")
def modularity(graph: InteractionGraph) -> float:
    """Greedy-community modularity (undirected projection).

    A proxy for functional-module structure in the communication graph: higher
    modularity indicates clearer sub-groups of densely-communicating agents.
    """
    g = graph.graph.to_undirected()
    if g.number_of_edges() == 0:
        return float("nan")
    communities = nx.community.greedy_modularity_communities(g)
    return float(nx.community.modularity(g, communities))


def compute_graph_metrics(graph: InteractionGraph, names: list[str]) -> dict[str, float]:
    """Compute the named graph metrics for ``graph``."""
    return {name: GRAPH_METRICS.get(name)(graph) for name in names}


__all__ = [
    "GRAPH_METRICS",
    "density",
    "global_clustering",
    "characteristic_path_length",
    "degree_centralisation",
    "modularity",
    "compute_graph_metrics",
]
