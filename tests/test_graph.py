"""Interaction graph + graph-theoretic metrics (functional pieces)."""

from itertools import permutations

import pytest

from communication.graph import InteractionGraph
from evaluation.graph_metrics import compute_graph_metrics

AGENTS = ["agent_0", "agent_1", "agent_2", "agent_3"]


def _complete_graph() -> InteractionGraph:
    g = InteractionGraph(AGENTS)
    g.set_edges(permutations(AGENTS, 2), weight=1.0)
    return g


def test_graph_construction_and_weights():
    g = _complete_graph()
    assert g.agents == AGENTS
    assert set(g.neighbours("agent_0")) == set(AGENTS) - {"agent_0"}
    g.set_weight("agent_1", "agent_0", 2.5)
    assert g.weight("agent_1", "agent_0") == 2.5


def test_unknown_agent_rejected():
    g = InteractionGraph(AGENTS)
    with pytest.raises(KeyError):
        g.add_edge("agent_0", "ghost")


def test_adjacency_matrix_shape():
    g = _complete_graph()
    mat = g.adjacency_matrix()
    assert mat.shape == (len(AGENTS), len(AGENTS))
    # complete directed graph: all off-diagonal entries set, diagonal zero
    assert mat.diagonal().sum() == 0
    assert (mat > 0).sum() == len(AGENTS) * (len(AGENTS) - 1)


def test_graph_metrics_on_complete_graph():
    g = _complete_graph()
    metrics = compute_graph_metrics(
        g, ["density", "global_clustering", "characteristic_path_length"]
    )
    assert metrics["density"] == pytest.approx(1.0)
    assert metrics["global_clustering"] == pytest.approx(1.0)
    assert metrics["characteristic_path_length"] == pytest.approx(1.0)


def test_metrics_selected_by_name():
    g = _complete_graph()
    metrics = compute_graph_metrics(g, ["density", "modularity", "degree_centralisation"])
    assert set(metrics) == {"density", "modularity", "degree_centralisation"}
