"""A NetworkX-backed interaction graph with plastic edge weights.

The :class:`InteractionGraph` is the shared state that topology, protocol and
plasticity all operate on:

* **topology** decides which directed edges *exist* in a given step,
* **protocol** reads edge weights to aggregate incoming messages,
* **plasticity** updates edge weights from pre/post communication activity.

Edge weight ``w[i, j]`` is the (plastic) strength of the connection from sender
``i`` to receiver ``j``. This class is fully functional; it carries no learned
parameters and therefore has no torch dependency.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx

from core.types import AgentID


class InteractionGraph:
    """A directed, weighted communication graph over a fixed set of agents."""

    def __init__(self, agent_ids: Sequence[AgentID]) -> None:
        self._agents: list[AgentID] = list(agent_ids)
        self._index = {a: i for i, a in enumerate(self._agents)}
        self._g = nx.DiGraph()
        self._g.add_nodes_from(self._agents)

    # -- construction from a weight matrix --------------------------------
    @classmethod
    def from_weight_matrix(
        cls,
        agent_ids: Sequence[AgentID],
        matrix,
        threshold: float = 0.0,
        receiver_by_sender: bool = True,
    ) -> "InteractionGraph":
        """Build a weighted graph from a dense ``[N, N]`` matrix.

        By default ``matrix[i, j]`` is read as the weight from sender ``j`` to
        receiver ``i`` (the convention used by the communication layer), and an
        edge ``j -> i`` is added when the weight exceeds ``threshold``. Self-loops
        are skipped.
        """
        import numpy as np

        graph = cls(agent_ids)
        mat = np.asarray(matrix, dtype=float)
        n = len(graph._agents)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                weight = mat[i, j]
                if weight <= threshold:
                    continue
                src, dst = (j, i) if receiver_by_sender else (i, j)
                graph.add_edge(graph._agents[src], graph._agents[dst], weight=float(weight))
        return graph

    # -- basic structure ---------------------------------------------------
    @property
    def agents(self) -> list[AgentID]:
        return list(self._agents)

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying (mutable) NetworkX graph."""
        return self._g

    def set_edges(self, edges: Iterable[tuple[AgentID, AgentID]], weight: float = 1.0) -> None:
        """Replace all edges with ``edges`` (each given default ``weight``)."""
        self._g.remove_edges_from(list(self._g.edges()))
        for src, dst in edges:
            self._validate(src, dst)
            self._g.add_edge(src, dst, weight=float(weight))

    def add_edge(self, src: AgentID, dst: AgentID, weight: float = 1.0) -> None:
        self._validate(src, dst)
        self._g.add_edge(src, dst, weight=float(weight))

    def neighbours(self, dst: AgentID) -> list[AgentID]:
        """Return the senders that have an incoming edge to ``dst``."""
        return list(self._g.predecessors(dst))

    # -- weights -----------------------------------------------------------
    def weight(self, src: AgentID, dst: AgentID) -> float:
        return float(self._g[src][dst]["weight"])

    def set_weight(self, src: AgentID, dst: AgentID, value: float) -> None:
        self._validate(src, dst)
        if self._g.has_edge(src, dst):
            self._g[src][dst]["weight"] = float(value)
        else:
            self._g.add_edge(src, dst, weight=float(value))

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the underlying weighted directed graph.

        The copy is safe to hand to any NetworkX algorithm or exporter (e.g.
        ``nx.write_graphml``, ``nx.pagerank``) without affecting this object.
        """
        return self._g.copy()

    def adjacency_matrix(self):
        """Return the dense weighted adjacency matrix as a NumPy array.

        Row = sender, column = receiver, ordered by :pyattr:`agents`.
        """
        import numpy as np

        n = len(self._agents)
        mat = np.zeros((n, n), dtype=float)
        for src, dst, data in self._g.edges(data=True):
            mat[self._index[src], self._index[dst]] = data.get("weight", 1.0)
        return mat

    # -- helpers -----------------------------------------------------------
    def _validate(self, *ids: AgentID) -> None:
        for a in ids:
            if a not in self._index:
                raise KeyError(f"Unknown agent id {a!r}; known: {self._agents}")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"InteractionGraph(n_agents={len(self._agents)}, "
            f"n_edges={self._g.number_of_edges()})"
        )


__all__ = ["InteractionGraph"]
