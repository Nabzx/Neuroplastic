"""Adaptive, graph-based inter-agent communication.

Sub-modules:

* :mod:`communication.message`   -- the message data structure.
* :mod:`communication.graph`     -- a NetworkX-backed interaction graph with
  plastic edge weights.
* :mod:`communication.topology`  -- strategies that decide *who talks to whom*
  (fully-connected, k-nearest, static, adaptive/learned).
* :mod:`communication.protocol`  -- strategies that decide *how* received
  messages are aggregated (mean, attention, GNN).
* :mod:`communication.channel`   -- ties topology + protocol together and
  routes messages for one step.
* :mod:`communication.adaptive`  -- learned, weighted (attention) edge matrix
  for adaptive communication (PyTorch; imported on demand).
* :mod:`communication.statistics` -- statistics of a weighted communication graph.
"""

from communication.graph import InteractionGraph
from communication.message import Message
from communication.statistics import weight_matrix_statistics

__all__ = ["Message", "InteractionGraph", "weight_matrix_statistics"]
