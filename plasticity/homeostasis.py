"""Homeostatic plasticity: keep plastic weights bounded and well-conditioned.

Pure Hebbian learning is unstable -- weights grow without bound. Biological
synapses counteract this with homeostatic mechanisms (e.g. synaptic scaling).
Here we implement a simple, functional reference: clip negative weights to zero
and rescale each receiver's incoming weights so they sum to at most ``target``.
"""

from __future__ import annotations

from communication.graph import InteractionGraph


def apply_synaptic_scaling(graph: InteractionGraph, target: float = 1.0) -> None:
    """Clip negatives and L1-normalise incoming weights per receiver.

    After scaling, for every receiver the sum of incoming edge weights is at
    most ``target`` (receivers already under budget are left unchanged, which
    lets weak connections stay weak rather than being inflated).
    """
    for receiver in graph.agents:
        senders = graph.neighbours(receiver)
        if not senders:
            continue
        weights = []
        for src in senders:
            w = max(0.0, graph.weight(src, receiver))
            graph.set_weight(src, receiver, w)
            weights.append(w)
        total = sum(weights)
        if total > target and total > 0.0:
            scale = target / total
            for src in senders:
                graph.set_weight(src, receiver, graph.weight(src, receiver) * scale)


__all__ = ["apply_synaptic_scaling"]
