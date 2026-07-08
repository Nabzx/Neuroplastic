"""Hebbian-inspired plasticity rules.

This module provides a **first-pass reference** Hebbian rule that is functional
(NumPy-only) so the plasticity -> graph -> protocol pipeline can be exercised
end-to-end with synthetic messages. The richer variants intended for the study
(vector outer-product Hebbian, three-factor / neuromodulated updates, and
Oja-normalised learning) are specified in ``docs/experiment_plan.md``; the
non-reference ones are left as clearly-marked placeholders.

Reference rule ("correlation" Hebbian on scalar co-activity)::

    dw_ij = lr * pre_i * post_j - decay * w_ij

where ``pre_i`` is the activity of sender *i*'s outgoing message and ``post_j``
is the activity of receiver *j*'s aggregated context, both summarised here as
the L2 norm of the corresponding vector. Homeostatic scaling (optional) then
keeps each receiver's incoming weights bounded.
"""

from __future__ import annotations

from typing import Any, Mapping

from communication.graph import InteractionGraph
from communication.message import Message
from core.types import AgentID
from plasticity.base import PLASTICITY_REGISTRY, PlasticityRule
from plasticity.homeostasis import apply_synaptic_scaling
from plasticity.modulation import modulation_factor


def _activity(vector: Any) -> float:
    """Summarise a message/context vector as a non-negative scalar activity."""
    if vector is None:
        return 0.0
    import numpy as np

    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


@PLASTICITY_REGISTRY.register("hebbian")
class HebbianRule(PlasticityRule):
    """Reference correlation-Hebbian update on the interaction graph."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.homeostasis = getattr(config, "homeostasis", True)

    def update(
        self,
        graph: InteractionGraph,
        messages: Mapping[AgentID, Message],
        contexts: Mapping[AgentID, Any],
        step: int,
    ) -> None:
        gate = modulation_factor(self.config, step=step)
        for src, dst in list(graph.graph.edges()):
            msg = messages.get(src)
            pre = _activity(msg.content if msg is not None else None)
            post = _activity(contexts.get(dst))
            w = graph.weight(src, dst)
            dw = gate * self.learning_rate * pre * post - self.decay * w
            graph.set_weight(src, dst, w + dw)

        if self.homeostasis:
            apply_synaptic_scaling(graph)


@PLASTICITY_REGISTRY.register("oja")
class OjaRule(PlasticityRule):
    """Oja's rule: Hebbian learning with multiplicative normalisation.

    Placeholder. Oja's rule stabilises weight growth without an explicit decay
    term; the vectorised form is specified in docs/experiment_plan.md.
    """

    def update(self, graph, messages, contexts, step) -> None:  # pragma: no cover - deferred
        raise NotImplementedError(
            "OjaRule.update is a placeholder; the normalised update is specified "
            "in docs/experiment_plan.md."
        )


__all__ = ["HebbianRule", "OjaRule"]
