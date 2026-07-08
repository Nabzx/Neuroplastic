"""The unit of inter-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.types import AgentID, ArrayLike


@dataclass
class Message:
    """A message emitted by one agent during a communication round.

    Attributes
    ----------
    sender:
        Id of the emitting agent.
    content:
        The learned message vector (typically a ``torch.Tensor`` /
        ``numpy.ndarray`` of length ``communication.message_dim``).
    step:
        Environment timestep the message was produced at.
    meta:
        Free-form auxiliary data (e.g. attention logits) for logging/analysis.
    """

    sender: AgentID
    content: ArrayLike
    step: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        shape = getattr(self.content, "shape", None)
        descr = f"shape={tuple(shape)}" if shape is not None else "content=<opaque>"
        return f"Message(sender={self.sender!r}, step={self.step}, {descr})"


__all__ = ["Message"]
