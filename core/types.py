"""Project-wide type aliases.

Kept deliberately loose while the interfaces stabilise. Concrete environments
and policy networks will refine these (e.g. an ``Observation`` is typically a
``numpy.ndarray`` or a nested dict of arrays, depending on the benchmark).
"""

from __future__ import annotations

from typing import Any, TypeAlias

#: Stable identifier for an agent within an episode (matches PettingZoo agent
#: ids, which are strings such as ``"agent_0"``).
AgentID: TypeAlias = str

#: A single agent's observation. The concrete type is environment-specific.
Observation: TypeAlias = Any

#: A single agent's action. The concrete type is environment-specific.
Action: TypeAlias = Any

#: Anything array-like (``numpy.ndarray``, ``torch.Tensor``, nested lists).
#: Left as ``Any`` so that importing :mod:`core` never forces a numpy/torch
#: import.
ArrayLike: TypeAlias = Any

__all__ = ["AgentID", "Observation", "Action", "ArrayLike"]
