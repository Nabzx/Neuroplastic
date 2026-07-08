"""Shared, cross-cutting primitives for Neuroplastic Collective Intelligence.

This package intentionally depends on *none* of the domain packages
(``agents``, ``communication``, ``plasticity`` ...). It provides the small
building blocks that all of them rely on:

* :class:`~core.registry.Registry` -- a lightweight name -> factory registry
  that makes the project config-driven and extensible.
* :func:`~core.seeding.set_global_seed` -- deterministic seeding across
  ``random``, ``numpy`` and ``torch``.
* shared type aliases (:data:`~core.types.AgentID` and friends).
* :func:`~core.logging_utils.get_logger` -- a consistent logger factory.
"""

from core.registry import Registry
from core.seeding import set_global_seed
from core.types import Action, AgentID, ArrayLike, Observation

__all__ = [
    "Registry",
    "set_global_seed",
    "AgentID",
    "Observation",
    "Action",
    "ArrayLike",
]
