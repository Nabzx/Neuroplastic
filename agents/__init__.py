"""Agents: policy networks and the interface the training loop drives.

The public surface is intentionally small: a :class:`~agents.base.BaseAgent`
interface, an ``AGENT_REGISTRY`` for config-driven construction, and a
:func:`~agents.base.make_agent` factory. Concrete architectures live in their
own modules and self-register.
"""

from agents.base import AGENT_REGISTRY, BaseAgent, make_agent

__all__ = ["BaseAgent", "AGENT_REGISTRY", "make_agent"]
