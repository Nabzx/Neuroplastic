"""Training-time helpers: device resolution, tensor packing, comm masks.

All heavy imports (torch/numpy) are performed inside the functions so this module
can be imported without the RL stack present.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.types import AgentID


def resolve_device(name: str) -> str:
    """Resolve ``"auto" | "cpu" | "cuda"`` to a concrete device string."""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def stack_by_agent(
    values: Mapping[AgentID, Any],
    order: Sequence[AgentID],
    device: str,
    dtype: str = "float32",
) -> Any:
    """Stack a per-agent mapping into a tensor in a fixed agent order.

    Returns a tensor of shape ``[N, ...]`` (rows follow ``order``).
    """
    import numpy as np
    import torch

    arrays = [np.asarray(values[agent]) for agent in order]
    stacked = np.stack(arrays, axis=0)
    torch_dtype = getattr(torch, dtype)
    return torch.as_tensor(stacked, dtype=torch_dtype, device=device)


def build_communication_mask(
    agent_ids: Sequence[AgentID],
    comm_config: Any,
    device: str,
) -> Any:
    """Build a fixed, row-normalised communication mask from the topology.

    Returns a ``[N, N]`` float tensor where entry ``[i, j]`` is the weight from
    sender ``j`` to receiver ``i`` (rows sum to 1 over each receiver's
    neighbours), or ``None`` when communication is disabled.

    The mask is derived from the existing communication abstractions
    (:class:`~communication.graph.InteractionGraph` + the configured topology),
    so adding a new fixed topology automatically yields a usable mask.
    """
    if not getattr(comm_config, "enabled", True):
        return None

    import numpy as np
    import torch

    from communication.graph import InteractionGraph
    from communication.topology import make_topology

    graph = InteractionGraph(list(agent_ids))
    topology = make_topology(comm_config)
    topology.reset()
    topology.update(graph, {}, 0)  # fixed topologies ignore messages/step

    adjacency = graph.adjacency_matrix()          # [N, N], sender x receiver
    receiver_by_sender = adjacency.T.copy()        # [N, N], receiver x sender
    row_sums = receiver_by_sender.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0                # avoid div-by-zero (isolated node)
    receiver_by_sender /= row_sums
    return torch.as_tensor(receiver_by_sender, dtype=torch.float32, device=device)


__all__ = ["resolve_device", "stack_by_agent", "build_communication_mask"]
