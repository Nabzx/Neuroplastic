"""PyTorch policy network for the learning baselines.

A parameter-shared actor-critic with an **optional single-round communication
vector**. The same module powers all three baseline communication settings; only
the adjacency passed to :meth:`PolicyNetwork.forward` differs:

* ``adjacency=None``                -> no communication (zero context),
* dense row-normalised adjacency    -> fully-connected mean messaging,
* sparse row-normalised adjacency   -> fixed sparse-graph mean messaging.

Design notes
------------
* **Separate actor and critic trunks.** They do *not* share an encoder/core.
  Sharing lets the (large-magnitude) value loss dominate the shared
  representation and starves the policy gradient -- so, like CleanRL's PPO, we
  keep them independent.
* **Messages are learned by the policy.** The message head sits on the actor's
  encoding, and the critic consumes the aggregated context **detached**, so value
  gradients never reshape the communication -- messages are shaped purely by the
  policy objective.
* **Identical architecture across settings.** No-communication simply zeroes the
  context, so parameter counts match and the baselines stay comparable.

Forward pass (vectorised over the ``N`` agents)::

    a_emb    = ActorEncoder(obs)
    message  = MessageHead(a_emb)
    context  = adjacency @ message          # mean of neighbours' messages
    logits   = ActorHead(ActorCore([a_emb; context]))
    value    = CriticHead(CriticCore([CriticEncoder(obs); context.detach()]))
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orthogonal(layer: nn.Linear, gain: float = 2.0**0.5) -> nn.Linear:
    """Orthogonal weight init + zero bias (standard for stable PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ObservationEncoder(nn.Module):
    """Encode a flat observation into a latent embedding."""

    def __init__(self, obs_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal(nn.Linear(obs_dim, embedding_dim)),
            nn.Tanh(),
            _orthogonal(nn.Linear(embedding_dim, embedding_dim)),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MessageHead(nn.Module):
    """Map an observation embedding to an outgoing message vector."""

    def __init__(self, in_dim: int, message_dim: int) -> None:
        super().__init__()
        self.proj = _orthogonal(nn.Linear(in_dim, message_dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.proj(embedding)


def _trunk(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """A small two-layer Tanh MLP trunk."""
    return nn.Sequential(
        _orthogonal(nn.Linear(in_dim, hidden_dim)),
        nn.Tanh(),
        _orthogonal(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
    )


class PolicyNetwork(nn.Module):
    """Parameter-shared communicating actor-critic (feedforward).

    A single instance is applied to every agent (parameter sharing) -- the
    standard baseline for homogeneous cooperative agents.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        obs_embedding_dim: int = 64,
        message_dim: int = 16,
        recurrent: bool = False,
        comm_mode: str = "none",
        attention_dim: int = 32,
        num_agents: int | None = None,
        structural_mask: torch.Tensor | None = None,
        plasticity_config: object | None = None,
    ) -> None:
        super().__init__()
        self.message_dim = message_dim
        self.recurrent = recurrent
        self.comm_mode = comm_mode  # "none" | "fixed" | "adaptive" | "plastic"
        self._last_weights: torch.Tensor | None = None

        # Actor (with communication) -- also produces the messages.
        self.actor_encoder = ObservationEncoder(obs_dim, obs_embedding_dim)
        self.message_head = MessageHead(obs_embedding_dim, message_dim)
        self.actor_core = _trunk(obs_embedding_dim + message_dim, hidden_dim)
        self.actor_head = _orthogonal(nn.Linear(hidden_dim, action_dim), gain=0.01)

        # Critic (separate trunk; consumes the context detached).
        self.critic_encoder = ObservationEncoder(obs_dim, obs_embedding_dim)
        self.critic_core = _trunk(obs_embedding_dim + message_dim, hidden_dim)
        self.critic_head = _orthogonal(nn.Linear(hidden_dim, 1), gain=1.0)

        # Weighted-communication submodules -- built only for the chosen mode.
        self.adaptive = None
        self.plastic = None
        if comm_mode == "adaptive":
            from communication.adaptive import AdaptiveCommunication

            self.adaptive = AdaptiveCommunication(obs_embedding_dim, message_dim, attention_dim)
        elif comm_mode == "plastic":
            from plasticity.plastic_edges import PlasticEdges

            self.plastic = PlasticEdges(num_agents, structural_mask, plasticity_config)

    def _communicate(
        self, embeddings: torch.Tensor, messages: torch.Tensor, adjacency: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(context, weights)`` for the configured communication mode.

        ``adjacency`` is a row-normalised uniform mask for ``"fixed"`` mode and a
        binary structural mask (candidate edges) for ``"adaptive"`` mode.
        """
        if self.comm_mode == "adaptive":
            return self.adaptive(embeddings, messages, adjacency)
        if self.comm_mode == "plastic":
            # Aggregation weights come from the persistent Hebbian edge matrix.
            return self.plastic(messages), None
        if self.comm_mode == "fixed" and adjacency is not None:
            # context_i = sum_j adjacency[i, j] * messages_j  (uniform mean)
            return torch.einsum("ij,bjm->bim", adjacency, messages), None
        context = torch.zeros(
            *embeddings.shape[:-1], self.message_dim,
            device=embeddings.device, dtype=embeddings.dtype,
        )
        return context, None

    def forward(
        self,
        obs: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, value, messages)`` for ``obs`` of shape ``[B, N, obs_dim]``.

        ``adjacency`` is the communication mask (row-normalised for ``"fixed"``,
        binary structural for ``"adaptive"``), or ``None`` for no communication.
        """
        if self.recurrent:  # pragma: no cover - reserved for a later milestone
            raise NotImplementedError(
                "Recurrent core is a future extension; the baseline network is "
                "feedforward. Set agent.recurrent=false."
            )

        a_emb = self.actor_encoder(obs)                       # [B, N, E]
        messages = self.message_head(a_emb)                   # [B, N, M]
        context, weights = self._communicate(a_emb, messages, adjacency)
        self._last_weights = None if weights is None else weights.detach()

        logits = self.actor_head(self.actor_core(torch.cat([a_emb, context], dim=-1)))

        c_emb = self.critic_encoder(obs)
        value = self.critic_head(
            self.critic_core(torch.cat([c_emb, context.detach()], dim=-1))
        ).squeeze(-1)                                          # [B, N]

        return logits, value, messages

    @torch.no_grad()
    def messages_only(self, obs: torch.Tensor) -> torch.Tensor:
        """Return just the outgoing messages ``[B, N, M]`` (for plasticity)."""
        return self.message_head(self.actor_encoder(obs))

    @torch.no_grad()
    def edge_weights(
        self, obs: torch.Tensor, adjacency: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return the adaptive edge-weight matrix ``[B, N, N]`` (or ``None``).

        Only meaningful in ``"adaptive"`` mode; used to snapshot the current
        communication graph for statistics/export without an actor/critic pass.
        """
        if self.comm_mode != "adaptive":
            return None
        a_emb = self.actor_encoder(obs)
        return self.adaptive.edge_weights(a_emb, adjacency)


__all__ = [
    "ObservationEncoder",
    "MessageHead",
    "PolicyNetwork",
]
