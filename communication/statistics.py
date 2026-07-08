"""Statistics for a weighted communication graph.

Operates on a plain ``[N, N]`` weight matrix (receiver-by-sender), so it works on
either a fixed uniform mask or a snapshot of adaptive attention weights, and
needs neither torch nor a live model. These are the quantities worth watching
during training to see how information flow is organising itself:

* ``edge_density``      -- fraction of possible directed edges that are active,
* ``mean_weight`` / ``max_weight`` -- magnitude of active connections,
* ``weight_entropy``    -- mean (over receivers) Shannon entropy of the incoming
  weight distribution (low = each agent listens to a few peers),
* ``effective_degree``  -- mean perplexity ``2**entropy`` = effective number of
  senders each agent attends to.
"""

from __future__ import annotations

import numpy as np

from core.types import ArrayLike


def weight_matrix_statistics(matrix: ArrayLike, threshold: float = 1e-3) -> dict[str, float]:
    """Return summary statistics of a ``[N, N]`` weighted-graph matrix.

    ``matrix[i, j]`` is the weight from sender ``j`` to receiver ``i``. The
    diagonal (self-loops) is ignored.
    """
    w = np.asarray(matrix, dtype=float)
    n = w.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)

    active = (w > threshold) & off_diagonal
    possible = max(1, n * (n - 1))
    positive = w[off_diagonal & (w > 0)]

    entropies: list[float] = []
    effective_degrees: list[float] = []
    for i in range(n):
        row = w[i][off_diagonal[i]]
        row = row[row > 0]
        if row.size == 0:
            continue
        p = row / row.sum()
        entropy = float(-(p * np.log2(p)).sum())
        entropies.append(entropy)
        effective_degrees.append(float(2.0**entropy))

    return {
        "comm_edge_density": float(active.sum() / possible),
        "comm_mean_weight": float(positive.mean()) if positive.size else 0.0,
        "comm_max_weight": float(w[off_diagonal].max()) if n > 1 else 0.0,
        "comm_weight_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "comm_effective_degree": float(np.mean(effective_degrees)) if effective_degrees else 0.0,
    }


__all__ = ["weight_matrix_statistics"]
