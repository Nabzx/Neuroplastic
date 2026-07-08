"""Edge-weight stability over training.

Given a time-ordered stack of communication weight matrices ``[T, N, N]``
(as recorded in ``runs/<name>/edge_weights.npz``), quantify how much the
communication graph churns during training. A *fixed* graph is perfectly stable
(``stability == 1``, ``drift == 0``); a *plastic* graph evolves, so these
numbers reveal how strongly -- and how persistently -- the topology adapts.
"""

from __future__ import annotations

import numpy as np

from core.types import ArrayLike

_EPS = 1e-12
_NAN_REPORT = {
    "edge_weight_stability": float("nan"),
    "edge_weight_drift": float("nan"),
    "edge_weight_final_shift": float("nan"),
    "edge_weight_variance": float("nan"),
}


def edge_weight_stability(snapshots: ArrayLike | None) -> dict[str, float]:
    """Return stability statistics for a ``[T, N, N]`` stack of weight matrices.

    * ``edge_weight_stability`` -- mean cosine similarity between consecutive
      matrices (1 = unchanging).
    * ``edge_weight_drift`` -- mean absolute change per step (0 = unchanging).
    * ``edge_weight_final_shift`` -- mean absolute difference between the last and
      first matrix (net movement).
    * ``edge_weight_variance`` -- mean per-edge variance over training.

    All statistics ignore self-loops. Returns NaNs if there are fewer than two
    snapshots (e.g. the no-communication setting).
    """
    if snapshots is None:
        return dict(_NAN_REPORT)
    weights = np.asarray(snapshots, dtype=float)
    if weights.ndim != 3 or weights.shape[0] < 2:
        return dict(_NAN_REPORT)

    n = weights.shape[1]
    off_diagonal = ~np.eye(n, dtype=bool)
    flat = weights[:, off_diagonal]  # [T, E]

    drift = float(np.abs(np.diff(weights, axis=0))[:, off_diagonal].mean())
    final_shift = float(np.abs(weights[-1] - weights[0])[off_diagonal].mean())
    variance = float(flat.var(axis=0).mean())

    # cosine similarity between consecutive (flattened) matrices
    cosines = []
    for prev, curr in zip(flat[:-1], flat[1:]):
        denom = np.linalg.norm(prev) * np.linalg.norm(curr)
        cosines.append(float(prev @ curr / denom) if denom > _EPS else 1.0)

    return {
        "edge_weight_stability": float(np.mean(cosines)),
        "edge_weight_drift": drift,
        "edge_weight_final_shift": final_shift,
        "edge_weight_variance": variance,
    }


__all__ = ["edge_weight_stability"]
