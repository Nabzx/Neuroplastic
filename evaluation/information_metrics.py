"""Information-theoretic measures of communication.

These quantify *how much* and *what* information flows through the channel.
Estimating information quantities from finite samples is subtle; we provide
simple, well-understood histogram (plug-in) estimators as a functional baseline
and flag the bias caveats. Transfer entropy -- the key directed-flow measure --
is left as a placeholder because a defensible estimator (KSG / lagged
conditional MI with bias correction) is a milestone in its own right.

All estimators take plain NumPy arrays of recorded activity, so they are
decoupled from the training stack.
"""

from __future__ import annotations

from core.registry import Registry
from core.types import ArrayLike

INFORMATION_METRICS: Registry = Registry("information_metric")


def shannon_entropy(samples: ArrayLike, bins: int = 16) -> float:
    """Plug-in Shannon entropy (bits) of ``samples`` via histogram binning.

    Note: the plug-in estimator is negatively biased for small samples; use as a
    relative measure across matched conditions rather than an absolute value.
    """
    import numpy as np

    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    total = p.sum()
    if total == 0:
        return float("nan")
    p = p[p > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_information(x: ArrayLike, y: ArrayLike, bins: int = 16) -> float:
    """Plug-in mutual information (bits) between paired samples ``x`` and ``y``."""
    import numpy as np

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or x.size != y.size:
        return float("nan")
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    total = joint.sum()
    if total == 0:
        return float("nan")
    p_xy = joint / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x * p_y)
        terms = p_xy * np.log2(ratio)
    return float(np.nansum(terms))


def transfer_entropy(source: ArrayLike, target: ArrayLike, lag: int = 1) -> float:  # pragma: no cover - deferred
    """Directed information flow ``source -> target`` (placeholder).

    Requires a lagged conditional-MI estimator with bias correction; specified in
    docs/experiment_plan.md.
    """
    raise NotImplementedError(
        "transfer_entropy is a placeholder; the estimator is specified in "
        "docs/experiment_plan.md."
    )


# Register for discoverability. "message_entropy" is entropy applied to the
# recorded message stream; the evaluator supplies the array.
INFORMATION_METRICS.register("message_entropy", shannon_entropy)
INFORMATION_METRICS.register("mutual_information", mutual_information)
INFORMATION_METRICS.register("transfer_entropy", transfer_entropy)


__all__ = [
    "INFORMATION_METRICS",
    "shannon_entropy",
    "mutual_information",
    "transfer_entropy",
]
