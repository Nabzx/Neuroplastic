"""Statistical helpers for comparing conditions across seeds.

Reporting standard for this project: every headline comparison comes with a
bootstrap confidence interval over seeds and an effect size, not just a mean.
``bootstrap_ci`` is functional; a significance test is provided when SciPy is
available and otherwise raises with an actionable message.
"""

from __future__ import annotations

from typing import Sequence


def bootstrap_ci(
    samples: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` via percentile bootstrap.

    Parameters
    ----------
    samples:
        Per-seed metric values.
    confidence:
        Two-sided confidence level (default 0.95).
    n_resamples:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.
    """
    import numpy as np

    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(x.mean()), float(lower), float(upper))


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Mann-Whitney U test between two conditions (requires SciPy)."""
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "mann_whitney requires SciPy. Install analysis extras: "
            "`pip install -e .[analysis]`."
        ) from exc
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


__all__ = ["bootstrap_ci", "mann_whitney"]
