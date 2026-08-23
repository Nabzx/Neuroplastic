"""Statistical helpers for comparing mechanisms across seeds.

Every headline comparison carries a bootstrap CI over seeds and an effect size,
not just a mean. Significance uses a SciPy-free permutation test (few seeds ->
low power; interpret accordingly).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(
    samples: Sequence[float], confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` via percentile bootstrap."""
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(x.mean()), float(lo), float(hi))


def summarise(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Return ``{mean, std, ci_low, ci_high, n}`` over per-seed values (NaNs dropped)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    _, lo, hi = bootstrap_ci(x, confidence=confidence) if x.size > 1 else (mean, mean, mean)
    return {"mean": mean, "std": std, "ci_low": float(lo), "ci_high": float(hi), "n": int(x.size)}


def permutation_test(
    a: Sequence[float], b: Sequence[float], n_permutations: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Two-sided permutation test on the difference of means (SciPy-free)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return {"difference": float("nan"), "p_value": float("nan"), "n_a": int(x.size), "n_b": int(y.size)}
    observed = abs(x.mean() - y.mean())
    combined = np.concatenate([x, y])
    n_a = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        if abs(combined[:n_a].mean() - combined[n_a:].mean()) >= observed - 1e-12:
            count += 1

    # Cohen's d (pooled) effect size -- reported alongside the p-value.
    if x.size > 1 and y.size > 1:
        pooled = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2))
        cohens_d = float((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0
    else:
        cohens_d = float("nan")

    return {
        "difference": float(x.mean() - y.mean()),
        "cohens_d": cohens_d,
        "p_value": float((count + 1) / (n_permutations + 1)),
        "n_a": int(n_a),
        "n_b": int(y.size),
    }


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as input).

    Controls the family-wise error rate across a family of comparisons and is more
    powerful than plain Bonferroni. NaNs pass through unchanged.
    """
    p = list(pvalues)
    valid = [(i, v) for i, v in enumerate(p) if v == v]
    m = len(valid)
    adjusted = list(p)
    running = 0.0
    for rank, (idx, value) in enumerate(sorted(valid, key=lambda t: t[1])):
        running = max(running, (m - rank) * value)
        adjusted[idx] = min(1.0, running)
    return adjusted


__all__ = ["bootstrap_ci", "summarise", "permutation_test", "holm_bonferroni"]
