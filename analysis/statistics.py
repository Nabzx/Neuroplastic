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


def summarise(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Return ``{mean, std, ci_low, ci_high, n}`` for a set of per-seed values.

    ``std`` is the sample standard deviation; the CI is a percentile bootstrap.
    NaNs are dropped. With a single value the CI collapses to the value.
    """
    import numpy as np

    x = np.asarray([v for v in values], dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if x.size > 1:
        _, lo, hi = bootstrap_ci(x, confidence=confidence)
    else:
        lo = hi = mean
    return {"mean": mean, "std": std, "ci_low": float(lo), "ci_high": float(hi), "n": int(x.size)}


def permutation_test(
    a: Sequence[float], b: Sequence[float], n_permutations: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Two-sided permutation test on the difference of means (SciPy-free).

    Returns ``{difference, p_value, n_a, n_b}`` where ``difference = mean(a) -
    mean(b)`` and ``p_value`` is the fraction of random label shuffles whose
    absolute mean-difference is at least the observed one. NaN p-value if either
    group is empty. With very few seeds the test has little power -- interpret
    accordingly.
    """
    import numpy as np

    x = np.asarray([v for v in a], dtype=float)
    y = np.asarray([v for v in b], dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
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
    return {
        "difference": float(x.mean() - y.mean()),
        "p_value": float((count + 1) / (n_permutations + 1)),
        "n_a": int(n_a),
        "n_b": int(y.size),
    }


__all__ = ["bootstrap_ci", "mann_whitney", "summarise", "permutation_test"]
