"""Diagnostics for loss of plasticity in continually-trained networks."""

from diagnostics.plasticity import (
    dormant_fraction,
    effective_rank,
    representation_diagnostics,
    weight_magnitude,
)

__all__ = [
    "dormant_fraction",
    "effective_rank",
    "weight_magnitude",
    "representation_diagnostics",
]
