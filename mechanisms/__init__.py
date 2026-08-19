"""Biological plasticity mechanisms and baselines.

Each mechanism modifies a network *during* continual training (after each
optimiser step) to preserve plasticity. They register under ``MECHANISM_REGISTRY``
and are selected by name. The concrete biological mechanisms (homeostatic
scaling, structural plasticity) and baselines (L2, shrink-and-perturb, continual
backprop, ReDo) are added in Phase 2.
"""

from mechanisms.base import MECHANISM_REGISTRY, Mechanism, make_mechanism

__all__ = ["Mechanism", "MECHANISM_REGISTRY", "make_mechanism"]
