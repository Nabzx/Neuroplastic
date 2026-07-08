"""Neuroplastic mechanisms for the communication weights.

The plasticity rules here adapt the *edge weights* of the interaction graph from
communication activity, in the spirit of computational-neuroscience learning
rules (Hebbian, Oja) with optional neuromodulation and homeostasis.

* :mod:`plasticity.base`        -- the ``PlasticityRule`` interface + registry.
* :mod:`plasticity.hebbian`     -- Hebbian and Oja rules.
* :mod:`plasticity.modulation`  -- reward-gated / three-factor modulation.
* :mod:`plasticity.homeostasis` -- synaptic scaling to keep weights bounded.
"""

from plasticity.base import PLASTICITY_REGISTRY, PlasticityRule, make_plasticity

__all__ = ["PlasticityRule", "PLASTICITY_REGISTRY", "make_plasticity"]
