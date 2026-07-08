"""Evaluation of emergent behaviour.

The project characterises emergent coordination and specialisation with three
families of measures:

* :mod:`evaluation.graph_metrics`       -- graph-theoretic structure of the
  discovered interaction topology (functional; NetworkX-based).
* :mod:`evaluation.information_metrics`  -- information flow through the channel
  (entropy, mutual information, transfer entropy).
* :mod:`evaluation.coordination`         -- task-level coordination outcomes.

:mod:`evaluation.evaluator` orchestrates them over recorded rollouts.
"""

from evaluation.evaluator import Evaluator

__all__ = ["Evaluator"]
