"""Evaluation of emergent behaviour.

The project characterises emergent coordination and specialisation with three
families of measures:

* :mod:`evaluation.graph_metrics`       -- graph-theoretic structure of the
  discovered interaction topology (functional; NetworkX-based).
* :mod:`evaluation.information_metrics`  -- information flow through the channel
  (entropy, mutual information, transfer entropy).
* :mod:`evaluation.coordination`         -- task-level coordination outcomes.

:mod:`evaluation.evaluator` orchestrates them over recorded rollouts, and
:mod:`evaluation.report` bundles coordination, graph-structure and edge-weight
stability metrics into a single run report (usable from a saved run directory).
"""

from evaluation.evaluator import Evaluator
from evaluation.report import (
    evaluate_run,
    flatten_report,
    format_comparison,
    graph_report,
    load_run,
)

__all__ = [
    "Evaluator",
    "evaluate_run",
    "load_run",
    "graph_report",
    "flatten_report",
    "format_comparison",
]
