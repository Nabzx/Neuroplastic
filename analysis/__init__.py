"""Post-hoc analysis of experiment results.

* :mod:`analysis.experiment`               -- result records + loading.
* :mod:`analysis.topology_analysis`        -- how the interaction graph evolves.
* :mod:`analysis.specialisation_analysis`  -- functional-role differentiation.
* :mod:`analysis.statistics`               -- CIs and significance tests.

Analysis is deliberately separate from :mod:`evaluation`: evaluation *computes*
metrics during/after a run; analysis *interprets* collections of them (across
seeds, conditions and training time).
"""

__all__ = []
