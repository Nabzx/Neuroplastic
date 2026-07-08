"""Plotting utilities for interaction graphs and training dynamics.

Matplotlib is imported lazily inside each function so the package can be imported
(and the rest of the project used headless) without a plotting backend.

* :mod:`visualisation.dynamics`    -- reward / metric curves, hub centrality.
* :mod:`visualisation.graphs`      -- edge-weight heatmaps, communication graphs.
* :mod:`visualisation.run_figures` -- generate the standard figure set from logs.
* :mod:`visualisation.dashboards`  -- composite multi-panel figures (placeholder).
"""

from visualisation.dynamics import plot_centrality_over_time, plot_training_curves
from visualisation.graphs import draw_communication_graph, plot_edge_weight_heatmap
from visualisation.run_figures import visualise_runs

__all__ = [
    "plot_training_curves",
    "plot_centrality_over_time",
    "plot_edge_weight_heatmap",
    "draw_communication_graph",
    "visualise_runs",
]
