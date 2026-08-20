"""Continual-learning task streams for studying (loss of) plasticity.

The streams are deliberately small and synthetic so a rigorous multi-seed study
runs on a CPU in minutes. The primary benchmark is permuted-input regression --
a synthetic analogue of Continual Permuted MNIST that needs no downloaded data
and reliably exposes loss of plasticity in vanilla networks.
"""

from data.streams import PermutedRegressionStream, TaskStep

__all__ = ["PermutedRegressionStream", "TaskStep"]
