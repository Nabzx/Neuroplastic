"""Synthetic continual task streams.

``PermutedRegressionStream`` presents a sequence of related regression *tasks*.
A single fixed random *teacher* network defines a nonlinear target; each task
applies a fresh random permutation of the input features before the teacher, so
every task requires the learner to re-fit a genuinely new input->output mapping
(exactly the mechanism behind Continual Permuted MNIST). Measuring how well the
learner fits *task k* as *k* grows reveals loss of plasticity: a healthy learner
keeps its per-task error flat; a plastic-losing one gets steadily worse.

Everything is generated on the fly from a seeded RNG, so runs are reproducible
and need no external data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TaskStep:
    """One mini-batch from the stream, with task bookkeeping."""

    x: torch.Tensor          # [B, input_dim]
    y: torch.Tensor          # [B]
    task_id: int
    step_in_task: int
    boundary: bool           # True on the first step of a new task


class PermutedRegressionStream:
    """A sequence of permuted-input regression tasks from a fixed teacher.

    Parameters
    ----------
    input_dim:
        Number of input features.
    teacher_hidden:
        Hidden width of the frozen teacher (nonlinear target complexity).
    num_tasks:
        Number of tasks in the stream.
    task_length:
        Examples per task.
    batch_size:
        Online mini-batch size.
    seed:
        RNG seed (teacher weights, permutations and inputs are all derived from it).
    """

    def __init__(
        self,
        input_dim: int = 16,
        teacher_hidden: int = 64,
        num_tasks: int = 300,
        task_length: int = 200,
        batch_size: int = 1,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.task_length = task_length
        self.batch_size = batch_size
        self.device = device
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

        # Frozen teacher: sign-nonlinearity hidden layer (a linear-threshold unit
        # network) + linear readout. Weights are fixed for the whole stream.
        self._w1 = torch.randn(teacher_hidden, input_dim, generator=self._gen)
        self._w2 = torch.randn(teacher_hidden, generator=self._gen)
        # Per-task input permutations, precomputed for reproducibility.
        self._perms = [torch.randperm(input_dim, generator=self._gen) for _ in range(num_tasks)]

        # Normalise the target scale so MSE is comparable across settings.
        probe = self._teacher(self._sample_inputs(2048), self._perms[0])
        self._y_std = float(probe.std().clamp_min(1e-6))

    # -- target ------------------------------------------------------------
    def _sample_inputs(self, n: int) -> torch.Tensor:
        # Rademacher {-1, +1} inputs -- clean for linear-threshold teachers.
        bits = torch.randint(0, 2, (n, self.input_dim), generator=self._gen)
        return bits.float() * 2.0 - 1.0

    def _teacher(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        hidden = torch.sign(x[:, perm] @ self._w1.t())
        return hidden @ self._w2

    # -- iteration ---------------------------------------------------------
    def steps(self):
        """Yield :class:`TaskStep` mini-batches across the whole task sequence."""
        n_batches = max(1, self.task_length // self.batch_size)
        for task_id in range(self.num_tasks):
            perm = self._perms[task_id]
            for b in range(n_batches):
                x = self._sample_inputs(self.batch_size)
                y = self._teacher(x, perm) / self._y_std
                yield TaskStep(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    task_id=task_id,
                    step_in_task=b,
                    boundary=(b == 0),
                )

    def probe_batch(self, n: int = 512) -> torch.Tensor:
        """A fixed input batch for measuring representation diagnostics."""
        gen = torch.Generator(device="cpu").manual_seed(12345)
        bits = torch.randint(0, 2, (n, self.input_dim), generator=gen)
        return (bits.float() * 2.0 - 1.0).to(self.device)


__all__ = ["PermutedRegressionStream", "TaskStep"]
