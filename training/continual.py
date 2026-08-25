"""Online continual training over a task stream, with plasticity diagnostics.

Trains a model online (one mini-batch at a time) across a sequence of tasks and
records, per task, how well the network *can* fit that task (its late-task loss)
plus representation diagnostics. A network that keeps its per-task loss flat has
retained plasticity; one whose per-task loss climbs has lost it.

A :class:`PlasticityMechanism` (see :mod:`mechanisms`) can be attached to modify
the network after each optimiser step (homeostatic scaling, structural
plasticity, ...). ``mechanism=None`` is the vanilla baseline.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn

from diagnostics.plasticity import representation_diagnostics


class PlasticityMechanism(Protocol):
    """Hook interface for a biological plasticity mechanism."""

    requires_activations: bool

    def observe(self, activations: list[torch.Tensor]) -> None: ...
    def observe_context(self, network_input: torch.Tensor,
                        activations: list[torch.Tensor], loss: float) -> None: ...
    def before_optimizer_step(self, model: nn.Module, step_index: int) -> None: ...
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None: ...


class ContinualTrainer:
    """Run online continual learning and collect per-task plasticity metrics."""

    def __init__(
        self,
        model: nn.Module,
        stream: Any,
        optimizer: torch.optim.Optimizer,
        mechanism: PlasticityMechanism | None = None,
        loss_fn: nn.Module | None = None,
        metric_fn=None,
        device: str = "cpu",
        diag_tau: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.stream = stream
        self.optimizer = optimizer
        self.mechanism = mechanism
        self.device = device
        self.diag_tau = diag_tau
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metric_fn = metric_fn                 # e.g. accuracy for classification

    def run(self) -> list[dict[str, float]]:
        """Train across the whole stream; return one metric record per task."""
        probe = self.stream.probe_batch().to(self.device)
        wants_acts = bool(getattr(self.mechanism, "requires_activations", False))
        wants_ctx = bool(getattr(self.mechanism, "requires_context", False))

        history: list[dict[str, float]] = []
        current_task = -1
        losses: list[float] = []
        metrics: list[float] = []
        examples_seen = 0
        step_index = 0

        for step in self.stream.steps():
            if step.boundary and current_task >= 0:
                history.append(self._record(current_task, losses, metrics, examples_seen, probe))
                losses, metrics = [], []
            current_task = step.task_id

            self.model.train()
            if wants_acts:
                pred, activations = self.model(step.x, return_activations=True)
            else:
                pred, activations = self.model(step.x), None
            loss = self.loss_fn(pred, step.y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.mechanism is not None:
                self.mechanism.before_optimizer_step(self.model, step_index)
            self.optimizer.step()

            if self.mechanism is not None:
                if wants_acts and activations is not None:
                    acts = [a.detach() for a in activations]
                    self.mechanism.observe(acts)
                    if wants_ctx:
                        self.mechanism.observe_context(step.x.detach(), acts, float(loss.item()))
                self.mechanism.after_optimizer_step(self.model, step_index)

            losses.append(float(loss.item()))
            if self.metric_fn is not None:
                metrics.append(float(self.metric_fn(pred.detach(), step.y)))
            examples_seen += step.x.shape[0]
            step_index += 1

        if losses:
            history.append(self._record(current_task, losses, metrics, examples_seen, probe))
        return history

    def _record(
        self, task_id: int, losses: list[float], metrics: list[float], examples_seen: int, probe: torch.Tensor
    ) -> dict[str, float]:
        half = max(1, len(losses) // 2)
        diagnostics = representation_diagnostics(self.model, probe, tau=self.diag_tau)
        record = {
            "task": task_id,
            "examples_seen": examples_seen,
            "train_loss_mean": float(sum(losses) / len(losses)),
            "train_loss_late": float(sum(losses[half:]) / len(losses[half:])),
            **diagnostics,
        }
        if metrics:
            record["accuracy_late"] = float(sum(metrics[half:]) / len(metrics[half:]))
        return record


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy for ``[B, C]`` logits against ``[B]`` int targets."""
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


__all__ = ["ContinualTrainer", "PlasticityMechanism", "accuracy"]
