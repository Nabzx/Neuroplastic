"""Tests for the Permuted-MNIST benchmark and the classification training path."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from data.streams import TaskStep  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer, accuracy  # noqa: E402


def test_accuracy():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    targets = torch.tensor([0, 1, 0])                 # last one wrong
    assert accuracy(logits, targets) == pytest.approx(2 / 3)


class _FakeClassStream:
    """A tiny synthetic classification stream (no MNIST download needed)."""

    def __init__(self, dim=6, classes=3, num_tasks=3, task_length=20, batch_size=5):
        self.dim, self.classes = dim, classes
        self.num_tasks, self.task_length, self.batch_size = num_tasks, task_length, batch_size
        self._g = torch.Generator().manual_seed(0)

    def steps(self):
        for t in range(self.num_tasks):
            for b in range(self.task_length // self.batch_size):
                x = torch.randn(self.batch_size, self.dim, generator=self._g)
                y = torch.randint(0, self.classes, (self.batch_size,), generator=self._g)
                yield TaskStep(x=x, y=y, task_id=t, step_in_task=b, boundary=(b == 0))

    def probe_batch(self, n=32):
        return torch.randn(n, self.dim, generator=torch.Generator().manual_seed(1))


def test_classification_trainer_records_accuracy():
    stream = _FakeClassStream()
    net = MLP(input_dim=6, hidden_dim=16, output_dim=3, num_hidden_layers=1)
    trainer = ContinualTrainer(
        net, stream, torch.optim.SGD(net.parameters(), lr=0.05),
        loss_fn=torch.nn.CrossEntropyLoss(), metric_fn=accuracy,
    )
    history = trainer.run()
    assert len(history) == 3
    assert "accuracy_late" in history[0]
    assert 0.0 <= history[0]["accuracy_late"] <= 1.0


def test_permuted_mnist_stream_if_available():
    try:
        from data.mnist import PermutedMNISTStream

        stream = PermutedMNISTStream(num_tasks=2, task_length=32, batch_size=16, seed=0)
    except Exception as exc:  # network unavailable / download failed
        pytest.skip(f"MNIST unavailable: {exc}")
    steps = list(stream.steps())
    assert steps[0].x.shape == (16, 784)
    assert steps[0].y.dtype == torch.int64
    assert int(steps[0].y.min()) >= 0 and int(steps[0].y.max()) <= 9
    assert stream.probe_batch(20).shape == (20, 784)
