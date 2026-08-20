"""Continual Permuted-MNIST stream.

A recognised loss-of-plasticity benchmark: a sequence of tasks, each a fixed
random permutation of the 784 input pixels. MNIST is downloaded once (from the
PyTorch S3 mirror) and cached; no torchvision dependency. Per-task accuracy /
loss measures the learner's *current* fitting ability, so a plastic-losing
network fits later tasks worse.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

from data.streams import TaskStep

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {"images": "train-images-idx3-ubyte.gz", "labels": "train-labels-idx1-ubyte.gz"}
_MEAN, _STD = 0.1307, 0.3081


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic = struct.unpack(">I", fh.read(4))[0]
        if magic == 2051:                                   # images
            n, rows, cols = struct.unpack(">III", fh.read(12))
            return np.frombuffer(fh.read(), dtype=np.uint8).reshape(n, rows * cols).copy()
        if magic == 2049:                                   # labels
            struct.unpack(">I", fh.read(4))
            return np.frombuffer(fh.read(), dtype=np.uint8).copy()
        raise ValueError(f"Unexpected IDX magic {magic} in {path}")


def load_mnist(cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X [60000, 784] float32 normalised, y [60000] int64)``, cached."""
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "neuroplastic" / "mnist"
    cache.mkdir(parents=True, exist_ok=True)
    for filename in _FILES.values():
        path = cache / filename
        if not path.exists():
            request = urllib.request.Request(_MIRROR + filename, headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    x = _read_idx(cache / _FILES["images"]).astype("float32")
    y = _read_idx(cache / _FILES["labels"]).astype("int64")
    x = (x / 255.0 - _MEAN) / _STD
    return x, y


class PermutedMNISTStream:
    """A sequence of pixel-permuted MNIST classification tasks."""

    def __init__(
        self,
        num_tasks: int = 150,
        task_length: int = 1000,
        batch_size: int = 16,
        seed: int = 0,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        x, y = load_mnist(cache_dir)
        self._x = torch.from_numpy(x)
        self._y = torch.from_numpy(y)
        self.input_dim = 784
        self.output_dim = 10
        self.num_tasks = num_tasks
        self.task_length = task_length
        self.batch_size = batch_size
        self.device = device
        self._n = self._x.shape[0]
        self._gen = torch.Generator().manual_seed(seed)
        self._perms = [torch.randperm(784, generator=self._gen) for _ in range(num_tasks)]

    def steps(self):
        n_batches = max(1, self.task_length // self.batch_size)
        for task_id in range(self.num_tasks):
            perm = self._perms[task_id]
            for b in range(n_batches):
                idx = torch.randint(0, self._n, (self.batch_size,), generator=self._gen)
                yield TaskStep(
                    x=self._x[idx][:, perm].to(self.device),
                    y=self._y[idx].to(self.device),
                    task_id=task_id,
                    step_in_task=b,
                    boundary=(b == 0),
                )

    def probe_batch(self, n: int = 512) -> torch.Tensor:
        gen = torch.Generator().manual_seed(999)
        idx = torch.randint(0, self._n, (n,), generator=gen)
        return self._x[idx].to(self.device)


__all__ = ["PermutedMNISTStream", "load_mnist"]
