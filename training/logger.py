"""Lightweight training metric logger: append-only CSV + console.

Writes one row per training iteration to ``<output_dir>/<run_name>/metrics.csv``
and echoes a formatted line to the console logger. The CSV is the durable record
of reward and episode-length curves for later comparison across baselines.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


class MetricLogger:
    """Stream scalar training metrics to CSV and the console."""

    def __init__(self, output_dir: str, run_name: str, console_logger: Any) -> None:
        self.dir = Path(output_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.csv"
        self.console = console_logger
        self._file = None
        self._writer: csv.DictWriter | None = None

    def log(self, row: Mapping[str, Any], verbose: bool = True) -> None:
        """Append ``row`` to the CSV (header written once) and optionally print."""
        if self._writer is None:
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        if verbose:
            message = (
                "iter=%(iteration)d steps=%(env_steps)d "
                "ep_return=%(episode_return_mean).3f ep_len=%(episode_length_mean).1f "
                "| pg=%(policy_loss).3f v=%(value_loss).3f ent=%(entropy).3f"
                % _defaults(row)
            )
            if "comm_effective_degree" in row:
                message += (
                    " | comm_deg=%(comm_effective_degree).2f "
                    "dens=%(comm_edge_density).2f H=%(comm_weight_entropy).2f" % row
                )
            if "plast_modulation" in row:
                message += (
                    " | plast_mod=%(plast_modulation).3f "
                    "w=%(plast_mean_weight).3f dP=%(plast_update_norm).4f" % row
                )
            self.console.info(message)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def _defaults(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing keys so the console format string never raises."""
    out = dict(row)
    for key, default in (
        ("iteration", 0),
        ("env_steps", 0),
        ("episode_return_mean", float("nan")),
        ("episode_length_mean", float("nan")),
        ("policy_loss", float("nan")),
        ("value_loss", float("nan")),
        ("entropy", float("nan")),
    ):
        out.setdefault(key, default)
    return out


__all__ = ["MetricLogger"]
