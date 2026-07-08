"""YAML config loading with inheritance and command-line overrides.

Features
--------
* ``extends:`` -- a config may inherit from another via a relative path, so
  ablations are expressed as small diffs against ``default.yaml``.
* deep merge -- child values override parent values key-by-key (nested dicts
  are merged, not replaced wholesale).
* dotted overrides -- ``training.lr=1e-4`` style strings (typically from the
  CLI) are parsed with YAML semantics and applied last.

The result is always validated against :class:`~configs.schema.ExperimentConfig`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from configs.schema import ExperimentConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dict (empty file -> ``{}``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config {path} must define a mapping at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load ``path``, resolving any ``extends:`` chain into a single dict."""
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'extends' detected at {path}")
    _seen.add(path)

    raw = load_yaml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent_path = (path.parent / parent_ref).resolve()
    parent = _resolve_extends(parent_path, _seen)
    return deep_merge(parent, raw)


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` from a ``"a.b"`` key, creating dicts."""
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        node = cursor.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping")
        cursor = node
    cursor[keys[-1]] = value


def _parse_scalar(raw: str) -> Any:
    """Parse an override value with YAML semantics, coercing numeric strings.

    YAML 1.1 (PyYAML) does not recognise unscientific-notation floats such as
    ``1e-4`` as numbers, leaving them as strings. Since the CLI advertises exactly
    that form (``-o training.lr=1e-4``), we fall back to int/float coercion for
    strings that look numeric, while genuinely non-numeric strings pass through.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` override strings, parsing values as YAML."""
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not of the form key=value")
        key, _, raw_value = item.partition("=")
        _set_dotted(result, key.strip(), _parse_scalar(raw_value))
    return result


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> ExperimentConfig:
    """Load ``path`` (resolving ``extends``), apply overrides, and validate.

    Parameters
    ----------
    path:
        Path to a YAML config.
    overrides:
        Optional iterable of ``"dotted.key=value"`` strings applied last.
    """
    data = _resolve_extends(Path(path))
    if overrides:
        data = apply_overrides(data, overrides)
    return ExperimentConfig.from_dict(data)


__all__ = [
    "load_yaml",
    "deep_merge",
    "apply_overrides",
    "load_config",
]
