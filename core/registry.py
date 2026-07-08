"""A minimal, typed component registry.

The registry is the backbone of the project's extensibility story. New agents,
environments, plasticity rules, communication topologies and protocols are all
registered under a string key and then selected *by name* from a config file.
Adding a new variant therefore never requires editing the training loop -- only
writing a class and decorating it.

Example
-------
>>> from core.registry import Registry
>>> RULES: Registry = Registry("plasticity_rule")
>>> @RULES.register("hebbian")
... class HebbianRule:
...     ...
>>> RULES.get("hebbian")            # doctest: +ELLIPSIS
<class '...HebbianRule'>
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A ``name -> object`` map with decorator-style registration.

    Parameters
    ----------
    name:
        Human-readable name of the registry, used in error messages.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str, obj: T | None = None) -> Callable[[T], T] | T:
        """Register ``obj`` under ``key``.

        Usable either as a decorator (``@registry.register("foo")``) or as a
        direct call (``registry.register("foo", Foo)``). Raises :class:`KeyError`
        on duplicate keys so that clashes fail loudly at import time.
        """

        def _add(target: T) -> T:
            if key in self._entries:
                raise KeyError(
                    f"{key!r} is already registered in the {self._name!r} registry"
                )
            self._entries[key] = target
            return target

        if obj is None:
            return _add
        return _add(obj)

    def get(self, key: str) -> T:
        """Return the object registered under ``key`` or raise ``KeyError``."""
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"Unknown {self._name} {key!r}. "
                f"Registered options: {sorted(self._entries)}"
            ) from None

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Registry(name={self._name!r}, entries={self.keys()})"


__all__ = ["Registry"]
