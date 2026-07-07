"""
backends/registry.py

Plugin registry for inpainting backends.

Usage — registering a backend
------------------------------
    from backends.registry import backend_registry
    from backends.base import InpaintingBackend

    @backend_registry.register("my_backend")
    class MyBackend(InpaintingBackend):
        ...

Usage — building a backend by name
------------------------------------
    backend = backend_registry.build("lama", device="cuda")

Built-in backends are registered at the bottom of this file so they
are available as soon as ``backends.registry`` is imported.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from backends.base import InpaintingBackend


class BackendRegistry:
    """
    String-keyed registry mapping backend names to their classes.

    Identical structure to ``OCRRegistry`` — keeping both consistent
    makes the codebase easier to reason about.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[InpaintingBackend]] = {}

    def register(
        self, name: str
    ) -> Callable[[Type[InpaintingBackend]], Type[InpaintingBackend]]:
        """
        Class decorator that adds the decorated class under *name*.

        Example
        -------
            @backend_registry.register("lama")
            class LamaBackend(InpaintingBackend): ...
        """
        def decorator(cls: Type[InpaintingBackend]) -> Type[InpaintingBackend]:
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any) -> InpaintingBackend:
        """
        Instantiate and return the backend registered under *name*.

        Parameters
        ----------
        name : str
            Registry key, e.g. ``"lama"``.
        **kwargs
            Forwarded to the backend ``__init__``.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Unknown inpainting backend '{name}'. "
                f"Available backends: [{available}]"
            )
        return self._registry[name](**kwargs)

    def available(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._registry)


# Singleton registry instance
backend_registry = BackendRegistry()

# ---------------------------------------------------------------------------
# Register built-in backends
# ---------------------------------------------------------------------------
import backends.lama_backend   # noqa: E402, F401
import backends.mock_backend   # noqa: E402, F401
