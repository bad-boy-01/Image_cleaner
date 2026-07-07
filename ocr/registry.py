"""
ocr/registry.py

Plugin registry for OCR backends.

Usage — registering a backend
------------------------------
    from ocr.registry import ocr_registry
    from ocr.base import OCRBackend

    @ocr_registry.register("my_backend")
    class MyOCRBackend(OCRBackend):
        ...

Usage — resolving a backend by name
-------------------------------------
    backend = ocr_registry.build("easyocr", languages=["ko","en"])

All built-in backends are registered at the bottom of this file so
they are available as soon as ``ocr.registry`` is imported.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from ocr.base import OCRBackend


class OCRRegistry:
    """
    A simple string-keyed registry that maps backend names to their classes.

    Registering a backend is idempotent — re-registering the same name
    replaces the previous entry and does not raise.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[OCRBackend]] = {}

    def register(self, name: str) -> Callable[[Type[OCRBackend]], Type[OCRBackend]]:
        """
        Class decorator.  Adds the decorated class to the registry
        under ``name``.

        Example
        -------
            @ocr_registry.register("easyocr")
            class EasyOCRBackend(OCRBackend): ...
        """
        def decorator(cls: Type[OCRBackend]) -> Type[OCRBackend]:
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any) -> OCRBackend:
        """
        Instantiate the backend registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key, e.g. ``"easyocr"``.
        **kwargs
            Forwarded to the backend's ``__init__``.

        Raises
        ------
        KeyError
            If ``name`` is not in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Unknown OCR backend '{name}'. "
                f"Available backends: [{available}]"
            )
        return self._registry[name](**kwargs)

    def available(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._registry)


# Singleton registry instance used throughout the application
ocr_registry = OCRRegistry()

# ---------------------------------------------------------------------------
# Register built-in backends
# Import here (after the registry is created) to avoid circular imports.
# ---------------------------------------------------------------------------
import ocr.easyocr_backend   # noqa: E402, F401  (registers on import)
