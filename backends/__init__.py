"""
backends/__init__.py
"""
from .base import InpaintingBackend
from .registry import backend_registry

__all__ = ["InpaintingBackend", "backend_registry"]
