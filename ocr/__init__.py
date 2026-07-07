"""
ocr/__init__.py
"""
from .base import OCRBackend
from .registry import ocr_registry

__all__ = ["OCRBackend", "ocr_registry"]
