"""
ocr/base.py

Abstract interface that every OCR backend must implement.

Adding a new OCR backend requires only:
  1. Subclass ``OCRBackend``.
  2. Implement ``load``, ``detect``, ``unload``.
  3. Register with ``@ocr_registry.register("my_backend")``.

The rest of the pipeline is unaffected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.models import TextDetection


class OCRBackend(ABC):
    """
    Abstract base class for OCR detection backends.

    Lifecycle
    ---------
    Backends are loaded once before the first image and unloaded after
    the last image.  Do not assume ``load`` is called more than once per
    pipeline run.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Initialise the backend: download weights, build the model graph,
        move tensors to the target device, etc.

        This method is always called before the first ``detect`` call.
        """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextDetection]:
        """
        Run text detection on a single RGB uint8 image.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W, 3), dtype uint8, colour order RGB.

        Returns
        -------
        list[TextDetection]
            Unfiltered detections from the backend.  Language filtering
            is handled downstream by ``mask.language_filter``.
        """

    @abstractmethod
    def unload(self) -> None:
        """
        Release resources: delete model tensors, free GPU memory, close
        file handles, etc.
        """

    # ------------------------------------------------------------------ #
    # Optional lifecycle hooks (no-op defaults)
    # ------------------------------------------------------------------ #

    def warmup(self, dummy: np.ndarray) -> None:
        """
        Optional: run a dummy forward pass to pre-warm CUDA kernels.
        Called once after ``load``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
