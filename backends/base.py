"""
backends/base.py

Abstract interface that every inpainting backend must implement.

Lifecycle contract
------------------
``load_model()``
    Called once before the first image in a batch is processed.
    Must initialise all model weights, move tensors to the target
    device, and optionally warm up CUDA kernels.

``inpaint_patch(patch, mask)``
    Called for every patch extracted from every image.
    Must NOT modify the backend's persistent state.
    Must return an RGB uint8 array of the same shape as *patch*.

``unload_model()``
    Called after the last image in the batch has been processed.
    Must release GPU memory so other processes can use the device.

GPU memory tracking
-------------------
``memory_usage_mb()`` is an optional helper that backends may implement
to expose current GPU reserved memory.  The pipeline logger calls it
before and after ``load_model`` / ``unload_model`` if available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InpaintingBackend(ABC):
    """
    Abstract base class for inpainting backends.

    All concrete implementations must register themselves via
    ``backends.registry.backend_registry``.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Initialise and load the model into memory / onto the GPU.

        This method is always called before the first ``inpaint_patch``
        call within a pipeline run.
        """

    @abstractmethod
    def inpaint_patch(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct the masked region of *patch*.

        Parameters
        ----------
        patch : np.ndarray
            RGB uint8 array of shape (H, W, 3).  Pixel values in the
            masked region may be arbitrary (they are discarded by the model).
        mask : np.ndarray
            uint8 array of shape (H, W) with values {0, 255}.
            255 = pixels to be inpainted; 0 = context pixels to preserve.

        Returns
        -------
        np.ndarray
            RGB uint8 array of shape (H, W, 3).  The backend is expected
            to return a fully reconstructed image; the paste-back stage
            will apply the mask selector before writing anything to the
            final output.
        """

    @abstractmethod
    def unload_model(self) -> None:
        """
        Release all GPU / CPU resources held by this backend.

        After this method returns, the backend must be in a state where
        ``load_model`` can be called again safely.
        """

    # ------------------------------------------------------------------ #
    # Optional instrumentation helpers
    # ------------------------------------------------------------------ #

    def memory_usage_mb(self) -> float | None:
        """
        Return current GPU memory reserved by this backend in MiB,
        or ``None`` if this backend does not track GPU memory.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / (1024 ** 2)
        except ImportError:
            pass
        return None

    def warmup(self, size: tuple[int, int] = (64, 64)) -> None:
        """
        Optional: run a dummy inpaint call to pre-warm CUDA kernels.
        The default implementation creates a small black patch + full mask.
        """
        dummy_patch = np.zeros((*size, 3), dtype=np.uint8)
        dummy_mask = np.full(size, 255, dtype=np.uint8)
        self.inpaint_patch(dummy_patch, dummy_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
