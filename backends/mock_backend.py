"""
backends/mock_backend.py

A deterministic mock inpainting backend for unit tests and dry-runs.

Instead of running any AI model, it fills the masked region with the
local median colour of the surrounding non-masked context pixels.  This
produces a visually rough but deterministic result that is sufficient
for:
  - Unit tests that verify the pipeline without a GPU.
  - Dry-runs to validate OCR + masking quality before committing to
    full model inference.

Registration
------------
This module registers itself with ``backend_registry`` on import.
"""

from __future__ import annotations

import numpy as np

from backends.base import InpaintingBackend
from backends.registry import backend_registry


@backend_registry.register("mock")
class MockBackend(InpaintingBackend):
    """
    Median-fill mock backend.  Requires no GPU or model weights.

    Parameters
    ----------
    fill_value : int or None
        If an integer (0–255), fills the masked region with a flat
        grey value instead of the local median.  Useful for debugging
        to make the filled regions immediately visible.
    """

    def __init__(self, fill_value: int | None = None) -> None:
        self.fill_value = fill_value

    # ------------------------------------------------------------------ #
    # Lifecycle (no-op — no model to load)
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Inpainting
    # ------------------------------------------------------------------ #

    def inpaint_patch(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fill masked pixels with either a flat value or the local median.

        Parameters
        ----------
        patch : np.ndarray
            RGB uint8 (H, W, 3).
        mask : np.ndarray
            uint8 (H, W) with values {0, 255}.

        Returns
        -------
        np.ndarray
            RGB uint8 (H, W, 3) with masked pixels replaced.
        """
        result = patch.copy()
        mask_bool = mask > 0  # (H, W) bool

        if self.fill_value is not None:
            result[mask_bool] = self.fill_value
        else:
            # Median of non-masked context pixels
            context_pixels = patch[~mask_bool]  # (N, 3)
            if len(context_pixels) > 0:
                median_colour = np.median(context_pixels, axis=0).astype(np.uint8)
            else:
                median_colour = np.array([128, 128, 128], dtype=np.uint8)
            result[mask_bool] = median_colour

        return result

    def memory_usage_mb(self) -> float | None:
        return 0.0
