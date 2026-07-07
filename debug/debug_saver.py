"""
debug/debug_saver.py

Saves intermediate pipeline artefacts when debug mode is enabled.

Saved artefacts
---------------
For each page:

  <debug_dir>/<page_stem>/
      01_raw_masks.png        -- all raw OCR mask outlines overlaid on image
      02_expanded_masks.png   -- masks after morphological dilation
      03_patch_<N>.png        -- each patch crop with mask overlay
      04_inpainted_<N>.png    -- inpainted patch result
      05_diff.png             -- pixel diff (original vs cleaned)
      06_rogue_pixels.png     -- pixels that changed outside approved regions

All images are saved as PNG regardless of the original format so that
they are lossless and easy to inspect.

This module has no impact on the pipeline output — it is purely
observational.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from core.models import ImagePage, MaskRegion, Patch, RepairResult
from validator.pixel_validator import ValidationResult


class DebugSaver:
    """
    Writes debug visualisations to ``debug_dir / page_stem /``.

    Parameters
    ----------
    debug_dir : Path
        Root directory for all debug artefacts.
    """

    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = Path(debug_dir)

    def _page_dir(self, page: ImagePage) -> Path:
        d = self.debug_dir / page.path.stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ #
    # Save methods
    # ------------------------------------------------------------------ #

    def save_raw_masks(self, page: ImagePage, regions: List[MaskRegion]) -> None:
        """Overlay raw mask outlines on the original image."""
        canvas = page.original_image.copy()
        for region in regions:
            # Draw mask contours in red
            contours, _ = cv2.findContours(
                region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)
        self._save(canvas, self._page_dir(page) / "01_raw_masks.png")

    def save_expanded_masks(self, page: ImagePage, regions: List[MaskRegion]) -> None:
        """Overlay expanded mask outlines on the original image."""
        canvas = page.original_image.copy()
        for region in regions:
            contours, _ = cv2.findContours(
                region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 128, 255), 2)
        self._save(canvas, self._page_dir(page) / "02_expanded_masks.png")

    def save_patch(self, page: ImagePage, patch: Patch, index: int) -> None:
        """Save a patch crop with the mask overlaid as a semi-transparent tint."""
        crop = patch.image_crop.copy()
        tint = np.zeros_like(crop)
        tint[patch.mask_crop > 0] = (0, 200, 100)
        blended = cv2.addWeighted(crop, 0.7, tint, 0.3, 0)
        self._save(blended, self._page_dir(page) / f"03_patch_{index:03d}.png")

    def save_inpainted_patch(
        self, page: ImagePage, result: RepairResult, index: int
    ) -> None:
        """Save the inpainted patch output."""
        self._save(
            result.inpainted_crop,
            self._page_dir(page) / f"04_inpainted_{index:03d}.png",
        )

    def save_diff(
        self,
        page: ImagePage,
        validation: ValidationResult,
    ) -> None:
        """Save the diff image and rogue-pixel overlay."""
        pd = self._page_dir(page)

        if validation.diff_image is not None:
            # Amplify the diff for visibility (×10)
            amplified = np.clip(validation.diff_image.astype(np.uint32) * 10, 0, 255).astype(np.uint8)
            diff_rgb = cv2.applyColorMap(amplified, cv2.COLORMAP_HOT)
            self._save(diff_rgb, pd / "05_diff.png")

        if validation.rogue_mask is not None and validation.rogue_mask.any():
            canvas = page.original_image.copy()
            canvas[validation.rogue_mask] = (255, 0, 255)  # magenta = rogue pixels
            self._save(canvas, pd / "06_rogue_pixels.png")

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save(image: np.ndarray, path: Path) -> None:
        """Write *image* (RGB uint8) to *path* as PNG via OpenCV (BGR)."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
