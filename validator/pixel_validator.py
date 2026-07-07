"""
validator/pixel_validator.py

Validates that the cleaned image differs from the original *only* inside
the approved repair regions.

Algorithm
---------
1. Compute a per-pixel absolute difference between ``original`` and
   ``cleaned``: ``diff = |original - cleaned|``.
2. Identify pixels where the maximum channel diff > ``tolerance``:
   ``changed = diff.max(axis=2) > tolerance``.
3. Build an ``approved_mask`` from all MaskRegions (union of all masks).
4. Compute ``rogue = changed & ~approved_mask``.
5. If ``rogue.sum() > 0`` → validation fails.

The ``tolerance`` parameter absorbs trivial rounding errors from JPEG
round-trips (typically ≤ 2 per channel) without hiding genuine bugs.

Design
------
The validator is deliberately stateless.  It takes plain numpy arrays
and returns a ``ValidationResult`` dataclass — making it trivially
unit-testable without any pipeline plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from core.models import MaskRegion


@dataclass
class ValidationResult:
    """
    Result of a single pixel-level validation check.

    Attributes
    ----------
    passed : bool
        True if no unexpected pixel changes were detected.
    rogue_pixel_count : int
        Number of pixels that changed outside the approved mask.
    approved_pixel_count : int
        Number of pixels inside approved regions that changed.
    diff_image : np.ndarray | None
        Grayscale diff image for debug output.  None unless
        ``return_diff=True`` was passed to ``validate``.
    rogue_mask : np.ndarray | None
        Boolean mask highlighting rogue pixels.  None unless
        ``return_diff=True`` was passed to ``validate``.
    """
    passed: bool
    rogue_pixel_count: int
    approved_pixel_count: int
    diff_image: np.ndarray | None = None  # (H, W) uint8
    rogue_mask: np.ndarray | None = None  # (H, W) bool


def validate(
    original: np.ndarray,
    cleaned: np.ndarray,
    approved_regions: List[MaskRegion],
    tolerance: int = 2,
    return_diff: bool = False,
) -> ValidationResult:
    """
    Compare *original* and *cleaned* images, asserting that changes are
    confined to *approved_regions*.

    Parameters
    ----------
    original : np.ndarray
        Original RGB uint8 image (H, W, 3).
    cleaned : np.ndarray
        Cleaned RGB uint8 image (H, W, 3).  Must have the same shape.
    approved_regions : list[MaskRegion]
        All mask regions that were submitted for inpainting.
    tolerance : int
        Per-channel absolute difference below which a pixel change is
        considered negligible (absorbs JPEG round-trip noise).
    return_diff : bool
        If True, include ``diff_image`` and ``rogue_mask`` in the result
        for debug visualisation.

    Returns
    -------
    ValidationResult
    """
    if original.shape != cleaned.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs cleaned {cleaned.shape}."
        )

    # ------------------------------------------------------------------ #
    # Step 1: Compute per-pixel change map
    # ------------------------------------------------------------------ #
    diff = np.abs(original.astype(np.int32) - cleaned.astype(np.int32))  # (H, W, 3)
    max_diff = diff.max(axis=2)                                            # (H, W)
    changed = max_diff > tolerance                                          # (H, W) bool

    # ------------------------------------------------------------------ #
    # Step 2: Build approved mask (union of all repair regions)
    # ------------------------------------------------------------------ #
    h, w = original.shape[:2]
    approved_mask = np.zeros((h, w), dtype=bool)
    for region in approved_regions:
        approved_mask |= region.mask > 0

    # ------------------------------------------------------------------ #
    # Step 3: Identify rogue changes
    # ------------------------------------------------------------------ #
    rogue = changed & ~approved_mask
    rogue_count = int(rogue.sum())
    approved_count = int((changed & approved_mask).sum())

    passed = rogue_count == 0

    diff_image: np.ndarray | None = None
    rogue_mask_arr: np.ndarray | None = None

    if return_diff:
        diff_image = np.clip(max_diff, 0, 255).astype(np.uint8)
        rogue_mask_arr = rogue

    return ValidationResult(
        passed=passed,
        rogue_pixel_count=rogue_count,
        approved_pixel_count=approved_count,
        diff_image=diff_image,
        rogue_mask=rogue_mask_arr,
    )
