"""
mask/expander.py

Morphologically dilates binary masks by a configurable radius.

Purpose
-------
OCR polygons are tight fits around detected glyphs.  A small dilation
ensures that anti-aliased glyph edges and slight OCR alignment errors
are fully covered before inpainting, reducing the risk of leaving
residual ink at region boundaries.

The dilation is applied independently to each MaskRegion and operates
only within the mask's bounding box for efficiency.  The full-image
mask array is updated in-place.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from core.models import MaskRegion


def expand_masks(
    regions: List[MaskRegion],
    expansion_px: int,
) -> List[MaskRegion]:
    """
    Dilate every MaskRegion's mask by *expansion_px* pixels.

    Parameters
    ----------
    regions : list[MaskRegion]
        Mask regions to expand.  Modified *in-place*; the same list is
        returned for chaining convenience.
    expansion_px : int
        Dilation radius in pixels.  0 means no expansion.

    Returns
    -------
    list[MaskRegion]
        The same list with updated masks and bounding boxes.
    """
    if expansion_px <= 0:
        return regions

    kernel = _make_kernel(expansion_px)

    for region in regions:
        # Dilate the full-image mask
        region.mask = cv2.dilate(region.mask, kernel, iterations=1)

        # Recompute tight bounding box from the expanded mask
        ys, xs = np.nonzero(region.mask)
        if len(xs) > 0:
            region.bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )

    return regions


def _make_kernel(radius: int) -> np.ndarray:
    """
    Create a circular structuring element for morphological dilation.

    A circle is preferred over a rectangle because it produces more
    uniform expansion in all directions, which reduces the chance of
    accidentally masking non-text art just outside a rectangular box.
    """
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (diameter, diameter),
    )
    return kernel
