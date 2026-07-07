"""
patch/extractor.py

Extracts local image patches around each MaskRegion, ready for inpainting.

Dynamic patching strategy
--------------------------
Rather than using a fixed patch size (e.g. always 512×512), the patch
dimensions are computed from the mask bounding box plus configurable
padding:

    raw_w = (x_max - x_min) + 2 * patch_padding
    raw_h = (y_max - y_min) + 2 * patch_padding

    # Align to model requirements (most models need multiples of 8)
    patch_w = ceil(raw_w / patch_align) * patch_align
    patch_h = ceil(raw_h / patch_align) * patch_align

    # Hard cap to prevent OOM
    patch_w = min(patch_w, max_patch_size)
    patch_h = min(patch_h, max_patch_size)

Benefits
--------
* Small text regions produce small patches → less GPU memory.
* Large speech bubbles produce larger patches → better context for inpainting.
* All patches are divisible by ``patch_align`` → compatible with all major
  inpainting models without resizing.

Important
---------
The patch is cropped from the image at its *natural* size.  The patch
is never resized or scaled before being passed to the backend.  If the
backend requires a specific size, that is the backend's responsibility.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from core.models import ImagePage, MaskRegion, Patch
from config.settings import Settings


def extract_patches(
    page: ImagePage,
    regions: List[MaskRegion],
    settings: Settings,
) -> List[Patch]:
    """
    Produce one ``Patch`` per ``MaskRegion``.

    Parameters
    ----------
    page : ImagePage
        The page being processed.  ``page.image`` is read but not modified.
    regions : list[MaskRegion]
        Mask regions to extract patches for.
    settings : Settings
        Pipeline configuration providing padding, alignment, and size cap.

    Returns
    -------
    list[Patch]
        One Patch per region, in the same order as *regions*.
    """
    h, w = page.image.shape[:2]
    patches: List[Patch] = []

    for region in regions:
        patch = _extract_one(page.image, region, h, w, settings)
        patches.append(patch)

    return patches


def _extract_one(
    image: np.ndarray,
    region: MaskRegion,
    img_h: int,
    img_w: int,
    settings: Settings,
) -> Patch:
    """Extract and return a single Patch for *region*."""
    x0_bb, y0_bb, x1_bb, y1_bb = region.bbox

    # ------------------------------------------------------------------ #
    # Compute padded, aligned, capped crop coordinates
    # ------------------------------------------------------------------ #
    pad = settings.patch_padding
    align = settings.patch_align
    cap = settings.max_patch_size

    # Padded extents (before alignment)
    raw_x0 = x0_bb - pad
    raw_y0 = y0_bb - pad
    raw_x1 = x1_bb + pad
    raw_y1 = y1_bb + pad

    # Dimensions
    raw_w = raw_x1 - raw_x0
    raw_h = raw_y1 - raw_y0

    # Align upward
    aligned_w = math.ceil(raw_w / align) * align
    aligned_h = math.ceil(raw_h / align) * align

    # Apply hard cap
    aligned_w = min(aligned_w, cap)
    aligned_h = min(aligned_h, cap)

    # Centre the aligned patch on the padded extents
    centre_x = (raw_x0 + raw_x1) // 2
    centre_y = (raw_y0 + raw_y1) // 2

    x0 = centre_x - aligned_w // 2
    y0 = centre_y - aligned_h // 2
    x1 = x0 + aligned_w
    y1 = y0 + aligned_h

    # Clamp to image boundaries
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)

    # ------------------------------------------------------------------ #
    # Crop image and mask
    # ------------------------------------------------------------------ #
    image_crop = image[y0:y1, x0:x1].copy()
    mask_crop = region.mask[y0:y1, x0:x1].copy()

    return Patch(
        image_crop=image_crop,
        mask_crop=mask_crop,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        source_region=region,
    )
