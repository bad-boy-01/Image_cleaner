"""
patch/paster.py

Pastes inpainted patch crops back into the full image.

Critical invariant
------------------
Only pixels that are non-zero in the *mask_crop* are written back.
All other pixels in the patch are ignored, even if the inpainting model
produced slightly different values for them (e.g. from context padding).

This guarantees pixel-perfect preservation of everything outside the
approved repair regions and is the enforcement layer for the spec's
highest-priority requirement.

Implementation
--------------
For each RepairResult:
  1. Build a boolean selector from ``patch.mask_crop > 0``.
  2. Apply the selector to write ``inpainted_crop`` pixels into
     ``page.image`` at the patch's (x0, y0)→(x1, y1) coordinates.
  3. Count modified pixels for metrics.

Failed repairs (``result.success == False``) are skipped; the original
pixels remain.
"""

from __future__ import annotations

from typing import List

import numpy as np

from core.models import ImagePage, RepairResult


def paste_results(
    page: ImagePage,
    results: List[RepairResult],
) -> int:
    """
    Write inpainted patches back into *page.image*.

    Parameters
    ----------
    page : ImagePage
        The page being processed.  ``page.image`` is modified in-place.
    results : list[RepairResult]
        Inpainting results from the backend stage.

    Returns
    -------
    int
        Total number of pixels that were actually changed across all patches.
    """
    total_changed = 0

    for result in results:
        if not result.success:
            continue

        patch = result.patch
        y0, y1 = patch.y0, patch.y1
        x0, x1 = patch.x0, patch.x1

        # Boolean selector from mask (True = belongs to repair region)
        selector = patch.mask_crop > 0  # shape (H, W)

        # Extract the region currently in the output image
        current_region = page.image[y0:y1, x0:x1]

        # Compute changed pixel count before writing
        inpainted = result.inpainted_crop
        diff = np.any(current_region[selector] != inpainted[selector], axis=-1)
        changed = int(diff.sum())

        # Write inpainted pixels only where the mask is active
        current_region[selector] = inpainted[selector]
        page.image[y0:y1, x0:x1] = current_region

        result.changed_pixel_count = changed
        total_changed += changed

    return total_changed
