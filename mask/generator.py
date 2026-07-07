"""
mask/generator.py

Converts a list of TextDetection objects into binary MaskRegion objects.

Strategy
--------
1. Each TextDetection's polygon is rasterised onto a full-image boolean
   canvas using OpenCV's ``fillPoly``.
2. Detections whose bounding boxes overlap significantly are *merged*
   into a single MaskRegion to reduce patch count and avoid split
   repairs on adjacent speech bubbles.
3. Each merged region's tight bounding box is recomputed from the
   non-zero pixels of the merged mask.

This module has no dependency on any AI model or OCR backend.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from core.models import MaskRegion, TextDetection


def generate_masks(
    detections: List[TextDetection],
    image_shape: Tuple[int, int],
    merge_iou_threshold: float = 0.0,
) -> List[MaskRegion]:
    """
    Convert OCR detections to merged binary MaskRegion objects.

    Parameters
    ----------
    detections : list[TextDetection]
        Detections that have been filtered and are scheduled for removal.
    image_shape : tuple[int, int]
        ``(height, width)`` of the full image.
    merge_iou_threshold : float
        Bounding-box IoU above which two detections are merged into one
        region.  0.0 means any overlap triggers a merge.

    Returns
    -------
    list[MaskRegion]
        One MaskRegion per merged group.  Each mask covers the full
        image dimensions but is non-zero only inside the merged region.
    """
    if not detections:
        return []

    h, w = image_shape

    # ------------------------------------------------------------------ #
    # Step 1: Rasterise each detection polygon into a per-detection mask
    # ------------------------------------------------------------------ #
    individual_masks: List[np.ndarray] = []
    for det in detections:
        canvas = np.zeros((h, w), dtype=np.uint8)
        pts = det.polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(canvas, [pts], 255)
        individual_masks.append(canvas)

    # ------------------------------------------------------------------ #
    # Step 2: Merge overlapping masks (union-find by bbox overlap)
    # ------------------------------------------------------------------ #
    groups = _merge_overlapping(detections, individual_masks, merge_iou_threshold)

    # ------------------------------------------------------------------ #
    # Step 3: Build MaskRegion for each group
    # ------------------------------------------------------------------ #
    regions: List[MaskRegion] = []
    for det_indices in groups:
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for idx in det_indices:
            merged_mask = np.maximum(merged_mask, individual_masks[idx])

        # Tight bounding box from non-zero pixels
        ys, xs = np.nonzero(merged_mask)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        regions.append(
            MaskRegion(
                mask=merged_mask,
                bbox=bbox,
                source_detections=[detections[i] for i in det_indices],
            )
        )

    return regions


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    Compute bounding-box Intersection over Union.

    Boxes are (x_min, y_min, x_max, y_max).
    Returns 0.0 if there is no overlap.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _merge_overlapping(
    detections: List[TextDetection],
    masks: List[np.ndarray],
    iou_threshold: float,
) -> List[List[int]]:
    """
    Group detection indices by overlapping bounding boxes using union-find.
    """
    n = len(detections)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    bboxes = [det.bbox for det in detections]

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou(bboxes[i], bboxes[j]) > iou_threshold:
                union(i, j)

    # Collect groups
    groups: dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())
