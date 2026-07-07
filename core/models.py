"""
core/models.py

Canonical data models for the image-cleaning pipeline.

These dataclasses are the *lingua franca* between pipeline stages.
Every module speaks in these types — never in raw numpy arrays or
dictionaries — which makes each stage independently testable and
the contracts between stages explicit.

Hierarchy
---------
    ImagePage
        └─ list[TextDetection]   (from OCR)
            └─ list[MaskRegion]  (after mask generation + filtering)
                └─ list[Patch]   (after patch extraction)
                    └─ RepairResult  (after inpainting + paste-back)
    PageMetrics  (aggregated after the full page is processed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level detection primitives
# ---------------------------------------------------------------------------

@dataclass
class TextDetection:
    """
    A single text region returned by an OCR backend.

    Attributes
    ----------
    polygon : np.ndarray
        Ordered (N, 2) array of (x, y) vertex coordinates in pixel space.
        For axis-aligned boxes, N=4; free-form polygons may have more vertices.
    text : str
        The OCR-decoded text string (used only for language classification).
    language : str
        ISO 639-1 language code inferred by the OCR backend, e.g. ``"ko"``.
    confidence : float
        Detection confidence in [0, 1].
    script : str
        Unicode script name inferred from ``text``, e.g. ``"Hangul"``,
        ``"Latin"``, ``"CJK"``.  Populated by the language filter.
    """
    polygon: np.ndarray          # shape (N, 2), dtype int32
    text: str
    language: str
    confidence: float
    script: str = ""

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box: (x_min, y_min, x_max, y_max)."""
        xs = self.polygon[:, 0]
        ys = self.polygon[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Mask primitives
# ---------------------------------------------------------------------------

@dataclass
class MaskRegion:
    """
    A binary repair mask derived from one or more ``TextDetection`` objects.

    The mask is always stored relative to the *full image* coordinate system
    so it can be overlaid or validated without coordinate translation.

    Attributes
    ----------
    mask : np.ndarray
        Boolean/uint8 array with the same H×W as the source image.
        255 (or True) = pixel belongs to the region to be repaired.
    bbox : tuple
        Axis-aligned tight bounding box of the non-zero mask area:
        ``(x_min, y_min, x_max, y_max)``.
    source_detections : list[TextDetection]
        The OCR detections that were merged to form this region.
    """
    mask: np.ndarray             # shape (H, W), dtype uint8 {0, 255}
    bbox: Tuple[int, int, int, int]
    source_detections: List[TextDetection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Patch primitives
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    """
    A rectangular crop of the image and its corresponding binary mask,
    ready to be sent to an inpainting backend.

    Coordinates are in *full-image* pixel space so the patch can be
    pasted back without ambiguity.

    Attributes
    ----------
    image_crop : np.ndarray
        RGB uint8 array of shape (H, W, 3).
    mask_crop : np.ndarray
        uint8 array of shape (H, W) with values {0, 255}.
    x0, y0 : int
        Top-left corner of the patch in the full image.
    x1, y1 : int
        Bottom-right corner (exclusive) of the patch in the full image.
    source_region : MaskRegion
        The mask region that triggered this patch.
    """
    image_crop: np.ndarray       # (H, W, 3) uint8
    mask_crop: np.ndarray        # (H, W)    uint8
    x0: int
    y0: int
    x1: int
    y1: int
    source_region: MaskRegion

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) of this patch."""
        return self.x1 - self.x0, self.y1 - self.y0


# ---------------------------------------------------------------------------
# Repair result
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """
    The output produced by an inpainting backend for a single ``Patch``.

    Attributes
    ----------
    patch : Patch
        The input patch that was inpainted.
    inpainted_crop : np.ndarray
        The model's output: same shape as ``patch.image_crop``.
    changed_pixel_count : int
        Number of pixels that differ between the original crop and
        the inpainted crop (informational, populated by the paster).
    success : bool
        False if the backend raised an exception; the original crop is
        kept and the failure is logged.
    error : str, optional
        Error message if ``success=False``.
    """
    patch: Patch
    inpainted_crop: np.ndarray   # (H, W, 3) uint8
    changed_pixel_count: int = 0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Page-level container
# ---------------------------------------------------------------------------

@dataclass
class ImagePage:
    """
    Represents a single image page as it flows through the pipeline.

    ``image`` is always stored as a *mutable* RGB uint8 numpy array.
    The original pixels are captured in ``original_image`` before any
    inpainting takes place so the validator can compare them.

    Attributes
    ----------
    path : Path
        Absolute path to the source image file.
    image : np.ndarray
        Current (mutable) RGB uint8 array — modified in-place by the
        paste-back stage.
    original_image : np.ndarray
        Immutable snapshot taken immediately after loading.
    fmt : str
        Lowercase format string: ``"png"`` or ``"jpeg"``.
    dpi : tuple[float, float] | None
        Original DPI metadata, e.g. ``(300.0, 300.0)``.
    extra_metadata : dict
        Any additional Pillow ``info`` dict entries (e.g. ICC profile).
    detections : list[TextDetection]
        Populated by the OCR stage.
    mask_regions : list[MaskRegion]
        Populated by the mask generation + expansion + filter stages.
    patches : list[Patch]
        Populated by the patch extraction stage.
    repair_results : list[RepairResult]
        Populated by the inpainting + paste-back stage.
    """
    path: Path
    image: np.ndarray            # (H, W, 3) uint8, mutable
    original_image: np.ndarray   # (H, W, 3) uint8, immutable snapshot
    fmt: str                     # "png" | "jpeg"
    dpi: Optional[Tuple[float, float]]
    extra_metadata: dict = field(default_factory=dict)
    detections: List[TextDetection] = field(default_factory=list)
    mask_regions: List[MaskRegion] = field(default_factory=list)
    patches: List[Patch] = field(default_factory=list)
    repair_results: List[RepairResult] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def shape(self) -> Tuple[int, int]:
        """(height, width) of the image."""
        h, w = self.image.shape[:2]
        return h, w


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class PageMetrics:
    """
    Structured per-page metrics collected during pipeline execution.

    Attributes
    ----------
    page_name : str
        Filename of the processed page.
    processing_time_s : float
        Wall-clock seconds from load to save.
    ocr_detection_count : int
        Total number of raw OCR detections.
    non_english_count : int
        Detections retained after language filtering.
    mask_region_count : int
        Number of merged mask regions.
    mask_coverage_pct : float
        Percentage of the image area covered by all masks.
    patch_count : int
        Number of patches sent to the inpainting backend.
    repair_success_count : int
        Patches that completed without error.
    repair_failure_count : int
        Patches that failed (original crop retained).
    changed_pixels_total : int
        Sum of changed pixels across all repair results.
    validation_passed : bool
        Whether the pixel validator found no unexpected changes.
    validation_rogue_pixels : int
        Pixels outside approved regions that changed (0 if passed).
    """
    page_name: str
    processing_time_s: float = 0.0
    ocr_detection_count: int = 0
    non_english_count: int = 0
    mask_region_count: int = 0
    mask_coverage_pct: float = 0.0
    patch_count: int = 0
    repair_success_count: int = 0
    repair_failure_count: int = 0
    changed_pixels_total: int = 0
    validation_passed: bool = True
    validation_rogue_pixels: int = 0

    def as_dict(self) -> dict:
        """Serialise to a plain dict for JSON/SQLite storage."""
        return {
            "page_name": self.page_name,
            "processing_time_s": round(self.processing_time_s, 3),
            "ocr_detection_count": self.ocr_detection_count,
            "non_english_count": self.non_english_count,
            "mask_region_count": self.mask_region_count,
            "mask_coverage_pct": round(self.mask_coverage_pct, 4),
            "patch_count": self.patch_count,
            "repair_success_count": self.repair_success_count,
            "repair_failure_count": self.repair_failure_count,
            "changed_pixels_total": self.changed_pixels_total,
            "validation_passed": int(self.validation_passed),
            "validation_rogue_pixels": self.validation_rogue_pixels,
        }


@dataclass
class PipelineTimer:
    """Simple context-manager wall-clock timer."""
    _start: float = field(default_factory=time.monotonic, init=False)

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def __enter__(self) -> "PipelineTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        pass
