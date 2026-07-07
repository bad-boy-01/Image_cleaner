"""
core/pipeline.py

Per-page pipeline orchestrator.

This module owns the logic of a single image pass:

    load → OCR → filter → mask → expand → patch → inpaint → paste → validate → save

It is intentionally free of any concrete OCR or inpainting logic.
Every backend is injected via the abstract interfaces, so the pipeline
does not change when backends are swapped.

Sequential processing
---------------------
Images are processed one at a time to keep GPU memory under control on
Kaggle.  Models are loaded once before the first page and unloaded once
after the last page.  The caller controls this lifecycle.

Error handling
--------------
If inpainting a single patch fails, the exception is caught, the
original pixels are kept, and processing continues.  A failed patch is
recorded in ``RepairResult.success = False``.

If a full page fails catastrophically (e.g. OOM), the exception
propagates to the caller which decides whether to abort or skip.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from backends.base import InpaintingBackend
from config.settings import Settings
from core.models import (
    ImagePage,
    MaskRegion,
    PageMetrics,
    Patch,
    PipelineTimer,
    RepairResult,
)
from debug.debug_saver import DebugSaver
from mask.expander import expand_masks
from mask.generator import generate_masks
from mask.language_filter import filter_non_english
from ocr.base import OCRBackend
from patch.extractor import extract_patches
from patch.paster import paste_results
from utils import image_io
from utils import logger
from validator.pixel_validator import ValidationResult, validate


class PagePipeline:
    """
    Processes a single ``ImagePage`` through the full cleaning pipeline.

    Parameters
    ----------
    ocr : OCRBackend
        Pre-loaded OCR backend.
    backend : InpaintingBackend
        Pre-loaded inpainting backend.
    settings : Settings
        Pipeline configuration.
    debug_saver : DebugSaver | None
        Debug artefact writer.  None when ``settings.debug_mode`` is False.
    """

    def __init__(
        self,
        ocr: OCRBackend,
        backend: InpaintingBackend,
        settings: Settings,
        debug_saver: Optional[DebugSaver] = None,
    ) -> None:
        self.ocr = ocr
        self.backend = backend
        self.settings = settings
        self.debug_saver = debug_saver

    def run(self, image_path: Path, output_path: Path) -> PageMetrics:
        """
        Execute the full cleaning pipeline for one image file.

        Parameters
        ----------
        image_path : Path
            Source image (PNG/JPG).
        output_path : Path
            Where the cleaned image should be saved.

        Returns
        -------
        PageMetrics
            Structured metrics for this page.
        """
        cfg = self.settings
        metrics = PageMetrics(page_name=image_path.name)

        with PipelineTimer() as timer:
            # ---------------------------------------------------------- #
            # 1. Load
            # ---------------------------------------------------------- #
            logger.step(f"Loading image: {image_path.name}")
            page = image_io.load_image(image_path)

            # ---------------------------------------------------------- #
            # 2. OCR detection
            # ---------------------------------------------------------- #
            logger.step("Detecting text regions (OCR)")
            detections = self.ocr.detect(page.image)
            metrics.ocr_detection_count = len(detections)
            logger.sub_step(f"{len(detections)} raw detections")

            # ---------------------------------------------------------- #
            # 3. Language filter
            # ---------------------------------------------------------- #
            logger.step("Filtering non-target languages")
            target_detections = filter_non_english(detections, cfg)
            metrics.non_english_count = len(target_detections)
            logger.sub_step(
                f"{len(target_detections)} region(s) scheduled for removal"
            )
            page.detections = target_detections

            if not target_detections:
                logger.sub_step("No non-English text found — page is clean")
                image_io.save_image(page, output_path, cfg.jpeg_quality)
                metrics.processing_time_s = timer.elapsed()
                return metrics

            # ---------------------------------------------------------- #
            # 4. Mask generation
            # ---------------------------------------------------------- #
            logger.step("Generating binary masks")
            h, w = page.image.shape[:2]
            raw_regions = generate_masks(target_detections, (h, w))
            logger.sub_step(f"{len(raw_regions)} region(s) after merging")

            if cfg.debug_mode and self.debug_saver:
                self.debug_saver.save_raw_masks(page, raw_regions)

            # ---------------------------------------------------------- #
            # 5. Mask expansion
            # ---------------------------------------------------------- #
            logger.step(f"Expanding masks by {cfg.mask_expansion_px}px")
            expanded_regions = expand_masks(raw_regions, cfg.mask_expansion_px)
            page.mask_regions = expanded_regions

            if cfg.debug_mode and self.debug_saver:
                self.debug_saver.save_expanded_masks(page, expanded_regions)

            metrics.mask_region_count = len(expanded_regions)
            metrics.mask_coverage_pct = _compute_coverage(expanded_regions, h, w)

            # ---------------------------------------------------------- #
            # 6. Patch extraction
            # ---------------------------------------------------------- #
            logger.step("Extracting local patches")
            patches = extract_patches(page, expanded_regions, cfg)
            page.patches = patches
            metrics.patch_count = len(patches)
            logger.sub_step(f"{len(patches)} patch(es) extracted")

            # ---------------------------------------------------------- #
            # 7. Inpainting
            # ---------------------------------------------------------- #
            logger.step("Running inpainting backend")
            results: List[RepairResult] = []
            for i, patch in enumerate(patches):
                logger.sub_step(
                    f"Cleaning patch {i + 1}/{len(patches)} "
                    f"({patch.size[0]}×{patch.size[1]}px)"
                )
                result = _inpaint_one(patch, self.backend)
                results.append(result)

                if cfg.debug_mode and self.debug_saver:
                    self.debug_saver.save_patch(page, patch, i)
                    if result.success:
                        self.debug_saver.save_inpainted_patch(page, result, i)

                if not result.success:
                    logger.warning(f"Patch {i + 1} failed: {result.error}")
                    metrics.repair_failure_count += 1
                else:
                    metrics.repair_success_count += 1

            page.repair_results = results

            # ---------------------------------------------------------- #
            # 8. Paste back
            # ---------------------------------------------------------- #
            logger.step("Pasting inpainted regions back into image")
            total_changed = paste_results(page, results)
            metrics.changed_pixels_total = total_changed
            logger.sub_step(f"{total_changed} pixel(s) modified")

            # ---------------------------------------------------------- #
            # 9. Pixel validation
            # ---------------------------------------------------------- #
            if cfg.enable_validator:
                logger.step("Running pixel validation")
                val_result = validate(
                    original=page.original_image,
                    cleaned=page.image,
                    approved_regions=expanded_regions,
                    tolerance=cfg.validator_tolerance,
                    return_diff=cfg.debug_mode,
                )
                metrics.validation_passed = val_result.passed
                metrics.validation_rogue_pixels = val_result.rogue_pixel_count

                if not val_result.passed:
                    logger.warning(
                        f"Validation failed: {val_result.rogue_pixel_count} rogue "
                        f"pixel(s) changed outside approved regions."
                    )
                else:
                    logger.sub_step(
                        f"Validation passed — {val_result.approved_pixel_count} "
                        f"approved pixel(s) modified"
                    )

                if cfg.debug_mode and self.debug_saver:
                    self.debug_saver.save_diff(page, val_result)

            # ---------------------------------------------------------- #
            # 10. Save
            # ---------------------------------------------------------- #
            logger.step(f"Saving cleaned image → {output_path.name}")
            image_io.save_image(page, output_path, cfg.jpeg_quality)

            metrics.processing_time_s = timer.elapsed()

        logger.page_done(image_path.name, metrics.processing_time_s)
        return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _inpaint_one(patch: Patch, backend: InpaintingBackend) -> RepairResult:
    """Call the backend for a single patch, catching any exception."""
    try:
        inpainted = backend.inpaint_patch(patch.image_crop, patch.mask_crop)
        return RepairResult(
            patch=patch,
            inpainted_crop=inpainted,
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        return RepairResult(
            patch=patch,
            inpainted_crop=patch.image_crop.copy(),  # keep original
            success=False,
            error=str(exc),
        )


def _compute_coverage(
    regions: List[MaskRegion], h: int, w: int
) -> float:
    """
    Compute the percentage of the image area covered by all mask regions.
    """
    if not regions:
        return 0.0
    combined = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        combined = np.maximum(combined, region.mask)
    covered = int((combined > 0).sum())
    total = h * w
    return (covered / total) * 100.0 if total > 0 else 0.0
