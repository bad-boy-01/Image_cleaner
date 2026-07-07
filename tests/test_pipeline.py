"""
tests/test_pipeline.py

Integration tests for the full per-page pipeline using the mock backend.

These tests run without a GPU and without real OCR — they verify the
wiring, data flow, and pixel-preservation invariant of the full pipeline.

The mock OCR backend is injected directly (bypassing the registry) to
simulate specific detection scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from backends.mock_backend import MockBackend
from config.settings import Settings
from core.models import ImagePage, TextDetection
from core.pipeline import PagePipeline
from ocr.base import OCRBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeOCR(OCRBackend):
    """OCR backend that returns a preconfigured list of detections."""

    def __init__(self, detections: List[TextDetection]) -> None:
        self._detections = detections

    def load(self) -> None: pass
    def unload(self) -> None: pass

    def detect(self, image: np.ndarray) -> List[TextDetection]:
        return self._detections


def _make_detection(x0: int, y0: int, x1: int, y1: int,
                    script: str = "Hangul") -> TextDetection:
    polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
    return TextDetection(
        polygon=polygon, text="한", language="ko",
        confidence=0.9, script=script,
    )


def _make_white_image(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _default_settings(tmp_path: Path) -> Settings:
    return Settings(
        input_zip=tmp_path / "in.zip",
        output_zip=tmp_path / "out.zip",
        work_dir=tmp_path / "work",
        inpainting_backend="mock",
        enable_validator=True,
        debug_mode=False,
        resume=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineNoText:
    def test_clean_page_saves_unmodified(self, tmp_path: Path) -> None:
        """A page with no non-English text should be saved pixel-identical."""
        # OCR returns only English
        ocr = _FakeOCR([_make_detection(10, 10, 50, 50, script="Latin")])
        backend = MockBackend()
        backend.load_model()
        cfg = _default_settings(tmp_path)

        # Write a test image
        src = tmp_path / "001.png"
        from PIL import Image
        arr = _make_white_image()
        Image.fromarray(arr).save(str(src))

        out = tmp_path / "001_out.png"
        pipeline = PagePipeline(ocr=ocr, backend=backend, settings=cfg)
        metrics = pipeline.run(src, out)

        # No changes expected
        assert metrics.non_english_count == 0
        assert metrics.patch_count == 0
        assert metrics.validation_passed

        # Output should exist and dimensions match
        result = np.array(Image.open(str(out)).convert("RGB"))
        np.testing.assert_array_equal(arr, result)


class TestPipelineWithText:
    def test_pixels_outside_mask_unchanged(self, tmp_path: Path) -> None:
        """
        After inpainting, every pixel outside the mask must be identical
        to the original — this is the core pixel-preservation invariant.
        """
        h, w = 200, 200
        arr = _make_white_image(h, w)
        # Draw a black rectangle simulating text
        arr[50:80, 60:120] = 0

        src = tmp_path / "001.png"
        from PIL import Image
        Image.fromarray(arr).save(str(src))

        # OCR finds the text region
        det = _make_detection(60, 50, 120, 80, script="Hangul")
        ocr = _FakeOCR([det])
        backend = MockBackend(fill_value=200)  # fills with grey=200
        backend.load_model()

        cfg = _default_settings(tmp_path)
        cfg = cfg.model_copy(update={"mask_expansion_px": 2})

        out = tmp_path / "001_out.png"
        pipeline = PagePipeline(ocr=ocr, backend=backend, settings=cfg)
        metrics = pipeline.run(src, out)

        assert metrics.non_english_count >= 1
        assert metrics.patch_count >= 1

        result = np.array(Image.open(str(out)).convert("RGB"))

        # Build the approved mask (union of all masks with expansion)
        from mask.generator import generate_masks
        from mask.expander import expand_masks
        regions = generate_masks([det], (h, w))
        expand_masks(regions, 2)
        approved = np.zeros((h, w), dtype=bool)
        for r in regions:
            approved |= r.mask > 0

        # Every pixel OUTSIDE the approved mask must be identical
        outside_original = arr[~approved]
        outside_result = result[~approved]
        np.testing.assert_array_equal(
            outside_original, outside_result,
            err_msg="Pixels outside approved mask changed — pixel-preservation violated!"
        )

    def test_metrics_populated(self, tmp_path: Path) -> None:
        """Metrics should be filled in for a page that has text."""
        src = tmp_path / "002.png"
        from PIL import Image
        Image.fromarray(_make_white_image()).save(str(src))

        det = _make_detection(20, 20, 80, 60, script="CJK")
        ocr = _FakeOCR([det])
        backend = MockBackend()
        backend.load_model()

        cfg = _default_settings(tmp_path)
        out = tmp_path / "002_out.png"
        pipeline = PagePipeline(ocr=ocr, backend=backend, settings=cfg)
        metrics = pipeline.run(src, out)

        assert metrics.ocr_detection_count == 1
        assert metrics.non_english_count == 1
        assert metrics.mask_region_count == 1
        assert metrics.patch_count == 1
        assert metrics.repair_success_count == 1
        assert metrics.repair_failure_count == 0
        assert metrics.processing_time_s > 0


class TestPatchExtractor:
    def test_patch_alignment(self, tmp_path: Path) -> None:
        """Patch dimensions must be divisible by patch_align."""
        from config.settings import Settings
        from core.models import ImagePage
        from patch.extractor import extract_patches
        from mask.generator import generate_masks

        cfg = Settings(input_zip="x.zip", patch_align=8, patch_padding=16)
        det = _make_detection(10, 10, 37, 53)  # odd-sized region

        from pathlib import Path as P
        arr = _make_white_image(200, 200)
        page = ImagePage(
            path=P("test.png"), image=arr.copy(), original_image=arr.copy(),
            fmt="png", dpi=None,
        )
        regions = generate_masks([det], (200, 200))
        patches = extract_patches(page, regions, cfg)

        assert len(patches) == 1
        pw, ph = patches[0].size
        assert pw % 8 == 0, f"Patch width {pw} not divisible by 8"
        assert ph % 8 == 0, f"Patch height {ph} not divisible by 8"
