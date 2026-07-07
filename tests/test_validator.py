"""
tests/test_validator.py

Unit tests for the pixel validator.

The validator is purely functional (numpy in, dataclass out), so it is
easy to test exhaustively without any pipeline plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.models import MaskRegion
from validator.pixel_validator import validate


def _make_image(h: int = 50, w: int = 50, fill: int = 128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _make_region(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> MaskRegion:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    bbox = (x0, y0, x1, y1)
    return MaskRegion(mask=mask, bbox=bbox, source_detections=[])


class TestValidatorPass:
    def test_identical_images_pass(self) -> None:
        img = _make_image()
        result = validate(img, img.copy(), approved_regions=[])
        assert result.passed
        assert result.rogue_pixel_count == 0

    def test_changes_inside_approved_region_pass(self) -> None:
        h, w = 50, 50
        original = _make_image(h, w, fill=100)
        cleaned = original.copy()
        region = _make_region(h, w, 10, 10, 20, 20)
        # Modify pixels inside the approved region
        cleaned[10:20, 10:20] = 200
        result = validate(original, cleaned, approved_regions=[region])
        assert result.passed
        assert result.rogue_pixel_count == 0

    def test_tolerance_absorbs_jpeg_noise(self) -> None:
        h, w = 50, 50
        original = _make_image(h, w, fill=128)
        cleaned = original.copy()
        # Add ±2 noise everywhere (simulating JPEG round-trip)
        cleaned[:, :, 0] = np.clip(cleaned[:, :, 0].astype(int) + 2, 0, 255)
        result = validate(original, cleaned, approved_regions=[], tolerance=2)
        assert result.passed

    def test_multiple_approved_regions(self) -> None:
        h, w = 100, 100
        original = _make_image(h, w)
        cleaned = original.copy()
        r1 = _make_region(h, w, 0, 0, 20, 20)
        r2 = _make_region(h, w, 50, 50, 70, 70)
        cleaned[0:20, 0:20] = 255
        cleaned[50:70, 50:70] = 0
        result = validate(original, cleaned, approved_regions=[r1, r2])
        assert result.passed


class TestValidatorFail:
    def test_changes_outside_approved_region_fail(self) -> None:
        h, w = 50, 50
        original = _make_image(h, w, fill=100)
        cleaned = original.copy()
        region = _make_region(h, w, 10, 10, 20, 20)  # approved: top-left area
        # Modify pixels OUTSIDE the approved region
        cleaned[40, 40] = 200
        result = validate(original, cleaned, approved_regions=[region], tolerance=0)
        assert not result.passed
        assert result.rogue_pixel_count >= 1

    def test_no_approved_regions_any_change_fails(self) -> None:
        h, w = 50, 50
        original = _make_image(h, w)
        cleaned = original.copy()
        cleaned[25, 25] = (0, 0, 0)
        result = validate(original, cleaned, approved_regions=[], tolerance=0)
        assert not result.passed

    def test_shape_mismatch_raises(self) -> None:
        original = _make_image(50, 50)
        cleaned = _make_image(60, 50)
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate(original, cleaned, approved_regions=[])


class TestValidatorDiff:
    def test_return_diff_populates_images(self) -> None:
        h, w = 50, 50
        original = _make_image(h, w, fill=100)
        cleaned = original.copy()
        region = _make_region(h, w, 10, 10, 20, 20)
        cleaned[10:20, 10:20] = 200
        result = validate(original, cleaned, approved_regions=[region],
                          return_diff=True)
        assert result.diff_image is not None
        assert result.diff_image.shape == (h, w)
        assert result.rogue_mask is not None

    def test_return_diff_false_leaves_none(self) -> None:
        img = _make_image()
        result = validate(img, img.copy(), approved_regions=[], return_diff=False)
        assert result.diff_image is None
        assert result.rogue_mask is None
