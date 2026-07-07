"""
tests/test_mask.py

Unit tests for mask generation, expansion, and language filtering.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.models import TextDetection
from mask.expander import expand_masks
from mask.generator import generate_masks
from mask.language_filter import filter_non_english


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_det(x0: int, y0: int, x1: int, y1: int,
              script: str = "Hangul", lang: str = "ko") -> TextDetection:
    polygon = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32
    )
    return TextDetection(
        polygon=polygon, text="한", language=lang,
        confidence=0.9, script=script,
    )


# ---------------------------------------------------------------------------
# language_filter
# ---------------------------------------------------------------------------

class TestLanguageFilter:
    def _settings(self, remove_english: bool = False):
        from config.settings import Settings
        return Settings(
            input_zip="x.zip",
            target_scripts=["Hangul", "Hiragana", "Katakana", "CJK"],
            remove_english=remove_english,
        )

    def test_keeps_cjk(self) -> None:
        dets = [_make_det(0, 0, 10, 10, script="CJK", lang="zh")]
        result = filter_non_english(dets, self._settings())
        assert len(result) == 1

    def test_keeps_hangul(self) -> None:
        dets = [_make_det(0, 0, 10, 10, script="Hangul", lang="ko")]
        result = filter_non_english(dets, self._settings())
        assert len(result) == 1

    def test_drops_latin_by_default(self) -> None:
        dets = [_make_det(0, 0, 10, 10, script="Latin", lang="en")]
        result = filter_non_english(dets, self._settings())
        assert len(result) == 0

    def test_removes_latin_when_flag_set(self) -> None:
        dets = [_make_det(0, 0, 10, 10, script="Latin", lang="en")]
        result = filter_non_english(dets, self._settings(remove_english=True))
        assert len(result) == 1

    def test_mixed_list(self) -> None:
        dets = [
            _make_det(0, 0, 10, 10, script="Hangul"),
            _make_det(20, 20, 30, 30, script="Latin", lang="en"),
            _make_det(40, 40, 50, 50, script="CJK", lang="zh"),
        ]
        result = filter_non_english(dets, self._settings())
        assert len(result) == 2
        scripts = {d.script for d in result}
        assert "Latin" not in scripts


# ---------------------------------------------------------------------------
# generate_masks
# ---------------------------------------------------------------------------

class TestGenerateMasks:
    def test_single_detection(self) -> None:
        det = _make_det(10, 10, 30, 30)
        regions = generate_masks([det], image_shape=(100, 100))
        assert len(regions) == 1
        region = regions[0]
        assert region.mask.shape == (100, 100)
        # The bbox should cover the polygon
        x0, y0, x1, y1 = region.bbox
        assert x0 <= 10 and y0 <= 10
        assert x1 >= 30 and y1 >= 30

    def test_mask_is_nonzero_inside_polygon(self) -> None:
        det = _make_det(10, 10, 30, 30)
        regions = generate_masks([det], image_shape=(100, 100))
        mask = regions[0].mask
        # Centre of the polygon should be filled
        assert mask[20, 20] == 255

    def test_overlapping_detections_merged(self) -> None:
        det1 = _make_det(10, 10, 40, 40)
        det2 = _make_det(35, 35, 60, 60)  # overlaps with det1
        regions = generate_masks([det1, det2], image_shape=(100, 100))
        # Should be merged into one region
        assert len(regions) == 1
        assert len(regions[0].source_detections) == 2

    def test_non_overlapping_detections_separate(self) -> None:
        det1 = _make_det(0, 0, 10, 10)
        det2 = _make_det(50, 50, 60, 60)
        regions = generate_masks([det1, det2], image_shape=(100, 100))
        assert len(regions) == 2

    def test_empty_detections(self) -> None:
        regions = generate_masks([], image_shape=(100, 100))
        assert regions == []


# ---------------------------------------------------------------------------
# expand_masks
# ---------------------------------------------------------------------------

class TestExpandMasks:
    def test_expansion_increases_coverage(self) -> None:
        det = _make_det(40, 40, 60, 60)
        regions_before = generate_masks([det], image_shape=(100, 100))
        coverage_before = regions_before[0].mask.sum()

        regions_after = generate_masks([det], image_shape=(100, 100))
        expand_masks(regions_after, expansion_px=5)
        coverage_after = regions_after[0].mask.sum()

        assert coverage_after > coverage_before

    def test_zero_expansion_unchanged(self) -> None:
        det = _make_det(40, 40, 60, 60)
        regions = generate_masks([det], image_shape=(100, 100))
        mask_before = regions[0].mask.copy()
        expand_masks(regions, expansion_px=0)
        np.testing.assert_array_equal(regions[0].mask, mask_before)

    def test_bbox_updated_after_expansion(self) -> None:
        det = _make_det(40, 40, 60, 60)
        regions = generate_masks([det], image_shape=(100, 100))
        bbox_before = regions[0].bbox
        expand_masks(regions, expansion_px=5)
        bbox_after = regions[0].bbox
        assert bbox_after[0] < bbox_before[0] or bbox_after[1] < bbox_before[1]
