"""
tests/test_models.py

Unit tests for core data models.

Tests cover:
  - TextDetection.bbox property
  - ImagePage.shape property
  - PageMetrics.as_dict() serialisation
  - PipelineTimer context manager
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.models import (
    ImagePage,
    MaskRegion,
    PageMetrics,
    PipelineTimer,
    TextDetection,
)


class TestTextDetection:
    def test_bbox_axis_aligned(self) -> None:
        polygon = np.array([[10, 20], [50, 20], [50, 60], [10, 60]], dtype=np.int32)
        det = TextDetection(polygon=polygon, text="테스트", language="ko",
                            confidence=0.9, script="Hangul")
        assert det.bbox == (10, 20, 50, 60)

    def test_bbox_non_rectangular(self) -> None:
        # Parallelogram-like polygon
        polygon = np.array([[5, 30], [25, 10], [45, 30], [25, 50]], dtype=np.int32)
        det = TextDetection(polygon=polygon, text="漢字", language="zh",
                            confidence=0.8, script="CJK")
        x0, y0, x1, y1 = det.bbox
        assert x0 == 5 and y0 == 10 and x1 == 45 and y1 == 50


class TestImagePage:
    def _make_page(self, h: int = 100, w: int = 80) -> ImagePage:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        from pathlib import Path
        return ImagePage(
            path=Path("test.png"),
            image=arr.copy(),
            original_image=arr.copy(),
            fmt="png",
            dpi=(72.0, 72.0),
        )

    def test_shape(self) -> None:
        page = self._make_page(h=120, w=90)
        assert page.shape == (120, 90)

    def test_name(self) -> None:
        page = self._make_page()
        assert page.name == "test.png"

    def test_defaults(self) -> None:
        page = self._make_page()
        assert page.detections == []
        assert page.mask_regions == []
        assert page.patches == []
        assert page.repair_results == []


class TestPageMetrics:
    def test_as_dict_keys(self) -> None:
        m = PageMetrics(page_name="001.png")
        d = m.as_dict()
        expected_keys = {
            "page_name", "processing_time_s", "ocr_detection_count",
            "non_english_count", "mask_region_count", "mask_coverage_pct",
            "patch_count", "repair_success_count", "repair_failure_count",
            "changed_pixels_total", "validation_passed", "validation_rogue_pixels",
        }
        assert expected_keys == set(d.keys())

    def test_validation_passed_serialised_as_int(self) -> None:
        m = PageMetrics(page_name="001.png", validation_passed=True)
        assert m.as_dict()["validation_passed"] == 1
        m2 = PageMetrics(page_name="001.png", validation_passed=False)
        assert m2.as_dict()["validation_passed"] == 0


class TestPipelineTimer:
    def test_elapsed_positive(self) -> None:
        with PipelineTimer() as t:
            time.sleep(0.05)
        assert t.elapsed() >= 0.05

    def test_context_manager_returns_self(self) -> None:
        with PipelineTimer() as t:
            assert isinstance(t, PipelineTimer)
