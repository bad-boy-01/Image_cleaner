"""
utils/metrics.py

Aggregates per-page PageMetrics into a pipeline-level summary and
provides formatted human-readable reports.

Usage
-----
    from utils.metrics import MetricsCollector

    collector = MetricsCollector()
    # ... process pages ...
    collector.record(page_metrics)
    collector.print_summary()
    collector.save_json(Path("metrics.json"))
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from core.models import PageMetrics


class MetricsCollector:
    """
    Accumulates ``PageMetrics`` records and computes summary statistics.
    """

    def __init__(self) -> None:
        self._records: List[PageMetrics] = []
        self._pipeline_start = time.monotonic()

    def record(self, metrics: PageMetrics) -> None:
        """Append a completed page's metrics."""
        self._records.append(metrics)

    # ------------------------------------------------------------------ #
    # Aggregate statistics
    # ------------------------------------------------------------------ #

    @property
    def total_pages(self) -> int:
        return len(self._records)

    @property
    def total_processing_time(self) -> float:
        return sum(m.processing_time_s for m in self._records)

    @property
    def avg_processing_time(self) -> float:
        if not self._records:
            return 0.0
        return self.total_processing_time / len(self._records)

    @property
    def total_ocr_detections(self) -> int:
        return sum(m.ocr_detection_count for m in self._records)

    @property
    def total_non_english(self) -> int:
        return sum(m.non_english_count for m in self._records)

    @property
    def total_mask_regions(self) -> int:
        return sum(m.mask_region_count for m in self._records)

    @property
    def total_patches(self) -> int:
        return sum(m.patch_count for m in self._records)

    @property
    def total_repair_failures(self) -> int:
        return sum(m.repair_failure_count for m in self._records)

    @property
    def validation_failures(self) -> int:
        return sum(1 for m in self._records if not m.validation_passed)

    @property
    def avg_mask_coverage_pct(self) -> float:
        if not self._records:
            return 0.0
        return sum(m.mask_coverage_pct for m in self._records) / len(self._records)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        wall_time = time.monotonic() - self._pipeline_start
        lines = [
            "",
            "  ┌─────────────────────────────────────────────┐",
            "  │           Pipeline Metrics Summary           │",
            "  ├─────────────────────────────────────────────┤",
            f"  │ Pages processed        : {self.total_pages:<18} │",
            f"  │ Wall-clock time        : {wall_time:<17.1f}s │",
            f"  │ Avg time/page          : {self.avg_processing_time:<17.1f}s │",
            f"  │ Total OCR detections   : {self.total_ocr_detections:<18} │",
            f"  │ Non-English retained   : {self.total_non_english:<18} │",
            f"  │ Mask regions generated : {self.total_mask_regions:<18} │",
            f"  │ Patches inpainted      : {self.total_patches:<18} │",
            f"  │ Repair failures        : {self.total_repair_failures:<18} │",
            f"  │ Validation failures    : {self.validation_failures:<18} │",
            f"  │ Avg mask coverage      : {self.avg_mask_coverage_pct:<17.2f}% │",
            "  └─────────────────────────────────────────────┘",
            "",
        ]
        print("\n".join(lines), flush=True)

    def as_dict(self) -> dict:
        """Serialise summary + per-page records to a dict."""
        return {
            "summary": {
                "total_pages": self.total_pages,
                "total_processing_time_s": round(self.total_processing_time, 3),
                "avg_processing_time_s": round(self.avg_processing_time, 3),
                "total_ocr_detections": self.total_ocr_detections,
                "total_non_english": self.total_non_english,
                "total_mask_regions": self.total_mask_regions,
                "total_patches": self.total_patches,
                "total_repair_failures": self.total_repair_failures,
                "validation_failures": self.validation_failures,
                "avg_mask_coverage_pct": round(self.avg_mask_coverage_pct, 4),
            },
            "pages": [m.as_dict() for m in self._records],
        }

    def save_json(self, path: Path) -> None:
        """Write the full metrics report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)
