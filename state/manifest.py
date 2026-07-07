"""
state/manifest.py

SQLite-based resumable processing manifest.

Purpose
-------
Kaggle notebooks can time out or be interrupted mid-run.  The manifest
records which pages have been successfully completed so a resumed run
can skip them and continue from where it left off.

Schema
------
Table: ``pages``

    page_name     TEXT  PRIMARY KEY   -- filename (e.g. "003.png")
    status        TEXT                -- "completed" | "failed" | "skipped"
    processing_time_s REAL
    ocr_detections INT
    non_english    INT
    mask_regions   INT
    patches        INT
    repair_failures INT
    validation_passed INT
    rogue_pixels   INT
    completed_at   TEXT              -- ISO 8601 timestamp

Usage
-----
    manifest = Manifest(Path("_work/manifest.db"))
    manifest.open()

    if manifest.is_done("003.png"):
        logger.step("003.png already done — skipping")
    else:
        # ... process page ...
        manifest.mark_done(metrics)

    manifest.close()
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from core.models import PageMetrics


class Manifest:
    """
    SQLite-backed manifest for resumable page processing.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.  Created on first open.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Open the database connection and create the schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    def close(self) -> None:
        """Commit any pending writes and close the connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Manifest":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def is_done(self, page_name: str) -> bool:
        """Return True if *page_name* was previously completed successfully."""
        if self._conn is None:
            raise RuntimeError("Manifest is not open.")
        row = self._conn.execute(
            "SELECT status FROM pages WHERE page_name = ?", (page_name,)
        ).fetchone()
        return row is not None and row[0] == "completed"

    def mark_done(self, metrics: PageMetrics) -> None:
        """Record a successfully completed page."""
        self._upsert(metrics, status="completed")

    def mark_failed(self, page_name: str, reason: str = "") -> None:
        """Record a page that failed processing."""
        if self._conn is None:
            raise RuntimeError("Manifest is not open.")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute(
            """
            INSERT INTO pages (page_name, status, completed_at)
            VALUES (?, 'failed', ?)
            ON CONFLICT(page_name) DO UPDATE
               SET status = 'failed', completed_at = excluded.completed_at
            """,
            (page_name, ts),
        )
        self._conn.commit()

    def completed_pages(self) -> list[str]:
        """Return list of all completed page names."""
        if self._conn is None:
            raise RuntimeError("Manifest is not open.")
        rows = self._conn.execute(
            "SELECT page_name FROM pages WHERE status = 'completed'"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _create_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_name         TEXT PRIMARY KEY,
                status            TEXT NOT NULL DEFAULT 'pending',
                processing_time_s REAL,
                ocr_detections    INTEGER,
                non_english       INTEGER,
                mask_regions      INTEGER,
                patches           INTEGER,
                repair_failures   INTEGER,
                validation_passed INTEGER,
                rogue_pixels      INTEGER,
                completed_at      TEXT
            )
            """
        )
        self._conn.commit()

    def _upsert(self, metrics: PageMetrics, status: str) -> None:
        assert self._conn is not None
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute(
            """
            INSERT INTO pages (
                page_name, status, processing_time_s, ocr_detections,
                non_english, mask_regions, patches, repair_failures,
                validation_passed, rogue_pixels, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(page_name) DO UPDATE SET
                status            = excluded.status,
                processing_time_s = excluded.processing_time_s,
                ocr_detections    = excluded.ocr_detections,
                non_english       = excluded.non_english,
                mask_regions      = excluded.mask_regions,
                patches           = excluded.patches,
                repair_failures   = excluded.repair_failures,
                validation_passed = excluded.validation_passed,
                rogue_pixels      = excluded.rogue_pixels,
                completed_at      = excluded.completed_at
            """,
            (
                metrics.page_name,
                status,
                metrics.processing_time_s,
                metrics.ocr_detection_count,
                metrics.non_english_count,
                metrics.mask_region_count,
                metrics.patch_count,
                metrics.repair_failure_count,
                int(metrics.validation_passed),
                metrics.validation_rogue_pixels,
                ts,
            ),
        )
        self._conn.commit()
