"""
app.py

Main entry point for the AI Image Cleaner.

Orchestrates the full pipeline:
  1. Validate configuration
  2. Extract input ZIP
  3. Load OCR + inpainting backends (once)
  4. Process each page sequentially
  5. Resume from manifest if interrupted
  6. Rebuild output ZIP
  7. Print metrics summary
  8. Save metrics JSON

Usage
-----
    python app.py                          # uses .env / environment defaults
    python app.py --input chapter1.zip     # override input ZIP
    python app.py --backend mock           # use mock backend (no GPU)
    python app.py --no-resume              # reprocess all pages
    python app.py --debug                  # enable debug artefacts
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from backends.registry import backend_registry
from config.settings import Settings
from core.pipeline import PagePipeline
from debug.debug_saver import DebugSaver
from ocr.registry import ocr_registry
from state.manifest import Manifest
from utils import logger
from utils.metrics import MetricsCollector
from utils.zip_handler import build_zip, extract_zip


def build_settings_from_args(args: argparse.Namespace) -> Settings:
    """Merge CLI arguments with Settings (env/defaults)."""
    overrides: dict = {}
    if args.input:
        overrides["input_zip"] = args.input
    if args.output:
        overrides["output_zip"] = args.output
    if args.backend:
        overrides["inpainting_backend"] = args.backend
    if args.ocr:
        overrides["ocr_backend"] = args.ocr
    if args.debug:
        overrides["debug_mode"] = True
    if args.no_resume:
        overrides["resume"] = False
    if args.cpu:
        overrides["gpu_device"] = "cpu"
    return Settings(**overrides)


def run(cfg: Settings) -> int:
    """
    Execute the full cleaning pipeline.

    Returns
    -------
    int
        Exit code.  0 = success, 1 = partial failure, 2 = fatal error.
    """
    pipeline_start = time.monotonic()

    # ------------------------------------------------------------------ #
    # Prepare directories
    # ------------------------------------------------------------------ #
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = cfg.work_dir / "output"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    debug_saver = DebugSaver(cfg.debug_dir) if cfg.debug_mode else None

    # ------------------------------------------------------------------ #
    # Extract input ZIP
    # ------------------------------------------------------------------ #
    logger.step(f"Extracting input ZIP: {cfg.input_zip}")
    try:
        image_paths = extract_zip(cfg.input_zip, cfg.work_dir / "input")
    except (FileNotFoundError, Exception) as exc:
        logger.error(f"Failed to extract ZIP: {exc}")
        return 2

    if not image_paths:
        logger.error("No supported image files found in the input ZIP.")
        return 2

    logger.pipeline_start(len(image_paths))

    # ------------------------------------------------------------------ #
    # Load backends (once — shared across all pages)
    # ------------------------------------------------------------------ #
    logger.step(f"Loading OCR backend: {cfg.ocr_backend}")
    ocr = ocr_registry.build(
        cfg.ocr_backend,
        languages=cfg.ocr_languages,
        gpu=cfg.gpu_device != "cpu",
        confidence_threshold=cfg.confidence_threshold,
    )
    ocr.load()

    logger.step(f"Loading inpainting backend: {cfg.inpainting_backend}")
    inpainting = backend_registry.build(
        cfg.inpainting_backend,
        device=cfg.gpu_device,
    )
    inpainting.load_model()
    logger.gpu_memory("after model load", inpainting.memory_usage_mb())

    # Warm up CUDA kernels
    logger.step("Warming up inpainting backend")
    inpainting.warmup()

    # ------------------------------------------------------------------ #
    # Pipeline + manifest
    # ------------------------------------------------------------------ #
    page_pipeline = PagePipeline(
        ocr=ocr,
        backend=inpainting,
        settings=cfg,
        debug_saver=debug_saver,
    )
    collector = MetricsCollector()
    exit_code = 0

    with Manifest(cfg.manifest_path) as manifest:
        for page_index, image_path in enumerate(image_paths, start=1):
            page_name = image_path.name
            output_path = output_images_dir / page_name

            logger.page_start(page_name, page_index, len(image_paths))

            # Resume check
            if cfg.resume and manifest.is_done(page_name):
                logger.step(f"Already completed — skipping (resume mode)")
                # Copy existing output if available
                if output_path.exists():
                    collector.record(__skipped_metrics(page_name))
                continue

            try:
                metrics = page_pipeline.run(image_path, output_path)
                manifest.mark_done(metrics)
                collector.record(metrics)

                if not metrics.validation_passed:
                    exit_code = max(exit_code, 1)

            except Exception as exc:  # noqa: BLE001
                logger.error(f"Page '{page_name}' failed: {exc}")
                manifest.mark_failed(page_name, str(exc))
                exit_code = max(exit_code, 1)

            # ETA estimation
            logger.eta(
                completed=page_index,
                total=len(image_paths),
                elapsed_s=time.monotonic() - pipeline_start,
            )

    # ------------------------------------------------------------------ #
    # Unload backends
    # ------------------------------------------------------------------ #
    logger.step("Unloading backends")
    inpainting.unload_model()
    logger.gpu_memory("after model unload", inpainting.memory_usage_mb())
    ocr.unload()

    # ------------------------------------------------------------------ #
    # Rebuild output ZIP
    # ------------------------------------------------------------------ #
    logger.step(f"Building output ZIP: {cfg.output_zip}")
    build_zip(output_images_dir, cfg.output_zip)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    collector.print_summary()
    metrics_path = cfg.work_dir / "metrics.json"
    collector.save_json(metrics_path)
    logger.step(f"Metrics saved → {metrics_path}")

    total_elapsed = time.monotonic() - pipeline_start
    logger.pipeline_done(
        total_pages=collector.total_pages,
        elapsed_s=total_elapsed,
        output_zip=str(cfg.output_zip),
    )

    return exit_code


def __skipped_metrics(page_name: str):
    """Return a placeholder metrics record for a skipped page."""
    from core.models import PageMetrics
    return PageMetrics(page_name=page_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image_cleaner",
        description="AI-powered image text removal pipeline.",
    )
    parser.add_argument("--input", "-i", help="Path to input ZIP archive.")
    parser.add_argument("--output", "-o", help="Path for output ZIP archive.")
    parser.add_argument(
        "--backend", "-b",
        choices=backend_registry.available(),
        help="Inpainting backend to use.",
    )
    parser.add_argument(
        "--ocr",
        choices=ocr_registry.available(),
        help="OCR backend to use.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug artefact saving.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess all pages even if already in the manifest.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (disables GPU).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    cfg = build_settings_from_args(args)
    sys.exit(run(cfg))
