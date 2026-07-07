"""
config/settings.py

Central configuration for the AI Image Cleaner pipeline.

All settings can be overridden via environment variables or a .env file.
The field names map 1-to-1 to environment variable names (upper-cased),
e.g. MASK_EXPANSION_PX=6 overrides the default.

Usage
-----
    from config.settings import Settings
    cfg = Settings()                    # from environment / defaults
    cfg = Settings(debug_mode=True)     # inline override
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Unified configuration for the image-cleaning pipeline.

    All path fields accept both strings and Path objects.
    All numeric fields are validated to be within sensible bounds.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    input_zip: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the input ZIP archive. "
            "Mutually exclusive with input_folder. "
            "One of input_zip or input_folder must be provided."
        ),
    )
    input_folder: Optional[Path] = Field(
        default=None,
        description=(
            "Path to a folder containing image files directly. "
            "Use this when Kaggle has already unzipped your dataset. "
            "Mutually exclusive with input_zip."
        ),
    )
    output_zip: Path = Field(
        default=Path("cleaned_chapter.zip"),
        description="Path where the output ZIP will be written.",
    )
    work_dir: Path = Field(
        default=Path("_work"),
        description="Scratch directory for extracted images and temp files.",
    )

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    ocr_backend: Literal["easyocr", "paddleocr"] = Field(
        default="easyocr",
        description="Text detection backend.",
    )
    ocr_languages: List[str] = Field(
        default=["ko", "ja", "ch_sim", "en"],
        description="Languages to scan for.",
    )
    confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Discard OCR detections below this confidence score.",
    )

    # ------------------------------------------------------------------ #
    # Language filtering
    # ------------------------------------------------------------------ #
    remove_english: bool = Field(
        default=False,
        description=(
            "When True, English text regions are also cleaned. "
            "Reserved for future use."
        ),
    )
    target_scripts: List[str] = Field(
        default=["Hangul", "Hiragana", "Katakana", "CJK"],
        description=(
            "Unicode script names whose detections are scheduled for removal. "
            "Used by the language filter when remove_english=False."
        ),
    )

    # ------------------------------------------------------------------ #
    # Masking
    # ------------------------------------------------------------------ #
    mask_expansion_px: int = Field(
        default=8,
        ge=0,
        le=50,
        description="Morphological dilation radius applied to raw OCR masks (pixels).",
    )

    # ------------------------------------------------------------------ #
    # Patch extraction
    # ------------------------------------------------------------------ #
    patch_padding: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Extra padding added around each mask bounding box before cropping.",
    )
    patch_align: int = Field(
        default=8,
        ge=1,
        description=(
            "Patch dimensions are rounded up to the nearest multiple of this value. "
            "Most inpainting models require dimensions divisible by 8."
        ),
    )
    max_patch_size: int = Field(
        default=1024,
        ge=64,
        description="Hard cap on patch width/height to prevent OOM on large regions.",
    )

    # ------------------------------------------------------------------ #
    # Inpainting
    # ------------------------------------------------------------------ #
    inpainting_backend: Literal["lama", "mock"] = Field(
        default="lama",
        description="Inpainting backend identifier. Must be registered in the backend registry.",
    )
    gpu_device: str = Field(
        default="cuda",
        description="Torch device string, e.g. 'cuda', 'cuda:0', 'cpu'.",
    )

    # ------------------------------------------------------------------ #
    # Output quality
    # ------------------------------------------------------------------ #
    jpeg_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG save quality (only applies to .jpg/.jpeg inputs).",
    )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    enable_validator: bool = Field(
        default=True,
        description="Run pixel-level diff validation after each image is processed.",
    )
    validator_tolerance: int = Field(
        default=2,
        ge=0,
        le=255,
        description=(
            "Per-channel absolute difference below which a pixel change is tolerated "
            "(accounts for JPEG round-trip noise)."
        ),
    )

    # ------------------------------------------------------------------ #
    # State / resumability
    # ------------------------------------------------------------------ #
    manifest_path: Path = Field(
        default=Path("_work/manifest.db"),
        description="SQLite database used for resumable processing.",
    )
    resume: bool = Field(
        default=True,
        description="Skip pages that are already recorded as completed in the manifest.",
    )

    # ------------------------------------------------------------------ #
    # Debug
    # ------------------------------------------------------------------ #
    debug_mode: bool = Field(
        default=False,
        description=(
            "When True, save intermediate artefacts: raw masks, expanded masks, "
            "patch previews, diff images, and validation overlays."
        ),
    )
    debug_dir: Path = Field(
        default=Path("_debug"),
        description="Directory where debug artefacts are written.",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator(
        "input_zip", "input_folder", "output_zip",
        "work_dir", "manifest_path", "debug_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        return Path(str(v))
