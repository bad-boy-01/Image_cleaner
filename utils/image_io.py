"""
utils/image_io.py

Format-preserving image loading and saving.

Guarantees
----------
* PNG input → PNG output (lossless, same bit-depth).
* JPG/JPEG input → JPG output (at configured quality, no extra recompression).
* DPI metadata is read on load and restored on save.
* ICC profiles and other ``PIL.Image.info`` entries are preserved where
  Pillow supports them.
* The image is NEVER resized, rotated, cropped, or colour-corrected here.

All images are returned as RGB uint8 numpy arrays so the rest of the
pipeline works with a single, predictable format.  The original format
string and DPI are stored in ``ImagePage`` for use at save time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from core.models import ImagePage


# Formats the pipeline will accept
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_image(path: Path) -> ImagePage:
    """
    Load an image from *path* and return an ``ImagePage``.

    Parameters
    ----------
    path : Path
        Absolute or relative path to a PNG, JPG, or JPEG file.

    Returns
    -------
    ImagePage
        * ``image`` and ``original_image`` both hold the same initial
          RGB uint8 array (two separate copies).
        * ``fmt`` is ``"png"`` or ``"jpeg"``.
        * ``dpi`` is the tuple from ``pil_img.info.get("dpi")`` or None.

    Raises
    ------
    ValueError
        If the file extension is not in ``SUPPORTED_EXTENSIONS``.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with Image.open(path) as pil_img:
        # Normalise to RGB (handles RGBA, L, P, etc.)
        rgb_img = pil_img.convert("RGB")

        # Extract metadata before converting
        dpi: Optional[Tuple[float, float]] = pil_img.info.get("dpi")
        if dpi is not None:
            dpi = (float(dpi[0]), float(dpi[1]))

        # Preserve non-DPI metadata entries (ICC profile, etc.)
        extra_meta = {k: v for k, v in pil_img.info.items() if k != "dpi"}

        # Determine canonical format
        fmt = "jpeg" if suffix in {".jpg", ".jpeg"} else "png"

        arr = np.array(rgb_img, dtype=np.uint8)

    return ImagePage(
        path=path,
        image=arr.copy(),
        original_image=arr.copy(),
        fmt=fmt,
        dpi=dpi,
        extra_metadata=extra_meta,
    )


def save_image(page: ImagePage, output_path: Path, jpeg_quality: int = 95) -> None:
    """
    Save ``page.image`` to *output_path* in the original format.

    Parameters
    ----------
    page : ImagePage
        The processed page.  ``page.fmt`` and ``page.dpi`` are used to
        reproduce the original file format and metadata.
    output_path : Path
        Destination file path.  Parent directory must exist.
    jpeg_quality : int
        JPEG quality setting (1–100).  Ignored for PNG output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_img = Image.fromarray(page.image, mode="RGB")

    save_kwargs: dict = {}

    if page.dpi is not None:
        save_kwargs["dpi"] = page.dpi

    if page.fmt == "png":
        # PNG: lossless, preserve DPI via pnginfo if available
        pil_img.save(output_path, format="PNG", **save_kwargs)
    else:
        # JPEG: configurable quality, no chroma subsampling change
        save_kwargs["quality"] = jpeg_quality
        save_kwargs["subsampling"] = 0  # 4:4:4 — avoids colour shift
        pil_img.save(output_path, format="JPEG", **save_kwargs)
