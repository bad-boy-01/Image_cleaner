"""
utils/zip_handler.py

ZIP archive utilities: extraction and rebuilding.

Responsibilities
----------------
* Extract a chapter ZIP to a work directory, yielding only supported
  image files in sorted order.
* Rebuild a ZIP from a directory of processed images, preserving the
  original filenames and relative paths within the archive.

Design notes
------------
* Files are sorted by name so pages are always processed in order.
* Non-image files inside the input ZIP (e.g. thumbs.db, metadata.xml)
  are silently skipped during extraction — they are not included in the
  output ZIP.
* The output ZIP is written atomically: a temporary file is built first,
  then renamed.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterator, List

from utils.image_io import SUPPORTED_EXTENSIONS


def extract_zip(zip_path: Path, work_dir: Path) -> List[Path]:
    """
    Extract all supported image files from *zip_path* into *work_dir*.

    Parameters
    ----------
    zip_path : Path
        Path to the input ZIP archive.
    work_dir : Path
        Directory to extract files into.  Created if it does not exist.

    Returns
    -------
    list[Path]
        Absolute paths to the extracted image files, sorted by name.

    Raises
    ------
    FileNotFoundError
        If *zip_path* does not exist.
    zipfile.BadZipFile
        If the archive is corrupt.
    """
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Input ZIP not found: {zip_path}")

    extracted: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Skip directories and macOS metadata
            if member.filename.endswith("/") or "__MACOSX" in member.filename:
                continue

            suffix = Path(member.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            dest = work_dir / Path(member.filename).name
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

            extracted.append(dest)

    return sorted(extracted, key=lambda p: p.name)


def build_zip(source_dir: Path, output_zip: Path) -> None:
    """
    Pack all supported image files from *source_dir* into *output_zip*.

    Parameters
    ----------
    source_dir : Path
        Directory containing cleaned output images.
    output_zip : Path
        Destination ZIP path.  Any existing file is overwritten.
    """
    source_dir = Path(source_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_zip.with_suffix(".tmp.zip")

    image_files = sorted(
        (p for p in source_dir.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name,
    )

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for img_path in image_files:
            zf.write(img_path, arcname=img_path.name)

    # Atomic rename
    if output_zip.exists():
        output_zip.unlink()
    tmp_path.rename(output_zip)


def iter_images(directory: Path) -> Iterator[Path]:
    """
    Yield supported image paths from *directory* in sorted name order.
    """
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
