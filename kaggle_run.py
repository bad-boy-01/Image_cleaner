"""
kaggle_run.py

Kaggle Notebook runner — designed to be pasted as notebook cells or run
as a standalone script in a Kaggle session.

How to use in a Kaggle Notebook
--------------------------------
Option A — Single-script execution:
    Run this file directly:
        !python kaggle_run.py

Option B — Cell-by-cell (recommended for interactive use):
    Copy each section delimited by ### CELL N ### into its own notebook cell.
    This lets you re-run individual stages without re-running the whole pipeline.

Kaggle-specific notes
---------------------
* Input dataset should be uploaded to Kaggle as a dataset and mounted at
  ``/kaggle/input/<dataset-name>/chapter.zip``.
* Output will be written to ``/kaggle/working/cleaned_chapter.zip``.
* GPU: ensure the notebook's accelerator is set to GPU (T4) in Settings.
* Internet: enable internet in notebook settings to allow model downloads
  on first run.  Subsequent runs use cached weights.
* Session limits: the manifest at ``/kaggle/working/_work/manifest.db``
  survives between sessions (if saved as an output dataset), enabling
  true resumability across Kaggle's 12-hour GPU limit.
"""

import os
import sys

# ---------------------------------------------------------------------------
### CELL 1 — Install dependencies
# ---------------------------------------------------------------------------
# Uncomment and run this cell once per Kaggle session.
# After the kernel restarts, run remaining cells normally.

# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
#     "easyocr",
#     "simple-lama-inpainting",
#     "pydantic>=2.0",
#     "pydantic-settings>=2.0",
#     "opencv-python-headless",
# ])
# print("✔ Dependencies installed")


# ---------------------------------------------------------------------------
### CELL 2 — Configuration
# ---------------------------------------------------------------------------

# Kaggle paths
KAGGLE_INPUT_DIR  = "/kaggle/input"
KAGGLE_WORKING    = "/kaggle/working"

# ── Edit these three lines ──────────────────────────────────────────────────
INPUT_DATASET   = "your-dataset-name"   # the Kaggle dataset slug
CHAPTER_ZIP     = "chapter.zip"         # filename inside the dataset
INPAINT_BACKEND = "lama"               # "lama" or "mock" (for testing)
OCR_BACKEND     = "easyocr"
# ────────────────────────────────────────────────────────────────────────────

# Detect whether we are inside Kaggle
IS_KAGGLE = os.path.exists(KAGGLE_INPUT_DIR)

if IS_KAGGLE:
    INPUT_ZIP  = os.path.join(KAGGLE_INPUT_DIR, INPUT_DATASET, CHAPTER_ZIP)
    OUTPUT_ZIP = os.path.join(KAGGLE_WORKING, "cleaned_chapter.zip")
    WORK_DIR   = os.path.join(KAGGLE_WORKING, "_work")
    DEBUG_DIR  = os.path.join(KAGGLE_WORKING, "_debug")
else:
    # Local development fallback
    INPUT_ZIP  = "chapter.zip"
    OUTPUT_ZIP = "cleaned_chapter.zip"
    WORK_DIR   = "_work"
    DEBUG_DIR  = "_debug"

print(f"Input  : {INPUT_ZIP}")
print(f"Output : {OUTPUT_ZIP}")
print(f"Work   : {WORK_DIR}")


# ---------------------------------------------------------------------------
### CELL 3 — Add project root to Python path (if not installed as a package)
# ---------------------------------------------------------------------------

# When running from /kaggle/working, the project directory must be on the path.
# Adjust REPO_PATH if the project is in a different location.
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

print(f"Project root: {REPO_PATH}")


# ---------------------------------------------------------------------------
### CELL 4 — Build Settings and run the pipeline
# ---------------------------------------------------------------------------

from config.settings import Settings
from app import run

cfg = Settings(
    input_zip=INPUT_ZIP,
    output_zip=OUTPUT_ZIP,
    work_dir=WORK_DIR,
    debug_dir=DEBUG_DIR,
    manifest_path=f"{WORK_DIR}/manifest.db",
    inpainting_backend=INPAINT_BACKEND,
    ocr_backend=OCR_BACKEND,
    # GPU is always cuda in Kaggle; switch to "cpu" for testing
    gpu_device="cuda" if IS_KAGGLE else "cpu",
    # Set to True to save debug artefacts for mask inspection
    debug_mode=False,
    # Set to False to reprocess all pages from scratch
    resume=True,
)

exit_code = run(cfg)
print(f"\nPipeline finished with exit code {exit_code}")


# ---------------------------------------------------------------------------
### CELL 5 — Verify output (optional)
# ---------------------------------------------------------------------------
# Uncomment to inspect the output ZIP contents.

# import zipfile
# with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
#     for info in zf.infolist():
#         print(f"  {info.filename:40s}  {info.file_size:>10,} bytes")
