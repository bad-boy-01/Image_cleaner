# AI Image Cleaner

A professional, modular Python pipeline for AI-assisted **text removal from images**. Designed to run in a **Kaggle Notebook** with GPU acceleration, and equally usable locally.

Removes non-English text (Hangul, Japanese, Chinese, and other CJK scripts) from image pages using OCR-based detection + AI inpainting, while **pixel-perfectly preserving** everything outside the cleaned regions.

---

## Key Features

| Feature | Detail |
|---|---|
| **Pixel-perfect preservation** | Only masked pixels are ever modified. Enforced at paste-back *and* validated by a diff checker. |
| **Format fidelity** | PNG → PNG (lossless). JPG → JPG (same quality). DPI and metadata preserved. |
| **Plugin architecture** | Swap OCR or inpainting backends with a single config line. Adding a new backend requires zero pipeline changes. |
| **Dynamic patching** | Patch size grows from the mask bounding box — no fixed 512×512 assumption. Aligned to model requirements automatically. |
| **Resumable sessions** | SQLite manifest records completed pages. Interrupted Kaggle sessions resume where they left off. |
| **Structured metrics** | Per-page and pipeline-level stats: OCR count, mask coverage, patch count, validation results, processing time. |
| **Debug mode** | Saves raw masks, expanded masks, patch previews, inpainted results, diff heatmaps, and rogue-pixel overlays. |

---

## Project Structure

```
image_cleaner/
├── app.py                    # CLI entry point
├── kaggle_run.py             # Kaggle Notebook runner (cell-by-cell)
├── .env.example              # Config template → copy to .env
├── requirements.txt
├── pyproject.toml
│
├── input/                    # ← Drop your chapter.zip here (local use)
│
├── config/
│   └── settings.py           # Pydantic settings (env-file + CLI override)
│
├── core/
│   ├── models.py             # TextDetection, MaskRegion, Patch, RepairResult,
│   │                         #   ImagePage, PageMetrics, PipelineTimer
│   └── pipeline.py           # Per-page orchestrator (backend-injected)
│
├── ocr/
│   ├── base.py               # Abstract OCRBackend
│   ├── registry.py           # Plugin registry
│   └── easyocr_backend.py    # Default OCR implementation
│
├── mask/
│   ├── generator.py          # OCR polygon → binary mask (union-find merge)
│   ├── expander.py           # Morphological dilation
│   └── language_filter.py    # CJK vs Latin script filter
│
├── patch/
│   ├── extractor.py          # Dynamic bbox+padding+alignment crop
│   └── paster.py             # Mask-selective pixel paste-back
│
├── backends/
│   ├── base.py               # Abstract InpaintingBackend + GPU lifecycle
│   ├── registry.py           # Plugin registry
│   ├── lama_backend.py       # LaMa (simple-lama-inpainting)
│   └── mock_backend.py       # Median-fill mock (no GPU, for testing)
│
├── validator/
│   └── pixel_validator.py    # Diff validator — detects rogue pixel changes
│
├── state/
│   └── manifest.py           # SQLite resumable session manifest
│
├── debug/
│   └── debug_saver.py        # Intermediate artefact writer
│
├── utils/
│   ├── image_io.py           # Format/DPI-preserving Pillow I/O
│   ├── zip_handler.py        # ZIP extraction and rebuild
│   ├── logger.py             # ANSI progress logger with ETA
│   └── metrics.py            # MetricsCollector + JSON export
│
└── tests/
    ├── test_models.py
    ├── test_mask.py
    ├── test_validator.py
    └── test_pipeline.py
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your input ZIP

Drop your chapter ZIP into the `input/` folder:

```
input/
└── chapter1.zip
```

The ZIP should contain image files at its root or in a single subfolder:

```
chapter1.zip
├── 001.png
├── 002.png
└── 003.jpg
```

### 3. Run (local, no GPU — mock backend)

```bash
python app.py --input input/chapter1.zip --backend mock
```

### 4. Run (local or Kaggle, GPU — LaMa backend)

```bash
python app.py --input input/chapter1.zip --backend lama
```

Output is written to `cleaned_chapter.zip` by default.

### 5. Override any setting

```bash
python app.py \
  --input  input/chapter1.zip \
  --output output/chapter1_cleaned.zip \
  --backend lama \
  --debug
```

Or set values in a `.env` file (copy `.env.example` → `.env`).

---

## Kaggle Notebook Usage

1. Upload your chapter ZIP as a **Kaggle Dataset**.
2. Add the dataset to your notebook.
3. Open `kaggle_run.py` and edit the three config lines at the top of **Cell 2**:
   ```python
   INPUT_DATASET = "your-dataset-name"   # Kaggle dataset slug
   CHAPTER_ZIP   = "chapter.zip"
   INPAINT_BACKEND = "lama"
   ```
4. Paste each `### CELL N ###` section into its own notebook cell.
5. Run **Cell 1** once to install deps, then restart and run **Cells 2–5**.

> **Resumability**: The manifest at `/kaggle/working/_work/manifest.db` persists across sessions if saved as an output dataset. Re-running the notebook will skip already-completed pages automatically.

---

## Configuration Reference

All settings can be set via environment variables, a `.env` file, or inline `Settings(...)` kwargs.

| Setting | Default | Description |
|---|---|---|
| `INPUT_ZIP` | `chapter.zip` | Path to input ZIP archive |
| `OUTPUT_ZIP` | `cleaned_chapter.zip` | Path for output ZIP |
| `WORK_DIR` | `_work` | Scratch directory (auto-created) |
| `OCR_BACKEND` | `easyocr` | OCR backend name |
| `OCR_LANGUAGES` | `["ko","ja","ch_sim","en"]` | Languages passed to the OCR backend |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum OCR detection confidence |
| `REMOVE_ENGLISH` | `false` | Also remove Latin-script text |
| `TARGET_SCRIPTS` | `["Hangul","Hiragana","Katakana","CJK"]` | Scripts scheduled for removal |
| `MASK_EXPANSION_PX` | `4` | Morphological dilation radius (px) |
| `PATCH_PADDING` | `32` | Padding added around each mask bbox (px) |
| `PATCH_ALIGN` | `8` | Patch dimensions rounded to this multiple |
| `MAX_PATCH_SIZE` | `1024` | Hard cap on patch width/height (px) |
| `INPAINTING_BACKEND` | `lama` | Inpainting backend name |
| `GPU_DEVICE` | `cuda` | Torch device string |
| `JPEG_QUALITY` | `95` | JPEG save quality (1–100) |
| `ENABLE_VALIDATOR` | `true` | Run pixel diff validation |
| `VALIDATOR_TOLERANCE` | `2` | Per-channel diff tolerance (absorbs JPEG noise) |
| `RESUME` | `true` | Skip already-completed pages |
| `DEBUG_MODE` | `false` | Save intermediate debug artefacts |
| `DEBUG_DIR` | `_debug` | Directory for debug artefacts |

---

## Pipeline Stages

```
Input ZIP
  ↓ extract_zip()
Load image  →  ImagePage (RGB uint8, DPI preserved)
  ↓ OCRBackend.detect()
TextDetection[]  (polygon, text, script, confidence)
  ↓ filter_non_english()
Target detections  (CJK / non-Latin only)
  ↓ generate_masks()
MaskRegion[]  (rasterised polygons, union-find merged)
  ↓ expand_masks()
Expanded MaskRegion[]  (morphological dilation)
  ↓ extract_patches()
Patch[]  (dynamic bbox+padding+align crop)
  ↓ InpaintingBackend.inpaint_patch()
RepairResult[]
  ↓ paste_results()          ← mask-selective write only
ImagePage.image updated
  ↓ validate()               ← diff check: only approved pixels changed?
ValidationResult
  ↓ save_image()             ← format/DPI preserved
  ↓ build_zip()
cleaned_chapter.zip
```

---

## Adding a New Backend

### New inpainting backend

```python
# backends/my_backend.py
from backends.base import InpaintingBackend
from backends.registry import backend_registry

@backend_registry.register("my_backend")
class MyBackend(InpaintingBackend):
    def load_model(self) -> None: ...
    def inpaint_patch(self, patch, mask): ...
    def unload_model(self) -> None: ...
```

Then use it with `--backend my_backend` or `INPAINTING_BACKEND=my_backend`.

### New OCR backend

```python
# ocr/my_ocr.py
from ocr.base import OCRBackend
from ocr.registry import ocr_registry

@ocr_registry.register("my_ocr")
class MyOCR(OCRBackend):
    def load(self) -> None: ...
    def detect(self, image) -> list[TextDetection]: ...
    def unload(self) -> None: ...
```

No other files need to change.

---

## Running Tests

```bash
pytest
```

Tests use the **mock backend** and injected fake OCR — no GPU or model weights required.

```
tests/test_models.py     — data models, bbox, timer
tests/test_mask.py       — language filter, mask generation, expansion
tests/test_validator.py  — pass/fail, tolerance, diff output
tests/test_pipeline.py   — integration, pixel-preservation invariant
```

---

## Runtime Directories

| Directory | Created by | Contents |
|---|---|---|
| `input/` | You (manually) | Your input ZIP files |
| `_work/input/` | Pipeline | Extracted image pages (temporary) |
| `_work/output/` | Pipeline | Cleaned images before ZIP rebuild |
| `_work/manifest.db` | Pipeline | SQLite resumable session state |
| `_work/metrics.json` | Pipeline | Per-page + summary metrics |
| `_debug/` | Pipeline (debug mode) | Mask overlays, diffs, patch previews |

---

## License

MIT
