"""
ocr/easyocr_backend.py

EasyOCR implementation of the OCRBackend interface.

EasyOCR is the default backend because:
  - It supports Korean, Japanese, Simplified Chinese, and Latin in a
    single model call.
  - It returns polygon coordinates (not just axis-aligned boxes).
  - It runs on CUDA out of the box.
  - It requires no API key or internet access at inference time.

Registration
------------
This module registers itself with ``ocr_registry`` on import.
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional

import numpy as np

from core.models import TextDetection
from ocr.base import OCRBackend
from ocr.registry import ocr_registry


@ocr_registry.register("easyocr")
class EasyOCRBackend(OCRBackend):
    """
    EasyOCR-based text detector.

    Parameters
    ----------
    languages : list[str]
        EasyOCR language codes to enable, e.g. ``["ko","ja","ch_sim","en"]``.
    gpu : bool
        Whether to use the GPU.  Defaults to True.
    confidence_threshold : float
        Detections below this score are discarded before returning.
    model_storage_directory : str, optional
        Override the default EasyOCR model cache directory.
        Useful on Kaggle where ``/root/.EasyOCR`` may not be writable.
    """

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = True,
        confidence_threshold: float = 0.5,
        model_storage_directory: Optional[str] = None,
    ) -> None:
        self.languages = languages or ["ko", "ja", "ch_sim", "en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.model_storage_directory = model_storage_directory
        self._reader: object | None = None  # easyocr.Reader, typed as object to avoid import

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Initialise the EasyOCR Reader (downloads weights on first run)."""
        import easyocr  # lazy import — only required when this backend is used

        kwargs: dict = dict(gpu=self.gpu)
        if self.model_storage_directory:
            kwargs["model_storage_directory"] = self.model_storage_directory

        self._reader = easyocr.Reader(self.languages, **kwargs)

    def unload(self) -> None:
        """Delete the reader and free GPU memory."""
        self._reader = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> List[TextDetection]:
        """
        Run EasyOCR on *image* and return a list of TextDetections.

        EasyOCR returns a list of (bbox, text, confidence) tuples where
        bbox is a list of four [x, y] corner points.
        """
        if self._reader is None:
            raise RuntimeError("EasyOCRBackend.load() must be called before detect().")

        # EasyOCR expects a numpy array in BGR *or* RGB — we pass RGB and
        # let EasyOCR handle it.
        raw_results = self._reader.readtext(image, detail=1, paragraph=False)

        detections: List[TextDetection] = []
        for bbox, text, confidence in raw_results:
            if confidence < self.confidence_threshold:
                continue

            # bbox is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            polygon = np.array(bbox, dtype=np.int32)  # shape (4, 2)

            script = _dominant_script(text)
            language = _script_to_language(script)

            detections.append(
                TextDetection(
                    polygon=polygon,
                    text=text,
                    language=language,
                    confidence=float(confidence),
                    script=script,
                )
            )

        return detections

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #

    def warmup(self, dummy: np.ndarray) -> None:
        """Run a tiny dummy detection to initialise CUDA kernels."""
        tiny = np.zeros((64, 64, 3), dtype=np.uint8)
        self._reader.readtext(tiny, detail=1, paragraph=False)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_script(text: str) -> str:
    """
    Determine the dominant Unicode script in *text* by counting characters.

    Returns one of: "Hangul", "Hiragana", "Katakana", "CJK", "Latin", "Other".
    """
    counts: dict[str, int] = {
        "Hangul": 0,
        "Hiragana": 0,
        "Katakana": 0,
        "CJK": 0,
        "Latin": 0,
        "Other": 0,
    }
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0xA960 <= cp <= 0xA97F:
            counts["Hangul"] += 1
        elif 0x3040 <= cp <= 0x309F:
            counts["Hiragana"] += 1
        elif 0x30A0 <= cp <= 0x30FF:
            counts["Katakana"] += 1
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
              or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
            counts["CJK"] += 1
        elif ch.isalpha() and unicodedata.category(ch).startswith("L"):
            try:
                script = unicodedata.name(ch, "").split()[0]
                if script == "LATIN":
                    counts["Latin"] += 1
                else:
                    counts["Other"] += 1
            except Exception:
                counts["Other"] += 1
        else:
            counts["Other"] += 1

    # Return the script with the most characters, defaulting to "Other"
    dominant = max(counts, key=lambda k: counts[k])
    return dominant if counts[dominant] > 0 else "Other"


def _script_to_language(script: str) -> str:
    """Map dominant script name to ISO 639-1 language code."""
    mapping = {
        "Hangul": "ko",
        "Hiragana": "ja",
        "Katakana": "ja",
        "CJK": "zh",
        "Latin": "en",
    }
    return mapping.get(script, "unknown")
