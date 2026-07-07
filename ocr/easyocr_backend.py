"""
ocr/easyocr_backend.py

EasyOCR implementation of the OCRBackend interface.

EasyOCR language compatibility
-------------------------------
EasyOCR enforces strict language grouping rules.  CJK languages
(ch_sim, ch_tra, ja, ko) must each be loaded in a separate Reader
and can only be combined with Latin-script languages (like "en").
You cannot mix e.g. ch_sim + ko in one Reader — it raises ValueError.

This backend handles that by creating one Reader per CJK language
(paired with English), plus one Reader for all remaining Latin-script
languages.  Results from all readers are merged and deduplicated.

Example — languages=["ko","ja","ch_sim","en"] produces three readers:
  Reader A: ["ko",     "en"]
  Reader B: ["ja",     "en"]
  Reader C: ["ch_sim", "en"]
  (no separate latin reader since "en" is already covered above)

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


# CJK languages that each require their own EasyOCR Reader
_CJK_LANGUAGES = {"ch_sim", "ch_tra", "ja", "ko"}


@ocr_registry.register("easyocr")
class EasyOCRBackend(OCRBackend):
    """
    EasyOCR-based text detector.

    Creates one EasyOCR Reader per CJK language (each paired with 'en')
    to comply with EasyOCR's language-compatibility constraints.

    Parameters
    ----------
    languages : list[str]
        EasyOCR language codes, e.g. ``["ko", "ja", "ch_sim", "en"]``.
    gpu : bool
        Whether to use CUDA.
    confidence_threshold : float
        Detections below this score are discarded.
    model_storage_directory : str, optional
        Override the EasyOCR model cache directory (useful on Kaggle).
    dedup_iou_threshold : float
        IoU threshold above which duplicate detections (from multiple
        readers scanning the same region) are suppressed.
    """

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = True,
        confidence_threshold: float = 0.5,
        model_storage_directory: Optional[str] = None,
        dedup_iou_threshold: float = 0.5,
    ) -> None:
        self.languages = languages or ["ko", "ja", "ch_sim", "en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.model_storage_directory = model_storage_directory
        self.dedup_iou_threshold = dedup_iou_threshold
        self._readers: list = []   # list of easyocr.Reader instances

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """
        Initialise one EasyOCR Reader per compatible language group.

        Each CJK language gets its own Reader paired with English.
        Any remaining non-CJK languages share one Reader.
        """
        import easyocr  # lazy import

        reader_kwargs: dict = dict(gpu=self.gpu)
        if self.model_storage_directory:
            reader_kwargs["model_storage_directory"] = self.model_storage_directory

        requested = set(self.languages)
        cjk_requested = requested & _CJK_LANGUAGES
        latin_requested = requested - _CJK_LANGUAGES  # e.g. {"en", "fr"}

        self._readers = []

        # One reader per CJK language, each paired with English
        for lang in sorted(cjk_requested):
            lang_list = [lang]
            if "en" not in lang_list:
                lang_list.append("en")  # required companion for CJK
            self._readers.append(easyocr.Reader(lang_list, **reader_kwargs))

        # One shared reader for any pure Latin-script languages
        # (skip if "en" already covered by CJK readers above)
        remaining_latin = latin_requested - {"en"} if cjk_requested else latin_requested
        if remaining_latin:
            lang_list = sorted(remaining_latin)
            if "en" not in lang_list:
                lang_list.append("en")
            self._readers.append(easyocr.Reader(lang_list, **reader_kwargs))

        # If the user only asked for "en" and no CJK, create one en-only reader
        if not self._readers:
            self._readers.append(easyocr.Reader(["en"], **reader_kwargs))

    def unload(self) -> None:
        """Delete all readers and free GPU memory."""
        self._readers.clear()
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
        Run all readers on *image* and return merged, deduplicated detections.
        """
        if not self._readers:
            raise RuntimeError("EasyOCRBackend.load() must be called before detect().")

        all_detections: List[TextDetection] = []

        for reader in self._readers:
            raw = reader.readtext(image, detail=1, paragraph=False)
            for bbox, text, confidence in raw:
                if confidence < self.confidence_threshold:
                    continue
                polygon = np.array(bbox, dtype=np.int32)
                script = _dominant_script(text)
                language = _script_to_language(script)
                all_detections.append(
                    TextDetection(
                        polygon=polygon,
                        text=text,
                        language=language,
                        confidence=float(confidence),
                        script=script,
                    )
                )

        # Deduplicate overlapping detections from multiple readers
        return _deduplicate(all_detections, self.dedup_iou_threshold)

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #

    def warmup(self, dummy: np.ndarray | None = None) -> None:
        """Run a tiny dummy pass on each reader to pre-warm CUDA kernels."""
        tiny = np.zeros((64, 64, 3), dtype=np.uint8)
        for reader in self._readers:
            reader.readtext(tiny, detail=1, paragraph=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_script(text: str) -> str:
    """
    Determine the dominant Unicode script in *text* by character count.

    Returns one of: "Hangul", "Hiragana", "Katakana", "CJK", "Latin", "Other".
    """
    counts: dict[str, int] = {
        "Hangul": 0, "Hiragana": 0, "Katakana": 0,
        "CJK": 0, "Latin": 0, "Other": 0,
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
        elif ch.isalpha():
            try:
                name = unicodedata.name(ch, "")
                counts["Latin"] += 1 if name.startswith("LATIN") else 1  # count as Latin if alpha
            except Exception:
                counts["Other"] += 1
        else:
            counts["Other"] += 1

    dominant = max(counts, key=lambda k: counts[k])
    return dominant if counts[dominant] > 0 else "Other"


def _script_to_language(script: str) -> str:
    """Map dominant script name to an ISO 639-1 language code."""
    return {
        "Hangul": "ko",
        "Hiragana": "ja",
        "Katakana": "ja",
        "CJK": "zh",
        "Latin": "en",
    }.get(script, "unknown")


def _bbox_iou(a_poly: np.ndarray, b_poly: np.ndarray) -> float:
    """Approximate IoU between two 4-point polygons via their axis-aligned bboxes."""
    ax0, ay0 = a_poly[:, 0].min(), a_poly[:, 1].min()
    ax1, ay1 = a_poly[:, 0].max(), a_poly[:, 1].max()
    bx0, by0 = b_poly[:, 0].min(), b_poly[:, 1].min()
    bx1, by1 = b_poly[:, 0].max(), b_poly[:, 1].max()

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def _deduplicate(
    detections: List[TextDetection],
    iou_threshold: float,
) -> List[TextDetection]:
    """
    Remove duplicate detections from overlapping readers.

    When two detections have IoU > *iou_threshold*, keep the one with
    the higher confidence.  Uses a simple greedy NMS approach.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending so higher-confidence detections win
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[TextDetection] = []

    for candidate in sorted_dets:
        suppressed = False
        for accepted in kept:
            if _bbox_iou(candidate.polygon, accepted.polygon) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    return kept

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
