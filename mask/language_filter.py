"""
mask/language_filter.py

Filters a list of TextDetection objects, keeping only those whose
dominant Unicode script matches the configured target scripts.

The filter operates purely on the ``script`` field already set by the
OCR backend — no secondary model or API call is needed.

Design notes
------------
* The set of target scripts is fully configurable via ``Settings.target_scripts``.
* English (Latin-script) detections are kept or removed based on
  ``Settings.remove_english``.
* Adding a new script (e.g. Devanagari) requires only adding the name
  to ``target_scripts`` in the config — no code changes.
"""

from __future__ import annotations

from typing import List

from core.models import TextDetection
from config.settings import Settings


# Scripts that are considered "non-target" (i.e. always kept unless
# remove_english is True and the script is Latin).
_LATIN_SCRIPTS = {"Latin"}


def filter_non_english(
    detections: List[TextDetection],
    settings: Settings,
) -> List[TextDetection]:
    """
    Return only the detections that should be *removed* from the image.

    A detection is scheduled for removal when:
      - Its ``script`` is in ``settings.target_scripts`` (e.g. CJK, Hangul), OR
      - ``settings.remove_english`` is True AND its script is Latin.

    Parameters
    ----------
    detections : list[TextDetection]
        Raw detections from the OCR backend (all languages).
    settings : Settings
        Pipeline configuration.

    Returns
    -------
    list[TextDetection]
        Subset of *detections* that are flagged for removal.
    """
    target_set = set(settings.target_scripts)
    results: List[TextDetection] = []

    for det in detections:
        script = det.script

        if script in target_set:
            results.append(det)
            continue

        if settings.remove_english and script in _LATIN_SCRIPTS:
            results.append(det)

    return results
