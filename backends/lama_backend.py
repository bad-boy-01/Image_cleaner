"""
backends/lama_backend.py

LaMa (Large Mask inpainting) backend using the ``simple-lama-inpainting``
package, which wraps the pretrained LaMa model with a clean Python API.

LaMa is well-suited to this use case because:
  - It was specifically designed for large mask inpainting (e.g. text bubbles).
  - It runs efficiently on Kaggle T4 GPUs.
  - It requires no authentication or API key.
  - The ``simple-lama-inpainting`` wrapper handles model download automatically.

Install
-------
    pip install simple-lama-inpainting

Model weights are downloaded on first call to ``load_model()`` and cached
in ``~/.cache/simple_lama``.

Registration
------------
This module registers itself with ``backend_registry`` on import.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backends.base import InpaintingBackend
from backends.registry import backend_registry


@backend_registry.register("lama")
class LamaBackend(InpaintingBackend):
    """
    LaMa inpainting backend via ``simple-lama-inpainting``.

    Parameters
    ----------
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model: Optional[object] = None  # SimpleLama instance

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Download (if necessary) and load LaMa weights."""
        from simple_lama_inpainting import SimpleLama  # lazy import

        self._model = SimpleLama(device=self.device)

    def unload_model(self) -> None:
        """Delete the model and free GPU memory."""
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inpainting
    # ------------------------------------------------------------------ #

    def inpaint_patch(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Run LaMa inpainting on a single patch.

        Parameters
        ----------
        patch : np.ndarray
            RGB uint8 (H, W, 3).
        mask : np.ndarray
            uint8 (H, W) with values {0, 255}.
            255 = region to reconstruct.

        Returns
        -------
        np.ndarray
            RGB uint8 (H, W, 3) with masked region reconstructed.
        """
        if self._model is None:
            raise RuntimeError(
                "LamaBackend.load_model() must be called before inpaint_patch()."
            )

        from PIL import Image  # lazy import

        # simple-lama-inpainting accepts PIL Images
        pil_image = Image.fromarray(patch, mode="RGB")
        pil_mask = Image.fromarray(mask, mode="L")

        result_pil = self._model(pil_image, pil_mask)

        result = np.array(result_pil.convert("RGB"), dtype=np.uint8)

        # Ensure output shape matches input (defensive check)
        if result.shape[:2] != patch.shape[:2]:
            # Resize back if the model changed dimensions internally
            from PIL import Image as PILImage
            result_pil_resized = result_pil.resize(
                (patch.shape[1], patch.shape[0]),
                PILImage.LANCZOS,
            )
            result = np.array(result_pil_resized.convert("RGB"), dtype=np.uint8)

        return result
