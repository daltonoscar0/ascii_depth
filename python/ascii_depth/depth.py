"""
depth.py — MiDaS monocular depth estimation.

MiDaS (Mixed Dataset Sampling) is a model from Intel ISL that predicts
relative inverse depth from a single RGB image.  "Relative" means the
values are not in real-world units (metres) but are proportional to 1/depth,
so nearer objects have *higher* values.  `renderer.py` inverts this when
mapping to characters so that near == dense == bright.

Model choice
------------
We use `MiDaS_small`, a lightweight EfficientNet-based variant (~100 MB).
It runs at ~15–25 FPS on CPU and >30 FPS on CUDA, making it well suited
for real-time terminal rendering.  Swap the model name and transform to
`DPT_Large` / `dpt_transform` for higher accuracy at the cost of speed.

Caching
-------
The model and its transform are loaded once and stored as module-level
globals.  Subsequent calls to `estimate_depth()` skip the load step.
Torch Hub also caches the downloaded weights in `~/.cache/torch/hub/`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# These are set on the first call to _load_model() and reused for every
# subsequent frame, avoiding the ~1-second load penalty per frame.

_model: Optional[torch.nn.Module] = None
_transform = None                           # callable: np.ndarray → Tensor
_device: Optional[torch.device] = None


def _load_model(
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, object, torch.device]:
    """
    Load MiDaS_small from Torch Hub (or return the cached instance).

    Parameters
    ----------
    device : str, optional
        ``'cpu'`` or ``'cuda'``.  When omitted, CUDA is used if available,
        otherwise CPU.

    Returns
    -------
    model : torch.nn.Module
        The MiDaS model in eval mode, moved to `device`.
    transform : callable
        The matching preprocessing transform.  Accepts a uint8 RGB numpy
        array and returns a batched float tensor ready for the model.
    device : torch.device
        The device the model lives on.
    """
    global _model, _transform, _device

    # Return immediately if already loaded (common case after first frame).
    if _model is not None:
        return _model, _transform, _device  # type: ignore[return-value]

    # Pick the compute device.
    chosen = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load the model ---
    # `trust_repo=True` suppresses the Hub security prompt on first run.
    # MiDaS_small uses an EfficientNet backbone and runs efficiently on CPU.
    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        pretrained=True,
        trust_repo=True,
    )
    model.to(chosen)
    model.eval()  # disable dropout / batch-norm training behaviour

    # --- Load the matching preprocessing transform ---
    # Each MiDaS variant requires its own transform (different input size,
    # normalisation stats, etc.).  `small_transform` matches MiDaS_small.
    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
    )
    transform = transforms.small_transform

    # Store in module globals so the next call skips this block entirely.
    _model, _transform, _device = model, transform, chosen
    return model, transform, chosen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_depth(
    bgr_frame: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Run MiDaS_small on a single BGR camera frame and return a depth map.

    Parameters
    ----------
    bgr_frame : np.ndarray
        uint8 array of shape (H, W, 3) in OpenCV's default BGR colour order.
    device : str, optional
        ``'cpu'`` or ``'cuda'``.  Defaults to auto-detection.

    Returns
    -------
    np.ndarray
        float32 array of shape (H, W) containing raw (non-normalised)
        inverse-depth values.  Higher values = closer to the camera.
        The range varies per frame; call `normalize_depth()` from the Rust
        extension before mapping to characters.
    """
    model, transform, dev = _load_model(device)

    # MiDaS expects an RGB uint8 array, not OpenCV's BGR.
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # The transform resizes and normalises the image for the model's input
    # size, then adds a batch dimension: shape becomes (1, 3, H', W').
    input_batch = transform(rgb).to(dev)

    with torch.no_grad():
        prediction = model(input_batch)

        # The model output is at the model's internal resolution (e.g. 256×256
        # for MiDaS_small).  Interpolate back to the original frame size so
        # the depth map aligns pixel-for-pixel with the camera image.
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),                          # add channel dim
            size=(bgr_frame.shape[0], bgr_frame.shape[1]),   # target: H × W
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # remove batch + channel dims → shape (H, W)

    return prediction.cpu().numpy().astype(np.float32)
