"""
capture.py — OpenCV camera capture, wrapped as a context manager.

Usage
-----
    with CameraCapture(device=0, fps=30) as cam:
        for frame in cam.frames():
            process(frame)   # frame is a BGR uint8 numpy array

The context manager is the sole owner of the VideoCapture object.
__exit__ is the one and only place the camera is released.
"""

import time
from typing import Generator

import cv2
import numpy as np


class CameraCapture:
    """
    Thin wrapper around cv2.VideoCapture that:
      - Opens the device on __enter__ and releases it on __exit__.
      - Negotiates the requested resolution and FPS with the driver
        (the driver may silently cap these to hardware limits).
      - Throttles the frame generator so downstream code never receives
        frames faster than `target_fps`, even if the camera runs faster.
    """

    def __init__(
        self,
        device: int = 0,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """
        Parameters
        ----------
        device : int
            Camera index passed to cv2.VideoCapture (0 = first webcam).
        fps : float
            Target frame rate.  Actual rate may be lower if depth inference
            or rendering is the bottleneck.
        width, height : int
            Requested capture resolution.  The driver may choose the closest
            supported resolution instead.
        """
        self.device = device
        self.target_fps = fps
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._frame_duration = 1.0 / fps

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._last_frame_time = time.monotonic()
        return self

    def __exit__(self, *_) -> None:
        """
        Release the camera — the single authoritative cleanup point.

        On macOS, AVFoundation (which OpenCV uses under the hood) does not
        turn off the camera indicator until the underlying capture session is
        stopped AND the Python VideoCapture object is fully destroyed.

        Strategy:
          1. Swap `self._cap` out to a local variable and set `self._cap = None`
             immediately.  This ensures no other code path (generator finally,
             exception traceback frame, etc.) can see a non-None `_cap` after
             this point, preventing any double-release races.
          2. Call `release()` on the local reference.
          3. Let the local go out of scope at the end of __exit__, dropping
             the last Python reference and triggering the C++ destructor that
             signals AVFoundation to stop the session.
        """
        cap = self._cap       # move the reference to a local
        self._cap = None      # CameraCapture no longer owns it
        if cap is not None:
            cap.release()
            # `cap` drops to refcount 0 here (end of __exit__ scope),
            # destroying the C++ VideoCapture and releasing the OS camera lock.

    # ------------------------------------------------------------------
    # Frame generator
    # ------------------------------------------------------------------

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield raw BGR uint8 frames from the camera, one at a time.

        Camera ownership and cleanup belong exclusively to __exit__.
        This generator does NOT release the camera — it only reads frames.
        If the generator is abandoned mid-iteration (break, exception, Ctrl-C),
        Python will eventually call generator.close(), which is fine; the
        camera will be cleaned up when the enclosing `with` block exits.

        Yields
        ------
        np.ndarray
            Shape (height, width, 3), dtype uint8, colour order BGR.
        """
        if self._cap is None:
            raise RuntimeError("Call frames() inside a 'with' block.")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                # Camera stream ended or device failed.
                break

            yield frame

            # Throttle: sleep for any remaining time in the target frame interval.
            elapsed = time.monotonic() - self._last_frame_time
            sleep_for = self._frame_duration - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_frame_time = time.monotonic()
