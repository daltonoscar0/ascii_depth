"""
main.py — Entry point for the ascii-depth terminal renderer.

Run with:
    python -m ascii_depth.main [options]

Threading model
---------------
MiDaS inference (~50–100 ms on CPU) is the dominant bottleneck.  Running it
synchronously in the render loop would cap the display at ~10–20 FPS and
cause every frame to stall while waiting for PyTorch to finish.

Instead, a dedicated `_DepthWorker` daemon thread runs inference continuously:

    Main thread (single loop)          Worker thread (inference)
    ──────────────────────────         ─────────────────────────
    capture frame                      wait for new frame
    submit frame to worker  ─────────► run MiDaS on it
    if first result ready:   ◄──────── store result
      render with latest depth
    repeat immediately                 repeat immediately

The worker always processes the *most recently submitted* frame.  If the main
thread submits faster than inference completes, older frames are silently
dropped.  If inference is slower than capture (typical on CPU), each rendered
frame reuses the last known depth while the next one is being computed.

This decoupling lets the terminal update as fast as the Rust renderer can
write strings to stdout (~5–15 ms per frame), independent of inference speed.

FPS counter
-----------
The counter measures end-to-end wall time for the render loop (capture +
render, not inference), which is what the user actually sees update.

Keyboard interrupt (Ctrl-C) stops the worker thread and restores the terminal.

Signal handling
---------------
Python's default SIGHUP handler (sent when the terminal window is closed)
and SIGTERM handler (sent by `kill`) both terminate the process immediately
without running `finally` blocks or `__exit__` methods.  That leaves the
camera active and the terminal in a broken state.

Both signals are remapped to raise KeyboardInterrupt in the main thread,
routing them through the existing try/except/finally cleanup path.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import numpy as np

from ascii_depth.capture import CameraCapture
from ascii_depth.depth import estimate_depth
from ascii_depth.renderer import DEFAULT_CHARSET, render_frame

# ANSI cursor control sequences.
_HIDE_CURSOR  = "\033[?25l"         # hide cursor during animation
_RESET_CURSOR = "\033[0m\033[?25h"  # reset colour + restore cursor on exit


def _fps_line(fps: float) -> str:
    """
    Return an ANSI string that overwrites the top-left corner with the FPS
    counter without disturbing the rest of the frame.

    `\\033[1;1H` moves the cursor to row 1, column 1.  Because render_frame()
    reserves `terminal_lines - 1` rows for ASCII art, the counter lands on
    the first art row — no extra scroll line is ever added.
    """
    return f"\033[1;1H\033[97mFPS: {fps:5.1f}\033[0m"


# ---------------------------------------------------------------------------
# Background depth worker
# ---------------------------------------------------------------------------

class _DepthWorker(threading.Thread):
    """
    Daemon thread that runs MiDaS inference continuously.

    The main loop calls `submit(frame)` to hand off the latest camera frame.
    The worker picks it up, runs inference, and stores the result where
    `get_depth()` can retrieve it.

    Only the *most recent* submitted frame is ever processed.  If a new frame
    arrives before the previous inference finishes, the previous frame is
    silently discarded.  This is the correct behaviour: we always want to
    display the scene as it looks *now*, not catch up on a backlog.
    """

    def __init__(self, device: str | None) -> None:
        super().__init__(daemon=True)  # daemon=True: thread dies with the process
        self._device = device

        # Shared state guarded by _lock.
        self._lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None   # latest unprocessed frame
        self._latest_depth:  np.ndarray | None = None   # latest completed depth map

        # Event set as soon as the first depth result is ready.
        # The render loop waits on this before drawing the first frame so it
        # never renders a blank/None depth.
        self.first_result_ready = threading.Event()

        self._stop = threading.Event()

    def submit(self, frame: np.ndarray) -> None:
        """Hand the worker a new camera frame.  Thread-safe."""
        with self._lock:
            self._pending_frame = frame   # overwrites any not-yet-processed frame

    def get_depth(self) -> np.ndarray | None:
        """Return the latest completed depth map, or None before first result."""
        with self._lock:
            return self._latest_depth

    def stop(self) -> None:
        """Signal the worker to exit after its current inference completes."""
        self._stop.set()

    def run(self) -> None:
        """
        Worker body.  Spins in a tight loop:
          1. Grab the latest pending frame (if any).
          2. Run MiDaS inference on it.
          3. Store the result so the render loop can pick it up.
        """
        while not self._stop.is_set():
            # Atomically grab and clear the pending frame.
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                # No new frame yet — yield the CPU briefly rather than spinning.
                time.sleep(0.001)
                continue

            # Run inference.  This is the slow step (~50–100 ms on CPU).
            depth = estimate_depth(frame, device=self._device)

            with self._lock:
                self._latest_depth = depth

            # Signal the render loop that at least one depth map is available.
            self.first_result_ready.set()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time ASCII depth map renderer (MiDaS + Rust).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--width",   type=int,   default=None, help="Character columns (default: terminal width).")
    p.add_argument("--height",  type=int,   default=None, help="Character rows (default: terminal height − 1).")
    p.add_argument("--fps",     type=float, default=30.0, help="Camera capture target FPS.")
    p.add_argument("--charset", type=str,   default=DEFAULT_CHARSET, help="Sparse→dense character set.")
    p.add_argument("--device",  type=str,   default=None, help="PyTorch device: 'cpu' or 'cuda' (auto-detected).")
    p.add_argument("--camera",  type=int,   default=0,    help="Camera device index.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _signal_handler(signum: int, frame: object) -> None:
    """
    Convert SIGHUP / SIGTERM into KeyboardInterrupt so the main loop's
    try/except/finally block runs and the camera + terminal are cleaned up.

    Without this, closing the terminal window (SIGHUP) or running
    `kill <pid>` (SIGTERM) would terminate the process immediately,
    bypassing __exit__ and leaving the macOS camera indicator active.
    """
    raise KeyboardInterrupt


def main() -> None:
    args = parse_args()

    # Remap SIGHUP (terminal window closed) and SIGTERM (kill) so they
    # trigger our cleanup path instead of hard-terminating the process.
    signal.signal(signal.SIGHUP,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("Loading MiDaS model… (first run may download weights)", flush=True)

    # Start the depth worker before opening the camera so PyTorch model loading
    # happens in the background while OpenCV initialises the device.
    worker = _DepthWorker(device=args.device)
    worker.start()

    # Hide cursor for a cleaner animation; restored in the finally block.
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    # Rolling window of render-loop wall-clock times (not inference times).
    # 30 samples ≈ 1 second of history at typical frame rates.
    frame_times: list[float] = []

    try:
        with CameraCapture(device=args.camera, fps=args.fps, width=640, height=480) as cam:
            # Single loop: submit every frame to the worker and render once the
            # first depth result is available.  Using one loop avoids calling
            # cam.frames() twice, which would trigger the generator's finally
            # block and release the camera on the first break.
            for bgr_frame in cam.frames():
                worker.submit(bgr_frame)

                # Block cheaply until the first inference completes.
                # Subsequent iterations return immediately (event stays set).
                if not worker.first_result_ready.wait(timeout=0):
                    continue  # still waiting for the first depth map

                t0 = time.monotonic()
                depth = worker.get_depth()
                render_frame(depth, charset=args.charset, width=args.width, height=args.height)

                frame_times.append(time.monotonic() - t0)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                sys.stdout.write(_fps_line(avg_fps))
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass  # clean exit via finally
    finally:
        worker.stop()
        sys.stdout.write(_RESET_CURSOR + "\n")
        sys.stdout.flush()
        print("Bye.")


if __name__ == "__main__":
    main()
