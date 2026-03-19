"""
renderer.py — Terminal ASCII frame renderer.

Pipeline for one frame
----------------------
1. Call `ascii_depth_rs.normalize_depth()` (Rust, parallel) to scale raw MiDaS
   output to [0, 1].  After normalisation, near objects → 1.0, far → 0.0.
2. Call `ascii_depth_rs.depth_to_ascii()` (Rust, parallel) to resample the depth
   map to terminal dimensions and map each cell to a character.  Near cells get
   dense characters (`@`, `#`, …); far cells get spaces (invisible on a black bg).
3. Wrap each row in an ANSI 24-bit white escape whose brightness tracks the
   normalised depth value: near → white (255, 255, 255), far → black (0, 0, 0).
4. Write the whole frame in a single `sys.stdout.write()` call.

Visual result
-------------
  Near objects  →  dense chars  @#%*+  →  bright white
  Far objects   →  spaces                →  black (terminal background)

ANSI escape sequences used
--------------------------
  \\033[H\\033[J        — move cursor to top-left and erase display
  \\033[38;2;R;G;Bm    — set foreground colour (24-bit / "true colour")
  \\033[0m             — reset all attributes
  \\033[?25l / ?25h    — hide / show cursor (used by main.py)
  \\033[row;colH       — absolute cursor position (used for FPS overlay)
"""

from __future__ import annotations

import shutil
import sys

import numpy as np

from ascii_depth import ascii_depth_rs  # compiled Rust extension

# Default character ramp: sparse (far/dark) → dense (near/bright).
# Index 0 is a space so fully-far pixels are invisible against a black terminal.
DEFAULT_CHARSET = " .:-=+*#%@"

# Pre-built escape sequences used every frame.
_CLEAR = "\033[H\033[J"  # home cursor + erase display
_RESET = "\033[0m"        # reset colour attributes


def _white(text: str, brightness: float) -> str:
    """
    Wrap `text` in a 24-bit ANSI white escape sequence scaled by `brightness`.

    Parameters
    ----------
    text : str
        One row of ASCII art characters.
    brightness : float
        Normalised depth value for this row, in [0, 1].
        1.0 → pure white (255, 255, 255) for near objects.
        0.0 → black (0, 0, 0) — invisible against a black terminal background.

    Returns
    -------
    str
        ANSI-escaped string ready to write to the terminal.

    Note
    ----
    One colour escape is emitted per *row* rather than per character.  This
    reduces frame byte-count by ~10×, which matters at 15–30 FPS.  The brightness
    is sampled from the centre pixel of each row so the gradient still tracks
    depth across the image even though individual characters within a row share
    one colour level.
    """
    v = int(brightness * 255)
    return f"\033[38;2;{v};{v};{v}m{text}"


def render_frame(
    depth_array: np.ndarray,
    charset: str = DEFAULT_CHARSET,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """
    Render one depth frame as ASCII art to the terminal.

    Parameters
    ----------
    depth_array : np.ndarray
        Raw 2D float32 array from `estimate_depth()`.  Any value range is
        accepted; normalisation to [0, 1] happens inside the Rust extension.
    charset : str
        Characters ordered sparse → dense (index 0 = far, last index = near).
    width : int, optional
        Character columns.  Defaults to current terminal width.
    height : int, optional
        Character rows.  Defaults to terminal height minus one row reserved
        for the FPS counter written by main.py.
    """
    # Query the live terminal size when dimensions are not pinned by the CLI.
    # `fallback` is used when stdout is not attached to a real terminal.
    if width is None or height is None:
        term = shutil.get_terminal_size(fallback=(80, 24))
        width = width or term.columns
        height = height or max(1, term.lines - 1)  # reserve one row for FPS

    # --- Step 1: normalise depth to [0, 1] via Rust ---
    # Returns a new array; input is unchanged.
    # After this call: near pixels ≈ 1.0, far pixels ≈ 0.0.
    normed: np.ndarray = ascii_depth_rs.normalize_depth(depth_array)

    # --- Step 2: resample + character mapping via Rust ---
    # Each of the `height` strings is `width` characters wide.
    # Near pixels → dense chars; far pixels → spaces (charset index 0).
    rows: list[str] = ascii_depth_rs.depth_to_ascii(normed, width, height, charset)

    # --- Step 3: build the full frame string ---
    # Opening with _CLEAR ensures leftover characters from a previous (larger)
    # frame or terminal resize don't bleed into the current one.
    buf_parts: list[str] = [_CLEAR]

    for row_idx, row_text in enumerate(rows):
        # Sample the normalised depth at the horizontal centre of this row.
        # This single sample drives the whole row's brightness, avoiding the
        # ~10× byte-count cost of per-character ANSI escapes.
        col_centre = width // 2
        brightness = float(normed[
            int((row_idx + 0.5) / height * normed.shape[0]),
            int((col_centre + 0.5) / width  * normed.shape[1]),
        ])
        # brightness is already in [0, 1] with near=1.0, so pass it directly:
        # no inversion needed here — the Rust extension handles charset ordering.
        buf_parts.append(_white(row_text, brightness))
        buf_parts.append("\n")

    buf_parts.append(_RESET)  # leave the terminal clean for the next write

    # --- Step 4: single write to stdout ---
    # One write() call per frame avoids per-row syscall overhead.
    sys.stdout.write("".join(buf_parts))
    sys.stdout.flush()
