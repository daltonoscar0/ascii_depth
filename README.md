# ascii-depth

Real-time terminal ASCII depth map renderer.
Point your webcam at anything and watch a live green-on-black depth heatmap render in your terminal — near objects in dense, bright characters; far objects in sparse, dim ones.

```
Near  →  @ # % * +  →  bright green
Far   →  . : - =    →  dim green
      →  (space)    →  background
```

Built with **MiDaS** (monocular depth estimation) and a **Rust extension** that handles per-frame normalisation and character mapping in parallel across all CPU cores.

---

## How it works

```
┌─────────────┐     BGR frame      ┌───────────────────┐
│  Webcam     │ ────────────────▶  │  MiDaS_small      │
│  (OpenCV)   │                    │  (PyTorch)         │
└─────────────┘                    └────────┬──────────┘
                                            │ raw inverse-depth
                                            │ float32 array (H × W)
                                            ▼
                                   ┌───────────────────┐
                                   │  Rust extension   │
                                   │  ascii_depth_rs   │
                                   │                   │
                                   │  normalize_depth()│  ← parallel min-max
                                   │  depth_to_ascii() │  ← parallel resample
                                   └────────┬──────────┘
                                            │ Vec<String>  (one per row)
                                            ▼
                                   ┌───────────────────┐
                                   │  renderer.py      │
                                   │                   │
                                   │  ANSI 24-bit green│
                                   │  single write()   │
                                   └───────────────────┘
                                            │
                                            ▼
                                       Terminal 🖥️
```

### Why Rust for the ASCII mapping?

The depth array at 640×480 contains ~300 k values that must be resampled to the terminal grid (often 200×50 or similar), looked up in a character table, and colour-escaped — every frame, 15–30 times per second.  Doing this in pure Python with per-element loops would be the bottleneck.  The Rust extension releases the Python GIL and processes all rows in parallel with [rayon](https://github.com/rayon-rs/rayon), keeping the mapping step under 2 ms regardless of terminal size.

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | ≥ 3.9 |
| Rust toolchain | stable (via rustup) |
| maturin | ≥ 1.6 |

A webcam accessible at `/dev/video0` (Linux) or the system default (macOS / Windows) is required.

---

## Installation

### 1 — Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2 — Install maturin

```bash
pip install maturin
```

### 3 — Build the Rust extension

```bash
cd ascii_depth
maturin develop --release
```

This compiles `src/lib.rs` into a native `.so` / `.pyd` extension and installs the whole `ascii_depth` Python package in editable mode.  The `--release` flag enables full optimisation (LTO, opt-level 3).

### 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration:** swap the `torch` line in `requirements.txt` for the appropriate CUDA wheel from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).  Set `--device cuda` when running.

### 5 — Run

```bash
python -m ascii_depth.main
```

Press **Ctrl-C** to quit.  The terminal cursor and colours are restored automatically.

---

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--width` | terminal width | Character columns |
| `--height` | terminal height − 1 | Character rows (one row reserved for FPS counter) |
| `--fps` | `30` | Target capture frame rate |
| `--charset` | `" .:-=+*#%@"` | Characters ordered sparse → dense |
| `--device` | auto | `cpu` or `cuda` |
| `--camera` | `0` | Camera device index |

Example — lower resolution for slower machines:

```bash
python -m ascii_depth.main --width 80 --height 40 --fps 15
```

Example — richer character ramp:

```bash
python -m ascii_depth.main --charset " ░▒▓█"
```

---

## Project structure

```
ascii_depth/
├── src/
│   └── lib.rs                  # Rust extension (PyO3 + rayon)
├── python/
│   └── ascii_depth/
│       ├── capture.py          # OpenCV camera context manager
│       ├── depth.py            # MiDaS depth inference (model cached after first load)
│       ├── renderer.py         # ANSI terminal renderer
│       └── main.py             # CLI entry point and main loop
├── Cargo.toml                  # Rust crate manifest
├── pyproject.toml              # Python package + maturin build config
└── requirements.txt
```

---

## Performance

| Stage | CPU (Apple M-series / modern x86) | CUDA GPU |
|---|---|---|
| MiDaS_small inference | ~15–25 FPS | ~30+ FPS |
| Rust normalise + map | < 2 ms | < 2 ms |
| Terminal write | < 1 ms | < 1 ms |

Depth inference is the bottleneck on CPU.  If you need higher frame rates, try reducing `--width` / `--height` (smaller terminal grid = faster Rust step) or switching to a GPU.

---

## Customisation tips

- **Different depth model:** edit `depth.py` — change `"MiDaS_small"` to `"DPT_Large"` and `small_transform` to `dpt_transform` for higher accuracy at ~4× the cost.
- **Colour scheme:** edit the `_green()` function in `renderer.py` to change the RGB values in the ANSI escape.
- **Character set:** any ordered string works as `--charset`; block elements (`░▒▓█`) or Braille patterns can give a denser look on capable terminals.
