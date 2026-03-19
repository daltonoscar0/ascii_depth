//! ascii_depth_rs — Rust extension module for fast depth-to-ASCII conversion.
//!
//! This crate is compiled into a native Python extension via PyO3 + maturin.
//! It exposes two functions to the Python runtime:
//!
//!   - `normalize_depth(array)` — min-max normalise a 2D f32 array to [0, 1]
//!   - `depth_to_ascii(array, width, height, charset)` — resample + map values to chars (near → dense)
//!
//! Both functions release the Python GIL and use rayon for data-parallel work,
//! so they run across all available CPU cores without blocking the interpreter.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a single normalised depth value to a character in `charset`.
///
/// MiDaS outputs *inverse* depth (disparity), so after `normalize_depth` the
/// values are already oriented as:
///   - Near objects → depth ≈ 1.0 → high charset index → dense characters (`@`, `#`, …)
///   - Far objects  → depth ≈ 0.0 → low charset index  → sparse characters (` `, `.`, …)
///
/// No inversion is applied here; the mapping is direct.
fn depth_to_char(depth: f32, charset: &[char]) -> char {
    // Clamp to guard against out-of-range values from the depth model.
    let clamped = depth.clamp(0.0, 1.0);

    // Scale to a charset index, rounding to the nearest integer.
    let idx = (clamped * (charset.len() - 1) as f32).round() as usize;

    // `.min()` is a safety belt in case floating-point rounding overshoots.
    charset[idx.min(charset.len() - 1)]
}

/// Nearest-neighbour resample from a flat row-major 2D array.
///
/// Maps destination pixel `(row, col)` in a `dst_rows × dst_cols` grid back
/// to the closest source pixel in a `src_rows × src_cols` array.
///
/// The `+ 0.5 / - 0.5` offset centres each destination pixel within the
/// source cell it maps to, avoiding the half-pixel bias that plain integer
/// scaling introduces at the edges.
fn sample(
    arr: &[f32],
    src_rows: usize,
    src_cols: usize,
    row: usize,
    col: usize,
    dst_rows: usize,
    dst_cols: usize,
) -> f32 {
    // Map destination row → source row, clamped to valid range.
    let sr = ((row as f32 + 0.5) / dst_rows as f32 * src_rows as f32 - 0.5)
        .round()
        .clamp(0.0, (src_rows - 1) as f32) as usize;

    // Map destination column → source column, clamped to valid range.
    let sc = ((col as f32 + 0.5) / dst_cols as f32 * src_cols as f32 - 0.5)
        .round()
        .clamp(0.0, (src_cols - 1) as f32) as usize;

    // Row-major index into the flat buffer.
    arr[sr * src_cols + sc]
}

// ---------------------------------------------------------------------------
// Python-exported functions
// ---------------------------------------------------------------------------

/// Resample a normalised 2D depth array to `width × height` and map each
/// value to a character from `charset`, returning one `String` per row.
///
/// # Arguments
/// * `depth_array` — 2D f32 numpy array, values expected in [0, 1].
/// * `width`       — Number of character columns in the output grid.
/// * `height`      — Number of character rows in the output grid.
/// * `charset`     — String of characters ordered sparse → dense
///                   (index 0 = far/background, last = near/foreground).
///
/// # Returns
/// A `Vec<String>` with `height` elements, each `width` characters wide,
/// ready to be printed line-by-line to the terminal.
///
/// # Performance
/// The GIL is released before entering the rayon parallel iterator so that
/// Python threads are not blocked while Rust processes the frame.
#[pyfunction]
fn depth_to_ascii(
    py: Python<'_>,
    depth_array: PyReadonlyArray2<f32>,
    width: usize,
    height: usize,
    charset: &str,
) -> PyResult<Vec<String>> {
    let arr = depth_array.as_array();
    let (src_rows, src_cols) = (arr.shape()[0], arr.shape()[1]);

    // Copy into an owned Vec so it is `Send` and can cross thread boundaries
    // inside rayon without holding a reference to the numpy buffer.
    let flat: Vec<f32> = arr.iter().copied().collect();
    let chars: Vec<char> = charset.chars().collect();

    // Release the GIL: rayon will spawn work on the thread pool, which must
    // not hold the GIL or Python will deadlock.
    let rows: Vec<String> = py.allow_threads(|| {
        (0..height)
            .into_par_iter()          // each row processed on a rayon thread
            .map(|row| {
                (0..width)
                    .map(|col| {
                        // Resample source depth to this terminal cell position.
                        let v = sample(&flat, src_rows, src_cols, row, col, height, width);
                        // Convert the depth value to its display character.
                        depth_to_char(v, &chars)
                    })
                    .collect()        // collect chars into a String for this row
            })
            .collect()                // collect all row Strings into a Vec
    });

    Ok(rows)
}

/// Min-max normalise a 2D f32 numpy array to the range [0.0, 1.0].
///
/// A new array is allocated and returned; the input is not modified.
///
/// # Why normalise in Rust?
/// MiDaS outputs raw inverse-depth values whose range varies frame-to-frame.
/// Normalising before ASCII mapping ensures the full charset is always used,
/// maximising contrast regardless of the scene's absolute depth range.
///
/// # Edge case
/// If all values in the array are identical (range == 0), the denominator is
/// clamped to 1e-8 to avoid a divide-by-zero; the output will be all zeros.
#[pyfunction]
fn normalize_depth<'py>(
    py: Python<'py>,
    depth_array: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = depth_array.as_array();

    // Own the data so rayon threads can access it without the GIL.
    let flat: Vec<f32> = arr.iter().copied().collect();

    // Find global min and max in parallel (two separate reductions).
    let (min_val, max_val) = py.allow_threads(|| {
        let min_val = flat
            .par_iter()
            .copied()
            .reduce(|| f32::INFINITY, f32::min);
        let max_val = flat
            .par_iter()
            .copied()
            .reduce(|| f32::NEG_INFINITY, f32::max);
        (min_val, max_val)
    });

    // Guard against a flat (constant-value) frame.
    let range = (max_val - min_val).max(1e-8);

    // Normalise every element: v_norm = (v - min) / range
    let normalized: Vec<f32> = py.allow_threads(|| {
        flat.par_iter().map(|&v| (v - min_val) / range).collect()
    });

    // Reshape the flat Vec back to the original 2D shape and hand it to numpy.
    let shape = [arr.shape()[0], arr.shape()[1]];
    let out = ndarray::Array2::from_shape_vec(shape, normalized)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(out.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// Register the extension module and its exported symbols.
///
/// The module name must match `module-name` in `pyproject.toml` (the leaf
/// component after the last dot), and the `.so` will be placed at the path
/// `ascii_depth/ascii_depth_rs.so` inside the installed package.
#[pymodule]
fn ascii_depth_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depth_to_ascii, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_depth, m)?)?;
    Ok(())
}
