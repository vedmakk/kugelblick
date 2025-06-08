"""kugelblick / calibration.py
================================
Calibrates unknown Kugelblick parameters (refractive index n and distance h
between the sphere and the image surface) based on a photo of a planar grid
taken through the sphere.

The routine uses a checkerboard pattern (OpenCV corner grid) as a reference
because its corners can be reliably detected. From the detected pixel
coordinates, a radial mapping rₒᵦₛ (camera image) ↔ rₚₗₐₙₑ (grid) is constructed.
Then, n, h, and a scaling factor s (px/mm) are determined using non-linear
optimization, so that the ray tracing model’s predicted rₒᵦₛ best matches the
measured values.

The output is the set of optimized parameters, which can be used directly in distort.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np

# Optional dependency – we import lazily because SciPy ist relativ groß und
# u. U. nicht installiert. Ein klarer Hinweis an den User genügt.
try:
    from scipy.optimize import minimize  # type: ignore

    _SCIPY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – in Test-Umgebungen evtl. ohne SciPy
    _SCIPY_AVAILABLE = False

from utils import generate_radial_mapping


# ---------------------------------------------------------------------------
# Corner detection helper
# ---------------------------------------------------------------------------


def _detect_chessboard_corners(
    img_path: Path,
    pattern_size: tuple[int, int],
):
    """Detect inner chessboard corners using OpenCV.

    Parameters
    ----------
    img_path : Path
        Path to calibration image.
    pattern_size : tuple[int, int]
        Number of inner corners per chessboard row and column (columns, rows),
        e.g. (7, 7).

    Returns
    -------
    np.ndarray
        Detected corner positions with shape (N, 2) and dtype float64.
    """

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(img_path)

    ret, corners = cv2.findChessboardCorners(img_gray, pattern_size)
    if not ret:
        raise RuntimeError(
            "Chessboard corners could not be detected. "
            "Ensure the calibration pattern is clearly visible and the correct pattern_size is specified."
        )

    # Refine to sub-pixel accuracy for better stability.
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

    return corners.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Optimisation target
# ---------------------------------------------------------------------------


def _calibration_objective(
    params: np.ndarray,
    r_plane_mm: np.ndarray,
    r_obs_px: np.ndarray,
    diameter_mm: float,
):
    """Error function – returns RMS difference between predicted and observed radii.

    Parameters
    ----------
    params : np.ndarray
        Vector ``[n, gap_mm, scale_px_per_mm]``.
    r_plane_mm : np.ndarray
        True radial distances of grid corners in **millimetres**.
    r_obs_px : np.ndarray
        Observed radial distances of grid corners in **pixels** (camera image).
    diameter_mm : float
        Known sphere diameter (mm).
    """

    n, gap_mm, scale = params

    if n <= 1.0 or scale <= 0.0 or gap_mm < 0.0:
        # Physically impossible – enforce large penalty.
        return 1e9

    radius_px = (diameter_mm / 2.0) * scale
    gap_px = gap_mm * scale

    # Build radial mapping once per iteration.
    r_obs_table, r_plane_table = generate_radial_mapping(
        r_max=int(round(radius_px)),
        gap_px=gap_px,
        refr_index=n,
        num_samples=2048,
    )

    # Interpolate predicted *observed* radii for the given r_plane values.
    # First, convert r_plane_mm to pixel scale.
    r_plane_px = r_plane_mm * scale

    # We need the inverse mapping (plane → obs).
    mask = ~np.isnan(r_plane_table)
    r_plane_valid = r_plane_table[mask]
    r_obs_valid = r_obs_table[mask]

    # Ensure ascending order for interp.
    idx_sort = np.argsort(r_plane_valid)
    r_plane_valid = r_plane_valid[idx_sort]
    r_obs_valid = r_obs_valid[idx_sort]

    r_pred_px = np.interp(r_plane_px, r_plane_valid, r_obs_valid, left=np.nan, right=np.nan)

    # Exclude NaNs (outside the physical region) from error computation.
    mask_valid = ~np.isnan(r_pred_px)

    if not np.any(mask_valid):
        return 1e9

    diff = r_pred_px[mask_valid] - r_obs_px[mask_valid]
    return float(np.sqrt(np.mean(diff**2)))


# ---------------------------------------------------------------------------
# Calibration driver
# ---------------------------------------------------------------------------


def calibrate(
    img_path: Path,
    pattern_size: tuple[int, int] = (7, 7),
    square_size_mm: float = 5.0,
    sphere_diameter_mm: float = 10.0,
):
    """Estimate refractive index *n* and gap_mm from a calibration image.

    Returns
    -------
    dict[str, float]
        Optimised parameters ``{"refractive_index": n, "distance_mm": gap_mm, "scale_px_per_mm": s}``.
    """

    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is required for calibration but is not installed.\n"
            "Install via 'pip install scipy'."
        )

    corners_px = _detect_chessboard_corners(img_path, pattern_size)

    # Build world-plane coordinates (assuming the board lies flat on print plane).
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 2), dtype=np.float64)
    objp[:, 0] = np.tile(np.arange(cols), rows) * square_size_mm
    objp[:, 1] = np.repeat(np.arange(rows), cols) * square_size_mm

    # Shift so that the grid centre is at (0,0).
    objp -= objp.mean(axis=0, keepdims=True)

    # Radii.
    r_plane_mm = np.hypot(objp[:, 0], objp[:, 1])

    # Image centre – approximate with mean of detected corners.
    centre_px = corners_px.mean(axis=0)
    r_obs_px = np.hypot(corners_px[:, 0] - centre_px[0], corners_px[:, 1] - centre_px[1])

    # ---------------------------------------------------------------------
    # Optimisation
    # ---------------------------------------------------------------------

    x0 = np.array([1.46, 0.0, 20.0])  # n, gap_mm, scale_px/mm  (heuristic start)

    bounds = [
        (1.3, 2.0),  # typical glass range
        (0.0, 5.0),  # gap 0–5 mm
        (0.1, 100.0),  # pixels per mm (depends on image resolution)
    ]

    res = minimize(
        _calibration_objective,
        x0,
        args=(r_plane_mm, r_obs_px, sphere_diameter_mm),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not res.success:
        raise RuntimeError("Calibration failed: " + res.message)

    n_opt, gap_mm_opt, scale_opt = res.x

    return {
        "refractive_index": float(n_opt),
        "distance_mm": float(gap_mm_opt),
        "scale_px_per_mm": float(scale_opt),
        "rms_px": float(res.fun),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate sphere parameters from a chessboard image.")

    p.add_argument("image", type=Path, help="Calibration photo (PNG/JPEG etc.)")

    p.add_argument("--pattern_cols", type=int, default=7, help="Number of inner corners per row (columns)")
    p.add_argument("--pattern_rows", type=int, default=7, help="Number of inner corners per column (rows)")
    p.add_argument("--square_size_mm", type=float, default=5.0, help="Edge length of one chessboard square in mm")
    p.add_argument("--sphere_diameter_mm", type=float, default=10.0, help="Known sphere diameter in mm")

    return p


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pattern_size = (args.pattern_cols, args.pattern_rows)

    res = calibrate(
        img_path=args.image,
        pattern_size=pattern_size,
        square_size_mm=args.square_size_mm,
        sphere_diameter_mm=args.sphere_diameter_mm,
    )

    print("Optimised parameters:")
    for k, v in res.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
