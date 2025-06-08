"""kugelblick / distort.py
=================================
CLI tool that pre-distorts an input PNG so that – after refraction through a
transparent glass sphere – it is perceived undistorted again.

The implementation follows an idealised optical model (Snell’s law, perfect
geometry, orthographic view). The main routine builds a radial lookup table via
ray-tracing, then uses ``cv2.remap`` to warp the image.

Example usage
-------------
python distort.py input.png output.png \
    --sphere_diameter_mm 10 \
    --distance_mm 0 \
    --refractive_index 1.46 \
    --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image

# Local imports
from utils import generate_radial_mapping, mm_to_px

# Calibration (optional)
try:
    from calibration import calibrate as _calibrate

    _CALIBRATION_AVAILABLE = True
except ImportError:
    _CALIBRATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_remap_arrays(
    width: int,
    height: int,
    radius_px: int,
    center_x: float,
    center_y: float,
    r_obs: np.ndarray,
    r_plane: np.ndarray,
):
    """Generate *map_x* and *map_y* arrays for ``cv2.remap``.

    Parameters
    ----------
    width, height
        Dimensions of the target (pre-distorted) image.
    radius_px
        Sphere radius in pixels.
    center_x, center_y
        Optical centre in pixel coordinates.
    r_obs, r_plane
        1-D arrays that define the radial mapping ``r_plane = g(r_obs)``.  The
        routine interpolates the inverse (observed radius for a given plane
        radius).
    """

    # Inverse mapping (plane radius -> observed radius).
    # r_plane may contain NaNs – ignore them during the interpolation setup.
    mask = ~np.isnan(r_plane)

    if not np.all(mask):
        # Ensure monotonicity within the valid range.
        r_plane_valid = r_plane[mask]
        r_obs_valid = r_obs[mask]
    else:
        r_plane_valid = r_plane
        r_obs_valid = r_obs

    # r_plane_valid should be monotonically increasing; enforce by sorting just
    # in case minor numerical jitter breaks the assumption.
    idx_sort = np.argsort(r_plane_valid)
    r_plane_valid = r_plane_valid[idx_sort]
    r_obs_valid = r_obs_valid[idx_sort]

    # Prepare coordinate grids for the destination image (print plane).
    yy, xx = np.mgrid[0:height, 0:width]

    # Shift origin to the centre of the sphere.
    dx = xx.astype(np.float64) - center_x
    dy = yy.astype(np.float64) - center_y
    r_p = np.hypot(dx, dy)

    # Inverse radial mapping via linear interpolation.
    r_o = np.interp(r_p, r_plane_valid, r_obs_valid, left=np.nan, right=np.nan)

    # Scale factors for the mapping.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = r_o / r_p

    # The centre pixel (r_p == 0) would lead to a division by zero – treat it
    # explicitly.
    scale[r_p == 0] = 0.0

    map_x = center_x + dx * scale
    map_y = center_y + dy * scale

    # Outside the sphere (r_p > radius_px) we keep the artwork unchanged:
    # remap coordinates equal identity.
    mask_outside = r_p > radius_px
    map_x[mask_outside] = xx[mask_outside]
    map_y[mask_outside] = yy[mask_outside]

    # OpenCV expects single-precision float32 maps.
    return map_x.astype(np.float32), map_y.astype(np.float32)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def distort(
    infile: Path,
    outfile: Path,
    sphere_diameter_mm: float,
    distance_mm: float,
    refractive_index: float,
    dpi: float,
):
    """Pre-distort *infile* and save the result to *outfile*."""

    # ---------------------------------------------------------------------
    # Load and prepare the input image
    # ---------------------------------------------------------------------
    img_in = Image.open(infile).convert("RGBA")  # Preserve alpha if present.
    width, height = img_in.size

    # Sphere parameters in pixel units.
    radius_px = int(round(mm_to_px(sphere_diameter_mm / 2.0, dpi)))
    gap_px = mm_to_px(distance_mm, dpi)

    # Optical centre (middle of the image).
    cx = width / 2.0
    cy = height / 2.0

    # ---------------------------------------------------------------------
    # Build radial lookup via ray-tracing.
    # ---------------------------------------------------------------------
    r_obs, r_plane = generate_radial_mapping(
        r_max=radius_px,
        gap_px=gap_px,
        refr_index=refractive_index,
    )

    # ---------------------------------------------------------------------
    # Construct OpenCV remapping arrays and apply the warp.
    # ---------------------------------------------------------------------
    map_x, map_y = _build_remap_arrays(
        width=width,
        height=height,
        radius_px=radius_px,
        center_x=cx,
        center_y=cy,
        r_obs=r_obs,
        r_plane=r_plane,
    )

    img_np = np.array(img_in)
    # cv2.remap expects the image in BGR(A) – we just pass the raw byte order
    # unchanged and convert back afterwards.
    warped = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    Image.fromarray(warped).save(outfile)
    print(f"Saved pre-distorted image to {outfile}")


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inverse sphere distortion tool")

    p.add_argument("input", type=Path, nargs="?", help="Input PNG file (skipped in calibration mode)")
    p.add_argument("output", type=Path, nargs="?", help="Output PNG file (skipped in calibration mode)")

    p.add_argument("--sphere_diameter_mm", type=float, default=10.0,
                   help="Sphere diameter in millimetres (default: 10 mm)")
    p.add_argument("--distance_mm", type=float, default=0.0,
                   help="Gap between sphere and print plane (mm)")
    p.add_argument("--refractive_index", type=float, default=1.46,
                   help="Refractive index of the sphere material")
    p.add_argument("--dpi", type=float, default=300.0,
                   help="Resolution used to convert mm -> px")

    # Place-holder for future calibration support.
    p.add_argument("--calibration_image", type=Path,
                   help="Run calibration on the specified image and exit")

    return p


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)


    # Handle calibration-only mode.
    if args.calibration_image is not None:
        if not _CALIBRATION_AVAILABLE:
            parser.error(
                "Calibration functionality requires scipy and calibration.py."
            )

        res = _calibrate(args.calibration_image)
        print("Calibration complete. Suggested parameters:")
        for k, v in res.items():
            print(f"  {k}: {v:.6f}")
        sys.exit(0)

    # Distortion mode requires positional arguments.
    if args.input is None or args.output is None:
        parser.error("input and output arguments are required unless --calibration_image is used.")

    distort(
        infile=args.input,
        outfile=args.output,
        sphere_diameter_mm=args.sphere_diameter_mm,
        distance_mm=args.distance_mm,
        refractive_index=args.refractive_index,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
