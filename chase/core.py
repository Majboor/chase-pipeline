"""Backward-compatibility shim for the pre-2.0 flat ``chase.core`` API.

The original ``chase-pipeline`` exposed everything from ``chase.core``.  The code
now lives in focused submodules (:mod:`chase.io`, :mod:`chase.calib`, …), but the
old names keep working so existing tutorials, notebooks, and Colab links don't
break.  New code should import from the submodules or the top-level package.
"""

from __future__ import annotations

import numpy as np

from . import _compat  # noqa: F401
from .calib.spatial import find_disk_center, recenter
from .calib.wavelength import resample_wavelength  # noqa: F401
from .io.download import download_file, resolve_inputs
from .io.fits_io import load_fits_data

TERMINAL_BANNER = """
============================================================
        CHASE Satellite Data Calibration Pipeline
============================================================
Core calibration logic for CHASE H-alpha / Fe I data.
============================================================
"""

__all__ = [
    "TERMINAL_BANNER",
    "download_file",
    "load_fits_from_url_or_txt",
    "load_fits_data",
    "find_disk_center",
    "recenter_image_cube",
    "pixel_to_arcsec",
    "align_subpixel",
    "extract_qs_region",
    "calibrate_wavelength",
    "normalize_intensity",
    "extract_subregion",
    "create_disk_mask",
    "run_pipeline",
]


def load_fits_from_url_or_txt(input_path, save_dir):
    """Deprecated alias for :func:`chase.io.resolve_inputs`."""
    return resolve_inputs(input_path, save_dir)


def recenter_image_cube(data_cube, threshold_ratio=0.2):
    """Original API: brightness-centroid recentre via integer roll."""
    h, w = data_cube.shape[1:]
    cx, cy, _ = find_disk_center(data_cube[0], threshold_ratio)
    dx, dy = w // 2 - cx, h // 2 - cy
    return recenter(data_cube, (dx, dy)), (dx, dy)


def pixel_to_arcsec(x, y, shape, scale=0.44):
    h, w = shape
    return (x - w // 2) * scale, (y - h // 2) * scale


def align_subpixel(ref, target, upsample_factor=100):
    """Original API: subpixel align via phase correlation + spline shift."""
    from scipy.ndimage import shift as ndshift
    from skimage.registration import phase_cross_correlation

    shift_yx, _, _ = phase_cross_correlation(ref, target, upsample_factor=upsample_factor)
    return ndshift(target, shift=shift_yx, order=3), shift_yx


def extract_qs_region(data_cube, size=100):
    h, w = data_cube.shape[1:]
    cx, cy, half = w // 2, h // 2, size // 2
    return data_cube[:, cy - half:cy + half, cx - half:cx + half]


def calibrate_wavelength(profile, ref_wavelength=656.28, dispersion=0.0025):
    center = np.argmin(profile)
    return ref_wavelength + (np.arange(len(profile)) - center) * dispersion


def normalize_intensity(cube, ref_cube):
    from .calib.intensity import normalize_intensity as _norm

    return _norm(cube, ref_cube)


def extract_subregion(cube, cx, cy, width, height):
    x1, x2 = max(cx - width // 2, 0), min(cx + width // 2, cube.shape[2])
    y1, y2 = max(cy - height // 2, 0), min(cy + height // 2, cube.shape[1])
    return cube[:, y1:y2, x1:x2]


def create_disk_mask(shape, cx, cy, radius):
    Y, X = np.ogrid[: shape[0], : shape[1]]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2


def run_pipeline(fits_file, **kwargs):
    """Deprecated single-file diagnostic runner.

    The modern entry point is :func:`chase.run_pipeline` with a
    :class:`chase.Config`.  This wrapper keeps the old single-file signature
    working for a directory or one FITS file.
    """
    import os

    from .config import Config
    from .pipeline import run_pipeline as _run

    fits_dir = fits_file if os.path.isdir(fits_file) else os.path.dirname(fits_file) or "."
    cfg = Config(fits_dir=fits_dir, out_dir=kwargs.get("fig_dir", "./figures"),
                 full_fov=True, optical_flow=False, gif=False, diagnostics=True)
    return _run(cfg)
