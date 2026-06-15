"""FITS cube reading and product saving.

CHASE RSM cubes store their image data in extension 1 (``hdul[1]``), with the
spectral axis as ``NAXIS3`` and a per-frame wavelength grid encoded by
``CRVAL3``/``CDELT3``.  ``BUNIT`` and ``DATE_OBS`` may be present.

These helpers stay deliberately low-level — a single file in, arrays out — so
they compose into the higher-level sequence loader and the modular SDK.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .. import _compat  # noqa: F401  (installs lzma stub before astropy import)

__all__ = [
    "load_cube",
    "load_fits_data",
    "wavelength_grid",
    "extract_disk_center",
    "save_cube_fits",
    "save_npz",
]


def wavelength_grid(header) -> np.ndarray:
    """Build the wavelength axis (Å) from a FITS header.

    ``wav[k] = CRVAL3 + k * CDELT3`` for ``k`` in ``0 .. NAXIS3-1``.
    """
    n = int(header["NAXIS3"])
    return np.arange(n) * header["CDELT3"] + header["CRVAL3"]


def load_cube(
    filepath: str,
    region: Optional[Tuple[int, int, int, int]] = None,
    dtype=np.float32,
) -> Tuple[np.ndarray, "object", np.ndarray]:
    """Load one FITS spectral cube with its header and wavelength grid.

    Parameters
    ----------
    filepath : str
        Path to a ``*HA.fits`` / ``*FE.fits`` cube.
    region : (y0, y1, x0, x1) or None, optional
        If given, only this spatial window is read (memory-mapped section slice,
        so large full-disk cubes are not pulled fully into RAM).
    dtype : numpy dtype, optional
        Output dtype (default ``float32`` — science data stays float).

    Returns
    -------
    cube : ndarray (nchannels, H, W)
    header : astropy FITS header
    wavelengths : ndarray (nchannels,) in Å

    Examples
    --------
    >>> cube, hdr, wav = load_cube("RSM..._HA.fits")
    >>> cube.shape, wav[0], wav[-1]
    ((118, 2313, 2304), 6559.4, 6565.1)
    """
    import astropy.io.fits as fits

    with fits.open(filepath, memmap=True, do_not_scale_image_data=False) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{filepath} has no image extension (hdul[1])")
        hdu = hdul[1]
        header = hdu.header
        wav = wavelength_grid(header)
        if region is None:
            cube = np.asarray(hdu.data, dtype=dtype)
        else:
            y0, y1, x0, x1 = region
            cube = np.asarray(hdu.section[:, y0:y1, x0:x1], dtype=dtype)
    return cube, header, wav


def load_fits_data(filepath: str) -> np.ndarray:
    """Backward-compatible loader: return ``hdul[1].data`` only."""
    cube, _, _ = load_cube(filepath, dtype=None if False else np.float32)
    return cube


def extract_disk_center(
    filepath: str,
    size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Extract the disk-center patch used for absolute calibration.

    Absolute intensity/wavelength calibration is always done at solar disk
    centre (``CRPIX1``/``CRPIX2``), because that location does not change frame
    to frame.  Returns a ``size × size`` cube centred on the reference pixel.

    Parameters
    ----------
    filepath : str
        FITS cube (typically the Fe I file).
    size : int, optional
        Patch side length in pixels (default 100).

    Returns
    -------
    cube : ndarray (nchannels, size, size)
    wavelengths : ndarray (nchannels,)
    center : (cy, cx) integer reference-pixel location used.
    """
    import astropy.io.fits as fits

    with fits.open(filepath, memmap=True) as hdul:
        hdu = hdul[1]
        header = hdu.header
        wav = wavelength_grid(header)
        H, W = int(header["NAXIS2"]), int(header["NAXIS1"])
        cx = int(round(header.get("CRPIX1", W / 2)))
        cy = int(round(header.get("CRPIX2", H / 2)))
        half = size // 2
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        cube = np.asarray(hdu.section[:, y0:y1, x0:x1], dtype=np.float32)
    return cube, wav, (cy, cx)


def save_cube_fits(cube: np.ndarray, path: str, header=None) -> str:
    """Write an aligned/calibrated cube to a FITS file (data in ``hdul[1]``)."""
    import astropy.io.fits as fits

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=np.asarray(cube, dtype=np.float32), header=header)
    fits.HDUList([primary, image]).writeto(path, overwrite=True)
    return path


def save_npz(path: str, compressed: bool = True, **arrays) -> str:
    """Save named arrays to an ``.npz`` archive (aligned cubes, wavelengths…)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path
