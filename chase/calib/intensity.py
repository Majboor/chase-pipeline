"""Intensity calibration: normalisation, scaling, and absolute (atlas) calib.

Three levels, cheapest first:

* :func:`normalize_intensity` — divide by the quiet-Sun mean (relative units).
* :func:`integral_scaling` — match the integral of two spectra (Davis's
  ``satprocess`` method; corrects frame-to-frame throughput drift).
* :func:`atlas_calibrate` — absolute DN→radiance calibration against the FTS
  solar atlas via **ISPy**, fitting the **line wings** (not the core, which is
  non-LTE/smeared) at disk centre (``mu=1.0``).  ISPy is an *optional* dependency
  — the rest of the package works without it.

:func:`planck_inversion` converts an absolute radiance to a brightness
temperature, used by the Fe I photospheric temperature maps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["normalize_intensity", "integral_scaling", "atlas_calibrate", "planck_inversion"]

# CGS physical constants
_H = 6.62607015e-27   # erg s
_C = 2.99792458e10    # cm/s
_K = 1.380649e-16     # erg/K


def normalize_intensity(cube: np.ndarray, ref_cube: np.ndarray, clip: Tuple[float, float] = (0, 2)) -> np.ndarray:
    """Normalise a cube by the mean of a quiet-Sun reference region."""
    out = cube / np.mean(ref_cube)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def integral_scaling(ref_spectrum: np.ndarray, target_spectrum: np.ndarray) -> float:
    """Integral-ratio intensity scale factor (Davis ``calculate_scaling``).

    Returns ``∫ref / ∫target`` over the overlap where both are non-zero, or 1.0
    if there is no usable overlap.  Multiply a cube by this to match throughput.

    Ported from ``satprocess`` (BSD-3, Finlay Davis).
    """
    from scipy.integrate import trapezoid

    ref = np.asarray(ref_spectrum, dtype=float)
    tgt = np.asarray(target_spectrum, dtype=float)
    mask = (ref != 0) & (tgt != 0)
    if not np.any(mask):
        return 1.0
    denom = trapezoid(tgt[mask])
    if denom == 0:
        return 1.0
    return float(trapezoid(ref[mask]) / denom)


def atlas_calibrate(
    wavelengths: np.ndarray,
    disk_center_cube: np.ndarray,
    wing_continuum_frac: float = 0.85,
) -> Tuple[float, float]:
    """Absolute DN→radiance calibration against the FTS atlas (ISPy).

    Fits the disk-centre quiet-Sun profile to the FTS solar atlas using only the
    line **wings** (channels where the atlas exceeds ``wing_continuum_frac`` of
    the continuum), at ``mu=1.0`` / ``calib_at_dc=True``.

    Parameters
    ----------
    wavelengths : ndarray (nchannels,)
        Wavelength axis of the disk-centre cube, in Å.
    disk_center_cube : ndarray (nchannels, H, W)
        Disk-centre Fe I cube (e.g. the 100×100 CRPIX patch, single frame).
    wing_continuum_frac : float, optional
        Threshold (fraction of continuum) above which a channel counts as wing.

    Returns
    -------
    ifact : float
        Multiply DN by this to get erg/(Hz·s·sr·cm²).
    woff : float
        Wavelength offset in Å.

    Raises
    ------
    ImportError
        If ISPy is not installed (``pip install 'chase-pipeline[atlas]'``).
    """
    try:
        from ISPy.spec import atlas as isp_atlas
        from ISPy.spec.calib import get_calibration as isp_get_calibration
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "atlas_calibrate requires ISPy. Install with "
            "`pip install 'chase-pipeline[atlas]'` or `pip install ISPy`."
        ) from e
    from scipy.interpolate import interp1d

    fe_avg = np.mean(disk_center_cube, axis=(1, 2))

    fts = isp_atlas.atlas()
    wave_fts, spec_fts, cont_fts = fts.get(
        wavelengths[0] - 0.5, wavelengths[-1] + 0.5, cgs=True, perHz=True
    )

    f_atlas = interp1d(wave_fts, spec_fts, kind="cubic", bounds_error=False, fill_value="extrapolate")
    atlas_at_obs = f_atlas(wavelengths)
    wing_idx = np.where(atlas_at_obs > wing_continuum_frac * cont_fts[0])[0]

    calibration = isp_get_calibration(
        wavelengths, fe_avg, wave_fts, spec_fts,
        mu=1.0, calib_at_dc=True, wave_idx=wing_idx,
    )
    return float(calibration[0]), float(calibration[1])


def planck_inversion(I_nu, wavelength_ang: float = 6569.0):
    """Invert the Planck function ``B_nu(T)`` to brightness temperature (CGS).

    ``T = hν / (k · ln(1 + 2hν³/(c²·I_ν)))``

    Parameters
    ----------
    I_nu : float or ndarray
        Spectral radiance in erg/(Hz·s·sr·cm²).
    wavelength_ang : float, optional
        Wavelength in Å (default 6569, the Fe I line).

    Returns
    -------
    T : float or ndarray
        Temperature in Kelvin.
    """
    lam_cm = wavelength_ang * 1e-8
    nu = _C / lam_cm
    coeff = 2.0 * _H * nu ** 3 / _C ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        T = _H * nu / (_K * np.log(coeff / I_nu + 1.0))
    return T
