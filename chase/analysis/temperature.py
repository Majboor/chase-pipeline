"""Temperature maps — split by line physics.

Two regimes, two methods:

* **Hα → chromospheric temperature** via the line *width* (Molnar et al. 2019,
  ApJ 881, 99).  Hα forms out of LTE, so a Planck inversion is invalid; instead
  the half-depth width maps onto an ALMA brightness temperature,
  ``T[K] = (width[Å] − 0.553) / 6.12e-5``.

* **Fe I → photospheric temperature** via a Planck inversion.  Fe I 6569 Å is a
  photospheric LTE line, so ``B_ν(T) = I_core`` holds once the data is on an
  absolute radiance scale (see :func:`chase.calib.atlas_calibrate`).  Three core
  estimators are offered:
  - :func:`fe_planck_temperature` — Gaussian fit to the line core (default);
  - :func:`fe_eb_temperature`     — Eddington–Barbier per-wavelength inversion;
  - :func:`fe_voigt_temperature`  — Voigt fit (robust when the Fe line is
    under-sampled).

References
----------
Molnar et al. 2019, ApJ 881, 99 (10.3847/1538-4357/ab2ba3).
Eddington–Barbier: aa21259-13 (A&A 2013).
ISPy FTS atlas: ISP-SST/ISPy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..calib.intensity import planck_inversion

__all__ = [
    "measure_halpha_width",
    "halpha_width_map",
    "width_to_temperature",
    "halpha_width_temperature",
    "fe_planck_temperature",
    "fe_eb_temperature",
    "fe_voigt_temperature",
]

_MOLNAR_OFFSET = 0.553
_MOLNAR_SLOPE = 6.12e-5


# ---------------------------------------------------------------------------
# Hα width → chromospheric temperature (Molnar 2019)
# ---------------------------------------------------------------------------
def _find_crossing(spectrum, wavelengths, indices, level):
    for j in range(len(indices) - 1):
        i0, i1 = indices[j], indices[j + 1]
        s0, s1 = spectrum[i0], spectrum[i1]
        if (s0 <= level <= s1) or (s1 <= level <= s0):
            if s1 == s0:
                return wavelengths[i0]
            frac = (level - s0) / (s1 - s0)
            return wavelengths[i0] + frac * (wavelengths[i1] - wavelengths[i0])
    return np.nan


def measure_halpha_width(spectrum, wavelengths, core_wavelength=6562.8, wing_offset=1.0):
    """Half-depth Hα width (Å) of one spectrum (Molnar 2019 definition).

    Finds the core minimum, measures the depth relative to the mean intensity at
    ``±wing_offset`` Å, and returns the wavelength separation where the profile
    crosses the half-depth level.  Returns NaN if the measurement is ill-defined.
    """
    from scipy.interpolate import interp1d

    core_mask = np.abs(wavelengths - core_wavelength) < 0.5
    if core_mask.sum() < 3:
        return np.nan
    region = np.where(core_mask)[0]
    local_min = region[np.argmin(spectrum[region])]
    core_wav = wavelengths[local_min]
    i_core = spectrum[local_min]

    wb, wr = core_wav - wing_offset, core_wav + wing_offset
    if wb < wavelengths[0] or wr > wavelengths[-1]:
        return np.nan
    f = interp1d(wavelengths, spectrum, kind="linear")
    depth = 0.5 * (float(f(wb)) + float(f(wr))) - i_core
    if depth <= 0:
        return np.nan
    level = i_core + 0.5 * depth

    blue = np.where(wavelengths < core_wav)[0][::-1]
    red = np.where(wavelengths > core_wav)[0]
    if len(blue) < 2 or len(red) < 2:
        return np.nan
    bx = _find_crossing(spectrum, wavelengths, blue, level)
    rx = _find_crossing(spectrum, wavelengths, red, level)
    if np.isnan(bx) or np.isnan(rx):
        return np.nan
    return rx - bx


def width_to_temperature(width):
    """Molnar 2019: ``T[K] = (width[Å] − 0.553) / 6.12e-5``."""
    return (width - _MOLNAR_OFFSET) / _MOLNAR_SLOPE


def halpha_width_map(ha_cube, wavelengths, core_wavelength=6562.8) -> np.ndarray:
    """Per-pixel Hα width map (Å) for one frame's cube ``(nch, H, W)``."""
    nch, H, W = ha_cube.shape
    width = np.full((H, W), np.nan, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            spec = ha_cube[:, y, x]
            if np.any(np.isnan(spec)) or np.all(spec == 0):
                continue
            width[y, x] = measure_halpha_width(spec, wavelengths, core_wavelength)
    return width


def halpha_width_temperature(
    ha_cube, wavelengths, core_wavelength=6562.8, t_min=4000.0, t_max=20000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Chromospheric temperature map from Hα width (Molnar 2019).

    Returns ``(temperature_map, width_map)``; temperatures outside
    ``[t_min, t_max]`` are masked to NaN.
    """
    width = halpha_width_map(ha_cube, wavelengths, core_wavelength)
    temp = width_to_temperature(width)
    temp[(temp < t_min) | (temp > t_max)] = np.nan
    return temp, width


# ---------------------------------------------------------------------------
# Fe I → photospheric temperature (Planck) — three estimators
# ---------------------------------------------------------------------------
def _gaussian(x, amp, mu, sigma, offset):
    return offset + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_fe_core_gaussian(spectrum, wavelengths, n_lowest=6):
    from scipy.optimize import curve_fit

    if np.any(np.isnan(spectrum)) or np.all(spectrum == 0):
        return np.nan
    lowest = np.sort(np.argsort(spectrum)[:n_lowest])
    wav_fit, spec_fit = wavelengths[lowest], spectrum[lowest]
    cont = spectrum.max()
    try:
        popt, _ = curve_fit(
            _gaussian, wav_fit, spec_fit,
            p0=[spec_fit.min() - cont, wav_fit[np.argmin(spec_fit)], 0.05, cont],
            maxfev=2000,
        )
        i_core = popt[3] + popt[0]
        return float(i_core) if i_core > 0 else np.nan
    except (RuntimeError, ValueError):
        return np.nan


def fe_planck_temperature(fe_cube, wavelengths, ifact, t_min=3000.0, t_max=8000.0):
    """Photospheric T from a Gaussian-core Planck inversion of Fe I.

    For each pixel: Gaussian-fit the line core to the ~6 lowest channels, scale
    DN→radiance with ``ifact`` (from :func:`chase.calib.atlas_calibrate`), and
    invert the Planck function at the Fe core wavelength.

    Returns ``(temperature_map, core_intensity_map)``.
    """
    nch, H, W = fe_cube.shape
    core_int = np.full((H, W), np.nan, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            core_int[y, x] = _fit_fe_core_gaussian(fe_cube[:, y, x], wavelengths)
    fe_core_wav = wavelengths[np.argmin(np.mean(fe_cube, axis=(1, 2)))]
    temp = planck_inversion(core_int * ifact, wavelength_ang=fe_core_wav).astype(np.float32)
    temp[(temp < t_min) | (temp > t_max)] = np.nan
    return temp, core_int


def fe_eb_temperature(fe_cube, wavelengths, ifact, t_min=3000.0, t_max=8000.0):
    """Eddington–Barbier photospheric T (per-wavelength Planck inversion).

    At disk centre (μ=1) ``I(λ) = B(T(τ_λ=1), λ)``, so every wavelength probes a
    different height.  Returns ``(temp_cube, temp_core, temp_continuum)`` where
    ``temp_cube`` is ``(nch, H, W)``, ``temp_core`` samples the per-pixel minimum,
    and ``temp_continuum`` averages the last three channels.
    """
    nch, H, W = fe_cube.shape
    I_cgs = fe_cube.astype(np.float64) * ifact
    temp_cube = np.empty_like(I_cgs, dtype=np.float32)
    for ich in range(nch):
        temp_cube[ich] = planck_inversion(I_cgs[ich], wavelengths[ich]).astype(np.float32)
    core_ch = np.argmin(fe_cube, axis=0)
    temp_core = np.take_along_axis(temp_cube, core_ch[np.newaxis], axis=0)[0]
    temp_cont = np.mean(temp_cube[-3:], axis=0)
    for arr in (temp_core, temp_cont):
        arr[(arr < t_min) | (arr > t_max)] = np.nan
    return temp_cube, temp_core, temp_cont


def _fit_fe_voigt(spectrum, wavelengths):
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile

    if np.any(np.isnan(spectrum)) or np.all(spectrum == 0):
        return np.nan, np.nan
    cont = float(np.mean(spectrum[-3:]))
    core_idx = int(np.argmin(spectrum))

    def model(wav, amp, center, sigma, gamma, continuum, slope):
        v = voigt_profile(wav - center, sigma, gamma)
        vpeak = voigt_profile(0.0, sigma, gamma)
        return (continuum + slope * (wav - center)) - amp * v / vpeak

    try:
        popt, _ = curve_fit(
            model, wavelengths, spectrum,
            p0=[cont - spectrum[core_idx], wavelengths[core_idx], 0.05, 0.02, cont, 0.0],
            bounds=(
                [0, wavelengths[0], 0.001, 0.001, 0, -2000],
                [cont * 2, wavelengths[-1], 0.5, 0.5, cont * 2, 2000],
            ),
            maxfev=5000,
        )
        i_core = popt[4] - popt[0]
        return (float(i_core), float(popt[1])) if i_core > 0 else (np.nan, np.nan)
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def fe_voigt_temperature(fe_cube, wavelengths, ifact, t_min=3000.0, t_max=8000.0):
    """Photospheric T from a Voigt-core Planck inversion of Fe I.

    Robust when the Fe I line is under-sampled (a Gaussian can miss the true
    core).  Returns ``(temperature_map, core_intensity_map, center_map)``.
    """
    nch, H, W = fe_cube.shape
    core_map = np.full((H, W), np.nan, dtype=np.float32)
    center_map = np.full((H, W), np.nan, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            ic, cen = _fit_fe_voigt(fe_cube[:, y, x].astype(np.float64), wavelengths)
            core_map[y, x], center_map[y, x] = ic, cen
    median_center = float(np.nanmedian(center_map))
    temp = planck_inversion(core_map * ifact, median_center).astype(np.float32)
    temp[(temp < t_min) | (temp > t_max)] = np.nan
    return temp, core_map, center_map
