"""High-level flare-sequence loader.

:func:`load_flare_sequence` is the workhorse the pipeline and TUI call: point it
at a folder of CHASE cubes, give it a patch (or full FOV), and it returns
aligned-ready float32 stacks plus the metadata downstream stages need.  It is a
refactor of the proven ``v50`` loader and preserves every physics rule:

* **shift-then-crop** with a padded buffer (see :func:`chase.calib.track_and_crop`);
* tracking on a **flare-free** channel (FE last wavelength, else HA continuum);
* **per-frame wavelength resampling** onto frame 0's grid;
* a **separate quiet-Sun background patch** for the contrast profile;
* an optional **separate align-patch** (small box for shift estimation, large box
  for the saved data).

Power users who want only one of these steps can call the modular functions in
:mod:`chase.calib` directly — this loader simply orchestrates them with the
streaming, memory-aware FITS reads that big full-disk cubes require.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import _compat  # noqa: F401
from ..calib.spatial import phase_shift
from ..calib.wavelength import resample_wavelength
from .download import discover_fits

__all__ = ["load_flare_sequence"]


def load_flare_sequence(
    fits_dir: str,
    patch: Optional[Sequence[int]] = None,
    align_patch: Optional[Sequence[int]] = None,
    core_wavelength: float = 6562.8,
    track: bool = True,
    pad: int = 20,
    resample: bool = True,
    verbose: bool = True,
):
    """Load and crop-track a CHASE HA(+FE) sequence from a directory.

    Parameters
    ----------
    fits_dir : str
        Directory of ``*HA.fits`` (and optional ``*FE.fits``) cubes.
    patch : [y0, y1, x0, x1] or None, optional
        Data crop window.  ``None`` → full FOV (no tracking).
    align_patch : [y0, y1, x0, x1] or None, optional
        Separate window for shift estimation (defaults to ``patch``).
    core_wavelength : float, optional
        Hα core wavelength in Å (default 6562.8).
    track : bool, optional
        Cross-correlation crop tracking on the flare-free channel.
    pad : int, optional
        Shift-then-crop padding in pixels.
    resample : bool, optional
        Per-frame wavelength resampling onto frame 0's grid.
    verbose : bool, optional

    Returns
    -------
    dict
        Keys: ``ha_cubes`` (nframes, nch_ha, H, W), ``fe_cubes`` (or None),
        ``cont_raw``, ``core_raw``, ``align_ref``, ``ha_bg`` (or None),
        ``wavelength_ha``, ``wavelength_fe``, ``times``, ``nframes``, ``H``,
        ``W``, ``nchannels_ha``, ``nchannels_fe``, ``core_idx``, ``patch``.
    """
    import astropy.io.fits as fits

    ha_names, fe_names = discover_fits(fits_dir)
    if not ha_names:
        raise FileNotFoundError(f"No *HA.fits files found in {fits_dir}")
    has_fe = len(fe_names) == len(ha_names) and len(fe_names) > 0
    nframes = len(ha_names)

    hdr0 = fits.open(ha_names[0])[1].header
    wavelength_ha = np.arange(hdr0["NAXIS3"]) * hdr0["CDELT3"] + hdr0["CRVAL3"]
    nch_ha = len(wavelength_ha)
    core_idx = int(np.argmin(np.abs(wavelength_ha - core_wavelength)))
    H_full, W_full = int(hdr0["NAXIS2"]), int(hdr0["NAXIS1"])

    if has_fe:
        hdr0_fe = fits.open(fe_names[0])[1].header
        wavelength_fe = np.arange(hdr0_fe["NAXIS3"]) * hdr0_fe["CDELT3"] + hdr0_fe["CRVAL3"]
        nch_fe = len(wavelength_fe)
    else:
        wavelength_fe, nch_fe = None, 0

    if patch is not None:
        y0, y1, x0, x1 = patch
    else:
        y0, y1, x0, x1 = 0, H_full, 0, W_full
    H, W = y1 - y0, x1 - x0
    ay0, ay1, ax0, ax1 = align_patch if align_patch is not None else (y0, y1, x0, x1)

    py0, py1 = max(0, y0 - pad), min(H_full, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(W_full, x1 + pad)
    oy, ox = y0 - py0, x0 - px0

    # Reference crop for tracking: prefer FE last wavelength (flare-free).
    if has_fe:
        ref_crop = fits.open(fe_names[0])[1].data[-1, ay0:ay1, ax0:ax1].astype(np.float32)
        if verbose:
            print("   Alignment reference: FE last wavelength (flare-free)")
    else:
        ref_crop = fits.open(ha_names[0])[1].data[-1, ay0:ay1, ax0:ax1].astype(np.float32)
        if verbose:
            print("   Alignment reference: HA continuum (FE not available)")

    # Quiet-Sun background patch, offset left by one patch width.
    bg_dx = -(x1 - x0)
    bx0, bx1 = x0 + bg_dx, x1 + bg_dx
    by0, by1 = y0, y1
    has_bg = patch is not None and bx0 >= 0 and by0 >= 0 and bx1 <= W_full and by1 <= H_full

    ha_cubes = np.zeros((nframes, nch_ha, H, W), dtype=np.float32)
    fe_cubes = np.zeros((nframes, nch_fe, H, W), dtype=np.float32) if has_fe else None
    ha_bg = np.zeros((nframes, nch_ha, H, W), dtype=np.float32) if has_bg else None
    cont_raw = np.zeros((nframes, H, W), dtype=np.float32)
    core_raw = np.zeros((nframes, H, W), dtype=np.float32)
    align_ref = np.zeros((nframes, H, W), dtype=np.float32)
    times = []

    for i in range(nframes):
        hdu_ha = fits.open(ha_names[i])[1]
        cube_ha = hdu_ha.data
        hdr_ha = hdu_ha.header
        wav_ha_i = np.arange(hdr_ha["NAXIS3"]) * hdr_ha["CDELT3"] + hdr_ha["CRVAL3"]

        if has_fe:
            hdu_fe = fits.open(fe_names[i])[1]
            cube_fe = hdu_fe.data
            hdr_fe = hdu_fe.header
            wav_fe_i = np.arange(hdr_fe["NAXIS3"]) * hdr_fe["CDELT3"] + hdr_fe["CRVAL3"]

        # Shift-then-crop: estimate integer shift on the align window.
        if patch is not None and track:
            cur = (cube_fe if has_fe else cube_ha)[-1, ay0:ay1, ax0:ax1].astype(np.float32)
            sy, sx = phase_shift(ref_crop, cur)
            sy, sx = int(round(sy)), int(round(sx))
            if verbose and (sy or sx):
                print(f"  Frame {i:>2d}: shift y={sy:+d}, x={sx:+d}")
        else:
            sy, sx = 0, 0
        cy, cx = oy - sy, ox - sx

        ha_crop = cube_ha[:, py0:py1, px0:px1][:, cy:cy + H, cx:cx + W].astype(np.float32)
        if resample:
            ha_crop = resample_wavelength(ha_crop, wav_ha_i, wavelength_ha)
        ha_cubes[i] = ha_crop
        cont_raw[i] = ha_cubes[i, -1]
        core_raw[i] = ha_cubes[i, core_idx]

        if has_bg:
            bg = cube_ha[:, by0 - sy:by1 - sy, bx0 - sx:bx1 - sx].astype(np.float32)
            ha_bg[i] = resample_wavelength(bg, wav_ha_i, wavelength_ha) if resample else bg

        if has_fe:
            fe_crop = cube_fe[:, py0:py1, px0:px1][:, cy:cy + H, cx:cx + W].astype(np.float32)
            if resample:
                fe_crop = resample_wavelength(fe_crop, wav_fe_i, wavelength_fe)
            fe_cubes[i] = fe_crop
            align_ref[i] = fe_cubes[i, -1]
        else:
            align_ref[i] = cont_raw[i]

        times.append(hdr_ha.get("DATE-OBS", hdr_ha.get("DATE_OBS", f"frame_{i}")))
        if verbose:
            print(f"  Loading FITS {i + 1}/{nframes}", end="\r")

    if verbose:
        print(f"\nLoaded {nframes} frames, {H}x{W} px; HA {nch_ha} ch, FE {nch_fe} ch")

    return {
        "ha_cubes": ha_cubes,
        "fe_cubes": fe_cubes,
        "cont_raw": cont_raw,
        "core_raw": core_raw,
        "align_ref": align_ref,
        "ha_bg": ha_bg,
        "wavelength_ha": wavelength_ha,
        "wavelength_fe": wavelength_fe,
        "times": times,
        "nframes": nframes,
        "H": H,
        "W": W,
        "nchannels_ha": nch_ha,
        "nchannels_fe": nch_fe,
        "core_idx": core_idx,
        "patch": np.array(patch) if patch is not None else np.array([]),
    }
