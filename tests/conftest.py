"""Synthetic FITS fixtures — small, fast, hermetic.

We never touch the real 25 x 375 MB cubes in unit tests.  Instead we synthesise
tiny CHASE-like HA/FE cubes (a few channels, 64x64) with a known sub-pixel
spatial shift and a known per-frame ``CRVAL3`` drift, so each physics rule can be
checked against ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

import chase._compat  # noqa: F401  (lzma stub before astropy)


def _gauss2d(H, W, cy, cx, sigma):
    y, x = np.ogrid[:H, :W]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))


def _absorption_spectrum(wav, core_wav, depth=0.6, width=0.4, continuum=1000.0):
    """A continuum with a Gaussian absorption line at ``core_wav``."""
    return continuum * (1.0 - depth * np.exp(-0.5 * ((wav - core_wav) / width) ** 2))


def make_cube(wav, H, W, core_wav, feature_xy, shift=(0.0, 0.0), continuum=1000.0, depth=0.6):
    """Build a (nch, H, W) cube: an absorption line modulated by a moving blob."""
    cy, cx = feature_xy
    blob = _gauss2d(H, W, cy + shift[0], cx + shift[1], sigma=6.0)
    spec = _absorption_spectrum(wav, core_wav, depth=depth, continuum=continuum)
    # brighten where the blob is; keep the absorption line shape
    cube = spec[:, None, None] * (1.0 + 0.5 * blob[None, :, :])
    return cube.astype(np.float32)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Write a small HA+FE sequence to ``tmp_path`` and return its metadata.

    Returns a dict with ``dir``, ``nframes``, ``H``, ``W``, ``nch_ha``,
    ``nch_fe``, ``core_wav``, ``shifts`` (true per-frame integer shifts),
    ``patch`` and ``feature``.
    """
    import astropy.io.fits as fits

    nframes, H, W = 4, 64, 64
    # HA grid is sampled finely enough (≈0.1 Å) for the Molnar half-depth width
    # measurement; the real instrument has 118 channels over the same span.
    nch_ha, nch_fe = 60, 8
    core_wav = 6562.8
    feature = (32, 38)  # cy, cx of the bright blob (the "flare")
    # True integer shifts injected per frame (frame 0 = reference)
    shifts = [(0, 0), (1, -2), (-1, 1), (2, 0)]

    ha_wav0 = np.linspace(6559.4, 6565.1, nch_ha)
    fe_wav0 = np.linspace(6567.0, 6571.0, nch_fe)
    ha_cdelt = ha_wav0[1] - ha_wav0[0]
    fe_cdelt = fe_wav0[1] - fe_wav0[0]

    d = tmp_path / "fits"
    d.mkdir()

    for i, (sy, sx) in enumerate(shifts):
        # Per-frame wavelength drift in CRVAL3 (the real instrument does this)
        crval_ha = ha_wav0[0] + 0.02 * i
        crval_fe = fe_wav0[0] + 0.02 * i
        wav_ha_i = np.arange(nch_ha) * ha_cdelt + crval_ha
        wav_fe_i = np.arange(nch_fe) * fe_cdelt + crval_fe

        ha = make_cube(wav_ha_i, H, W, core_wav, feature, shift=(sy, sx))
        fe = make_cube(wav_fe_i, H, W, 6569.0, feature, shift=(sy, sx), depth=0.4)

        for cube, wav0, crval, cdelt, tag in (
            (ha, ha_wav0, crval_ha, ha_cdelt, "HA"),
            (fe, fe_wav0, crval_fe, fe_cdelt, "FE"),
        ):
            # NAXIS*/PCOUNT/GCOUNT are filled in by astropy from the data shape;
            # we only add the WCS + observation keywords the loader reads.
            hdu = fits.ImageHDU(data=cube)
            hdu.header["CRVAL3"], hdu.header["CDELT3"] = crval, cdelt
            hdu.header["CRPIX1"], hdu.header["CRPIX2"] = W // 2, H // 2
            hdu.header["DATE_OBS"] = f"2023-03-29T02:1{i}:00"
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(d / f"RSM2023032900{i}_{tag}.fits", overwrite=True)

    return {
        "dir": str(d),
        "nframes": nframes,
        "H": H,
        "W": W,
        "nch_ha": nch_ha,
        "nch_fe": nch_fe,
        "core_wav": core_wav,
        "shifts": shifts,
        "patch": [18, 50, 22, 54],
        "feature": feature,
        "ha_wav0": ha_wav0,
        "fe_wav0": fe_wav0,
    }
