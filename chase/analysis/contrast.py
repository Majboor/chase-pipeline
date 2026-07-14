"""Spectral contrast profile (wavelength vs time).

The contrast map averages an aligned Hα cube over the spatial axes to get one
spectrum per frame, then expresses each as a fractional excess over a quiet-Sun
baseline.  The physically correct definition uses a **separate background patch**::

    contrast(t, λ) = (flare(t, λ) − bg(t, λ)) / bg(t, λ)  −  [same at frame 0]

which removes instrumental throughput drift that a naive ``frame_t/frame_0 − 1``
leaves in.  When no background patch is supplied it falls back to the naive form.

In the resulting wavelength-vs-time image the erupting CME shows up as a red
Doppler feature in the Hα wing — the science hook for the A&A letter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["contrast_profile"]


def contrast_profile(
    ha_cubes: np.ndarray,
    background: Optional[np.ndarray] = None,
    baseline_frame: int = 0,
    correct_drift: bool = True,
):
    """Compute the wavelength-vs-time contrast profile of an aligned cube.

    Parameters
    ----------
    ha_cubes : ndarray (nframes, nchannels, H, W)
        Aligned Hα spectral stack.
    background : ndarray (nframes, nchannels, H, W) or None, optional
        Quiet-Sun background patch (same shape).  If given, the
        ``(flare − bg)/bg`` form is used; otherwise ``frame_t/frame_0 − 1``.
    baseline_frame : int, optional
        Frame whose profile is subtracted as the pre-flare baseline.
    correct_drift : bool, optional
        Apply 1-D spectral-drift correction before averaging.

    Returns
    -------
    contrast : ndarray (nframes, nchannels)
        The contrast map (time along axis 0, wavelength along axis 1).
    mean_spec : ndarray (nframes, nchannels)
        Spatially-averaged flare spectra (post drift-correction).
    """
    cube = ha_cubes
    if correct_drift:
        from ..calib.wavelength import correct_spectral_drift

        cube, _ = correct_spectral_drift(cube, verbose=False)

    mean_spec = np.mean(cube, axis=(2, 3))  # (nframes, nch)

    if background is not None:
        mean_bg = np.mean(background, axis=(2, 3))
        mean_bg = np.where(mean_bg == 0, 1.0, mean_bg)
        contrast = (mean_spec - mean_bg) / mean_bg
        contrast = contrast - contrast[baseline_frame]
    else:
        ref = mean_spec[baseline_frame].copy()
        ref[ref == 0] = 1.0
        contrast = mean_spec / ref[np.newaxis, :] - 1.0

    return contrast, mean_spec
