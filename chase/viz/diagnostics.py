"""Per-step diagnostic figures (PNG) for the docs and the README.

Each plotter writes one PNG illustrating a single calibration stage, in the same
"one subpackage, one figure" spirit as the SpotiPy docs Alex asked us to follow.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import _compat  # noqa: F401

__all__ = [
    "plot_patch_overlay",
    "plot_qs_spectrum",
    "plot_shift_track",
    "plot_atlas_calibration",
]


def plot_patch_overlay(frame: np.ndarray, patch: Sequence[int], output_path: str,
                       align_patch: Optional[Sequence[int]] = None, title="Patch overlay") -> str:
    """Render frame 0 with the data (and optional align) patch box overlaid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.imshow(frame, origin="lower", cmap="hot")
    y0, y1, x0, x1 = patch
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec="cyan", fc="none", lw=2, label="data patch"))
    if align_patch is not None:
        ay0, ay1, ax0, ax1 = align_patch
        ax.add_patch(Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, ec="lime", fc="none", lw=2, ls="--", label="align patch"))
    ax.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_qs_spectrum(spectrum: np.ndarray, wavelengths: np.ndarray, output_path: str,
                     core_wavelength: Optional[float] = None, title="Quiet-Sun spectrum") -> str:
    """Plot a (quiet-Sun) spectral profile vs wavelength."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(wavelengths, spectrum, "k-")
    if core_wavelength is not None:
        ax.axvline(core_wavelength, color="r", ls="--", label="line core")
        ax.legend()
    ax.set_xlabel(r"Wavelength [$\rm\AA$]")
    ax.set_ylabel("Intensity [DN]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_shift_track(shifts, output_path: str, title="Crop-tracking shifts") -> str:
    """Plot per-frame integer (y, x) tracking shifts."""
    import matplotlib.pyplot as plt

    shifts = np.asarray(shifts)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(shifts[:, 0], "o-", label="shift y")
    ax.plot(shifts[:, 1], "s-", label="shift x")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Shift [px]")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_atlas_calibration(wavelengths_fe, disk_center_fe_cube, ifact, woff, output_path,
                           wing_continuum_frac: float = 0.85) -> str:
    """Overlay the calibrated CHASE Fe I profile on the FTS atlas (ISPy required)."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    from ISPy.spec import atlas as isp_atlas  # optional dep

    fe_avg = np.mean(disk_center_fe_cube, axis=(1, 2))
    fts = isp_atlas.atlas()
    wave_fts, spec_fts, cont_fts = fts.get(
        wavelengths_fe[0] - 0.5, wavelengths_fe[-1] + 0.5, cgs=True, perHz=True
    )
    f_atlas = interp1d(wave_fts, spec_fts, kind="cubic", bounds_error=False, fill_value="extrapolate")
    wing_mask = f_atlas(wavelengths_fe) > wing_continuum_frac * cont_fts[0]
    obs_cal = fe_avg * ifact
    wav_shifted = wavelengths_fe + woff

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(wave_fts, spec_fts, "k-", lw=1.5, alpha=0.7, label="FTS atlas")
    ax.plot(wav_shifted, obs_cal, "b-", lw=1.5, label="CHASE (calibrated)")
    ax.plot(wav_shifted[wing_mask], obs_cal[wing_mask], "ro", ms=6, label="wing points (fit)")
    ax.set_xlabel(r"Wavelength [$\rm\AA$]")
    ax.set_ylabel(r"$I_\nu$ [erg/(Hz s sr cm$^2$)]")
    ax.set_title(f"Fe I 6569 atlas calibration (ifact={ifact:.3e}, woff={woff:.4f} Å)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
