"""Animations: 2-panel flare GIF, contrast profile, temperature maps.

All writers expose a ``freeze_clim`` option.  When ``True`` the colour limits are
fixed across all frames (honest brightness comparison); when ``False`` each frame
is auto-scaled independently (better for seeing faint structure).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import _compat  # noqa: F401  (headless matplotlib)

__all__ = ["make_animation", "make_contrast_animation", "make_temperature_animation"]


PLATE_SCALE = 1.04  # arcsec / pixel (CHASE/HIS, Qiu et al. 2022)


def _extent(patch, plate_scale=PLATE_SCALE):
    """Axis extent in arcsec: pixel coordinates scaled by the plate scale."""
    patch = np.asarray(patch)
    if patch.size == 4:
        y0, y1, x0, x1 = [float(v) for v in patch]
        s = plate_scale
        return [x0 * s, x1 * s, y0 * s, y1 * s]
    return None


def make_animation(
    cont: np.ndarray,
    core: np.ndarray,
    times: Optional[Sequence] = None,
    patch=None,
    output_path: str = "animation.gif",
    fps: int = 4,
    cmap: str = "hot",
    freeze_clim: bool = False,
    clip_percentile: float = 99.5,
    titles=("Photosphere", "Chromosphere"),
) -> str:
    """2-panel Photosphere | Chromosphere GIF from aligned summary frames.

    Parameters
    ----------
    cont : ndarray (nframes, H, W)
        Continuum / photosphere panel (e.g. Hα last channel).
    core : ndarray (nframes, H, W)
        Hα core / chromosphere panel.
    times : sequence or None, optional
        Per-frame timestamp strings for the title.
    patch : array-like or None, optional
        ``[y0, y1, x0, x1]`` used to label axes in pixel coordinates.
    output_path : str, optional
    fps : int, optional
    cmap : str, optional
    freeze_clim : bool, optional
        Fix colour limits across frames (default False = per-frame autoscale).
    clip_percentile : float, optional
        Upper percentile used for the colour limit instead of the absolute
        maximum (default 99.5). A bright flare kernel or a cosmic-ray spike on a
        single frame contains a handful of pixels far brighter than everything
        else; scaling to the raw ``max`` then crushes the rest of that frame to a
        dark, "broken"-looking block. Clipping ``vmax`` to a high percentile keeps
        every frame's structure visible without special-casing any frame. Set to
        ``100`` to recover the old absolute-max behaviour.

    Returns
    -------
    str
        ``output_path``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    nframes = cont.shape[0]
    extent = _extent(patch)
    times = list(times) if times is not None else [f"Frame {i}" for i in range(nframes)]

    def _hi(arr):
        return np.nanpercentile(arr, clip_percentile) if clip_percentile < 100 else np.nanmax(arr)

    if freeze_clim:
        c_lo, c_hi = np.nanmin(cont), _hi(cont)
        k_lo, k_hi = np.nanmin(core), _hi(core)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), facecolor="white")
    im1 = ax1.imshow(cont[0], cmap=cmap, origin="lower", extent=extent,
                     interpolation="nearest")
    im2 = ax2.imshow(core[0], cmap=cmap, origin="lower", extent=extent,
                     interpolation="nearest")
    unit = "arcsec" if extent is not None else "px"
    for ax, t in zip((ax1, ax2), titles):
        ax.set_title(t)
        ax.set_xlabel(f"Solar X [{unit}]")
        ax.set_ylabel(f"Solar Y [{unit}]")
    suptitle = fig.suptitle(f"{times[0]}  —  Frame 0", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(i):
        im1.set_data(cont[i])
        im2.set_data(core[i])
        if freeze_clim:
            im1.set_clim(c_lo, c_hi)
            im2.set_clim(k_lo, k_hi)
        else:
            im1.set_clim(np.nanmin(cont[i]), _hi(cont[i]))
            im2.set_clim(np.nanmin(core[i]), _hi(core[i]))
        suptitle.set_text(f"{times[i]}  —  Frame {i}")
        return im1, im2, suptitle

    ani = FuncAnimation(fig, update, frames=nframes, interval=int(1000 / fps), blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)
    return output_path


def make_contrast_animation(
    ha_cubes: np.ndarray,
    wavelengths: np.ndarray,
    contrast: np.ndarray,
    core_idx: int,
    times: Optional[Sequence] = None,
    patch=None,
    output_path: str = "contrast_profile.gif",
    fps: int = 3,
    vmin: float = -0.15,
    vmax: float = 0.15,
    flarestart: Optional[int] = None,
    flarepeak: Optional[int] = None,
    flareclass: str = "",
) -> str:
    """3-panel Chromosphere | Photosphere | contrast(λ, t) GIF with a time cursor."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    nframes = ha_cubes.shape[0]
    extent = _extent(patch)
    times = list(times) if times is not None else [f"Frame {i}" for i in range(nframes)]
    extent_c = [wavelengths[0], wavelengths[-1], 0, nframes]

    fig = plt.figure(figsize=(14, 5), facecolor="white")
    ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
    im1 = ax1.imshow(ha_cubes[0, core_idx], origin="lower", cmap="hot", extent=extent)
    im2 = ax2.imshow(ha_cubes[0, -1], origin="lower", cmap="hot", extent=extent)
    ax1.set_title("Chromosphere")
    ax2.set_title("Photosphere")
    ax3.imshow(contrast, aspect="auto", origin="lower", cmap="PiYG",
               extent=extent_c, vmin=vmin, vmax=vmax, interpolation="none")
    if flarestart is not None:
        ax3.axhline(flarestart, color="black", lw=0.8)
    if flarepeak is not None:
        ax3.axhline(flarepeak, color="black", lw=0.8, ls="dashed")
    ax3.set_xlabel(r"Wavelength [$\rm\AA$]")
    ax3.set_ylabel("Time [frame]")
    ax3.set_ylim(0, nframes)
    ax3.set_title(flareclass)
    cursor = ax3.axhline(0, color="red", lw=2)
    suptitle = fig.suptitle(f"{times[0]}  —  Frame 0", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(i):
        im1.set_data(ha_cubes[i, core_idx])
        im1.set_clim(np.nanmin(ha_cubes[i, core_idx]), np.nanmax(ha_cubes[i, core_idx]))
        im2.set_data(ha_cubes[i, -1])
        im2.set_clim(np.nanmin(ha_cubes[i, -1]), np.nanmax(ha_cubes[i, -1]))
        cursor.set_ydata([i, i])
        suptitle.set_text(f"{times[i]}  —  Frame {i}")
        return im1, im2, cursor, suptitle

    ani = FuncAnimation(fig, update, frames=nframes, interval=400, blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)
    return output_path


def make_temperature_animation(
    ha_temp: np.ndarray,
    fe_temp: Optional[np.ndarray],
    core_imgs: np.ndarray,
    times: Optional[Sequence] = None,
    patch=None,
    output_path: str = "temperature.gif",
    fps: int = 3,
    ha_clim=(5000, 15000),
    fe_clim=(4000, 6000),
) -> str:
    """Temperature-map GIF: Hα chromosphere [+ Fe I photosphere] + Hα core ref."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    has_fe = fe_temp is not None
    ncols = 3 if has_fe else 2
    extent = _extent(patch)
    nframes = ha_temp.shape[0]
    times = list(times) if times is not None else [f"Frame {i}" for i in range(nframes)]

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), facecolor="white")
    im_ha = axes[0].imshow(ha_temp[0], origin="lower", cmap="inferno", extent=extent,
                           vmin=ha_clim[0], vmax=ha_clim[1])
    axes[0].set_title("Chromosphere (Hα width)")
    plt.colorbar(im_ha, ax=axes[0], shrink=0.8, label="T [K]")
    if has_fe:
        im_fe = axes[1].imshow(fe_temp[0], origin="lower", cmap="inferno", extent=extent,
                               vmin=fe_clim[0], vmax=fe_clim[1])
        axes[1].set_title("Photosphere (Fe I Planck)")
        plt.colorbar(im_fe, ax=axes[1], shrink=0.8, label="T [K]")
        core_ax = axes[2]
    else:
        core_ax = axes[1]
    im_core = core_ax.imshow(core_imgs[0], origin="lower", cmap="hot", extent=extent)
    core_ax.set_title("Hα Core")
    suptitle = fig.suptitle(f"{times[0]}  —  Frame 0", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(i):
        im_ha.set_data(ha_temp[i])
        im_core.set_data(core_imgs[i])
        im_core.set_clim(np.nanmin(core_imgs[i]), np.nanmax(core_imgs[i]))
        artists = [im_ha, im_core, suptitle]
        if has_fe:
            im_fe.set_data(fe_temp[i])
            artists.append(im_fe)
        suptitle.set_text(f"{times[i]}  —  Frame {i}")
        return artists

    ani = FuncAnimation(fig, update, frames=nframes, interval=400, blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)
    return output_path
