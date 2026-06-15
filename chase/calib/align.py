"""Fine alignment via dense optical flow.

Integer crop-tracking (:func:`chase.calib.spatial.track_and_crop`) removes whole
pixel jitter; residual sub-pixel wobble is removed here with a dense optical-flow
warp.

The flow field is computed **once per frame** from a single flare-free reference
channel converted to ``uint8`` (TV-L1 if OpenCV's ``optflow`` module is present,
else Farneback).  The resulting displacement map is then applied with
``cv2.remap`` (cubic) to **every** wavelength channel of the float32 science
cubes — the science data never leaves float precision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["create_optical_flow_solver", "calculate_optical_flow", "warp_cube", "optical_flow_align"]


def create_optical_flow_solver():
    """Return a TV-L1 optical-flow solver, or ``None`` to signal Farneback."""
    import cv2

    try:
        return cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        try:
            return cv2.DualTVL1OpticalFlow_create()
        except AttributeError:
            return None


def calculate_optical_flow(ref_u8: np.ndarray, cur_u8: np.ndarray, solver=None) -> np.ndarray:
    """Compute the dense flow field mapping ``cur`` onto ``ref`` (both uint8)."""
    import cv2

    if solver is not None:
        return solver.calc(ref_u8, cur_u8, None)
    return cv2.calcOpticalFlowFarneback(
        ref_u8, cur_u8, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _to_u8(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def warp_cube(cube: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Apply a precomputed remap to every channel of a cube (cubic, reflect)."""
    import cv2

    out = np.empty_like(cube)
    for ch in range(cube.shape[0]):
        out[ch] = cv2.remap(
            cube[ch], map_x, map_y,
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
        )
    return out


def optical_flow_align(
    cubes: np.ndarray,
    reference: Optional[np.ndarray] = None,
    ref_idx: int = 0,
    method: str = "tvl1",
    extra_cubes: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """Stabilise a frame sequence with optical-flow warping.

    Parameters
    ----------
    cubes : ndarray (nframes, nchannels, H, W)
        Science cubes to warp (e.g. the Hα stack).  Stays float32.
    reference : ndarray (nframes, H, W) or None, optional
        Per-frame flare-free image used to *compute* the flow.  If None, the last
        channel of ``cubes`` is used.  Pass the Fe I last wavelength here to
        avoid stabilising the eruption away.
    ref_idx : int, optional
        Index of the reference frame everything is aligned to.
    method : {"tvl1", "farneback"}, optional
        Optical-flow algorithm (TV-L1 falls back to Farneback if unavailable).
    extra_cubes : ndarray (nframes, nchannels2, H, W) or None, optional
        A second stack (e.g. Fe I) warped with the *same* per-frame map.
    verbose : bool, optional

    Returns
    -------
    aligned : ndarray, same shape as ``cubes``
    aligned_extra : ndarray or None
        Present only when ``extra_cubes`` is given.
    """
    import cv2  # noqa: F401  (ensure available early)

    nframes, nch, H, W = cubes.shape
    if reference is None:
        reference = cubes[:, -1, :, :]

    solver = create_optical_flow_solver() if method == "tvl1" else None
    ref_u8 = _to_u8(reference[ref_idx])

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    aligned = np.empty_like(cubes)
    aligned_extra = np.empty_like(extra_cubes) if extra_cubes is not None else None

    for i in range(nframes):
        cur_u8 = _to_u8(reference[i])
        flow = calculate_optical_flow(ref_u8, cur_u8, solver)
        map_x = (X + flow[..., 0]).astype(np.float32)
        map_y = (Y + flow[..., 1]).astype(np.float32)
        aligned[i] = warp_cube(cubes[i], map_x, map_y)
        if aligned_extra is not None:
            aligned_extra[i] = warp_cube(extra_cubes[i], map_x, map_y)
        if verbose:
            print(f"  optical-flow frame {i}/{nframes - 1} aligned")

    if aligned_extra is not None:
        return aligned, aligned_extra
    return aligned
