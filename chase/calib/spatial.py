"""Spatial calibration: disk centring and crop-window tracking.

Two disk-centring strategies are provided:

* :func:`hough_disk_center` — robust limb detection via a Hough-circle fit on
  the Canny edge of the solar limb.  Ported from Finlay Davis's ``satprocess``
  (BSD-3-Clause, see ``NOTICE``); far more reliable than a brightness centroid
  when flares or limb darkening skew the intensity distribution.
* :func:`find_disk_center` — the original brightness-centroid fallback (no
  ``scikit-image`` circle search required).

:func:`track_and_crop` implements the **shift-then-crop** rule that is critical
to the flare pipeline: load a padded region, estimate the integer shift on a
flare-free reference channel, then crop the final patch from *inside* the
buffer.  Cropping first and shifting afterwards reintroduces the antisymmetric
pre-flare colour artefact Alex caught.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "derotate",
    "derotate_crop",
    "find_disk_center",
    "hough_disk_center",
    "phase_shift",
    "recenter",
    "track_and_crop",
]


# ---------------------------------------------------------------------------
# Derotation to solar north
# ---------------------------------------------------------------------------

def derotate(image: np.ndarray, angle_deg: float,
             center: Tuple[float, float]) -> np.ndarray:
    """Rotate an image about ``center`` to align it with solar north.

    CHASE Level-1 headers record ``INST_ROT``, the angle of solar north with
    respect to the detector y-axis (~11 deg for the 2023 data). Unlike
    instruments where this offset must be fitted against an external reference
    (e.g. THEMIS, Peat et al. 2026), CHASE provides it in every file, so the
    correction is a single rotation about the disc centre
    (``CRPIX1``, ``CRPIX2``).

    Parameters
    ----------
    image : ndarray (H, W)
        A single spatial frame.
    angle_deg : float
        Rotation angle in degrees, OpenCV convention (positive rotates the
        image content clockwise when displayed with ``origin='lower'``).
        Pass the header's ``INST_ROT`` to bring solar north to +y. Validated
        against a co-temporal SDO/HMI continuum image (2023-03-29 02:41 UT):
        sunspot position angles in the detector frame differ from the north-up
        HMI reference by ~``INST_ROT`` and agree to within ~2 deg after this
        rotation.
    center : (cx, cy)
        Rotation centre in zero-based pixel coordinates
        (``CRPIX1 - 1``, ``CRPIX2 - 1`` for FITS one-based headers).

    Returns
    -------
    ndarray (H, W) float32
        The derotated frame. Corners that rotate out of the field are filled
        with 0.
    """
    import cv2

    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    return cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def derotate_crop(image: np.ndarray, ys: int, ye: int, xs: int, xe: int,
                  angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    """Derotate a full frame and return only the window ``[ys:ye, xs:xe]``.

    The rotation and the crop are composed into a single ``warpAffine`` call,
    so only the requested window is ever computed --- rotating a whole CHASE
    channel (2313x2304) to keep a 140x170 patch would waste two orders of
    magnitude of work. The window coordinates refer to the *derotated* frame,
    i.e. the same frame :func:`derotate` would return.
    """
    import cv2

    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    M[0, 2] -= xs
    M[1, 2] -= ys
    return cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M, (xe - xs, ye - ys),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# ---------------------------------------------------------------------------
# Disk centring
# ---------------------------------------------------------------------------
def find_disk_center(image: np.ndarray, threshold_ratio: float = 0.2):
    """Brightness-centroid disk centre (original ``chase`` method).

    Parameters
    ----------
    image : ndarray (H, W)
    threshold_ratio : float, optional
        Pixels brighter than ``threshold_ratio * max`` are counted as disk.

    Returns
    -------
    (cx, cy, mask) : int, int, ndarray
    """
    norm = image / np.max(image)
    mask = norm > threshold_ratio
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No disk detected.")
    return int(xs.mean()), int(ys.mean()), mask


# -- BSD 3-Clause License -----------------------------------------------------
# Copyright (c) 2025, Finlay Davis.  All rights reserved.
# Ported from FinlayDavis/satprocess (preprocess_image + detect_circles).
# -----------------------------------------------------------------------------
def hough_disk_center(
    image: np.ndarray,
    sigma: float = 2.0,
    reference_radius: Optional[float] = None,
    margin_percent: float = 2.0,
    min_radius: int = 500,
    max_radius: int = 1500,
) -> Tuple[float, float, float]:
    """Detect the solar disk centre via a Hough-circle limb fit.

    The image is Gaussian-blurred, Otsu-thresholded, and Canny-edged to isolate
    the limb ring; a Hough-circle transform then finds the best circle.  When a
    ``reference_radius`` is supplied (from a previously fitted frame), the radius
    search is constrained to ``reference_radius ± margin_percent`` for speed and
    stability.

    Parameters
    ----------
    image : ndarray (H, W)
    sigma : float, optional
        Gaussian blur sigma before thresholding.
    reference_radius : float or None, optional
        If given, restrict the radius search around this value.
    margin_percent : float, optional
        Half-width of the constrained radius band, in percent.
    min_radius, max_radius : int, optional
        Wide radius search bounds when ``reference_radius`` is None.

    Returns
    -------
    (cx, cy, radius) : float, float, float
    """
    from skimage.feature import canny
    from skimage.filters import gaussian, threshold_otsu
    from skimage.transform import hough_circle, hough_circle_peaks

    img = image.astype(np.float64)
    blurred = gaussian(img, sigma=sigma, preserve_range=True)
    try:
        thresh = threshold_otsu(blurred)
    except ValueError:
        thresh = blurred.mean()
    binary = blurred > thresh
    edges = canny(binary.astype(float), sigma=1)

    if reference_radius is not None:
        margin = reference_radius * margin_percent / 100.0
        lo = max(1, int(reference_radius - margin))
        hi = int(reference_radius + margin) + 1
    else:
        lo, hi = min_radius, max_radius
    radii = np.arange(lo, hi, 1)
    if len(radii) == 0:
        raise ValueError("Empty radius search range for Hough circle.")

    hough = hough_circle(edges, radii)
    _, cx, cy, rad = hough_circle_peaks(hough, radii, total_num_peaks=1)
    if len(cx) == 0:
        raise ValueError("No circle detected on the limb edge map.")
    return float(cx[0]), float(cy[0]), float(rad[0])


def recenter(cube: np.ndarray, shift_xy: Tuple[int, int]) -> np.ndarray:
    """Integer-roll every channel of a cube by ``(dx, dy)``."""
    dx, dy = shift_xy
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = np.roll(np.roll(cube[i], dy, axis=0), dx, axis=1)
    return out


# ---------------------------------------------------------------------------
# Crop-window tracking (shift-then-crop)
# ---------------------------------------------------------------------------
def phase_shift(reference: np.ndarray, target: np.ndarray, upsample_factor: int = 100):
    """Sub-pixel shift that best aligns ``target`` to ``reference``.

    Thin wrapper over :func:`skimage.registration.phase_cross_correlation`.

    Returns
    -------
    (sy, sx) : float, float
        Shift in pixels along (row, col).
    """
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(
        reference, target, upsample_factor=upsample_factor
    )
    return float(shift[0]), float(shift[1])


def track_and_crop(
    frames: Sequence[np.ndarray],
    patch: Sequence[int],
    ref_channel: int = -1,
    pad: int = 20,
    track: bool = True,
    align_patch: Optional[Sequence[int]] = None,
    reference_index: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Track a moving feature across frames and crop a stable patch.

    Implements **shift-then-crop**: for each frame a padded region around
    ``patch`` is taken, the integer shift relative to the reference frame is
    estimated on ``ref_channel`` (a flare-free channel — e.g. the Fe I or Hα
    continuum), and the final patch is cropped from inside the padded buffer at
    the shifted offset.  Every frame therefore covers identical physical
    content, which removes the spectral-averaging artefact that crop-then-shift
    produces.

    Parameters
    ----------
    frames : sequence of ndarray (nchannels, H, W)
        Per-frame spectral cubes (full FOV, or a region large enough to contain
        ``patch`` plus ``pad``).
    patch : [y0, y1, x0, x1]
        The data window to extract.
    ref_channel : int, optional
        Channel index used to estimate the shift (default ``-1`` = last channel,
        which is the flare-free continuum / photosphere).
    pad : int, optional
        Padding in pixels added on each side before shifting (default 20; must
        exceed the largest expected inter-frame shift).
    track : bool, optional
        If False, use a fixed window (zero shift) — useful when optical flow
        will do all the alignment.
    align_patch : [y0, y1, x0, x1] or None, optional
        A *separate, usually smaller* window used only for shift estimation
        (Alex's idea: small box for the shift, large box for the saved data,
        reducing wobble).  Defaults to ``patch``.
    reference_index : int, optional
        Frame whose patch defines the alignment reference.

    Returns
    -------
    cropped : ndarray (nframes, nchannels, h, w)
    shifts : list of (sy, sx) integer shifts applied per frame.
    """
    y0, y1, x0, x1 = patch
    H, W = y1 - y0, x1 - x0
    ay0, ay1, ax0, ax1 = align_patch if align_patch is not None else patch

    full_H, full_W = frames[reference_index].shape[1], frames[reference_index].shape[2]
    py0, py1 = max(0, y0 - pad), min(full_H, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(full_W, x1 + pad)
    oy, ox = y0 - py0, x0 - px0

    ref_crop = frames[reference_index][ref_channel, ay0:ay1, ax0:ax1]

    out = np.empty((len(frames), frames[0].shape[0], H, W), dtype=np.float32)
    shifts: List[Tuple[int, int]] = []
    for i, cube in enumerate(frames):
        if track:
            cur = cube[ref_channel, ay0:ay1, ax0:ax1]
            sy, sx = phase_shift(ref_crop, cur)
            sy, sx = int(round(sy)), int(round(sx))
        else:
            sy, sx = 0, 0
        shifts.append((sy, sx))
        cy, cx = oy - sy, ox - sx
        buf = cube[:, py0:py1, px0:px1]
        out[i] = buf[:, cy:cy + H, cx:cx + W].astype(np.float32)
    return out, shifts
