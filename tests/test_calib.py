"""Calibration physics: tracking, disk centring, resampling, Planck, Molnar."""

import numpy as np

from chase.calib import (
    find_disk_center,
    hough_disk_center,
    optical_flow_align,
    phase_shift,
    planck_inversion,
    resample_wavelength,
    track_and_crop,
)
from chase.analysis import width_to_temperature

# CGS constants matching chase.calib.intensity
_H, _C, _K = 6.62607015e-27, 2.99792458e10, 1.380649e-16


def _load_full_frames(synthetic_dataset):
    import astropy.io.fits as fits

    from chase.io import discover_fits

    ha, _ = discover_fits(synthetic_dataset["dir"])
    return [fits.open(p)[1].data.astype(np.float32) for p in ha]


def test_phase_shift_recovers_known_shift():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    shifted = np.roll(np.roll(img, 3, axis=0), -2, axis=1)
    sy, sx = phase_shift(img, shifted)
    assert round(sy) == -3 and round(sx) == 2


def test_track_and_crop_aligns_feature(synthetic_dataset):
    frames = _load_full_frames(synthetic_dataset)
    cropped, shifts = track_and_crop(frames, synthetic_dataset["patch"], ref_channel=-1, pad=10)
    assert cropped.shape[0] == len(frames)
    # After shift-then-crop every frame should match frame 0 (feature aligned)
    ref = cropped[0, -1]
    for i in range(1, len(frames)):
        diff = np.abs(cropped[i, -1] - ref).mean() / (ref.mean() + 1e-9)
        assert diff < 0.05, f"frame {i} not aligned: rel diff {diff:.3f}"
    # tracking found non-zero shifts for the shifted frames
    assert any(s != (0, 0) for s in shifts[1:])


def test_track_and_crop_fixed_window_when_untracked(synthetic_dataset):
    frames = _load_full_frames(synthetic_dataset)
    cropped, shifts = track_and_crop(frames, synthetic_dataset["patch"], track=False, pad=10)
    assert all(s == (0, 0) for s in shifts)


def test_hough_disk_center_on_synthetic_disk():
    from skimage.draw import disk

    img = np.zeros((128, 128), dtype=np.float32)
    rr, cc = disk((60, 70), 30, shape=img.shape)
    img[rr, cc] = 1.0
    cx, cy, rad = hough_disk_center(img, min_radius=20, max_radius=40)
    assert abs(cx - 70) <= 3 and abs(cy - 60) <= 3
    assert abs(rad - 30) <= 4


def test_find_disk_center_centroid():
    img = np.zeros((64, 64), dtype=np.float32)
    img[40, 25] = 10.0
    cx, cy, _ = find_disk_center(img, threshold_ratio=0.5)
    assert cx == 25 and cy == 40


def test_resample_wavelength_maps_shifted_grid_back():
    # A line at 6562.8 sampled on a drifted grid, resampled onto the base grid,
    # should put the minimum back at the same channel as the base grid.
    base = np.linspace(6559.4, 6565.1, 30)
    drift = base + 0.05
    core = 6562.8
    spec = 1.0 - 0.6 * np.exp(-0.5 * ((drift - core) / 0.4) ** 2)
    cube = np.tile(spec[:, None, None], (1, 4, 4)).astype(np.float32)
    out = resample_wavelength(cube, drift, base)
    base_spec = 1.0 - 0.6 * np.exp(-0.5 * ((base - core) / 0.4) ** 2)
    assert np.argmin(out[:, 0, 0]) == np.argmin(base_spec)


def test_optical_flow_align_runs(synthetic_dataset):
    from chase.io import load_flare_sequence

    seq = load_flare_sequence(synthetic_dataset["dir"], patch=synthetic_dataset["patch"],
                              verbose=False, pad=10)
    aligned = optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"], method="tvl1")
    assert aligned.shape == seq["ha_cubes"].shape
    assert np.isfinite(aligned).all()


def test_planck_inversion_recovers_6000K():
    lam = 6569e-8
    nu = _C / lam
    coeff = 2 * _H * nu ** 3 / _C ** 2
    I_nu = coeff / (np.exp(_H * nu / (_K * 6000.0)) - 1.0)
    T = planck_inversion(I_nu, wavelength_ang=6569.0)
    assert abs(T - 6000.0) < 1.0


def test_molnar_width_to_temperature_monotonic():
    widths = np.linspace(0.6, 2.0, 20)
    temps = width_to_temperature(widths)
    assert np.all(np.diff(temps) > 0)
    # ~10^4 K for a ~1.2 Å width (quiet chromosphere ballpark)
    assert 8000 < width_to_temperature(1.2) < 12000


def test_derotate_convention_and_roundtrip():
    """Pin the rotation convention (+angle = clockwise in origin='lower'
    display: east -> south for +90 deg) and check the operation inverts."""
    import numpy as np

    from chase.calib.spatial import derotate

    img = np.zeros((64, 64), np.float32)
    img[32, 50] = 100.0                      # east of centre
    out = derotate(img, 90.0, (31.5, 31.5))
    assert np.unravel_index(np.argmax(out), out.shape) == (13, 32)

    back = derotate(out, -90.0, (31.5, 31.5))
    assert np.unravel_index(np.argmax(back), back.shape) == (32, 50)
    assert back.max() > 90.0                 # interpolation loss stays small


def test_derotate_crop_matches_full_rotation():
    """The composed rotate+crop must equal cropping the fully rotated frame."""
    import numpy as np

    from chase.calib.spatial import derotate, derotate_crop

    rng = np.random.default_rng(7)
    img = rng.normal(1000.0, 50.0, (120, 140)).astype(np.float32)
    ang, ctr = 11.0, (69.5, 59.5)
    full = derotate(img, ang, ctr)[40:90, 30:100]
    win = derotate_crop(img, 40, 90, 30, 100, ang, ctr)
    assert win.shape == (50, 70)
    # cv2 quantises interpolation coefficients (fixed-point), so folding the
    # crop translation into the matrix changes results at the ~0.1% level;
    # the geometry itself is identical.
    d = np.abs(win - full)
    assert d.mean() < 0.1 and np.percentile(d, 99) < 2.5


def test_derotate_zero_angle_is_noop():
    import numpy as np

    from chase.calib.spatial import derotate_crop

    img = np.arange(48 * 48, dtype=np.float32).reshape(48, 48)
    win = derotate_crop(img, 10, 30, 5, 40, 0.0, (23.5, 23.5))
    assert np.allclose(win, img[10:30, 5:40], atol=1e-3)
