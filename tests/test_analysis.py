"""Analysis: contrast baseline + temperature maps on synthetic data."""

import numpy as np

from chase.analysis import contrast_profile, halpha_width_temperature, measure_halpha_width


def test_contrast_baseline_is_zero_at_frame0():
    # Construct an aligned cube whose frame 0 is the baseline.
    nframes, nch, H, W = 4, 12, 8, 8
    rng = np.random.default_rng(1)
    base = 1.0 - 0.6 * np.exp(-0.5 * ((np.arange(nch) - 6) / 2.0) ** 2)
    cube = np.empty((nframes, nch, H, W), dtype=np.float32)
    for i in range(nframes):
        cube[i] = (base[:, None, None] * (1 + 0.1 * i)) + 0.001 * rng.random((nch, H, W))
    contrast, _ = contrast_profile(cube, background=None, baseline_frame=0, correct_drift=False)
    assert np.allclose(contrast[0], 0.0, atol=1e-6)


def test_contrast_with_background_runs():
    nframes, nch, H, W = 3, 12, 8, 8
    base = 1.0 - 0.6 * np.exp(-0.5 * ((np.arange(nch) - 6) / 2.0) ** 2)
    cube = np.tile(base[None, :, None, None], (nframes, 1, H, W)).astype(np.float32)
    bg = cube * 0.9
    contrast, mean_spec = contrast_profile(cube, background=bg, baseline_frame=0, correct_drift=False)
    assert contrast.shape == (nframes, nch)
    assert np.allclose(contrast[0], 0.0, atol=1e-6)


def test_measure_halpha_width_positive():
    wav = np.linspace(6559.4, 6565.1, 60)
    spec = 1.0 - 0.6 * np.exp(-0.5 * ((wav - 6562.8) / 0.5) ** 2)
    width = measure_halpha_width(spec, wav, core_wavelength=6562.8)
    assert np.isfinite(width) and width > 0


def test_halpha_width_temperature_map(synthetic_dataset):
    from chase.io import load_flare_sequence

    seq = load_flare_sequence(synthetic_dataset["dir"], patch=synthetic_dataset["patch"],
                              verbose=False, pad=10)
    core_wav = seq["wavelength_ha"][seq["core_idx"]]
    temp, width = halpha_width_temperature(seq["ha_cubes"][0], seq["wavelength_ha"],
                                           core_wavelength=core_wav)
    assert temp.shape == seq["ha_cubes"][0].shape[1:]
    # at least some pixels yield a finite temperature
    assert np.isfinite(temp).any()
