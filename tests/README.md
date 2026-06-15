# Tests

Fast, hermetic unit tests — they synthesise tiny CHASE-like FITS cubes
(`tests/conftest.py`) and never touch the real multi-hundred-MB data.

```bash
pip install -e '.[dev]'
pytest -q
```

## Coverage

| Area | File | What it checks |
|------|------|----------------|
| Discovery / IO | `test_io.py` | dir/`.txt` discovery, `load_cube` shapes + header wavelength grid, region read, disk-center patch |
| Spatial | `test_calib.py` | `phase_shift` recovers a known shift; **shift-then-crop** aligns the moving feature; Hough + centroid disk centre |
| Wavelength | `test_calib.py` | resampling maps a drifted grid back onto the base grid |
| Optical flow | `test_calib.py` | TV-L1/Farneback align runs and stays finite |
| Temperature | `test_calib.py`, `test_analysis.py` | Planck inversion → ~6000 K; Molnar width→T monotonic; Hα width map |
| Contrast | `test_analysis.py` | baseline ≈ 0 at frame 0, background form runs |
| Orchestrator | `test_pipeline.py` | end-to-end run (npz + gif + contrast + temperature); TOML config round-trip |

`ISPy` (FTS-atlas absolute calibration) is optional; the temperature tests use
the Hα-width path or the `ifact=1` fallback so they pass without it.
