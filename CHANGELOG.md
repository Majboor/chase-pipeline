# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [2.0.1] — 2026-08-02

### Changed
- Docs rewritten code-first around the PyPI package: README, `docs/SDK.md`
  (usage snippets with expected outputs for every stage), and
  `docs/CONTRIBUTING.md` (adds the PyPI release procedure). No code changes.

## [2.0.0] — 2026-08-02 (first PyPI release, as `chasepy`)

A ground-up rebuild into a modular, multi-front-end package.

### Added
- **PyPI packaging as `chasepy`** — `pip install chasepy` installs the `chase`
  module and the `chase` / `chase-tui` commands.
- **Resume / step skipping** — `Config.resume` (CLI `--resume`, TUI checkbox)
  reuses `out_dir/aligned_data.npz` from a previous run, skipping the slow
  load + align stages so individual output steps can be rerun without
  restarting the whole pipeline. `chase --only gif,contrast` (implies
  `--resume`) runs just the listed output steps: `gif`, `contrast`,
  `temperature`, `npz`, `fits`, `diagnostics`. The TUI gains the matching
  per-output checkboxes (GIF, npz, FITS, diagnostics) plus a resume toggle,
  so single steps can be rerun from any front-end.
- **`correct_drift` toggle** — the contrast stage's data-driven spectral drift
  correction (previously always on) is now `Config.correct_drift`
  (CLI `--no-drift`, TUI checkbox); together with `resample_wavelength = false`
  this turns wavelength correction off entirely.
- The contrast profile is now saved to `out_dir/contrast_profile.npy` (it
  previously existed only in memory / in the contrast GIF).
- **Modular package layout** (`chase.io`, `chase.calib`, `chase.analysis`,
  `chase.viz`) — every stage is an importable function; nothing is a forced
  pipeline.
- **Python SDK** re-exported from `chase` (`load_cube`, `load_flare_sequence`,
  `track_and_crop`, `optical_flow_align`, `resample_wavelength`,
  `correct_spectral_drift`, `contrast_profile`, `halpha_width_temperature`,
  `fe_planck_temperature`, `atlas_calibrate`, `make_animation`, …).
- **TUI** (`chase-tui`, Textual) — data source → FOV → toggles → live-progress run
  → results; a thin layer over the SDK, with graceful degradation.
- **Config-file workflow** (`chase --config file.toml`) and a `Config` dataclass.
- **Shift-then-crop** crop tracking with a padded buffer and a separate optional
  alignment patch.
- **Optical-flow stabilisation** (TV-L1 / Farneback) applied to float32 cubes.
- **Per-frame wavelength resampling** onto a common grid.
- **Contrast profile** `(flare−bg)/bg − frame0` with a quiet-Sun background patch.
- **Temperature maps**: Hα width→T (Molnar 2019); Fe I →T via Planck (Gaussian),
  Eddington–Barbier, and Voigt estimators; absolute calibration against the FTS
  atlas (ISPy, optional).
- **`satprocess` integration** (BSD-3): Hough-circle disk centring, spectral
  cross-correlation + roll, integral intensity scaling, CSV shift-cache.
- **pytest** suite on synthetic FITS fixtures + GitHub Actions CI.
- **Docs**: data-acquisition tutorial, SDK reference, TUI guide, calibration
  science, contributing guide; README with real embedded outputs.

### Changed
- Disk centring now defaults to a robust Hough-circle limb fit (centroid kept as
  a fallback).
- `chase.core` is now a thin backward-compatibility shim re-exporting the old
  flat API.
- `pyproject.toml` packaging with `chase` and `chase-tui` entry points and
  `tui` / `atlas` / `dev` extras.

### Removed
- The 90 MB `assets/tutorial.gif` is no longer referenced (replaced by the real
  X2.1 `flare.gif`).

## [1.x] — original `chase-pipeline`
- Resumable download, brightness-centroid disk centring, QS spectral/intensity
  calibration, single-file argparse CLI, tutorial notebook.
