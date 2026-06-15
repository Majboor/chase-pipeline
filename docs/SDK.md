# Python SDK reference

Everything is importable from the top-level `chase` namespace, and every function
has a NumPy-style docstring (so `help(chase.track_and_crop)` is authoritative).
This page is the map.

```python
import chase
```

## I/O — `chase.io`

| Function | Signature (abridged) | Returns |
|----------|----------------------|---------|
| `download` / `download_file` | `(url, save_dir, max_retries=5)` | local path (resumable) |
| `resolve_inputs` | `(input_path, save_dir="./data")` | list of local FITS paths (dir / `.txt` / URL / file) |
| `discover_fits` | `(directory)` | `(ha_files, fe_files)` |
| `load_cube` | `(filepath, region=None)` | `(cube[nλ,H,W], header, wavelengths)` |
| `load_flare_sequence` | `(fits_dir, patch=None, align_patch=None, track=True, resample=True)` | dict of stacks + metadata |
| `extract_disk_center` | `(filepath, size=100)` | `(cube, wavelengths, (cy,cx))` — CRPIX patch |
| `save_cube_fits` / `save_npz` | `(…, path)` | path |

`load_flare_sequence` returns a dict with `ha_cubes` `(nframes, nλ_ha, H, W)`,
`fe_cubes` (or `None`), `cont_raw`, `core_raw`, `align_ref`, `ha_bg`,
`wavelength_ha`, `wavelength_fe`, `times`, `core_idx`, `patch`, …

## Calibration — `chase.calib`

### Spatial (`chase.calib.spatial`)
- `find_disk_center(image, threshold_ratio=0.2)` → `(cx, cy, mask)` — brightness centroid.
- `hough_disk_center(image, reference_radius=None, …)` → `(cx, cy, radius)` — robust limb fit *(satprocess, BSD-3)*.
- `phase_shift(reference, target, upsample_factor=100)` → `(sy, sx)`.
- `recenter(cube, (dx, dy))` → cube.
- `track_and_crop(frames, patch, ref_channel=-1, pad=20, track=True, align_patch=None)` → `(cropped, shifts)` — **shift-then-crop**.

### Alignment (`chase.calib.align`)
- `optical_flow_align(cubes, reference=None, method="tvl1", extra_cubes=None)` → aligned cube(s).
- `warp_cube(cube, map_x, map_y)`, `create_optical_flow_solver()`.

### Wavelength (`chase.calib.wavelength`)
- `resample_wavelength(cube, src_wav, dst_wav, kind="cubic")` → cube on `dst_wav`.
- `correct_spectral_drift(ha_cubes)` → `(corrected, shifts)`.
- `cross_correlation_shift(ref, target)`, `align_spectrum(ref_w, ref_i, tgt_w, tgt_i)` *(satprocess)*.

### Intensity (`chase.calib.intensity`)
- `normalize_intensity(cube, ref_cube, clip=(0,2))`.
- `integral_scaling(ref_spec, target_spec)` → factor *(satprocess)*.
- `atlas_calibrate(wavelengths, disk_center_cube)` → `(ifact, woff)` — absolute, **ISPy required**.
- `planck_inversion(I_nu, wavelength_ang=6569.0)` → T [K].

## Analysis — `chase.analysis`

- `contrast_profile(ha_cubes, background=None, baseline_frame=0, correct_drift=True)` → `(contrast, mean_spec)`.
- `halpha_width_temperature(ha_cube, wavelengths, core_wavelength=6562.8)` → `(T_map, width_map)` *(Molnar 2019)*.
- `measure_halpha_width`, `halpha_width_map`, `width_to_temperature`.
- `fe_planck_temperature(fe_cube, wavelengths, ifact)` → `(T_map, core_intensity)` *(Gaussian core)*.
- `fe_eb_temperature(...)` → `(T_cube, T_core, T_continuum)` *(Eddington–Barbier)*.
- `fe_voigt_temperature(...)` → `(T_map, core, center)` *(Voigt; robust to under-sampling)*.

## Visualisation — `chase.viz`

- `make_animation(cont, core, times=None, patch=None, output_path=…, freeze_clim=False)`.
- `make_contrast_animation(ha_cubes, wavelengths, contrast, core_idx, …)`.
- `make_temperature_animation(ha_temp, fe_temp, core_imgs, …)`.
- `chase.viz.diagnostics`: `plot_patch_overlay`, `plot_qs_spectrum`, `plot_shift_track`, `plot_atlas_calibration`.

## Orchestration

- `Config(...)` — dataclass of every option (see `examples/config.toml`).
- `load_config(path)` — read a TOML into a `Config`.
- `run_pipeline(config, progress=None)` — run the configured stages; returns a dict
  of in-memory products and output paths. `progress(stage, message)` is an optional
  callback for live UIs.

## Worked example

```python
import chase, numpy as np

seq = chase.load_flare_sequence("data/.../fits", patch=[940,1080,1870,2040],
                                align_patch=[980,1040,1920,1990])   # small align box
aligned, fe_aligned = chase.optical_flow_align(
    seq["ha_cubes"], reference=seq["align_ref"], extra_cubes=seq["fe_cubes"])

# absolute Fe I photospheric temperature
dc_cube, dc_wav, _ = chase.extract_disk_center("data/.../RSM...FE.fits")
ifact, woff = chase.atlas_calibrate(dc_wav, dc_cube)
T_phot, _ = chase.fe_planck_temperature(fe_aligned[24], seq["wavelength_fe"], ifact)
print(np.nanmedian(T_phot), "K")   # ~5100 K quiet photosphere
```

## Backward compatibility

The old flat API still works: `from chase.core import run_pipeline, load_fits_data,
find_disk_center, …`. New code should prefer the submodule / top-level imports above.
