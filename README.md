# chase-pipeline

Calibration and flare analysis for CHASE/HIS solar spectroscopy. One pip install gets you a Python SDK, a CLI, a TUI, and a config-file workflow, all driving the same functions.

[![tests](https://github.com/Majboor/chase-pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/Majboor/chase-pipeline/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/chasepy)](https://pypi.org/project/chasepy/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/flare.gif" width="760" alt="Stabilised 2-panel animation of the 2023-03-29 X2.1 flare"><br>
  <em>The 2023-03-29 X2.1 flare, stabilised by this pipeline. Photosphere (Hα continuum) left, chromosphere (Hα core) right. Rendered with <code>--flow-method farneback</code>; see <a href="docs/FLARE_STABILIZATION.md">docs/FLARE_STABILIZATION.md</a> for why not TV-L1.</em>
</p>

## Install

```bash
pip install chasepy              # installs the `chase` module + `chase` and `chase-tui` commands
pip install 'chasepy[tui]'       # + Textual TUI
pip install 'chasepy[atlas]'     # + ISPy, for absolute Fe I calibration
pip install 'chasepy[all]'       # everything incl. test deps
```

## 60 seconds to a result

One command. Point at a folder of `RSM*_HA.fits` (+ optional `*_FE.fits`) cubes:

```bash
chase /data/20230329_X12/fits --patch 940 1080 1870 2040 --contrast --temperature double
```

You get, in `./chase_out/`:

| File | What it is |
|---|---|
| `aligned_data.npz` | raw + aligned cubes, wavelengths, times. Also the resume checkpoint |
| `flare.gif` | photosphere / chromosphere animation |
| `contrast_profile.gif`, `contrast_profile.npy` | wavelength-vs-time flare contrast |
| `ha_temp_maps.npy`, `fe_temp_maps.npy`, `temperature.gif` | temperature maps |
| `diag_*.png` | patch overlay, quiet-Sun spectrum |

Same thing in Python:

```python
from chase import Config, run_pipeline

results = run_pipeline(Config(fits_dir="/data/20230329_X12/fits",
                              patch=[940, 1080, 1870, 2040],
                              contrast=True, temperature="double"))
results["aligned"]["ha"].shape   # (25, 118, 140, 170)  float32, stabilised
results["contrast"].shape        # (25, 118)
results["gif"]                   # './chase_out/flare.gif'
```

## Rerun one step, skip the slow part

Every run checkpoints the aligned cubes. Rerunning an output takes seconds, not the minutes load + align costs:

```bash
chase /data --out ./chase_out --only gif                        # re-render the animation
chase /data --out ./chase_out --only contrast,gif --no-drift    # contrast without drift correction
chase /data --out ./chase_out --only temperature --temperature fe_voigt
chase /data --out ./chase_out --resume --contrast               # resume + your usual flags
```

Steps for `--only`: `gif`, `contrast`, `temperature`, `npz`, `fits`, `diagnostics`. The checkpoint stores the cubes as cropped and aligned, so changing the patch or the resampling flag needs one fresh run without `--resume`.

```python
run_pipeline(Config(fits_dir="/data", out_dir="./chase_out",
                    resume=True, temperature="fe_voigt"))
```

## Use single functions

Nothing forces the full pipeline. Each stage is a plain function:

```python
import chase

# One cube
cube, header, wav = chase.load_cube("RSM20230329T..._HA.fits")
cube.shape                       # (118, H_full, W_full)  full disk, float32

# A tracked, wavelength-resampled sequence
seq = chase.load_flare_sequence("/data/fits", patch=[940, 1080, 1870, 2040])
seq["ha_cubes"].shape            # (25, 118, 140, 170)
seq["wavelength_ha"][seq["core_idx"]]   # ~6562.8  (Å, the Hα core channel)

# Stabilise. FE cubes get the same warp as HA
ha_al, fe_al = chase.optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"],
                                        method="farneback", extra_cubes=seq["fe_cubes"])

# Contrast profile: (flare - bg)/bg - frame0
contrast, spectra = chase.contrast_profile(ha_al, background=seq["ha_bg"])

# Chromospheric temperature from Hα width (Molnar et al. 2019)
T, width = chase.halpha_width_temperature(ha_al[24], seq["wavelength_ha"])

# Photospheric temperature, absolute-calibrated against the FTS atlas (needs ISPy)
dc_cube, dc_wav, _ = chase.extract_disk_center("RSM..._FE.fits")
ifact, woff = chase.atlas_calibrate(dc_wav, dc_cube)
T_phot, _ = chase.fe_planck_temperature(fe_al[24], seq["wavelength_fe"], ifact)
```

Turning calibrations off is explicit. Both wavelength corrections have a switch:

```python
seq = chase.load_flare_sequence("/data/fits", patch=[...], resample=False)  # keep raw λ grids
contrast, _ = chase.contrast_profile(ha_al, correct_drift=False)            # no xcorr shift
```

CLI equivalents: `--no-resample`, `--no-drift`. Also `--no-track` and `--no-optical-flow`.

Full function list with signatures: [docs/SDK.md](docs/SDK.md).

## TUI

```bash
chase-tui
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/tui_e2e_walkthrough.gif" width="800" alt="End-to-end chase-tui walkthrough">
</p>

Pick a data source, set the field of view, tick the calibrations and outputs you want, hit Run. Progress streams into the log pane. Every checkbox maps to one `Config` field; the TUI adds no logic of its own. Full-resolution video: [assets/tui_e2e_walkthrough.mp4](https://github.com/Majboor/chase-pipeline/raw/feat/unified-pipeline-tui-sdk/assets/tui_e2e_walkthrough.mp4).

## Config file

```bash
chase --config examples/config.toml
```

```toml
fits_dir = "/data/20230329_X12/fits"
out_dir  = "./chase_out"
patch    = [940, 1080, 1870, 2040]   # [y0, y1, x0, x1]
contrast = true
temperature = "double"               # halpha + fe_planck
resume = false                       # true: reuse the checkpoint, skip load + align
```

Every key mirrors a `Config` field. Full annotated example: [examples/config.toml](examples/config.toml).

## Why this exists (statement of need)

CHASE (the Chinese Hα Solar Explorer) scans the full solar disk in Hα (118 wavelengths, 6559.4 to 6565.1 Å) and Fe I (46 wavelengths near 6569 Å). The portal ships full-disk cubes with no co-alignment and a wavelength zero-point that drifts by ~0.8 channels per frame. Getting a science-ready region out of that takes real calibration work, which is why the data is under-used. Earlier scripts (the original `chase-pipeline`, Finlay Davis's `satprocess`) each solved part of the problem. This package merges them into one installable tool that preserves the physics: shift-then-crop tracking, per-frame wavelength resampling, alignment on a flare-free channel, absolute atlas calibration, and the correct temperature inversion per line. The reasoning behind each rule is in [docs/CALIBRATION.md](docs/CALIBRATION.md).

## What lives where

| Subpackage | Does | |
|---|---|---|
| `chase.io` | resumable download, FITS discovery, cube + sequence loading | <img src="assets/diag_patch.png" width="220"> |
| `chase.calib.spatial` / `.align` | shift-then-crop tracking, optical-flow stabilisation | <img src="assets/diag_optical_flow.png" width="220"> |
| `chase.calib.wavelength` | per-frame resampling onto a common grid, spectral-drift xcorr | <img src="assets/diag_qs_spectrum.png" width="220"> |
| `chase.calib.intensity` | quiet-Sun norm, integral scaling, FTS-atlas absolute calibration | <img src="assets/atlas_calibration.png" width="220"> |
| `chase.analysis.contrast` | wavelength-vs-time flare contrast | <img src="assets/contrast_profile.png" width="220"> |
| `chase.analysis.temperature` | Hα width to T (Molnar), Fe I to T (Planck / EB / Voigt) | <img src="assets/double_temp_map_peak.png" width="220"> |

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/temperature.gif" width="720" alt="Double temperature map animation"><br>
  <em>Double temperature map: chromosphere from Hα width (~10⁴ K), photosphere from Fe I Planck inversion (~5100 K, absolute-calibrated).</em>
</p>

## Documentation

- [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md): CHASE portal to processed cube, end to end
- [docs/SDK.md](docs/SDK.md): every function, with snippets
- [docs/TUI.md](docs/TUI.md): TUI walkthrough
- [docs/CALIBRATION.md](docs/CALIBRATION.md): the science behind each calibration step
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): dev setup, tests, style

## Development

```bash
git clone https://github.com/Majboor/chase-pipeline.git
cd chase-pipeline
pip install -e '.[all]'
pytest -q            # 27 tests, synthetic FITS fixtures, no big data needed
```

## Acknowledgements

Built with the guidance and corrections of **Dr. Alexander Pietrow** (AIP) and **Dr. Malcolm Druett**. The Hough-circle limb centring, spectral cross-correlation, integral intensity scaling, and CSV shift-cache are adapted from **Finlay Davis**'s [`satprocess`](https://github.com/FinlayDavis/satprocess) (BSD-3-Clause; see [`NOTICE`](NOTICE)). Affiliation support from **Prof. Dr. Sayed Amer Mahmood** (University of the Punjab). Data courtesy of the **CHASE/HIS** mission and the [Solar Science Data Center, Nanjing University](https://ssdc.nju.edu.cn/NdchaseSatellite).

## Citing

If you use this pipeline, cite the methods it builds on:

- Molnar et al. 2019, *ApJ* **881**, 99. Hα width to temperature ([10.3847/1538-4357/ab2ba3](https://doi.org/10.3847/1538-4357/ab2ba3))
- The Eddington-Barbier photospheric inversion (A&A 2013, aa21259-13)
- The **ISPy** FTS solar-atlas calibration ([ISP-SST/ISPy](https://github.com/ISP-SST/ISPy))

## License

MIT. Ported `satprocess` code stays BSD-3-Clause, listed in [`NOTICE`](NOTICE).
