# Calibration — the science, and why each step is the way it is

Every rule below was hard-won during debugging with Dr. Alexander Pietrow.
"Simplifying" any of them reintroduces a real artefact. This page explains what
each stage does and why.

## 1. Shift-then-crop (not crop-then-shift)

The flare patch drifts across the disk frame to frame. If you crop a fixed window
first and then shift, the **spectral average over the window mixes different
physical features**, producing an antisymmetric pink/green artefact in the
pre-flare frames. Instead we load a **padded** region (`patch ± ~20 px`), estimate
the integer shift on a flare-free channel, and **crop the final patch from inside
the buffer** at the shifted offset. Every frame then covers identical content.

→ `chase.calib.track_and_crop`, `chase.io.load_flare_sequence`.

## 2. Per-frame wavelength resampling

Each frame's `CRVAL3` drifts by ~0.04 Å (~0.8 channels) and `CDELT3` varies. On
the steep Hα wings, comparing the same *channel index* across frames manufactures
fake contrast. So each frame is resampled (cubic) onto **frame 0's wavelength
grid** before any spectral arithmetic.

→ `chase.calib.resample_wavelength`. Data-driven fallback when headers are
unreliable: `chase.calib.correct_spectral_drift`.

Both corrections are switchable when you want the untouched spectra:
`resample_wavelength = false` turns off the header-based resampling, and
`correct_drift = false` (CLI `--no-drift`) turns off the data-driven
correction inside the contrast stage.

## 3. Align on a flare-free channel

The eruption (frames ~22–24) is **real motion** — we must not stabilise it away.
Alignment is therefore computed on a channel with little flare signal: the **Fe I
last wavelength** (photospheric), falling back to the Hα continuum. The
photosphere is stable, so it makes a clean alignment reference while the
chromospheric flare is free to erupt.

## 4. Keep science data float32

Optical flow needs an 8-bit image to estimate displacement, so **only the
reference channel** is converted to `uint8`. The resulting displacement map is
applied with `cv2.remap` (cubic) to the **float32** science cubes — every Hα and
Fe I channel warped with the *same* map. The science data never loses precision.

→ `chase.calib.optical_flow_align`.

## 5. Contrast = (flare − background)/background − frame0

A naive `frame_t / frame_0 − 1` leaves instrumental throughput drift in the whole
field. The correct definition uses a **separate quiet-Sun background patch** (same
size, offset one patch-width to the side):

```
contrast(t, λ) = (flare(t, λ) − bg(t, λ)) / bg(t, λ)  −  [same at frame 0]
```

In the wavelength-vs-time map the erupting CME appears as a Doppler feature in the
Hα wing — the science hook.

→ `chase.analysis.contrast_profile`.

## 6. Absolute calibration at disk centre

Intensity and wavelength absolute calibration are done on the **100×100 patch
around `CRPIX1`/`CRPIX2`** (solar disk centre), because that location does not
change frame to frame. The flare patch is handled separately by the tracking
above.

→ `chase.io.extract_disk_center`.

## 7. Atlas calibration fits the wings, not the core

Raw DN cannot go into a Planck inversion. We fit the disk-centre quiet-Sun Fe I
profile to the **FTS solar atlas** (via ISPy `get_calibration`, `mu=1.0`,
`calib_at_dc=True`), using only the **line wings** — the core is non-LTE / smeared
and the CHASE Fe I line is under-sampled (it doesn't reach the true atlas
minimum). This yields a DN→radiance factor `ifact` (≈ 2.30×10⁻⁸ for this data) and
a small wavelength offset.

→ `chase.calib.atlas_calibrate`; chart: `chase.viz.plot_atlas_calibration`.

![atlas calibration](../assets/atlas_calibration.png)

## 8. Temperatures: different physics, different method

- **Hα → chromosphere via line *width*** (Molnar et al. 2019): Hα forms out of
  LTE, so a Planck inversion is invalid. The half-depth width maps onto an ALMA
  brightness temperature, `T = (width[Å] − 0.553) / 6.12×10⁻⁵`. Gives ~10⁴ K in
  the quiet chromosphere and >15,000 K at flare ribbons.
- **Fe I → photosphere via Planck** (LTE line): once on an absolute radiance
  scale, `B_ν(T) = I_core`. Three core estimators are provided —
  `fe_planck` (Gaussian core), `fe_eb` (Eddington–Barbier per-wavelength), and
  `fe_voigt` (robust to under-sampling). Quiet-Sun continuum recovers ~6025 K and
  the line core ~5,100 K — textbook photospheric values.

→ `chase.analysis.halpha_width_temperature`, `chase.analysis.fe_planck_temperature`,
`fe_eb_temperature`, `fe_voigt_temperature`, `chase.calib.planck_inversion`.

## 9. Save all channels

The pipeline saves **all 118 Hα + 46 Fe I channels** (not just a summary slice),
applying the same spatial warp to the Fe cube as the Hα cube.

## 10. Honest animations

Animations offer **frozen `vmin`/`vmax`** (`freeze_clim=True`) so brightness can
be compared across frames without per-frame autoscaling hiding the flare's rise.

## References

- Molnar et al. 2019, *ApJ* **881**, 99 — 10.3847/1538-4357/ab2ba3.
- Eddington–Barbier: A&A 2013, aa21259-13.
- ISPy FTS atlas: ISP-SST/ISPy.
- `satprocess` (Davis): Hough centring + xcorr + integral scaling + shift-cache.
