# Contributing

Thanks for helping improve `chase-pipeline`.

## Dev setup

```bash
git clone https://github.com/Majboor/chase-pipeline
cd chase-pipeline
pip install -e '.[all]'      # core + tui + atlas + dev
pytest -q                    # 22 tests, all on synthetic FITS fixtures
```

You do **not** need the real multi-hundred-MB CHASE cubes to develop or test —
`tests/conftest.py` synthesises tiny HA/FE cubes with a known shift and CRVAL3
drift.

## Project layout

```
chase/
  io/        download + discovery + cube/sequence loading
  calib/     spatial · align · wavelength · intensity
  analysis/  contrast · temperature
  viz/       animate · diagnostics
  config.py pipeline.py cli.py tui.py core.py(shim) shifts_cache.py
tests/  examples/  docs/  assets/
```

## Conventions

- **Modular first.** New functionality should be a small, pure function in the
  relevant submodule, re-exported from the package `__init__`. Avoid forcing it
  into the monolithic pipeline.
- **NumPy-style docstrings** on every public function (the docs auto-generate from
  these, SpotiPy-style). Keep parameter names and shapes explicit.
- **Don't break the physics.** See [CALIBRATION.md](CALIBRATION.md) — the ten
  rules there are requirements, not suggestions.
- **Keep science data float32**; only convert to uint8 where an algorithm demands
  it (optical-flow reference).
- **Ported `satprocess` code keeps its BSD-3 header** and is listed in
  [`NOTICE`](../NOTICE).

## Tests & CI

- Add a test for any new stage; prefer asserting against ground truth from the
  synthetic fixtures.
- CI runs `pytest` on Python 3.9 / 3.11 / 3.12 (`.github/workflows/test.yml`).
- Keep tests fast and hermetic — no network, no large data.

## Commits & PRs

- Small, readable, incremental commits (this project tracks for a JOSS submission,
  which values a clear history).
- Describe *why*, not just *what*. Reference the calibration rule a change touches.
- Run `pytest -q` before pushing.

## Optional dependencies

- `textual` — the TUI. Gate any TUI-only import.
- `ISPy` — absolute Fe I atlas calibration. Gate the import and degrade gracefully
  (the core pipeline must work without it).
