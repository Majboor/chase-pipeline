"""End-to-end orchestrator smoke tests on the synthetic fixture."""

import os

from chase import Config, run_pipeline


def test_run_pipeline_smoke_full_fov(synthetic_dataset, tmp_path):
    out = tmp_path / "out"
    cfg = Config(
        fits_dir=synthetic_dataset["dir"],
        out_dir=str(out),
        full_fov=True,
        optical_flow=True,
        gif=True,
        contrast=True,
        temperature="halpha",
        diagnostics=True,
    )
    results = run_pipeline(cfg, progress=lambda s, m: None)
    assert os.path.exists(results["npz"])
    assert os.path.exists(results["gif"])
    assert "contrast" in results
    assert os.path.exists(results["temperature_gif"])
    assert results["ha_temp"].shape[0] == synthetic_dataset["nframes"]


def test_run_pipeline_patch_no_opticalflow(synthetic_dataset, tmp_path):
    out = tmp_path / "out2"
    cfg = Config(
        fits_dir=synthetic_dataset["dir"],
        out_dir=str(out),
        patch=synthetic_dataset["patch"],
        optical_flow=False,
        gif=False,
        diagnostics=True,
    )
    cfg.resample_wavelength = True
    results = run_pipeline(cfg, progress=lambda s, m: None)
    assert os.path.exists(results["npz"])
    assert results["aligned"]["ha"].shape[0] == synthetic_dataset["nframes"]


def test_config_roundtrip_toml(tmp_path):
    from chase.config import load_config

    toml = tmp_path / "c.toml"
    toml.write_text(
        'fits_dir = "/data"\n'
        "patch = [10, 20, 30, 40]\n"
        "contrast = true\n"
        'temperature = "double"\n'
    )
    cfg = load_config(str(toml))
    assert cfg.fits_dir == "/data"
    assert cfg.patch == [10, 20, 30, 40]
    assert cfg.contrast is True
    assert cfg.temperature == "double"
