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


def test_resume_skips_load_and_align(synthetic_dataset, tmp_path, monkeypatch):
    """A second run with resume=True must not touch the FITS loader."""
    out = tmp_path / "out_resume"
    base = dict(fits_dir=synthetic_dataset["dir"], out_dir=str(out),
                patch=synthetic_dataset["patch"])
    run_pipeline(Config(**base, optical_flow=False, gif=False, diagnostics=False),
                 progress=lambda s, m: None)

    import chase.pipeline as pl

    def _boom(*a, **k):
        raise AssertionError("load_flare_sequence must not run when resuming")

    monkeypatch.setattr(pl, "load_flare_sequence", _boom)
    monkeypatch.setattr(pl, "optical_flow_align", _boom)

    results = run_pipeline(Config(**base, resume=True, gif=True, diagnostics=True),
                           progress=lambda s, m: None)
    assert os.path.exists(results["gif"])
    assert results["aligned"]["ha"].shape[0] == synthetic_dataset["nframes"]
    assert results["npz"].endswith("aligned_data.npz")


def test_resume_without_checkpoint_falls_back(synthetic_dataset, tmp_path):
    out = tmp_path / "out_fallback"
    cfg = Config(fits_dir=synthetic_dataset["dir"], out_dir=str(out),
                 patch=synthetic_dataset["patch"], resume=True,
                 optical_flow=False, gif=False, diagnostics=False)
    results = run_pipeline(cfg, progress=lambda s, m: None)
    assert os.path.exists(results["npz"])


def test_contrast_drift_toggle(synthetic_dataset, tmp_path, monkeypatch):
    """Config.correct_drift must reach contrast_profile; profile saved as .npy."""
    import chase.pipeline as pl

    seen = {}
    real = pl.contrast_profile

    def _spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(pl, "contrast_profile", _spy)
    cfg = Config(fits_dir=synthetic_dataset["dir"], out_dir=str(tmp_path / "out_drift"),
                 patch=synthetic_dataset["patch"], contrast=True, correct_drift=False,
                 optical_flow=False, gif=False, diagnostics=False)
    results = run_pipeline(cfg, progress=lambda s, m: None)
    assert seen["correct_drift"] is False
    assert os.path.exists(results["contrast_npy"])


def test_cli_only_selects_steps():
    from chase.cli import build_parser, config_from_args

    args = build_parser().parse_args(["/data", "--only", "gif,contrast"])
    cfg = config_from_args(args)
    assert cfg.gif is True and cfg.contrast is True
    assert cfg.save_npz is False and cfg.save_fits is False and cfg.diagnostics is False
    assert cfg.temperature == "none"
    assert cfg.resume is True


def test_cli_only_rejects_unknown_step():
    import pytest

    from chase.cli import build_parser, config_from_args

    args = build_parser().parse_args(["/data", "--only", "gif,bogus"])
    with pytest.raises(SystemExit):
        config_from_args(args)


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


def test_fits_export_and_alignment_diff(synthetic_dataset, tmp_path):
    """save_fits writes per-scan cubes with updated headers; diagnostics
    include the before/after alignment difference map."""
    import astropy.io.fits as fits

    out = tmp_path / "out_fits"
    cfg = Config(fits_dir=synthetic_dataset["dir"], out_dir=str(out),
                 patch=synthetic_dataset["patch"], optical_flow=True,
                 gif=False, save_fits=True, diagnostics=True)
    results = run_pipeline(cfg, progress=lambda s, m: None)

    paths = results["fits"]
    assert len(paths) == synthetic_dataset["nframes"]
    with fits.open(paths[0]) as h:
        hdr = h[1].header
        assert hdr["CUNIT1"] == "arcsec" and abs(hdr["CDELT1"] - 1.04) < 1e-9
        assert hdr["CTYPE3"] == "WAVE"
        assert any("optical-flow" in str(c) for c in hdr["HISTORY"])

    assert any("alignment_difference" in p for p in results["diagnostics"])
