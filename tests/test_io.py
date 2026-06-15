"""I/O: discovery, resolution, cube loading, disk-center extraction."""

import os

import numpy as np

from chase.io import (
    discover_fits,
    extract_disk_center,
    load_cube,
    resolve_inputs,
    wavelength_grid,
)


def test_discover_fits_pairs(synthetic_dataset):
    ha, fe = discover_fits(synthetic_dataset["dir"])
    assert len(ha) == synthetic_dataset["nframes"]
    assert len(fe) == synthetic_dataset["nframes"]
    assert all(p.endswith("HA.fits") for p in ha)


def test_resolve_inputs_directory(synthetic_dataset):
    files = resolve_inputs(synthetic_dataset["dir"])
    assert len(files) == 2 * synthetic_dataset["nframes"]


def test_resolve_inputs_txt(tmp_path, synthetic_dataset):
    ha, _ = discover_fits(synthetic_dataset["dir"])
    txt = tmp_path / "list.txt"
    txt.write_text("\n".join(ha) + "\n")
    files = resolve_inputs(str(txt))
    assert files == ha


def test_load_cube_shapes_and_wavelength(synthetic_dataset):
    ha, _ = discover_fits(synthetic_dataset["dir"])
    cube, hdr, wav = load_cube(ha[0])
    assert cube.shape == (synthetic_dataset["nch_ha"], synthetic_dataset["H"], synthetic_dataset["W"])
    assert cube.dtype == np.float32
    assert len(wav) == synthetic_dataset["nch_ha"]
    # wavelength grid reconstructed from header
    assert np.allclose(wav, wavelength_grid(hdr))
    assert wav[0] < wav[-1]


def test_load_cube_region(synthetic_dataset):
    ha, _ = discover_fits(synthetic_dataset["dir"])
    cube, _, _ = load_cube(ha[0], region=(10, 30, 5, 25))
    assert cube.shape == (synthetic_dataset["nch_ha"], 20, 20)


def test_extract_disk_center(synthetic_dataset):
    _, fe = discover_fits(synthetic_dataset["dir"])
    cube, wav, (cy, cx) = extract_disk_center(fe[0], size=20)
    assert cube.shape[1] <= 20 and cube.shape[2] <= 20
    # CRPIX of the synthetic data is the image centre
    assert abs(cy - synthetic_dataset["H"] // 2) <= 1
    assert abs(cx - synthetic_dataset["W"] // 2) <= 1
