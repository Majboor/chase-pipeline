"""
core.py — CHASE Satellite Calibration Core Module
Handles downloading, calibration steps, and utility functions.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import requests
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

# ------------------------
# Download with Resume
# ------------------------

def download_file(url, save_dir, max_retries=5, chunk_size=8192):
    os.makedirs(save_dir, exist_ok=True)
    local_filename = os.path.join(save_dir, url.split('/')[-1].split('?')[0])
    downloaded = os.path.getsize(local_filename) if os.path.exists(local_filename) else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    retries = 0

    while retries < max_retries:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0)) + downloaded
                mode = 'ab' if downloaded else 'wb'
                with open(local_filename, mode) as f, tqdm(
                    total=total,
                    initial=downloaded,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc=os.path.basename(local_filename)
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return local_filename
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            print(f"⚠️ Download error ({retries+1}/{max_retries}): {e}")
            retries += 1
            time.sleep(2)
    raise RuntimeError(f"❌ Failed to download {url} after {max_retries} retries.")

def load_fits_from_url_or_txt(input_path, save_dir):
    if input_path.lower().endswith('.txt'):
        with open(input_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    elif input_path.startswith("http"):
        urls = [input_path]
    else:
        # Local file
        return [input_path]

    files = []
    for url in urls:
        file = download_file(url, save_dir)
        files.append(file)
    return files

# ------------------------
# FITS Handling
# ------------------------

def load_fits_data(filepath):
    with fits.open(filepath) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"FITS file {filepath} does not contain a data HDU[1].")
        return hdul[1].data

# ------------------------
# Calibration Tools
# ------------------------

def find_disk_center(image, threshold_ratio=0.2):
    norm = image / np.max(image)
    mask = norm > threshold_ratio
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No disk detected in image.")
    return int(xs.mean()), int(ys.mean()), mask

def recenter_image_cube(cube, threshold_ratio=0.2):
    h, w = cube.shape[1:]
    cx, cy, _ = find_disk_center(cube[0], threshold_ratio)
    dx, dy = w//2 - cx, h//2 - cy
    shifted = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        shifted[i] = np.roll(np.roll(cube[i], dx, axis=1), dy, axis=0)
    return shifted, (dx, dy)

def pixel_to_arcsec(x, y, shape, scale=0.44):
    h, w = shape
    return (x - w//2) * scale, (y - h//2) * scale

def align_subpixel(ref, target, upsample_factor=100):
    shift_yx, _, _ = phase_cross_correlation(ref, target, upsample_factor=upsample_factor)
    aligned = shift(target, shift=shift_yx, order=3)
    return aligned, shift_yx

def extract_qs_region(cube, size=100):
    h, w = cube.shape[1:]
    cx, cy = w // 2, h // 2
    return cube[:, cy-size//2:cy+size//2, cx-size//2:cx+size//2]

def calibrate_wavelength(profile, ref_wavelength=656.28, dispersion=0.0025):
    center = np.argmin(profile)
    return ref_wavelength + (np.arange(len(profile)) - center) * dispersion

def normalize_intensity(cube, ref_cube):
    return np.clip(cube / np.mean(ref_cube), 0, 2)

def extract_subregion(cube, cx, cy, width, height):
    x1, x2 = max(cx-width//2, 0), min(cx+width//2, cube.shape[2])
    y1, y2 = max(cy-height//2, 0), min(cy+height//2, cube.shape[1])
    return cube[:, y1:y2, x1:x2]

def create_disk_mask(shape, cx, cy, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return (X - cx)**2 + (Y - cy)**2 <= radius**2

# ------------------------
# Pipeline Entrypoint
# ------------------------

def run_pipeline(fits_file,
                 do_spatial=True, do_subpixel=True,
                 do_spectral=True, do_intensity=True,
                 do_fov=True, save_figs=False,
                 threshold_ratio=0.2, subpixel_accuracy=100,
                 qs_box_size=100, fov_width=300, fov_height=300):

    print(f"\n--- Processing: {fits_file} ---")
    data = load_fits_data(fits_file)
    print("Data shape:", data.shape)

    if do_spatial:
        cx, cy, _ = find_disk_center(data[0], threshold_ratio)
        arcsec = pixel_to_arcsec(cx, cy, data.shape[1:])
        print("Disk center (arcsec):", arcsec)
        data, _ = recenter_image_cube(data, threshold_ratio)

    if do_subpixel:
        mid = data.shape[0] // 2
        ref, tgt = data[mid], data[mid]
        aligned, shift_val = align_subpixel(ref, tgt, upsample_factor=subpixel_accuracy)
        print("Subpixel shift:", shift_val)

    if do_spectral:
        qs_cube = extract_qs_region(data, size=qs_box_size)
        profile = np.mean(qs_cube, axis=(1, 2))
        wavelengths = calibrate_wavelength(profile)

    if do_intensity:
        data = normalize_intensity(data, qs_cube)

    if do_fov:
        cx, cy = data.shape[2] // 2, data.shape[1] // 2
        subcube = extract_subregion(data, cx, cy, fov_width, fov_height)
        mask = create_disk_mask(data.shape[1:], cx, cy, min(data.shape[1:])//2 - 100)

    if save_figs and do_spectral:
        plt.figure()
        plt.plot(wavelengths, profile)
        plt.title("QS Spectrum")
        plt.savefig(fits_file + "_qs_spectrum.png")

    if save_figs and do_fov:
        slice_index = min(70, data.shape[0] - 1)
        plt.figure()
        plt.imshow(subcube[slice_index], origin='lower', cmap='gray')
        plt.title(f"Sub-FOV Slice {slice_index}")
        plt.colorbar()
        plt.savefig(fits_file + f"_subcube{slice_index}.png")

    print("✅ Done.")
