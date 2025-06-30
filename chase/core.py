# chase/core.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import requests
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

TERMINAL_BANNER = """
============================================================
        🌞  CHASE Satellite Data Calibration Pipeline  🌞
============================================================
Terminal module containing core calibration logic for H-alpha data.
============================================================
"""

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
                with open(local_filename, mode) as f, tqdm(total=total, initial=downloaded, unit='B', unit_scale=True, desc=os.path.basename(local_filename)) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return local_filename
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"⚠️ Download error ({retries}/{max_retries}): {e}")
            time.sleep(2)
    raise RuntimeError(f"❌ Failed to download {url} after {max_retries} retries.")

def load_fits_from_url_or_txt(input_path, save_dir):
    if input_path.lower().endswith('.txt'):
        with open(input_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = [input_path]
    return [download_file(url, save_dir) if url.startswith("http") else url for url in urls]

def load_fits_data(filepath):
    with fits.open(filepath) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{filepath} has no image extension (hdul[1])")
        return hdul[1].data

def find_disk_center(image, threshold_ratio=0.2):
    norm = image / np.max(image)
    mask = norm > threshold_ratio
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No disk detected.")
    return int(xs.mean()), int(ys.mean()), mask

def recenter_image_cube(data_cube, threshold_ratio=0.2):
    h, w = data_cube.shape[1:]
    cx, cy, _ = find_disk_center(data_cube[0], threshold_ratio)
    dx, dy = w // 2 - cx, h // 2 - cy
    recentered = np.zeros_like(data_cube)
    for i in range(data_cube.shape[0]):
        recentered[i] = np.roll(np.roll(data_cube[i], dx, axis=1), dy, axis=0)
    return recentered, (dx, dy)

def pixel_to_arcsec(x, y, shape, scale=0.44):
    h, w = shape
    return (x - w // 2) * scale, (y - h // 2) * scale

def align_subpixel(ref, target, upsample_factor=100):
    shift_yx, _, _ = phase_cross_correlation(ref, target, upsample_factor=upsample_factor)
    aligned = shift(target, shift=shift_yx, order=3)
    return aligned, shift_yx

def extract_qs_region(data_cube, size=100):
    h, w = data_cube.shape[1:]
    cx, cy = w // 2, h // 2
    half = size // 2
    return data_cube[:, cy-half:cy+half, cx-half:cx+half]

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

def run_pipeline(
    fits_file,
    do_spatial=True,
    do_subpixel=True,
    do_spectral=True,
    do_intensity=True,
    do_fov=True,
    save_figs=True,
    fig_dir="./figures",
    threshold_ratio=0.2,
    subpixel_accuracy=100,
    qs_box_size=100,
    fov_width=300,
    fov_height=300
):
    print(f"\n--- Processing: {fits_file} ---")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(fits_file))[0]
    data = load_fits_data(fits_file)
    print("Data shape:", data.shape)

    original = data.copy()
    cx, cy = data.shape[2] // 2, data.shape[1] // 2
    mid = data.shape[0] // 2

    if do_spatial:
        cx, cy, _ = find_disk_center(data[0], threshold_ratio)
        arcsec = pixel_to_arcsec(cx, cy, data.shape[1:])
        print("Disk center (arcsec):", arcsec)
        data, (dx, dy) = recenter_image_cube(data, threshold_ratio)
        # Plot difference image
        diff = original[0] - data[0]
        plt.imshow(diff, cmap='seismic', origin='lower', vmin=-100, vmax=100)
        plt.title("Difference: Original - Recentered")
        plt.colorbar()
        plt.savefig(f"{fig_dir}/{base}_diff.png")
        plt.close()

    if do_subpixel:
        aligned, shift_val = align_subpixel(data[mid], data[mid], subpixel_accuracy)
        print("Subpixel shift:", shift_val)
        plt.subplot(1, 2, 1)
        plt.imshow(data[mid], cmap="gray", origin="lower")
        plt.title("Before Subpixel")
        plt.subplot(1, 2, 2)
        plt.imshow(aligned, cmap="gray", origin="lower")
        plt.title("After Subpixel")
        plt.savefig(f"{fig_dir}/{base}_subpixel_align.png")
        plt.close()

    if do_spectral:
        qs = extract_qs_region(data, size=qs_box_size)
        profile = np.mean(qs, axis=(1, 2))
        plt.plot(profile)
        plt.title("QS Profile")
        plt.savefig(f"{fig_dir}/{base}_qs_profile.png")
        plt.close()

        if do_spectral:
            wavelengths = calibrate_wavelength(profile)
            plt.plot(wavelengths, profile)
            plt.axvline(656.28, color="r", linestyle="--", label="H-alpha")
            plt.title("Wavelength Calibration")
            plt.xlabel("Wavelength (nm)")
            plt.legend()
            plt.savefig(f"{fig_dir}/{base}_wavelength_calibration.png")
            plt.close()

    if do_intensity:
        data = normalize_intensity(data, qs)
        plt.subplot(1, 2, 1)
        plt.imshow(original[mid], cmap="gray", origin="lower")
        plt.title("Before Intensity Norm")
        plt.subplot(1, 2, 2)
        plt.imshow(data[mid], cmap="gray", origin="lower", vmin=0, vmax=2)
        plt.title("After Intensity Norm")
        plt.savefig(f"{fig_dir}/{base}_intensity_normalization.png")
        plt.close()

    if do_fov:
        cx, cy = data.shape[2] // 2, data.shape[1] // 2
        sub = extract_subregion(data, cx, cy, fov_width, fov_height)
        plt.imshow(sub[mid], cmap="gray", origin="lower")
        plt.title("Sub-FOV Slice")
        plt.savefig(f"{fig_dir}/{base}_subfov_slice.png")
        plt.close()

    print("✅ Done.")
