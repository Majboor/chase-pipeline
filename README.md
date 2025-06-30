# 🌞 CHASE Satellite Data Calibration Pipeline

![CHASE Pipeline Demo](./assets/tutorial.gif)

The **CHASE Calibration Pipeline** is a terminal-based Python toolkit for downloading and processing solar flare data from the [Chinese Hα Solar Explorer (CHASE)](https://ssdc.nju.edu.cn/chase/).  
It includes **spatial**, **spectral**, **intensity**, and **field-of-view** calibration steps.

---

## 📦 Features

- ✅ Download FITS files directly or from a list (`.txt`)  
- ✅ Resume interrupted downloads with progress bar  
- ✅ Spatial calibration (disk center detection + recentering)  
- ✅ Sub-pixel image alignment using Fourier shift  
- ✅ Spectral calibration from QS profile  
- ✅ Intensity normalization and FOV extraction  
- ✅ CLI or importable Python API

---

## 🚀 Quick Start
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XSw8wow6hUDY-z8HOUtTTFK_kNtv7uj2?usp=sharing)

### 1. Clone the repo
```bash
git clone https://github.com/Majboor/chase-pipeline
cd chase-pipeline
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run from command line

```bash
python main.py --input "https://ssdc.nju.edu.cn/chase/2025/5/1/your_fits_file.fits?...sig..." --output ./downloads --save-figs
```

OR pass a `.txt` file with multiple URLs:

```bash
python main.py --input fits_list.txt --output ./downloads
```

---

## ⚙️ CLI Arguments

```bash
--input           Path to FITS URL or .txt list of URLs  
--output          Directory where FITS files will be downloaded (default: ./downloads)  
--threshold       Brightness threshold for disk detection (default: 0.2)  
--accuracy        Upsample factor for subpixel alignment (default: 100)  
--qsbox           Quiet Sun box size for spectral calibration (default: 100)  
--fov-width       Field of view width in pixels (default: 300)  
--fov-height      Field of view height in pixels (default: 300)  
--no-spatial      Skip spatial calibration  
--no-subpixel     Skip subpixel alignment  
--no-spectral     Skip spectral calibration  
--no-intensity    Skip intensity normalization  
--no-fov          Skip sub-FOV extraction  
--save            Save all diagnostic plots (e.g., QS profile, intensity maps)  
--fig-dir         Output directory for saved figures (default: ./figures)  

```

---

## 🧪 Example Notebook

You can explore the pipeline interactively in:

```bash
tutorial.ipynb
```

---

## 📸 Output Visuals (Diagnostics)

Below are sample outputs saved during a typical run with `--save-figs`:

| Image                                      | Description                                                                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| ![](./assets/initial_before_slice.png) | **Raw slice** from the original FITS data cube before any processing (typically slice 70).                                   |
| ![](./assets/disk_slice.png)           | Slice showing the **solar disk** and brightness structure before recentering. Useful for checking disk detectability.        |
| ![](./assets/recentered.png)           | The result after **spatial calibration**, where the disk is aligned to the center.                                           |
| ![](./assets/pixel.png)                | Pixel location of the detected **disk center**, later used to convert to arcseconds.                                         |
| ![](./assets/subpixel.png)             | Output of **sub-pixel alignment**, used to reduce jitter and motion between slices.                                          |
| ![](./assets/spectral_profi.png)       | Intermediate **Quiet Sun spectral profile** plot, showing intensity vs. slice index.                                         |
| ![](./assets/spectral_profile.png)     | Calibrated **spectral profile** with real wavelengths in Ångströms, aligned to H-alpha 656.28Å.                              |
| ![](./assets/spectral_slice.png)       | Example of a **spectral slice** used in calibration and wavelength assignment.                                               |
| ![](./assets/before_normal.png)        | A slice before **intensity normalization**, showing raw relative brightness.                                                 |
| ![](./assets/sub-fov.png)              | Cropped **field-of-view (FOV)** region after all calibrations. Typically centered on solar disk with dimensions set via CLI. |

Each image was generated from a single FITS cube and saved using the `--save-figs` flag.

---

## 📁 Project Structure

```
chase/
├── chase/
│   ├── __init__.py         # Package entry
│   ├── core.py             # All core calibration logic
│   ├── cli.py              # CLI parser
├── main.py                 # Entrypoint
├── requirements.txt        # Dependencies
├── README.md               # This file
├── tutorial.gif            # Visual usage guide
├── tutorial.ipynb          # Interactive notebook
└── assets/
    └── <images are here>/                # Diagnostic figures like spectra and FOVs
```

---

## 🤝 Acknowledgements

Developed as part of the CHASE calibration project supervised by:

* Prof. Dr. Sayed Amer Mahmood
* Dr. Alexander Pietrow
* Dr. Malcolm Druett

Data courtesy of the [CHASE Satellite](https://ssdc.nju.edu.cn/chase/).

---

## 📜 License

MIT License

