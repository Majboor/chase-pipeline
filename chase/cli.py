# chase/cli.py

import argparse
import os
from chase.core import (
    load_fits_from_url_or_txt,
    run_pipeline,
    TERMINAL_BANNER
)

def main():
    print(TERMINAL_BANNER)

    parser = argparse.ArgumentParser(
        description="CHASE H-alpha FITS Calibration Pipeline"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Single FITS URL or path to a .txt file containing multiple URLs"
    )
    parser.add_argument(
        "--output", "-o", default="./downloads",
        help="Directory where FITS files will be downloaded"
    )
    parser.add_argument(
        "--save", "-s", action="store_true",
        help="Save diagnostic plots (QS spectrum, sub-FOV slice, etc.)"
    )
    parser.add_argument(
        "--fig-dir", default="./figures",
        help="Directory to store output figures (default: ./figures)"
    )
    parser.add_argument("--no-spatial", action="store_true",
                        help="Skip spatial calibration")
    parser.add_argument("--no-subpixel", action="store_true",
                        help="Skip sub-pixel alignment")
    parser.add_argument("--no-spectral", action="store_true",
                        help="Skip spectral calibration")
    parser.add_argument("--no-intensity", action="store_true",
                        help="Skip intensity normalization")
    parser.add_argument("--no-fov", action="store_true",
                        help="Skip sub-FOV extraction")
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Threshold ratio for disk detection (default: 0.2)"
    )
    parser.add_argument(
        "--accuracy", type=int, default=100,
        help="Upsample factor for sub-pixel alignment (default: 100)"
    )
    parser.add_argument(
        "--qsbox", type=int, default=100,
        help="Quiet Sun box size for spectral calibration (default: 100)"
    )
    parser.add_argument(
        "--fov-width", type=int, default=300,
        help="Width of sub-field-of-view cutout (default: 300)"
    )
    parser.add_argument(
        "--fov-height", type=int, default=300,
        help="Height of sub-field-of-view cutout (default: 300)"
    )

    args = parser.parse_args()

    fits_files = load_fits_from_url_or_txt(args.input, args.output)
    if not fits_files:
        parser.error("No FITS files could be downloaded or found.")

    for fits_file in fits_files:
        run_pipeline(
            fits_file,
            do_spatial   = not args.no_spatial,
            do_subpixel  = not args.no_subpixel,
            do_spectral  = not args.no_spectral,
            do_intensity = not args.no_intensity,
            do_fov       = not args.no_fov,
            save_figs    = args.save,
            fig_dir= args.fig_dir,
            threshold_ratio   = args.threshold,
            subpixel_accuracy = args.accuracy,
            qs_box_size       = args.qsbox,
            fov_width         = args.fov_width,
            fov_height        = args.fov_height
        )

if __name__ == "__main__":
    main()
