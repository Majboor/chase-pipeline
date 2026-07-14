"""CSV shift-cache so reruns are fast.

Ported from FinlayDavis/satprocess (BSD-3-Clause, see ``NOTICE``).  One row per
physical file accumulates the spatial shift, wavelength shift, detected centre,
and intensity scale factor across calibration stages, keyed by the *cleaned*
filename (stage suffixes stripped).

CSV columns (order matters — ``shift_y`` precedes ``shift_x``)::

    filename,shift_y,shift_x,wavelength_shift,cx,cy,intensity_scaling
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Tuple

__all__ = ["clean_filename", "load_shifts", "save_shifts", "get_shifts", "update_shifts"]

_FIELDS = ["shift_y", "shift_x", "wavelength_shift", "cx", "cy", "intensity_scaling"]
_DEFAULT: Tuple = (0.0, 0.0, 0, 0.0, 0.0, 1.0)
_SUFFIXES = ["_aligned", "_wave", "_scaled", "_spatial"]


def clean_filename(filename: str) -> str:
    """Strip stage suffixes so all stages share one cache row per file."""
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    if ext == "":
        ext = ".fits"
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem + ".fits"


def load_shifts(shifts_file: str) -> Dict[str, Tuple]:
    """Read the shift cache into ``{cleaned_filename: tuple}``."""
    shifts: Dict[str, Tuple] = {}
    if not os.path.exists(shifts_file):
        return shifts
    with open(shifts_file, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            try:
                key = row[0]
                vals = (
                    float(row[1]), float(row[2]), int(float(row[3])),
                    float(row[4]), float(row[5]), float(row[6]),
                )
            except (IndexError, ValueError):
                key, vals = (row[0] if row else "?"), _DEFAULT
            shifts[key] = vals
    return shifts


def save_shifts(shifts: Dict[str, Tuple], shifts_file: str) -> None:
    """Write the shift cache (header + sorted rows)."""
    os.makedirs(os.path.dirname(os.path.abspath(shifts_file)) or ".", exist_ok=True)
    with open(shifts_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + _FIELDS)
        for key in sorted(shifts):
            writer.writerow([key, *shifts[key]])


def get_shifts(shifts: Dict[str, Tuple], filename: str) -> Tuple:
    """Return the cached row for ``filename`` (or the default tuple)."""
    return shifts.get(clean_filename(filename), _DEFAULT)


def update_shifts(shifts: Dict[str, Tuple], filename: str, **updates) -> Tuple:
    """Update named fields for ``filename`` in-place; return the new row."""
    key = clean_filename(filename)
    row = list(shifts.get(key, _DEFAULT))
    for name, value in updates.items():
        if name in _FIELDS:
            row[_FIELDS.index(name)] = value
    shifts[key] = tuple(row)
    return shifts[key]
