"""Data acquisition and discovery.

Three ways to point the pipeline at data, all funnelled through
:func:`resolve_inputs`:

* a **directory** of ``*HA.fits`` / ``*FE.fits`` cubes (the common case),
* a single ``.fits`` file,
* a ``.txt`` file of one download URL per line, or a bare ``http(s)://`` URL.

Downloads are **resumable** (HTTP ``Range`` header) with a ``tqdm`` progress bar
and a retry loop, ported from the original ``chase-pipeline`` ``core.py``.
"""

from __future__ import annotations

import glob
import os
import time
from typing import List, Optional, Tuple

__all__ = ["download_file", "resolve_inputs", "discover_fits"]


def download_file(
    url: str,
    save_dir: str,
    max_retries: int = 5,
    chunk_size: int = 8192,
) -> str:
    """Download ``url`` into ``save_dir``, resuming a partial file if present.

    Parameters
    ----------
    url : str
        HTTP(S) URL of the file to fetch.
    save_dir : str
        Destination directory (created if missing).
    max_retries : int, optional
        Number of times to retry on a transient network error.
    chunk_size : int, optional
        Streaming chunk size in bytes.

    Returns
    -------
    str
        Path to the downloaded file on disk.
    """
    import requests  # imported lazily so the package works offline
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)
    local_filename = os.path.join(save_dir, url.split("/")[-1].split("?")[0])
    downloaded = os.path.getsize(local_filename) if os.path.exists(local_filename) else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    retries = 0
    while retries < max_retries:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + downloaded
                mode = "ab" if downloaded else "wb"
                with open(local_filename, mode) as f, tqdm(
                    total=total,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(local_filename),
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return local_filename
        except requests.exceptions.RequestException as e:  # pragma: no cover - network
            retries += 1
            print(f"Download error ({retries}/{max_retries}): {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries.")


def resolve_inputs(input_path: str, save_dir: str = "./data") -> List[str]:
    """Turn a user-supplied source into a concrete list of local FITS paths.

    Accepts a directory, a single ``.fits`` file, a ``.txt`` URL list, or a bare
    URL.  URLs are downloaded into ``save_dir``; local paths pass through.

    Parameters
    ----------
    input_path : str
        Directory, ``.fits`` file, ``.txt`` URL list, or ``http(s)`` URL.
    save_dir : str, optional
        Where downloaded files are written.

    Returns
    -------
    list of str
        Local file paths, sorted.
    """
    if os.path.isdir(input_path):
        ha, fe = discover_fits(input_path)
        return sorted(ha + fe)

    if input_path.lower().endswith(".txt"):
        with open(input_path) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return [
            download_file(u, save_dir) if u.startswith("http") else u for u in urls
        ]

    if input_path.startswith("http"):
        return [download_file(input_path, save_dir)]

    # Single local file
    return [input_path]


def discover_fits(directory: str) -> Tuple[List[str], List[str]]:
    """Find paired ``*HA.fits`` and ``*FE.fits`` cubes in a directory.

    Parameters
    ----------
    directory : str
        Directory to scan (non-recursive).

    Returns
    -------
    (ha_files, fe_files) : tuple of list of str
        Sorted Hα and Fe I file lists.  ``fe_files`` may be empty.
    """
    d = str(directory).rstrip("/")
    ha = sorted(glob.glob(d + "/*HA.fits") + glob.glob(d + "/*ha.fits"))
    fe = sorted(glob.glob(d + "/*FE.fits") + glob.glob(d + "/*fe.fits"))
    return ha, fe


def pair_ha_fe(ha_files: List[str], fe_files: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Pair each HA file with its FE counterpart by timestamp/frame prefix."""
    pairs: List[Tuple[str, Optional[str]]] = []
    for ha in ha_files:
        stem = os.path.basename(ha).replace("HA.fits", "").replace("ha.fits", "")
        match = next(
            (fe for fe in fe_files if os.path.basename(fe).startswith(stem)), None
        )
        pairs.append((ha, match))
    return pairs
