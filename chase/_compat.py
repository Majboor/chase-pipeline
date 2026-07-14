"""Environment-compatibility shims.

Importing :mod:`chase._compat` (which happens automatically when you ``import
chase``) makes the package usable in two awkward-but-common situations:

1. **Python builds without the ``_lzma`` C extension.**  ``astropy`` imports
   :mod:`lzma` at load time, so on a pyenv/conda interpreter compiled without
   the xz headers an ``import astropy`` explodes with ``ModuleNotFoundError:
   _lzma``.  We install a minimal stub *before* astropy is imported.  The stub
   carries a real ``__spec__`` because :mod:`importlib` refuses to treat a bare
   ``ModuleType`` as a package.

2. **Headless matplotlib.**  We default ``MPLBACKEND`` to ``Agg`` and point
   ``MPLCONFIGDIR`` at a writable per-user cache so figure generation works on a
   server / in CI with no ``$HOME/.config`` write access.

Both shims are idempotent and only activate when needed, so importing this on a
fully-featured interpreter is a no-op.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

__all__ = ["ensure_lzma_stub", "ensure_headless_matplotlib"]


def ensure_lzma_stub() -> bool:
    """Install a stub :mod:`lzma` module if the real one is unavailable.

    Returns
    -------
    bool
        ``True`` if a stub was installed, ``False`` if the real module works.
    """
    try:  # pragma: no cover - depends on interpreter build
        import lzma  # noqa: F401

        return False
    except ImportError:
        from importlib.machinery import ModuleSpec
        from types import ModuleType

        stub = ModuleType("lzma")
        stub.__spec__ = ModuleSpec("lzma", loader=None)

        class _MissingLZMAFile:  # noqa: D401 - mirror of v50 stub
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "lzma is unavailable in this Python build; rebuild Python "
                    "with the xz/liblzma headers to read .xz-compressed FITS."
                )

        stub.LZMAFile = _MissingLZMAFile
        sys.modules["lzma"] = stub
        return True


def ensure_headless_matplotlib(config_dir: str | os.PathLike | None = None) -> None:
    """Configure matplotlib for headless rendering (Agg + writable cache)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "MPLCONFIGDIR" not in os.environ:
        if config_dir is None:
            config_dir = Path(tempfile.gettempdir()) / "chase_mplconfig"
        cfg = Path(config_dir)
        try:
            cfg.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cfg)
        except OSError:  # pragma: no cover - fall back to matplotlib default
            pass


# Apply on import; safe and idempotent.
ensure_lzma_stub()
ensure_headless_matplotlib()
