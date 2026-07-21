"""Prepare quiet, writable chart rendering for Analytics entry points."""

from __future__ import annotations

# Python imports
import logging
import os
from pathlib import Path
import tempfile


def quiet_matplotlib_startup() -> None:
    """Suppress known Matplotlib startup chatter in command-line success output."""
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def prepare_chart_rendering() -> None:
    """Set quiet, writable chart-rendering defaults before importing Matplotlib."""
    cache_root = Path(tempfile.gettempdir()) / "ppar_chart_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))
    quiet_matplotlib_startup()
