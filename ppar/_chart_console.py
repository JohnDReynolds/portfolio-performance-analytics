"""Console hygiene helpers for chart-rendering entry points."""

from __future__ import annotations

# Python imports
import logging


def quiet_matplotlib_startup() -> None:
    """Suppress known Matplotlib startup chatter in command-line success output."""
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
