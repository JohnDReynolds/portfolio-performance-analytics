"""Run the local PPAR analytics setup from Python.

This script is installed by ``ppar setup`` next to ``ppar.yaml`` and the
analytics CSV files. It is intentionally small: keep site-specific choices in
``ppar.yaml`` so scheduled jobs can stay stable while configuration changes.
"""

from __future__ import annotations

# Python imports
import os
from pathlib import Path
import tempfile

# Project imports
from ppar._chart_console import quiet_matplotlib_startup


SITE_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = SITE_DIRECTORY / "ppar.yaml"
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output"

# These defaults match the starter ``ppar.yaml``. You can change the values here
# for an automation-specific override, but most sites should change the YAML.
PORTFOLIO_CODE = "MEGA_ALPHA"
BENCHMARK_CODE = "MEGA_BENCH"
FREQUENCY = "quarterly"
CLASSIFICATION_NAME = "Economic Sector"


def main() -> None:
    """Create analytics output from the local setup-site CSV files."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Chart rendering libraries can create small cache files. Keep those cache
    # files in a temporary folder so the setup directory stays clean.
    original_cache_env = {
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="ppar_chart_cache_") as cache_directory:
            cache_path = Path(cache_directory)
            os.environ.setdefault("MPLCONFIGDIR", str(cache_path / "matplotlib"))
            os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "cache"))
            quiet_matplotlib_startup()

            # Import after chart cache settings are in place. Some chart
            # dependencies read cache-related environment variables at import.
            from ppar.axys import AxysData
            from ppar.demos.analytics_demo_outputs import (
                demo_frequency_from_string,
                print_analytics_demo_handoff,
                write_analytics_demo_outputs,
            )

            source_data = AxysData(CONFIG_PATH)
            portfolio = source_data.get_portfolio(PORTFOLIO_CODE)
            benchmark = source_data.get_portfolio(BENCHMARK_CODE)
            analytics = portfolio.to_analytics(
                benchmark,
                frequency=demo_frequency_from_string(FREQUENCY),
            )
            written_paths = write_analytics_demo_outputs(
                analytics,
                OUTPUT_DIRECTORY,
                sector_classification_name=CLASSIFICATION_NAME,
            )
            print_analytics_demo_handoff(OUTPUT_DIRECTORY, written_paths)
    finally:
        for key, original_value in original_cache_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


if __name__ == "__main__":
    main()
