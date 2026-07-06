"""Run the optional generic analytics sample from Python."""

from __future__ import annotations

# Python imports
import os
from pathlib import Path
import tempfile

# Project imports
from ppar._chart_console import quiet_matplotlib_startup


SITE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output"


def main() -> None:
    """Create analytics output from the optional generic setup data."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
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

            from ppar.analytics import Analytics
            from ppar.analytics.frequency import Frequency
            from ppar.demos.analytics_demo_outputs import (
                print_analytics_demo_handoff,
                write_analytics_demo_outputs,
            )

            analytics = Analytics(
                SITE_DIRECTORY / "performance" / "Mega-Cap Alpha Portfolio.csv",
                SITE_DIRECTORY / "performance" / "Mega-Cap Benchmark.csv",
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
                frequency=Frequency.QUARTERLY,
            )
            written_paths = write_analytics_demo_outputs(
                analytics,
                OUTPUT_DIRECTORY,
                sector_classification_name="Economic Sector",
                sector_classification_data_source=(
                    SITE_DIRECTORY / "classifications" / "Economic Sector.csv"
                ),
                sector_mapping_data_sources=(
                    SITE_DIRECTORY
                    / "mappings"
                    / "Security--to--Economic Sector.csv",
                    SITE_DIRECTORY
                    / "mappings"
                    / "Security--to--Economic Sector.csv",
                ),
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
