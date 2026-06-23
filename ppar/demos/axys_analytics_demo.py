"""Demonstrate Axys-backed analytics and attribution output."""

# Python imports
import os
from importlib.resources import as_file, files
from pathlib import Path
import time

_OUTPUT_DIRECTORY = Path("_demo_output") / "axys_analytics"
os.environ.setdefault("MPLCONFIGDIR", str(_OUTPUT_DIRECTORY / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUTPUT_DIRECTORY / ".cache"))

# Project imports
from ppar.axys import AxysData
from ppar.demos.analytics_outputs import (
    frequency_display_name,
    parse_demo_frequency_argument,
    print_analytics_demo_handoff,
    write_analytics_demo_outputs,
)

_PORTFOLIO_CODE = "MEGA_ALPHA"
_BENCHMARK_CODE = "MEGA_BENCH"


def main() -> None:
    """Run the Axys analytics demonstration with optional frequency selection.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()
    frequency = parse_demo_frequency_argument(
        description="Run the bundled Axys analytics demo.",
    )
    print(f"Using {frequency_display_name(frequency)} reporting.")

    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        axys_data = AxysData(axys_data_root / "axys_analytics.yaml")
        portfolio = axys_data.get_portfolio(_PORTFOLIO_CODE)
        benchmark = axys_data.get_portfolio(_BENCHMARK_CODE)
        analytics = portfolio.to_analytics(
            benchmark,
            frequency=frequency,
        )
        written_paths = write_analytics_demo_outputs(analytics, _OUTPUT_DIRECTORY)
        print_analytics_demo_handoff(_OUTPUT_DIRECTORY, written_paths)

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
