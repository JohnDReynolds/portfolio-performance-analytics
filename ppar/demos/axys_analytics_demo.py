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
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData
from ppar.demos.analytics_outputs import (
    print_analytics_demo_handoff,
    write_analytics_demo_outputs,
)

_PORTFOLIO_CODE = "MEGA_ALPHA"
_BENCHMARK_CODE = "MEGA_BENCH"
_DEMO_FREQUENCY = Frequency.QUARTERLY


def main() -> None:
    """Run the Axys analytics demonstration.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()

    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        axys_data = AxysData(axys_data_root / "axys_analytics.yaml")
        portfolio = axys_data.get_portfolio(_PORTFOLIO_CODE)
        benchmark = axys_data.get_portfolio(_BENCHMARK_CODE)
        analytics = portfolio.to_analytics(
            benchmark,
            frequency=_DEMO_FREQUENCY,
        )
        written_paths = write_analytics_demo_outputs(analytics, _OUTPUT_DIRECTORY)
        print_analytics_demo_handoff(_OUTPUT_DIRECTORY, written_paths)

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
