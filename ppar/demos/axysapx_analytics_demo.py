"""Demonstrate Axys/APX-backed analytics and attribution output."""

# Python imports
import os
from importlib.resources import as_file, files
from pathlib import Path

# Project imports
from ppar._chart_console import quiet_matplotlib_startup

_OUTPUT_DIRECTORY = Path("_demo_output") / "axysapx_analytics"
os.environ.setdefault("MPLCONFIGDIR", str(_OUTPUT_DIRECTORY / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUTPUT_DIRECTORY / ".cache"))
quiet_matplotlib_startup()

# Project imports that may initialize chart-rendering dependencies.
from ppar.axys import AxysData
from ppar.demos.analytics_demo_outputs import (
    parse_demo_frequency_argument,
    print_analytics_demo_handoff,
    write_analytics_demo_outputs,
)

_PORTFOLIO_CODE = "MEGA_ALPHA"
_BENCHMARK_CODE = "MEGA_BENCH"


def main() -> None:
    """Run the Axys/APX analytics demonstration with optional frequency selection.

    Raises:
        PpaError: If Axys/APX source validation, reconciliation, or analytics
            calculations fail.
    """
    frequency = parse_demo_frequency_argument(
        description="Run the bundled Axys/APX analytics demo.",
    )

    with as_file(files("ppar.demos.data") / "axysapx_analytics") as axys_data_root:
        axys_data = AxysData(axys_data_root / "axysapx_analytics.yaml")
        portfolio = axys_data.get_portfolio(_PORTFOLIO_CODE)
        benchmark = axys_data.get_portfolio(_BENCHMARK_CODE)
        analytics = portfolio.to_analytics(
            benchmark,
            frequency=frequency,
        )
        written_paths = write_analytics_demo_outputs(analytics, _OUTPUT_DIRECTORY)
        print_analytics_demo_handoff(_OUTPUT_DIRECTORY, written_paths)


if __name__ == "__main__":
    main()
