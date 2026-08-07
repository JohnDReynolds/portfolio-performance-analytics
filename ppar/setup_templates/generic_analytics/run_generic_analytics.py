"""Explore PPAR Analytics directly from a vendor-neutral Python workflow.

This workspace demonstrates constructing ``Analytics`` directly from generic
performance CSV files. Axys/APX users can instead use ``ppar analytics`` with
the setup-generated ``run_analytics.py``.
"""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys
from typing import Any

# Project imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Chart, View
from ppar.analytics.cli import write_html_file, write_png_file
from ppar.analytics.frequency import Frequency


# Anchor every relative path to this sample folder so the script behaves the
# same way regardless of the current working directory.
SITE_DIRECTORY: Path = Path(__file__).resolve().parent
OUTPUT_DIRECTORY: Path = SITE_DIRECTORY / "output"
CLASSIFICATION_NAME = "Economic Sector"
SECURITY_CLASSIFICATION_PATH = SITE_DIRECTORY / "classifications" / "Security.csv"
HOLIDAYS_PATH = SITE_DIRECTORY / "holidays.csv"


def main(argv: list[str] | None = None) -> int:
    """Create Analytics output from the generic workspace data.

    Returns:
        Process exit code. ``0`` indicates that review files were written.
    """
    _argument_parser().parse_args(argv)
    # ---------------------------------------------------------------------
    # 1. Construct Analytics directly from generic performance files
    # ---------------------------------------------------------------------
    # The generic sample uses explicit CSV paths instead of a ``ppar.yaml`` file.
    # The first file is the managed portfolio and the second is the benchmark.
    analytics: Analytics = Analytics(
        SITE_DIRECTORY / "performance" / "Mega-Cap Alpha Portfolio.csv",
        SITE_DIRECTORY / "performance" / "Mega-Cap Benchmark.csv",
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        frequency=Frequency.QUARTERLY,
        holidays=HOLIDAYS_PATH,
    )

    # These files define how each security rolls up to an economic sector. The
    # same mapping file is used for portfolio and benchmark in this sample.
    classification_data_source: Path = (
        SITE_DIRECTORY / "classifications" / f"{CLASSIFICATION_NAME}.csv"
    )
    mapping_data_sources: tuple[Path, Path] = (
        SITE_DIRECTORY / "mappings" / "Security--to--Economic Sector.csv",
        SITE_DIRECTORY / "mappings" / "Security--to--Economic Sector.csv",
    )

    written_paths: list[Path] = []

    # ---------------------------------------------------------------------
    # 2. Select the security and sector attribution outputs
    # ---------------------------------------------------------------------
    # Start with security attribution to review the individual names behind the
    # active return.
    attribution_by_security: Any = analytics.get_attribution(
        "Security",
        SECURITY_CLASSIFICATION_PATH,
    )
    written_paths.append(
        write_html_file(
            OUTPUT_DIRECTORY,
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    # Then roll securities up to the configured sector classification for the
    # cleaner, presentation-oriented views.
    attribution_by_sector: Any = analytics.get_attribution(
        CLASSIFICATION_NAME,
        classification_data_source,
        mapping_data_sources,
    )
    for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
        written_paths.append(
            write_html_file(
                OUTPUT_DIRECTORY,
                f"sector_{view.name.lower()}.html",
                attribution_by_sector.to_html(view),
            )
        )

    # Write the chart set as PNG files so each chart can be opened or reused on
    # its own.
    for chart in (
        Chart.OVERALL_CONTRIBUTION,
        Chart.OVERALL_ATTRIBUTION,
        Chart.SUBPERIOD_ATTRIBUTION,
        Chart.HEATMAP_ACTIVE_CONTRIBUTION,
        Chart.HEATMAP_ATTRIBUTION,
        Chart.CUMULATIVE_ATTRIBUTION,
        Chart.CUMULATIVE_RETURN,
    ):
        written_paths.append(
            write_png_file(
                OUTPUT_DIRECTORY,
                f"sector_{chart.name.lower()}.png",
                attribution_by_sector.to_chart(chart),
            )
        )

    # Risk statistics are calculated from the same portfolio and benchmark
    # return history used for attribution.
    risk_statistics: Any = analytics.get_riskstatistics()
    written_paths.append(
        write_html_file(
            OUTPUT_DIRECTORY,
            "risk_statistics.html",
            risk_statistics.to_html(),
        )
    )

    # ---------------------------------------------------------------------
    # 3. Hand the review files back to the user
    # ---------------------------------------------------------------------
    print("Open these files to review analytics output:")
    for path in written_paths:
        print(f"  {path.resolve()}")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return help for the Generic Analytics Python example."""
    return argparse.ArgumentParser(
        prog="python run_generic_analytics.py",
        description=(
            "Run the Generic Analytics Python example. This sample has no "
            "command-line settings; customize the visible Python constants and "
            "workflow instead."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
