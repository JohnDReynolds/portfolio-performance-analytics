"""Understand and customize the standard Performance Analytics workflow.

The normal command is ``ppar analytics ./analytics``. This script produces the
same output while showing the important Python objects and calculation steps.
PPAR handles command-line overrides, YAML validation, and routine file I/O so
this example can focus on the financial workflow.

This script accepts the same command-line options as ``ppar analytics``. Run
``python run_analytics.py -h`` to view them.
"""

from __future__ import annotations

# Python imports
from pathlib import Path
import sys
from typing import Any

# Third-party imports
import polars as pl

# Project imports
from ppar.analytics.attribution import Chart, View
from ppar.analytics.cli import (
    script_run_settings,
    write_html_file,
    write_png_file,
    write_risk_statistics_file,
)
import ppar.analytics.schema as cols
from ppar.axys_apx import AxysData


# All relative paths are anchored to this file. You can therefore run this
# script from any working directory.
SITE_DIRECTORY = Path(__file__).resolve().parent
SPECIFICATIONS_PATH = SITE_DIRECTORY / "ppar.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run the visible Python equivalent of ``ppar analytics``.

    Args:
        argv: Optional command-line overrides excluding the script name. These
            are the same workflow options accepted by ``ppar analytics``.

    Returns:
        Process exit code. ``0`` means all standard outputs were written.
    """
    # ---------------------------------------------------------------------
    # 1. Resolve configuration
    # ---------------------------------------------------------------------
    # PPAR reads the analytics settings in ppar.yaml, then applies any one-run
    # command-line overrides. Inspect ``settings`` in a debugger or print it
    # when you want to see the exact assumptions used for this run.
    settings = script_run_settings(SITE_DIRECTORY, argv)

    # ---------------------------------------------------------------------
    # 2. Load and reconcile the source data
    # ---------------------------------------------------------------------
    # AxysData reads period performance from portperf.csv and secperf.csv, then
    # resolves security names and classifications from secmast.csv as configured
    # in ppar.yaml. get_portfolio() reconciles the selected account and dates.
    source_data = AxysData(SPECIFICATIONS_PATH)
    portfolio = source_data.get_portfolio(
        settings.portfolio_code,
        from_date=settings.from_date,
        thru_date=settings.thru_date,
        classification_name=settings.classification_name,
    )
    benchmark = source_data.get_portfolio(
        settings.benchmark_code,
        from_date=settings.from_date,
        thru_date=settings.thru_date,
        classification_name=settings.classification_name,
    )

    # ---------------------------------------------------------------------
    # 3. Create the Analytics calculation
    # ---------------------------------------------------------------------
    # This is the central customization point. The risk assumptions flow into
    # RiskStatistics; frequency controls period consolidation throughout the
    # attribution and risk outputs.
    analytics = portfolio.to_analytics(
        benchmark,
        frequency=settings.frequency,
        holidays=settings.holidays_path,
        annual_minimum_acceptable_return=settings.annual_minimum_acceptable_return,
        annual_risk_free_rate=settings.annual_risk_free_rate,
        confidence_level=settings.confidence_level,
        portfolio_value=(settings.portfolio_value, settings.currency_symbol),
    )

    # ---------------------------------------------------------------------
    # 4. Select tables and charts
    # ---------------------------------------------------------------------
    # Edit these tuples to add, remove, or reorder standard outputs. This is
    # intentionally visible rather than hidden behind the CLI implementation.
    sector_views = (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION)
    sector_charts = (
        Chart.OVERALL_CONTRIBUTION,
        Chart.OVERALL_ATTRIBUTION,
        Chart.SUBPERIOD_ATTRIBUTION,
        Chart.HEATMAP_ACTIVE_CONTRIBUTION,
        Chart.HEATMAP_ATTRIBUTION,
        Chart.CUMULATIVE_ATTRIBUTION,
        Chart.CUMULATIVE_RETURN,
    )

    # Security attribution answers which holdings drove active performance.
    # Security names are reference data, so explicitly combine the filtered
    # secmast.csv lookups for the portfolio and benchmark before attribution.
    security_classification = pl.concat(
        [
            source_data.get_classification_sources(
                "Security", portfolio
            ).classification_data_source,
            source_data.get_classification_sources(
                "Security", benchmark
            ).classification_data_source,
        ],
        how="vertical",
    ).unique(subset=[cols.IDENTIFIER], keep="any")
    security_attribution: Any = analytics.get_attribution(
        "Security",
        security_classification,
    )

    # With no explicit classification, this uses the default classification
    # source resolved from ppar.yaml (Economic Sector in the starter).
    sector_attribution: Any = analytics.get_attribution()
    written_paths = [
        write_html_file(
            settings.output_directory,
            "security_overall_attribution.html",
            security_attribution.to_html(View.OVERALL_ATTRIBUTION),
        )
    ]
    for view in sector_views:
        written_paths.append(
            write_html_file(
                settings.output_directory,
                f"sector_{view.name.lower()}.html",
                sector_attribution.to_html(view),
            )
        )
    for chart in sector_charts:
        written_paths.append(
            write_png_file(
                settings.output_directory,
                f"sector_{chart.name.lower()}.png",
                sector_attribution.to_chart(chart),
            )
        )

    # RiskStatistics uses the same aligned return stream and assumptions as the
    # attribution calculation above. Native source periods intentionally skip
    # this fixed-frequency report.
    risk_statistics_path = write_risk_statistics_file(
        analytics,
        settings.output_directory,
        settings.frequency,
    )
    if risk_statistics_path is not None:
        written_paths.append(risk_statistics_path)

    # ---------------------------------------------------------------------
    # 5. Hand the review files back to the user
    # ---------------------------------------------------------------------
    print("Open these files to review analytics output:")
    for path in written_paths:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
