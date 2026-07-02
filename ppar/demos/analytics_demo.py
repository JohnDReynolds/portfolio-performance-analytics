"""Demonstrate ppar analytics, attribution, chart, and output features.

This module builds a sample ``Analytics`` instance from bundled demonstration
data, writes attribution tables, charts, and risk statistics to demo output
files, and exercises the main output-format methods.
"""

# Python imports
import os
from pathlib import Path

_OUTPUT_DIRECTORY = Path("_demo_output") / "analytics"
os.environ.setdefault("MPLCONFIGDIR", str(_OUTPUT_DIRECTORY / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUTPUT_DIRECTORY / ".cache"))
_DEFAULT_PERIODICITY = "q"

# Project Imports
from ppar.analytics import Analytics
from ppar.demos.analytics_outputs import (
    frequency_display_name,
    parse_demo_frequency_argument,
    print_analytics_demo_handoff,
    write_analytics_demo_outputs,
)
import ppar.demos.demo_data_sources as demo_data
from ppar.analytics.frequency import Frequency


def run_demo(periodicity: str) -> None:
    """Run the bundled ppar demonstration.

    The demo loads sample Mega-Cap portfolio and benchmark performance data,
    creates attribution results by security and economic sector, and writes
    formatted tables, charts, and ex-post risk statistics to
    ``_demo_output/analytics``.

    Args:
        periodicity: Reporting periodicity selector. Values starting with
            ``"q"`` or ``"Q"`` use quarterly reporting, values starting with
            ``"y"`` or ``"Y"`` use yearly reporting, and all other values use
            monthly reporting.

    Raises:
        PpaError: Propagated from ppar analytics, attribution, performance,
            classification, mapping, or risk-statistics validation if the demo
            data or requested calculations are invalid.
        OSError: Propagated if writing the demonstration CSV output fails.
    """
    frequency = _frequency_from_periodicity(periodicity)

    # Portfolio and benchmark data sources use narrow rows. The weights for each
    # time period must sum to 1.0. The time periods can be of any duration, and
    # column or row order does not matter. The "name" column is optional.
    #     Narrow layout:
    #         from_date, thru_date, identifier,        return, weight, name
    #         2023-12-31,      2024-01-31,       AAPL, -0.0422272121,    0.4, Apple Inc.
    #         2023-12-31,      2024-01-31,       MSFT,  0.0572811503,    0.6, Microsoft
    #         2024-01-31,      2024-02-29,       AAPL, -0.019793881,     0.7, Apple Inc.
    #         2024-01-31,      2024-02-29,       MSFT,  0.0403944092,    0.3, Microsoft
    #         ...
    # The data sources can be in any of the following formats:
    #     1. The path of a csv file containing the performance data.
    #     2. A pandas DataFrame containing the performance data.
    #     3. A polars DataFrame containing the performance data.
    portfolio_data_source = demo_data.performance_data_source(
        "Mega-Cap Alpha Portfolio.csv"
    )
    benchmark_data_source = demo_data.performance_data_source("Mega-Cap Benchmark.csv")

    # Set the classification names of the portfolio and benchmark data sources.
    portfolio_classification_name = "Security"
    benchmark_classification_name = "Security"

    # Get the Analytics instance.
    analytics = Analytics(
        portfolio_data_source,
        benchmark_data_source,
        portfolio_classification_name=portfolio_classification_name,
        benchmark_classification_name=benchmark_classification_name,
        frequency=frequency,
    )

    # Set the classification_name for another Attribution.
    classification_name = "Economic Sector"

    # Get the classification data source.  Here are sample source-data rows for the classification
    # source of an "Economic Sector" classification.  The unique identifier is in the first column,
    # and the name is in the second column.  There are no column headers.
    #     CO, Communication Services
    #     EN, Energy
    #     IT, Information Technology
    #     ...
    # The data source can be in any of the following formats:
    #     1. The path of a csv file containing the Classification data.
    #     2. A python dictionary containing the Classification data.
    #     3. A pandas DataFrame containing the Classification data.
    #     4. A polars DataFrame containing the Classification data.
    classification_data_source = demo_data.classification_data_source(classification_name)

    # Get a tuple of the mapping data sources (portfolio=0, benchmark=1).  They will provide
    # mappings from the classifications in the performance files (e.g. "Security") to the
    # Attribution classification (e.g. "Economic Sector").  Here are sample source-data rows for
    # the "Security" classification to the "Economic Sector" classification.  The unique identifier
    # of the "from" classification is in the first column, and the unique identifier of the "to"
    # classification is in the second column.  There are no column headers.
    #     AAPL, IT
    #     GOOG, CO
    #     XOM,  EN
    #     ...
    # The data source can be in any of the following formats:
    #     1. The path of a csv file containing the Mapping data.
    #     2. A python dictionary containing the Mapping data.
    #     3. A pandas DataFrame containing the Mapping data.
    #     4. A polars DataFrame containing the Mapping data.
    mapping_data_sources = demo_data.mapping_data_sources(analytics, classification_name)

    written_paths = write_analytics_demo_outputs(
        analytics,
        _OUTPUT_DIRECTORY,
        sector_classification_name=classification_name,
        sector_classification_data_source=classification_data_source,
        sector_mapping_data_sources=mapping_data_sources,
    )

    # Write a csv file of the attribution results by sector
    # attribution_by_sector.write_csv(view, file_path="demo_attribution_by_sector.csv")
    print_analytics_demo_handoff(_OUTPUT_DIRECTORY, written_paths)


def main() -> None:
    """Run the bundled ppar demonstration with optional frequency selection.

    Raises:
        PpaError: If analytics or output calculation fails for the bundled
            demonstration data.
        OSError: If demonstration output cannot be written or displayed.
    """
    frequency = parse_demo_frequency_argument(
        description="Run the bundled analytics demo.",
    )
    print(f"Using {frequency_display_name(frequency)} reporting.")
    run_demo(frequency.value)


def _frequency_from_periodicity(periodicity: str) -> Frequency:
    """Return the reporting frequency for an existing demo periodicity string."""
    first_character = (periodicity or _DEFAULT_PERIODICITY).strip().lower()[:1]
    if first_character == "m":
        return Frequency.MONTHLY
    if first_character == "y":
        return Frequency.YEARLY
    return Frequency.QUARTERLY


if __name__ == "__main__":
    main()
