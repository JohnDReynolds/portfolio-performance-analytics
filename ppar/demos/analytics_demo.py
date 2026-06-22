"""Demonstrate ppar analytics, attribution, chart, and output features.

This module builds a sample ``Analytics`` instance from bundled demonstration
data, writes attribution tables, charts, and risk statistics to demo output
files, and exercises the main output-format methods.
"""

# Python imports
from pathlib import Path

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Chart, View
import ppar.demos.demo_data_sources as demo_data
from ppar.analytics.frequency import Frequency
import ppar.utilities as util

_OUTPUT_DIRECTORY = Path("_demo_output") / "analytics"


def run_demo(periodicity: str) -> None:
    """Run the bundled ppar demonstration.

    The demo loads sample portfolio and benchmark performance data, optionally
    filters the monthly demo date range, creates attribution results by security
    and economic sector, and writes formatted tables, charts, and ex-post risk
    statistics to ``_demo_output/analytics``.

    Args:
        periodicity: Reporting periodicity selector. Values from with
            ``"q"`` or ``"Q"`` use quarterly reporting, values from with
            ``"y"`` or ``"Y"`` use yearly reporting, and all other values use
            monthly reporting.

    Raises:
        PpaError: Propagated from ppar analytics, attribution, performance,
            classification, mapping, or risk-statistics validation if the demo
            data or requested calculations are invalid.
        OSError: Propagated if writing the demonstration CSV output fails.
    """
    # Determine the frequency.
    if len(periodicity) < 1 or periodicity[0] not in ("q", "Q", "y", "Y"):
        frequency = Frequency.MONTHLY
    elif periodicity[0] in ("q", "Q"):
        frequency = Frequency.QUARTERLY
    else:
        frequency = Frequency.YEARLY

    written_paths: list[Path] = []

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
    portfolio_data_source = demo_data.performance_data_source("Large-Cap Alpha Portfolio.csv")
    benchmark_data_source = demo_data.performance_data_source("Large-Cap Benchmark.csv")

    # Set the classificcation names of the portfolio and benchmark data sources.
    portfolio_classification_name = "Security"
    benchmark_classification_name = "Security"

    # Get the Analytics instance.
    if frequency == Frequency.MONTHLY:
        # Filter on dates.
        analytics = Analytics(
            portfolio_data_source,
            benchmark_data_source,
            portfolio_classification_name=portfolio_classification_name,
            benchmark_classification_name=benchmark_classification_name,
            from_date="2023-01-01",
            thru_date="2024-02-29",
            frequency=frequency,
        )
    else:
        # Do not filter on dates.
        analytics = Analytics(
            portfolio_data_source,
            benchmark_data_source,
            portfolio_classification_name=portfolio_classification_name,
            benchmark_classification_name=benchmark_classification_name,
            frequency=frequency,
        )

    # Get the Attribution instance by Security.
    attribution_by_security = analytics.get_attribution()

    # Get an html string of the overall attribution results by Security.
    html = attribution_by_security.to_html(View.OVERALL_ATTRIBUTION)

    written_paths.append(_write_html("security_overall_attribution.html", html))

    # Set the classification_name for another Attribution.
    classification_name = "Economic Sector"

    # Get the classification data source.  Here is sample input data for the classification data
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
    # Attribution classification (e.g. "Economic Sector").  Here is sample input data for mapping
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

    # Get the Attribution by economic sector.
    attribution_by_sector = analytics.get_attribution(
        classification_name,
        classification_data_source,
        mapping_data_sources,
    )

    # Write selected attribution views.
    for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
        html = attribution_by_sector.to_html(view)
        written_paths.append(_write_html(f"sector_{view.name.lower()}.html", html))

    # Write selected attribution charts.
    for chart in (
        Chart.OVERALL_CONTRIBUTION,
        Chart.OVERALL_ATTRIBUTION,
        Chart.SUBPERIOD_ATTRIBUTION,
        Chart.HEATMAP_ACTIVE_CONTRIBUTION,
        Chart.HEATMAP_ATTRIBUTION,
        Chart.CUMULATIVE_ATTRIBUTION,
        Chart.CUMULATIVE_RETURN,
    ):
        png = attribution_by_sector.to_chart(chart)
        written_paths.append(_write_png(f"sector_{chart.name.lower()}.png", png))

    # Get the RiskStatistics instance.
    risk_statistics = analytics.get_riskstatistics()

    written_paths.append(_write_html("risk_statistics.html", risk_statistics.to_html()))

    # Get different formats of the OVERALL_ATTRIBUTION view output.
    view = View.OVERALL_ATTRIBUTION
    _ = attribution_by_sector.to_html(view)  # An html string
    _ = attribution_by_sector.to_json(view)  # A json string
    _ = attribution_by_sector.to_pandas(view)  # A pandas DataFrame
    _ = attribution_by_sector.to_polars(view)  # A polars DataFrame
    table = attribution_by_sector.to_table(view)  # A lightweight HTML table object.
    _ = table.as_raw_html(make_page=False)  # An html fragment without <html>/<body> tags.
    _ = attribution_by_sector.to_xml(view)  # Am xml string

    # Write a csv file of the attribution results by sector
    # attribution_by_sector.write_csv(view, file_path="demo_attribution_by_sector.csv")
    _print_demo_handoff(written_paths)


def _write_html(file_name: str, html: str) -> Path:
    """Write one demo HTML artifact."""
    path = _OUTPUT_DIRECTORY / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _write_png(file_name: str, png: bytes) -> Path:
    """Write one demo PNG artifact."""
    path = _OUTPUT_DIRECTORY / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _print_demo_handoff(paths: list[Path]) -> None:
    """Print generated analytics demo artifact paths."""
    print(f"Analytics demo output written to: {_OUTPUT_DIRECTORY.resolve()}")
    if paths:
        print("Open these files to review the demo output:")
        for path in paths:
            print(f"- {path.resolve()}")


def main() -> None:
    """Prompt for reporting periodicity and run the bundled ppar demonstration.

    Raises:
        PpaError: If analytics or output calculation fails for the bundled
            demonstration data.
        OSError: If demonstration output cannot be written or displayed.
    """
    reporting_periodicity = input("Monthly (m), Quarterly (q), or Yearly (y): ")
    run_demo(reporting_periodicity)


if __name__ == "__main__":
    main()
