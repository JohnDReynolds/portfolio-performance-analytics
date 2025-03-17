"""Sample demonstration script."""

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import Chart, View
import ppar.demo_data_sources as demo_data
from ppar.frequency import Frequency
import ppar.utilities as util


def run_demo(periodicity: str, tables_or_charts: str) -> None:
    """
    Display demo versions of tables or charts in a webbrowser.

    Args:
        periodicity (str): 'm'onthly, 'q'uarterly, or 'y'early
        tables_or_charts (str): If 'c' or 'C', then display charts.  Otherwise display tables.
    """
    # Determine the frequency.
    if len(periodicity) < 1 or periodicity[0] not in ("q", "Q", "y", "Y"):
        frequency = Frequency.MONTHLY
    elif periodicity[0] in ("q", "Q"):
        frequency = Frequency.QUARTERLY
    else:
        frequency = Frequency.YEARLY

    # Determine whether to display tables or charts.
    display_tables = len(tables_or_charts) < 1 or (tables_or_charts[0] not in ("c", "C"))

    # The portfolio and benchmark data sources can be in either of the 2 below layouts.  The
    # weights for each time period must sum to 1.0.  The equation SumOf(weight * return) ==
    # TotalReturn must be satisfied for each time period.  The column names must conform to the
    # ones in the below layouts.  The ordering of the columns or rows does not matter.  The
    # "name" column is optional.
    #     1. Narrow Layout:
    #         beginning_date, ending_date, identifier,        return, weight, name
    #         2023-12-31,      2024-01-31,       AAPL, -0.0422272121,    0.4, Apple Inc.
    #         2023-12-31,      2024-01-31,       MSFT,  0.0572811503,    0.6, Microsoft
    #         2024-01-31,      2024-02-29,       AAPL, -0.019793881,     0.7, Apple Inc.
    #         2024-01-31,      2024-02-29,       MSFT,  0.0403944092,    0.3, Microsoft
    #         ...
    #     2. Wide Layout:
    #         beginning_date, ending_date,      AAPL.ret,     MSFT.ret, AAPL.wgt, MSFT.wgt
    #         2023-12-31,      2024-01-31, -0.0422272121, 0.0572811503,      0.4,      0.6
    #         2024-01-31,      2024-02-29, -0.019793881,  0.0403944092,      0.7,      0.3
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
            beginning_date="2022-12-31",
            ending_date="2024-02-29",
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

    # Set the classification_name for the Attribution.
    classification_name = "Economic Sector"

    # Get the classifiation data source.  Here is sample input data for the classification data
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

    # Get the Attribution instance.
    attribution_by_sector = analytics.get_attribution(
        classification_name,
        classification_data_source,
        mapping_data_sources,
    )

    # Display the ouput in a webbrowser.
    if display_tables:
        # Display selected attribution views in a browser.
        for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
            html = attribution_by_sector.to_html(view)
            util.open_in_browser(html)

        # Get the Attribution instance by Security.  Since no classification name is specified,
        # no Classification or Mapping is necessary.  The security names will be taken from the
        # performance files.
        attribution_by_security = analytics.get_attribution()

        # Get an html string of the overall attribution results by Security.
        html = attribution_by_security.to_html(View.OVERALL_ATTRIBUTION)

        # Display the html string in a browser.
        util.open_in_browser(html)
    else:
        # Display some of the attribution charts in a browser.
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
            util.open_in_browser(png)

    if display_tables:
        # Get the RiskStatistics instance.
        risk_statistics = analytics.get_riskstatistics()

        # Display the risk statistics html in a browser.
        util.open_in_browser(risk_statistics.to_html())

    # Get different formats of the OVERALL_ATTRIBUTION view output.
    view = View.OVERALL_ATTRIBUTION
    _ = attribution_by_sector.to_html(view)  # An html string
    _ = attribution_by_sector.to_json(view)  # A json string
    _ = attribution_by_sector.to_pandas(view)  # A pandas DataFrame
    _ = attribution_by_sector.to_polars(view)  # A polars DataFrame
    _ = attribution_by_sector.to_table(view)  # A "great_table"
    _ = attribution_by_sector.to_xml(view)  # Am xml string
    attribution_by_sector.write_csv(view, "delete_me.csv")  # Write a csv file


########## Run the demo.
if __name__ == "__main__":
    reporting_periodicity = input("Monthly (m), Quarterly (q), or Yearly (y): ")
    display_tables_or_charts = input("Would you like to see tables (t) or charts (c): ")
    run_demo(reporting_periodicity, display_tables_or_charts)
