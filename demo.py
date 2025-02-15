""" Sample demonstration script. """

# Project Imports
from ppa.analytics import Analytics
from ppa.attribution import Attribution, Chart, View
import ppa.demo_data_sources as demo_data
from ppa.frequency import Frequency
import ppa.utilities as util


def _get_attribution(analytics: Analytics, classification_name: str = util.EMPTY) -> Attribution:
    """
    This is a "helper" function that retrieves an Attribution instance.  It can be customized to
    point to different data sources specific to an installation.  This particular function uses
    test classification data and test mapping data.  Alternatively, it could use dictionairies or
    DataFrames populated from another data source.

    Args:
        analytics (Analytics): The Analytics instance.
        classification_name (str, optional): The Classification name. Defaults to util.EMPTY.

    Returns:
        Attribution: The Attribution instance.
    """
    # Get the classification data source.
    classification_data_source = demo_data.classification_data_source(classification_name)

    # Get the mapping data sources.
    mapping_data_sources = demo_data.mapping_data_sources(analytics, classification_name)

    # Return the corresponding attribution instance.
    return analytics.get_attribution(
        classification_name,
        classification_data_source,
        mapping_data_sources,
    )


def _performance_data_source(performance_name: str) -> util.TypePerformanceDataSource:
    """
    This is a "helper" function that returns the performance data source for the given performance
    name.  It can be customized to point to different data sources specific to an installation.
    This particular function points to the file path of a csv file containing the performance data.
    Alternatively, it could return a pandas DataFrame or a polars DataFrame populated from another
    data source.

    Args:
        performance_name (str): The performance name.

    Returns:
        TypePerformanceDataSource: The performance data source.
    """
    return demo_data.performance_data_source(performance_name)


def run_demo(tables_or_charts: str):
    """
    Display demo versions of tables or charts in a webbrowser.

    Args:
        tables_or_charts (str): If 'c' or 'C', then display charts.  Otherwise display tables.
    """
    # Determine whether to display tables or charts.
    display_tables = len(tables_or_charts) < 1 or (tables_or_charts[0] not in ("c", "C"))

    # Get an Analytics instance using performance data from a custom data source (csv files).
    analytics = Analytics(
        portfolio_data_source=_performance_data_source("Large-Cap Alpha Portfolio"),
        benchmark_data_source=_performance_data_source("Large-Cap Benchmark"),
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        beginning_date="2022-12-31",
        ending_date="2024-02-29",
        frequency=Frequency.MONTHLY,
    )

    # Get the Attribution instance by Sector.
    attribution_by_sector = _get_attribution(analytics, "Gics Sector")

    # Display the ouput in a webbrowser.
    if display_tables:
        # Display selected attribution views in a browser.
        for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
            html = attribution_by_sector.to_html(view)
            util.open_in_browser(html)

        # Get the Attribution instance by Security
        attribution_by_security = _get_attribution(analytics, "Security")

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
    display_tables_or_charts = input("Would you like to see tables (t) or charts (c): ")
    run_demo(display_tables_or_charts)
