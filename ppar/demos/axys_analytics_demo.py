"""Demonstrate Axys-backed analytics and attribution output."""

# Python imports
import datetime as dt
from importlib.resources import as_file, files
import time

# Project imports
from ppar.attribution import View
from ppar.axys import AxysData
import ppar.utilities as util


def main() -> None:
    """Run the Axys analytics demonstration.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()

    with as_file(files("ppar.demo_data") / "axys") as axys_data_root:
        axys_data = AxysData(axys_data_root / "axys_column_mappings.yaml")

        for portfolio_code in ("PORT_SMALL", "PORT_LARGE"):
            # Specify dates and classification.
            portfolio = axys_data.get_portfolio(
                portfolio_code,
                from_date=dt.date(2024, 1, 1),
                thru_date=dt.date(2025, 12, 31),
                classification_name="Sector",
            )
            analytics = portfolio.to_analytics()
            attribution = analytics.get_attribution()
            _ = attribution.to_html(View.OVERALL_ATTRIBUTION)

        # Default to the dates and classification in the YAML file.
        portfolio = axys_data.get_portfolio("PORT_SMALL")
        benchmark = axys_data.get_portfolio("PORT_LARGE")
        analytics = portfolio.to_analytics(benchmark)
        attribution = analytics.get_attribution()
        html = attribution.to_html(View.OVERALL_ATTRIBUTION)
        util.open_in_browser(html)  # Total -0.0145 0.0460 0.0315

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
