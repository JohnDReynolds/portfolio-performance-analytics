"""Demonstrate Axys-backed analytics and attribution output."""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python imports
import datetime as dt
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Project imports
from ppar.attribution import View  # noqa: E402
from ppar.axys import AxysData  # noqa: E402
import ppar.utilities as util  # noqa: E402


def main() -> None:
    """Run the Axys analytics demonstration.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()

    axys_data = AxysData(_REPO_ROOT / "tests" / "data" / "axys_validation" / "ppar.yaml")

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
        html = attribution.to_html(View.OVERALL_ATTRIBUTION)
        # util.open_in_browser(html)

    # Default to the dates and classification in the YAML file.
    portfolio = axys_data.get_portfolio("PORT_SMALL")
    benchmark = axys_data.get_portfolio("PORT_LARGE")
    analytics = portfolio.to_analytics(benchmark)
    attribution = analytics.get_attribution()
    html = attribution.to_html(View.OVERALL_ATTRIBUTION)
    util.open_in_browser(html)  # Total −0.0145	0.0460	0.0315

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
