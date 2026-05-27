"""Demonstrate Axys-backed analytics and attribution output."""

# Imports below the repository path bootstrap are intentional for direct execution.
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

_CLASSIFICATION_NAME = "Sector2"  # "Security", "Sector1", "Sector2"
_PORTFOLIO_CODES = ("PORT_SMALL", "PORT_LARGE")
_AXYS_SPECIFICATIONS_PATH = _REPO_ROOT / "tests" / "data" / "axys_validation" / "axysdata.yaml"


def main() -> None:
    """Run the Axys analytics demonstration.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()

    axys_data = AxysData(
        _AXYS_SPECIFICATIONS_PATH,
    )

    for portfolio_code in _PORTFOLIO_CODES:
        portfolio = axys_data.get_portfolio(
            portfolio_code,
            from_date=dt.date(2024, 1, 1),
            thru_date=dt.date(2025, 12, 31),
            classification_name=_CLASSIFICATION_NAME,
        )
        analytics = portfolio.to_analytics()
        attribution = analytics.get_attribution()

        html = attribution.to_html(View.OVERALL_ATTRIBUTION)
        util.open_in_browser(html)

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
