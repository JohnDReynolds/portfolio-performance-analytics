"""As of May 2026, this is a work-in-progress.  It has not been published in the PyPI package."""

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
from ppar.analytics import Analytics  # noqa: E402
from ppar.attribution import View  # noqa: E402
from ppar.axysdata import AxysData  # noqa: E402
import ppar.utilities as util  # noqa: E402

_CLASSIFICATION_NAME = "Sector2"  # "Security", "Sector1", "Sector2"
_MAPPING_NAME = "SecurityToSector" if _CLASSIFICATION_NAME.startswith("Sector") else None
_PORTFOLIO_CODES = ("PORT_SMALL", "PORT_LARGE")
_SECPERF_CLASSIFICATION_NAME = "Security"  # Always "Security"
_AXYS_SPECIFICATIONS_PATH = _REPO_ROOT / "tests" / "data" / "axys_perf" / "axysdata.yaml"


def main() -> None:
    """Run the temporary Axys performance demonstration."""
    time_start = time.perf_counter()

    axys_data = AxysData(
        _AXYS_SPECIFICATIONS_PATH,
        from_date=dt.date(2024, 1, 1),
        thru_date=dt.date(2025, 12, 31),
        portfolio_codes=_PORTFOLIO_CODES,
        classification_names=("Security", "Sector1", "Sector2"),
        mapping_names=_MAPPING_NAME,
    )

    for portfolio_code in _PORTFOLIO_CODES:
        portfolio = axys_data.portfolios[portfolio_code]
        analytics = Analytics(
            portfolio_data_source=portfolio.secperf,
            portfolio_name=portfolio.portfolio_name,
            portfolio_classification_name=_SECPERF_CLASSIFICATION_NAME,
        )

        mapping_data_sources = (
            (
                axys_data.mapping_data_sources[_MAPPING_NAME],
                axys_data.mapping_data_sources[_MAPPING_NAME],
            )
            if _MAPPING_NAME is not None
            else None
        )

        attribution = analytics.get_attribution(
            classification_name=_CLASSIFICATION_NAME,
            classification_data_source=axys_data.classification_data_sources[_CLASSIFICATION_NAME],
            mapping_data_sources=mapping_data_sources,
        )

        html = attribution.to_html(View.OVERALL_ATTRIBUTION)
        util.open_in_browser(html)

    print("Time:", time.perf_counter() - time_start)


if __name__ == "__main__":
    main()
