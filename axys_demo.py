"""xxx"""

# Python imports
import datetime as dt
import time

# Project imports
from ppar.analytics import Analytics
from ppar.attribution import View
from ppar.axysdata import AxysData
import ppar.utilities as util


_CLASSIFICATION_NAME = "Sector2"  # "Security", "Sector1", "Sector2"
_MAPPING_NAME = "SecurityToSector" if _CLASSIFICATION_NAME.startswith("Sector") else None
_PORTFOLIO_CODES = ("PORT_SMALL", "PORT_LARGE")
_SECPERF_CLASSIFICATION_NAME = "Security"  # Always "Security"

time_start = time.perf_counter()

axys_data = AxysData(
    "tests/data/axys/axysdata.yaml",  # json",
    # "imex_portperf.csv",
    # "imex_secperf.csv",
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

    # continue  # 0.077

    # Get an html string of the overall attribution results by Security.
    html = attribution.to_html(View.OVERALL_ATTRIBUTION)

    # Display the html string in a browser.
    util.open_in_browser(html)

print("Time:", time.perf_counter() - time_start)  # 0.079 seconds
