"""xxx"""

# Python imports
import datetime as dt
import time

# Project imports
from ppar.analytics import Analytics
from ppar.attribution import View
from ppar.axysdata import AxysData
from ppar.errors import PpaError
import ppar.utilities as util

_CLASSIFICATION_SECURITY = "Security"
time_start = time.perf_counter()

# axys_data = AxysData(
#     "tests/data/axys/axysdata.json",
#     "imex_portperf.csv",
#     "imex_secperf.csv",
#     classification_name="Sector1",  # "Security", "Sector1", "Sector2"
#     mapping_name="SecurityToSector",
#     portfolio_code="PORT_SMALL",
#     # from_date=from_date,
#     # thru_date=thru_date,
# )
# analytics = Analytics(
#     portfolio_data_source=axys_data.secperf,  # .portfolio_data_source,
#     portfolio_name=axys_data.portfolio_name,
# )
# attribution = analytics.get_attribution(
#     classification_name=axys_data.classification_name,
#     classification_data_source=axys_data.classification_data_source,
#     mapping_data_sources=(axys_data.mapping_data_source, axys_data.mapping_data_source),
# )
# html = attribution.to_html(View.OVERALL_ATTRIBUTION)
# util.open_in_browser(html)


for portfolio_code in ["PORT_FAIL_EQUAL", "PORT_FAIL_HIGH", "PORT_LARGE", "PORT_SMALL"]:
    try:
        axys_data = AxysData(
            "tests/data/axys/axysdata.json",
            "imex_portperf.csv",
            "imex_secperf.csv",
            portfolio_code=portfolio_code,
            from_date=dt.date(2024, 1, 1),
            thru_date=dt.date(2024, 12, 31),
            classification_name=_CLASSIFICATION_SECURITY,
        )
    except PpaError as e:
        print(portfolio_code, e)
        continue
    # except Exception as e:
    #     print(portfolio_code, e)
    #     continue

    analytics = Analytics(
        portfolio_data_source=axys_data.secperf,  # .portfolio_data_source,
        portfolio_name=axys_data.portfolio_name,
        portfolio_classification_name=_CLASSIFICATION_SECURITY,
    )

    continue

    # Get the Attribution instance by Security.
    attribution = analytics.get_attribution()  # just for kicks
    attribution = analytics.get_attribution(
        classification_name=_CLASSIFICATION_SECURITY,
        classification_data_source=axys_data.classification_data_source,
    )

    # Get an html string of the overall attribution results by Security.
    html = attribution.to_html(View.OVERALL_ATTRIBUTION)

    # Display the html string in a browser.
    util.open_in_browser(html)

    print(portfolio_code, axys_data.unreconciled_periods)

print("Time:", time.perf_counter() - time_start)  # 0.079 seconds
