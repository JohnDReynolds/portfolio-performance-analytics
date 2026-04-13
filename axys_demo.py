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

time_start = time.perf_counter()

for portfolio_code in ["PORT_FAIL_EQUAL", "PORT_FAIL_HIGH", "PORT_LARGE", "PORT_SMALL"]:
    try:
        axys_data = AxysData(
            # "tests/data/axys/error_502_portperf.csv",``
            "tests/data/axys/imex_portperf.csv",
            "tests/data/axys/imex_secperf.csv",
            axysdata_json_path="tests/data/axys/axysdata.json",
            portfolio_code=portfolio_code,
            from_date=dt.date(2024, 1, 1),
            thru_date=dt.date(2024, 12, 31),
        )
    except PpaError as e:
        print(portfolio_code, e)
        continue
    # except Exception as e:
    #     print(portfolio_code, e)
    #     continue

    analytics = Analytics(axys_data.secperf, portfolio_name=axys_data.portfolio_name)

    # Get the Attribution instance by Security.
    attribution = analytics.get_attribution()

    # Get an html string of the overall attribution results by Security.
    html = attribution.to_html(View.OVERALL_ATTRIBUTION)

    # Display the html string in a browser.
    util.open_in_browser(html)

    print(portfolio_code, axys_data.unreconciled_periods)

print("Time:", time.perf_counter() - time_start)


###################################################################### OBSOLETE
# import json
# _PORTPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
#     "FROM_DATE",
#     "PORT_RETURN",
#     "PORTFOLIO_CODE",
#     "PORTFOLIO_NAME",
#     "THRU_DATE",
# )

# _SECPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
#     "BEGIN_MV",
#     "BEGIN_WEIGHT",
#     "CONTRIBUTION_W_X_R",
#     "FROM_DATE",
#     "PORTFOLIO_CODE",
#     "SEC_RETURN",
#     "SECURITY_ID",
#     "THRU_DATE",
# )

# dp = {
#     "FROM_DATE": "beginning_date",
#     "THRU_DATE": "ending_date",
#     "PORTFOLIO_CODE": "portfolio_code",
#     "PORTFOLIO_NAME": "portffolio_name",
#     "PORT_RETURN": "portfolio_return",
# }
# ds = {
#     "BEGIN_MV": "begin_mv",
#     "BEGIN_WEIGHT": "begin_weight",
#     "CONTRIBUTION_W_X_R": "contribution_w_x_r",
#     "FROM_DATE": "beginning_date",
#     "PORTFOLIO_CODE": "portfolio_code",
#     "SEC_RETURN": "return",
#     "SECURITY_ID": "identifier",
#     "THRU_DATE": "ending_date",
# }
# dn = {
#     "prefix_portfolio_code": " - ",
# }
# ddd = {
#     "processing_rules": dn,
#     "portperf_columns": dp,
#     "secperf_columns": ds,
# }
# with open("tests/data/axys/axysdata.json", "w", encoding="utf-8") as f:
#     json.dump(
#         ddd,
#         f,
#         indent=4,  # readable formatting
#         # sort_keys=True,    # deterministic output
#     )
