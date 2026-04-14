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


# self.assertTrue(
#     _axys_exception(
#         self,
#         errs.ERRORS[503],
#         "error_503_a_portperf.csv",
#         "error_503_a_secperf.csv",
#     )
# )

# q = AxysData(
#     "tests/data/axys/error_503_a_portperf.csv",
#     "tests/data/axys/error_503_a_secperf.csv",
#     axysdata_json_path="tests/data/axys/axysdata.json",
#     # test_util.axys_data_path(secperf_file_name),
#     # portfolio_code=portfolio_code,
#     # from_date=from_date,
#     # thru_date=thru_date,
# )


time_start = time.perf_counter()

for portfolio_code in ["PORT_FAIL_EQUAL", "PORT_FAIL_HIGH", "PORT_LARGE", "PORT_SMALL"]:
    try:
        axys_data = AxysData(
            "tests/data/axys/axysdata.json",
            "tests/data/axys/imex_portperf.csv",
            "tests/data/axys/imex_secperf.csv",
            portfolio_code=portfolio_code,
            from_date=dt.date(2024, 1, 1),
            thru_date=dt.date(2024, 12, 31),
            classification_name="Security",
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
