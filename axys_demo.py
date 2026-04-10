"""xxx"""

import datetime as dt
from ppar.axysdata import AxysData

# import axys.axys_util as axu
# portperf, secperf, derived_secperf = axu.load_and_validate_portperf_and_secperf(

axys_data = AxysData(
    "tests/data/axys/imex_portperf.csv",
    # "tests/data/axys/error_502_portperf.csv",
    "tests/data/axys/imex_secperf.csv",
    portfolio_code="PORT_FAIL_EQUAL",  # "PORT_SMALL",  #   # "PORT_FAIL_EQUAL",
    from_date=dt.date(2024, 1, 1),
    thru_date=dt.date(2024, 12, 31),
)
print(axys_data.derived_secperf)

# derived_secperf = axu._derive_secperf_for_all_periods(portperf, secperf)
# if derived_secperf is None:
#     print("Failed to derive secperf for all periods.")
# else:
#     print(derived_secperf)
