"""xxx"""

import datetime as dt
import axys_utilities as axu


portperf, secperf = axu.load_and_validate_portperf_and_secperf(
    "axys/data/imex_portperf.csv",
    "axys/data/imex_secperf.csv",
    portfolio_code="PORT_FAIL_HIGH", # "PORT_FAIL_EQUAL",
    from_date=dt.date(2024, 1, 1),
    thru_date=dt.date(2024, 12, 31),
)

derived_secperf = axu.derive_secperf_for_all_periods(portperf, secperf)
if derived_secperf is None:
    print("Failed to derive secperf for all periods.")
else:
    print(derived_secperf)
