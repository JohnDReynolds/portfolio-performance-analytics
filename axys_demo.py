"""xxx"""

import datetime as dt
import time
from ppar.axysdata import AxysData
from ppar.errors import PpaError

time_start = time.perf_counter()

for portfolio_code in ["PORT_FAIL_EQUAL", "PORT_FAIL_HIGH", "PORT_LARGE", "PORT_SMALL"]:
    try:
        axys_data = AxysData(
            # "tests/data/axys/error_502_portperf.csv",``
            "tests/data/axys/imex_portperf.csv",
            "tests/data/axys/imex_secperf.csv",
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
    print(portfolio_code, axys_data.unreconciled_periods)

print("Time:", time.perf_counter() - time_start)
