"""Focused validation tests for attribution construction and output limits."""

# Python Imports
import datetime as dt
import unittest

# Third-Party Imports
import polars as pl

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import View
import ppar.columns as cols
import ppar.errors as errs
from ppar.errors import PpaError


class TestAttributionValidation(unittest.TestCase):
    """Verify attribution-specific failures outside calculation invariants."""

    def test_return_below_negative_one_raises_error_203(self) -> None:
        """Attribution linking rejects a return less than negative one."""
        invalid_return = pl.DataFrame(
            {
                cols.BEGINNING_DATE: [dt.date(1979, 12, 14)],
                cols.ENDING_DATE: [dt.date(1979, 12, 15)],
                "aapl.ret": [-1.0521707668],
                "aapl.wgt": [1.0],
            }
        )

        with self.assertRaisesRegex(PpaError, errs.ERRORS[203]):
            Analytics(invalid_return, invalid_return.clone()).get_attribution()

    def test_large_detail_html_output_raises_error_204(self) -> None:
        """Overlarge detail HTML tables are rejected before rendering."""
        analytics = Analytics(
            test_util.performance_data_path("Magnificent 7"),
            test_util.performance_data_path("Large-Cap Portfolio"),
        )

        with self.assertRaisesRegex(PpaError, errs.ERRORS[204]):
            analytics.get_attribution().to_html(View.SUBPERIOD_ATTRIBUTION)


if __name__ == "__main__":
    unittest.main()
