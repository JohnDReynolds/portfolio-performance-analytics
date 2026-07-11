"""Focused validation tests for attribution construction and output limits."""

# Python Imports
import datetime as dt
import unittest
from unittest import mock

# Third-Party Imports
import polars as pl

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
import ppar.analytics.attribution as attribution_module
from ppar.analytics.attribution import View
import ppar.analytics.schema as cols
import ppar.errors as errs
from ppar.errors import PpaError


class TestAttributionValidation(unittest.TestCase):
    """Verify attribution-specific failures outside calculation invariants."""

    def test_return_below_negative_one_raises_error_203(self) -> None:
        """Attribution linking rejects a return less than negative one."""
        invalid_return = pl.DataFrame(
            {
                cols.FROM_DATE: [dt.date(1979, 12, 15)],
                cols.THRU_DATE: [dt.date(1979, 12, 15)],
                cols.IDENTIFIER: ["aapl"],
                cols.RETURN: [-1.0521707668],
                cols.WEIGHT: [1.0],
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

    def test_html_row_limit_accepts_1010_and_rejects_1011(self) -> None:
        """HTML output methods enforce the documented 1,010-row boundary."""
        analytics = Analytics(
            test_util.performance_data_path("Magnificent 7"),
            test_util.performance_data_path("Large-Cap Portfolio"),
        )
        attribution = analytics.get_attribution()

        with (
            mock.patch.object(
                attribution,
                "_fetch_dataframe",
                return_value=pl.DataFrame({"row": range(1_010)}),
            ),
            mock.patch.object(
                attribution_module.html_table,
                "attribution_html",
                return_value="html",
            ),
            mock.patch.object(
                attribution_module.html_table,
                "attribution_table",
                return_value=mock.sentinel.table,
            ),
        ):
            self.assertEqual(attribution.to_html(View.OVERALL_ATTRIBUTION), "html")
            self.assertIs(
                attribution.to_table(View.OVERALL_ATTRIBUTION),
                mock.sentinel.table,
            )

        with mock.patch.object(
            attribution,
            "_fetch_dataframe",
            return_value=pl.DataFrame({"row": range(1_011)}),
        ):
            for output_method in (attribution.to_html, attribution.to_table):
                with self.subTest(output_method=output_method.__name__):
                    with self.assertRaisesRegex(PpaError, errs.ERRORS[204]):
                        output_method(View.OVERALL_ATTRIBUTION)


if __name__ == "__main__":
    unittest.main()
