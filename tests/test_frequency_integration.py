"""Integration tests for consolidation across dates and report frequencies."""

# Legacy result checks exercise internal calculated frames.
# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

# Python Imports
import datetime as dt
import unittest

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
import ppar.schema as cols
from ppar.analytics.frequency import Frequency
from ppar.analytics.performance import Performance
import ppar.utilities as util


class TestFrequencyIntegration(unittest.TestCase):
    """Verify fixture-based consolidation and date-window workflows."""

    def test_crazy_frequency(self) -> None:
        """Irregular portfolio and benchmark frequency inputs align correctly."""
        analytics = Analytics(
            test_util.performance_data_path("case_mixed_frequency"),
            test_util.performance_data_path("case_crazy_frequency"),
        )

        self.assertEqual(
            len(test_util.get_attribution(analytics).to_pandas(View.SUBPERIOD_SUMMARY)),
            3,
        )

    def test_daily_to_monthly(self) -> None:
        """Daily performance consolidates to expected monthly attribution values."""
        analytics = Analytics(
            test_util.performance_data_path("big2_daily"),
            test_util.performance_data_path("Big 2"),
            from_date=dt.date(2021, 1, 1),
            frequency=Frequency.MONTHLY,
        )
        attribution = test_util.get_attribution(analytics)
        output = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)

        self.assertEqual(output[cols.FROM_DATE].item(0), dt.date(2021, 1, 1))
        self.assertEqual(output[cols.THRU_DATE].item(4), dt.date(2021, 3, 31))
        self.assertTrue(
            util.are_near(output[cols.TOTAL_EFFECT_SIMPLE].item(3), 0.0012545960452570828)
        )
        self.assertTrue(
            util.are_near(output[cols.SELECTION_EFFECT_SIMPLE].item(14), 0.001057705826113624)
        )

        detail = attribution._construct_df_for_detail_views(View.SUBPERIOD_ATTRIBUTION).collect()
        self.assertTrue(
            util.are_near(detail[cols.TOTAL_EFFECT_SMOOTHED].item(3), 0.002038295249203867)
        )
        self.assertTrue(
            util.are_near(detail[cols.SELECTION_EFFECT_SMOOTHED].item(14), 0.0015709213702753996)
        )

    def test_daily_to_quarterly(self) -> None:
        """Daily performance consolidates to expected quarterly attribution values."""
        analytics = Analytics(
            test_util.performance_data_path("big2_daily"),
            test_util.performance_data_path("Big 2"),
            from_date=dt.date(2021, 1, 1),
            frequency=Frequency.QUARTERLY,
        )
        attribution = test_util.get_attribution(analytics)
        output = attribution.to_polars(View.SUBPERIOD_SUMMARY)

        self.assertEqual(output[cols.FROM_DATE].item(0), dt.date(2021, 1, 1))
        self.assertEqual(output[cols.THRU_DATE].item(4), dt.date(2022, 3, 31))
        self.assertTrue(
            util.are_near(output[cols.TOTAL_EFFECT_SIMPLE].item(3), -0.0020721529010043226)
        )
        self.assertTrue(util.are_near(output[cols.PORTFOLIO_RETURN].item(8), 0.2401702546346276))
        self.assertTrue(
            util.are_near(
                attribution._df[cols.TOTAL_EFFECT_SMOOTHED].item(3), -0.002740959239265768
            )
        )

    def test_map_mixed_frequency(self) -> None:
        """Mixed-frequency inputs map correctly to an economic-sector view."""
        analytics = Analytics(
            test_util.performance_data_path("Magnificent 7"),
            test_util.performance_data_path("economic_sector_daily"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Economic Sector",
        )
        attribution = test_util.get_attribution(analytics, "Economic Sector")
        classifications = attribution.to_polars(View.OVERALL_ATTRIBUTION)[
            cols.CLASSIFICATION_IDENTIFIER
        ]

        self.assertEqual(classifications[:6].to_list(), ["CD", "CO", "EN", "HC", "IT", "MA"])
        self.assertEqual(attribution.to_polars(View.SUBPERIOD_SUMMARY).shape, (141, 11))

    def test_mixed_frequency(self) -> None:
        """Mixed-frequency input files align to their shared reporting periods."""
        analytics = Analytics(
            test_util.performance_data_path("case_mixed_frequency"),
            test_util.performance_data_path("case_monthly_frequency"),
        )

        self.assertEqual(
            len(test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)),
            3,
        )

    def test_monthly_to_yearly(self) -> None:
        """Monthly input consolidates to yearly reporting periods."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("big2_daily"),
            from_date=dt.date(2021, 1, 1),
            frequency=Frequency.YEARLY,
        )
        output = test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)

        self.assertEqual(len(output), 3)
        self.assertEqual(output[cols.FROM_DATE].item(0), dt.date(2021, 1, 1))
        self.assertEqual(output[cols.THRU_DATE].item(2), dt.date(2023, 12, 31))

    def test_specify_dates(self) -> None:
        """Explicit dates filter the fixture performance rows inclusively."""
        performance = Performance(
            test_util.performance_data_path("case_adjust_from_dates"),
            from_date="2023-01-31",
            thru_date="2023-02-28",
        )

        self.assertEqual(
            performance.period_totals()[cols.FROM_DATE].item(0), dt.date(2023, 1, 2)
        )
        self.assertEqual(
            performance.period_totals()[cols.THRU_DATE].to_list(),
            [dt.date(2023, 1, 31), dt.date(2023, 2, 12), dt.date(2023, 2, 28)],
        )


if __name__ == "__main__":
    unittest.main()
