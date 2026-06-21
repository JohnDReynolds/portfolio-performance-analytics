"""Focused in-memory tests for Analytics orchestration and attribution views."""

# Python Imports
import datetime as dt
import unittest

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
import ppar.columns as cols
import ppar.errors as errs
from ppar.errors import PpaError


_PERIODS = [
    (dt.date(2024, 1, 1), dt.date(2024, 1, 31)),
    (dt.date(2024, 2, 1), dt.date(2024, 2, 29)),
    (dt.date(2024, 3, 1), dt.date(2024, 3, 31)),
]


def _two_asset_performance() -> pl.DataFrame:
    """Return three periods of narrow-format performance data."""
    return pl.DataFrame(
        {
            cols.FROM_DATE: [period[0] for period in _PERIODS for _ in ("A", "B")],
            cols.THRU_DATE: [period[1] for period in _PERIODS for _ in ("A", "B")],
            cols.IDENTIFIER: ["A", "B"] * len(_PERIODS),
            cols.RETURN: [0.10, -0.05, 0.03, 0.02, -0.02, 0.04],
            cols.WEIGHT: [0.60, 0.40] * len(_PERIODS),
        }
    )


class TestAnalyticsContracts(unittest.TestCase):
    """Verify public orchestration behavior without external test data files."""

    def test_missing_benchmark_defaults_to_portfolio_classification_and_results(self) -> None:
        """A portfolio-only Analytics instance uses itself as its benchmark."""
        analytics = Analytics(
            _two_asset_performance(),
            portfolio_classification_name="Security",
        )

        attribution = analytics.get_attribution()
        summary = attribution.to_polars(View.SUBPERIOD_SUMMARY)

        self.assertEqual(analytics.classification_names(), ("Security", "Security"))
        self.assertTrue((summary[cols.ACTIVE_RETURN] == 0.0).all())
        self.assertTrue((summary[cols.TOTAL_EFFECT_SIMPLE] == 0.0).all())

    def test_date_window_keeps_only_periods_within_requested_bounds(self) -> None:
        """Date parameters constrain the aligned reportable periods."""
        analytics = Analytics(
            _two_asset_performance(),
            from_date="2024-02-01",
            thru_date=dt.date(2024, 3, 31),
        )

        summary = analytics.get_attribution().to_polars(View.SUBPERIOD_SUMMARY)

        self.assertEqual(
            summary[cols.FROM_DATE].to_list(),
            [dt.date(2024, 2, 1), dt.date(2024, 3, 1)],
        )
        self.assertEqual(
            summary[cols.THRU_DATE].to_list(),
            [dt.date(2024, 2, 29), dt.date(2024, 3, 31)],
        )

    def test_different_known_classifications_require_requested_target(self) -> None:
        """A caller must choose a target classification for unlike inputs."""
        analytics = Analytics(
            _two_asset_performance(),
            _two_asset_performance(),
            portfolio_classification_name="Security",
            benchmark_classification_name="Sector",
        )

        with self.assertRaisesRegex(PpaError, errs.ERRORS[252]):
            analytics.get_attribution()

    def test_repeated_attribution_retrieval_reuses_cached_instance(self) -> None:
        """Repeated requests for a classification reuse calculated attribution."""
        analytics = Analytics(
            _two_asset_performance(),
            portfolio_classification_name="Security",
        )

        first = analytics.get_attribution()
        second = analytics.get_attribution()

        self.assertIs(first, second)

    def test_detail_view_zero_fills_identifier_missing_from_benchmark(self) -> None:
        """Attribution aligns asymmetric holdings on one classification grid."""
        portfolio = _two_asset_performance().head(2)
        benchmark = portfolio.filter(pl.col(cols.IDENTIFIER) == "A").with_columns(
            pl.lit(1.0).alias(cols.WEIGHT)
        )

        detail = Analytics(portfolio, benchmark).get_attribution().to_polars(
            View.SUBPERIOD_ATTRIBUTION
        )
        b_row = detail.filter(pl.col(cols.CLASSIFICATION_IDENTIFIER) == "B")

        self.assertEqual(b_row.height, 1)
        self.assertEqual(b_row[cols.BENCHMARK_WEIGHT].item(), 0.0)
        self.assertEqual(b_row[cols.BENCHMARK_RETURN].item(), 0.0)
        self.assertEqual(b_row[cols.BENCHMARK_CONTRIB_SIMPLE].item(), 0.0)

    def test_total_rows_are_appended_only_to_aggregate_views(self) -> None:
        """Cumulative and overall views end in totals; detail views do not."""
        attribution = Analytics(_two_asset_performance()).get_attribution()

        cumulative = attribution.to_polars(View.CUMULATIVE_ATTRIBUTION)
        overall = attribution.to_polars(View.OVERALL_ATTRIBUTION)
        summary = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        detail = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)

        self.assertEqual(cumulative[cols.THRU_DATE].item(-1), "Total")
        self.assertEqual(overall[cols.CLASSIFICATION_NAME].item(-1), "Total")
        self.assertEqual(summary.height, len(_PERIODS))
        self.assertEqual(detail.height, 2 * len(_PERIODS))

    def test_overall_sorting_leaves_total_row_at_end(self) -> None:
        """Sorting orders holdings before the appended overall total row."""
        attribution = Analytics(_two_asset_performance()).get_attribution()

        overall = attribution.to_polars(
            View.OVERALL_ATTRIBUTION,
            columns_to_sort=cols.PORTFOLIO_CONTRIB_SMOOTHED,
            sort_descendings=True,
        )
        values = overall[cols.PORTFOLIO_CONTRIB_SMOOTHED][:-1].to_list()

        self.assertEqual(overall[cols.CLASSIFICATION_NAME].item(-1), "Total")
        self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
