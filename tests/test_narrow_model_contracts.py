"""Contract tests for the narrow performance and attribution model."""

# Python Imports
import datetime as dt
import math
from collections.abc import Mapping, Sequence
import unittest

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import View
import ppar.columns as cols
from ppar.frequency import Frequency

_Period = tuple[dt.date, dt.date]
_AssetValues = tuple[Sequence[float], Sequence[float]]

_MONTHLY_PERIODS: tuple[_Period, ...] = (
    (dt.date(2023, 12, 31), dt.date(2024, 1, 31)),
    (dt.date(2024, 1, 31), dt.date(2024, 2, 29)),
    (dt.date(2024, 2, 29), dt.date(2024, 3, 31)),
    (dt.date(2024, 3, 31), dt.date(2024, 4, 30)),
    (dt.date(2024, 4, 30), dt.date(2024, 5, 31)),
    (dt.date(2024, 5, 31), dt.date(2024, 6, 30)),
    (dt.date(2024, 6, 30), dt.date(2024, 7, 31)),
    (dt.date(2024, 7, 31), dt.date(2024, 8, 31)),
    (dt.date(2024, 8, 31), dt.date(2024, 9, 30)),
    (dt.date(2024, 9, 30), dt.date(2024, 10, 31)),
    (dt.date(2024, 10, 31), dt.date(2024, 11, 30)),
    (dt.date(2024, 11, 30), dt.date(2024, 12, 31)),
)


def _narrow_performance(
    periods: Sequence[_Period],
    assets: Mapping[str, _AssetValues],
) -> pl.DataFrame:
    """Create narrow performance rows from aligned asset return and weight values.

    Args:
        periods: Beginning and ending dates for each period.
        assets: Asset identifiers mapped to return and weight sequences aligned
            to ``periods``.

    Returns:
        Narrow Polars DataFrame accepted by ``Analytics`` and ``Performance``.
    """
    rows: list[dict[str, dt.date | str | float]] = []
    for period_index, (beginning_date, ending_date) in enumerate(periods):
        for identifier, (returns, weights) in assets.items():
            rows.append(
                {
                    cols.BEGINNING_DATE: beginning_date,
                    cols.ENDING_DATE: ending_date,
                    cols.IDENTIFIER: identifier,
                    cols.RETURN: returns[period_index],
                    cols.WEIGHT: weights[period_index],
                }
            )
    return pl.DataFrame(rows)


def _scalable_narrow_performance(identifier_count: int, period_count: int) -> pl.DataFrame:
    """Create a deterministic narrow workload for scalability assertions.

    Args:
        identifier_count: Number of equally weighted identifiers per period.
        period_count: Number of monthly periods to produce.

    Returns:
        Narrow performance rows with deterministic returns and normalized
        weights.
    """
    weight = 1.0 / identifier_count
    rows: list[dict[str, dt.date | str | float]] = []
    for period_index, (beginning_date, ending_date) in enumerate(
        _MONTHLY_PERIODS[:period_count]
    ):
        for identifier_index in range(identifier_count):
            rows.append(
                {
                    cols.BEGINNING_DATE: beginning_date,
                    cols.ENDING_DATE: ending_date,
                    cols.IDENTIFIER: f"S{identifier_index:04d}",
                    cols.RETURN: ((identifier_index % 11) - 5) / 1000.0
                    + ((period_index % 5) - 2) / 10000.0,
                    cols.WEIGHT: weight,
                }
            )
    return pl.DataFrame(rows)


def _assert_close(test_case: unittest.TestCase, actual: object, expected: float) -> None:
    """Assert that a summarized performance value is a float near its baseline."""
    test_case.assertIsInstance(actual, float)
    assert isinstance(actual, float)
    test_case.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))


class TestNarrowModelContracts(unittest.TestCase):
    """Verify core narrow-input output values and moderate workload behavior."""

    def test_attribution_values_from_narrow_inputs_are_stable(self) -> None:
        """Narrow attribution preserves representative summary and overall values."""
        periods = _MONTHLY_PERIODS[:3]
        portfolio = _narrow_performance(
            periods,
            {
                "A": ([0.08, -0.01, 0.03], [0.70, 0.55, 0.60]),
                "B": ([-0.02, 0.05, 0.01], [0.30, 0.45, 0.40]),
            },
        )
        benchmark = _narrow_performance(
            periods,
            {
                "A": ([0.06, 0.01, 0.02], [0.50, 0.50, 0.50]),
                "B": ([0.00, 0.03, -0.01], [0.50, 0.50, 0.50]),
            },
        )

        attribution = Analytics(portfolio, benchmark).get_attribution()
        summary = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        overall = attribution.to_polars(View.OVERALL_ATTRIBUTION)
        total = overall[-1]

        for actual, expected in zip(
            summary[cols.PORTFOLIO_RETURN],
            [0.05, 0.017, 0.022],
        ):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))
        for actual, expected in zip(
            summary[cols.BENCHMARK_RETURN],
            [0.03, 0.02, 0.005],
        ):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))
        for actual, expected in zip(
            summary[cols.ACTIVE_RETURN],
            [0.02, -0.003, 0.017],
        ):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(
                total[cols.ACTIVE_RETURN].item(),
                0.03548970000000007,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                total[cols.TOTAL_EFFECT_SMOOTHED].item(),
                total[cols.ACTIVE_RETURN].item(),
                abs_tol=1e-12,
            )
        )

    def test_mapping_values_from_narrow_inputs_are_stable(self) -> None:
        """Narrow security rows roll up to mapped attribution totals."""
        portfolio = _narrow_performance(
            _MONTHLY_PERIODS[:1],
            {"A": ([0.10], [0.60]), "B": ([-0.05], [0.40])},
        )
        benchmark = _narrow_performance(
            _MONTHLY_PERIODS[:1],
            {"A": ([0.06], [0.50]), "B": ([0.00], [0.50])},
        )
        analytics = Analytics(
            portfolio,
            benchmark,
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )

        detail = analytics.get_attribution(
            "Sector",
            {"TECH": "Technology"},
            ({"A": "TECH", "B": "TECH"}, {"A": "TECH", "B": "TECH"}),
        ).to_polars(View.SUBPERIOD_ATTRIBUTION)

        self.assertEqual(detail.height, 1)
        self.assertEqual(detail[cols.CLASSIFICATION_IDENTIFIER].item(), "TECH")
        self.assertAlmostEqual(detail[cols.PORTFOLIO_WEIGHT].item(), 1.0)
        self.assertAlmostEqual(detail[cols.PORTFOLIO_RETURN].item(), 0.04)
        self.assertAlmostEqual(detail[cols.BENCHMARK_RETURN].item(), 0.03)
        self.assertAlmostEqual(detail[cols.ACTIVE_RETURN].item(), 0.01)

    def test_consolidation_values_from_narrow_inputs_are_stable(self) -> None:
        """Narrow sub-monthly rows compound to stable monthly totals."""
        periods = (
            (dt.date(2023, 12, 31), dt.date(2024, 1, 15)),
            (dt.date(2024, 1, 15), dt.date(2024, 1, 31)),
            (dt.date(2024, 1, 31), dt.date(2024, 2, 15)),
            (dt.date(2024, 2, 15), dt.date(2024, 2, 29)),
        )
        performance = _narrow_performance(
            periods,
            {"A": ([0.01, 0.02, -0.03, 0.04], [1.0, 1.0, 1.0, 1.0])},
        )

        summary = Analytics(
            performance,
            performance,
            frequency=Frequency.MONTHLY,
        ).get_attribution().to_polars(View.SUBPERIOD_SUMMARY)

        self.assertEqual(summary.height, 2)
        for actual, expected in zip(
            summary[cols.PORTFOLIO_RETURN],
            [0.0302, 0.0088],
        ):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))
        _assert_close(self, summary[cols.ACTIVE_RETURN].abs().max(), 0.0)

    def test_risk_statistics_values_from_narrow_inputs_are_stable(self) -> None:
        """Narrow monthly inputs feed stable risk-statistics results."""
        portfolio = _narrow_performance(
            _MONTHLY_PERIODS,
            {"A": ([0.01, -0.02, 0.03, 0.01, -0.01, 0.02] * 2, [1.0] * 12)},
        )
        benchmark = _narrow_performance(
            _MONTHLY_PERIODS,
            {"A": ([0.005, -0.01, 0.02, 0.015, -0.005, 0.01] * 2, [1.0] * 12)},
        )
        output = Analytics(
            portfolio,
            benchmark,
            frequency=Frequency.MONTHLY,
        ).get_riskstatistics().to_polars()

        portfolio_values = dict(zip(output["column"], output["Portfolio"]))
        benchmark_values = dict(zip(output["column"], output["Benchmark"]))

        self.assertAlmostEqual(portfolio_values["Monthly Mean Return"], 0.006666666666666666)
        self.assertAlmostEqual(benchmark_values["Monthly Mean Return"], 0.005833333333333333)
        self.assertAlmostEqual(portfolio_values["Monthly Tracking Error"], 0.007861650943380503)

    def test_scalable_narrow_workload_preserves_shape_and_period_totals(self) -> None:
        """A moderate narrow workload establishes a future scalability contract."""
        identifier_count = 240
        period_count = 12
        performance = _scalable_narrow_performance(identifier_count, period_count)

        attribution = Analytics(performance, performance).get_attribution()
        detail = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)
        summary = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        expected_returns = (
            performance.group_by(cols.ENDING_DATE)
            .agg((pl.col(cols.RETURN) * pl.col(cols.WEIGHT)).sum().alias("expected"))
            .sort(cols.ENDING_DATE)["expected"]
        )

        self.assertEqual(performance.height, identifier_count * period_count)
        self.assertEqual(detail.height, identifier_count * period_count)
        self.assertEqual(summary.height, period_count)
        for actual, expected in zip(summary[cols.PORTFOLIO_RETURN], expected_returns):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))
        _assert_close(self, summary[cols.ACTIVE_RETURN].abs().max(), 0.0)


if __name__ == "__main__":
    unittest.main()
