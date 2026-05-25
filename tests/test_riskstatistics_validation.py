"""Focused validation tests for invalid and boundary risk-statistics inputs."""

# Python Imports
import datetime as dt
import math
import unittest

# Third-Party Imports
import numpy as np
import polars as pl

# Project Imports
import ppar.columns as cols
import ppar.errors as errs
from ppar.errors import PpaError
from ppar.frequency import Frequency
from ppar.performance import Performance
from ppar.riskstatistics import RiskStatistics


class TestRiskStatisticsValidation(unittest.TestCase):
    """Verify risk-statistics input rejection and calculation boundaries."""

    def test_unsupported_frequency_raises_error_402(self) -> None:
        """Daily-frequency risk statistics are not supported."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[402]):
            RiskStatistics(
                (np.array([np.nan, np.nan]), np.array([np.nan, np.nan])),
                Frequency.AS_OFTEN_AS_POSSIBLE,
            )

    def test_insufficient_returns_raise_error_403(self) -> None:
        """At least two observations are required to calculate statistics."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[403]):
            RiskStatistics(
                (np.array([np.nan]), np.array([np.nan])),
                Frequency.MONTHLY,
            )

    def test_mismatched_return_counts_raise_error_404(self) -> None:
        """Portfolio and benchmark arrays must contain matching periods."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[404]):
            RiskStatistics(
                (np.array([np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])),
                Frequency.MONTHLY,
            )

    def test_nan_returns_raise_error_405(self) -> None:
        """Missing observations are not valid calculated risk inputs."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[405]):
            RiskStatistics(
                (np.array([1.0, 2.0]), np.array([1.0, np.nan])),
                Frequency.MONTHLY,
            )

    def test_annualized_statistics_are_nan_for_less_than_one_year(self) -> None:
        """Annualized statistics require at least one year of return periods."""
        risk_statistics = RiskStatistics(
            (np.array([1, 2, 3]), np.array([4, 5, 6])),
            Frequency.QUARTERLY,
        )
        output = risk_statistics.to_polars()

        self.assertTrue(math.isnan(output["Portfolio"].item(2)))
        self.assertTrue(math.isnan(output["Benchmark"].item(2)))

    def test_parametric_var_uses_lower_tail_loss(self) -> None:
        """Value at risk reports lower-tail loss rather than upper-tail movement."""
        self.assertTrue(
            math.isclose(
                RiskStatistics._parametric_var(0.01, 0.02, 0.95, 100_000),
                2289.7072539029457,
            )
        )

    def test_performance_inputs_must_have_aligned_dates(self) -> None:
        """Direct Performance inputs require matching reporting periods."""
        portfolio = Performance(
            pl.DataFrame(
                {
                    cols.BEGINNING_DATE: [dt.date(2023, 1, 31), dt.date(2023, 2, 28)],
                    cols.ENDING_DATE: [dt.date(2023, 2, 28), dt.date(2023, 3, 31)],
                    "A.ret": [0.01, 0.02],
                    "A.wgt": [1.0, 1.0],
                }
            )
        )
        benchmark = Performance(
            pl.DataFrame(
                {
                    cols.BEGINNING_DATE: [dt.date(2023, 2, 28), dt.date(2023, 3, 31)],
                    cols.ENDING_DATE: [dt.date(2023, 3, 31), dt.date(2023, 4, 30)],
                    "A.ret": [0.03, 0.04],
                    "A.wgt": [1.0, 1.0],
                }
            )
        )

        with self.assertRaises(PpaError):
            RiskStatistics((portfolio, benchmark), Frequency.MONTHLY)


if __name__ == "__main__":
    unittest.main()
