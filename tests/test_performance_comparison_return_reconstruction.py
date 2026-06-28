"""Tests for portfolio return-reconstruction diagnostics."""

from __future__ import annotations

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.return_reconstruction import (
    DERIVED_RETURN_DIFFERENCE,
    END_VALUE_DIFFERENCE,
    NET_FLOW_DIFFERENCE,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_CATEGORY_FORMULA_DIFFERENCE,
    RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_STATUS_ALIGNED,
    RECONSTRUCTION_STATUS_DIFFERENT,
    RECONSTRUCTION_STATUS_MISSING_INPUTS,
    REPORTED_RETURN_DIFFERENCE,
    WEIGHTED_FLOW_DIFFERENCE,
    portfolio_return_reconstruction_checks,
    return_reconstruction_summary,
    security_return_reconstruction_checks,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison_portfolio.yaml"
)
_BASELINE_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison.yaml"
)
_EXPECTED_PORTFOLIO_RECONSTRUCTION_DIFFERENCES = {
    ("ALPHA", "2026-01-01", "2026-01-30"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("ALPHA", "2026-01-31", "2026-02-27"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("ALPHA", "2026-04-01", "2026-04-30"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("BALANCED", "2026-01-31", "2026-02-27"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("BALANCED", "2026-05-01", "2026-05-29"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("INCOME", "2026-01-31", "2026-02-27"): RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
    ("INCOME", "2026-04-01", "2026-04-30"): RECONSTRUCTION_CATEGORY_FORMULA_DIFFERENCE,
}


class TestPerformanceComparisonReturnReconstruction(unittest.TestCase):
    """Verify portfolio return-reconstruction diagnostics."""

    def test_missing_reconstruction_yaml_returns_empty_table(self) -> None:
        """Comparisons opt into reconstruction explicitly."""
        checks = portfolio_return_reconstruction_checks(_BASELINE_COMPARISON_PATH)
        security_checks = security_return_reconstruction_checks(
            _BASELINE_COMPARISON_PATH
        )
        summary = return_reconstruction_summary(_BASELINE_COMPARISON_PATH)

        self.assertTrue(checks.is_empty())
        self.assertTrue(security_checks.is_empty())
        self.assertTrue(summary.is_empty())

    def test_demo_reconstruction_checks_show_review_statuses(self) -> None:
        """Packaged demo produces aligned, different, and missing-input checks."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertTrue(
            {
                RECONSTRUCTION_STATUS_ALIGNED,
                RECONSTRUCTION_STATUS_DIFFERENT,
                RECONSTRUCTION_STATUS_MISSING_INPUTS,
            }.issubset(statuses)
        )

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertAlmostEqual(alpha_withdrawal[REPORTED_RETURN_DIFFERENCE], 0.0)
        self.assertNotAlmostEqual(
            alpha_withdrawal[REPORTED_RETURN_DIFFERENCE],
            alpha_withdrawal[DERIVED_RETURN_DIFFERENCE],
            places=6,
        )
        self.assertEqual(
            alpha_withdrawal[RECONSTRUCTION_CATEGORY],
            RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
        )

    def test_demo_portfolio_reconstruction_differences_are_intentional(self) -> None:
        """Unexpected reconstructed-return differences flag demo-data drift."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)
        different_rows = checks.filter(
            pl.col(RECONSTRUCTION_STATUS) == RECONSTRUCTION_STATUS_DIFFERENT
        )

        actual = {
            (
                row["portfolio_id"],
                row["from_date"].isoformat(),
                row["thru_date"].isoformat(),
            ): row[RECONSTRUCTION_CATEGORY]
            for row in different_rows.iter_rows(named=True)
        }

        self.assertEqual(actual, _EXPECTED_PORTFOLIO_RECONSTRUCTION_DIFFERENCES)

    def test_demo_security_reconstruction_checks_show_flow_inputs(self) -> None:
        """Security reconstruction treats buy/sell rows as security-level flows."""
        checks = security_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertTrue(
            {
                RECONSTRUCTION_STATUS_ALIGNED,
                RECONSTRUCTION_STATUS_DIFFERENT,
                RECONSTRUCTION_STATUS_MISSING_INPUTS,
            }.issubset(statuses)
        )

        alpha_aapl = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "AAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertGreater(alpha_aapl[END_VALUE_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[WEIGHTED_FLOW_DIFFERENCE], 0.0)
        self.assertEqual(
            alpha_aapl[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_DIFFERENT,
        )
        self.assertEqual(
            alpha_aapl[RECONSTRUCTION_CATEGORY],
            RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED,
        )

    def test_demo_reconstruction_summary_counts_available_checks(self) -> None:
        """Summary table counts portfolio and security reconstruction checks."""
        summary = return_reconstruction_summary(_PORTFOLIO_COMPARISON_PATH)

        check_types = set(summary.get_column("reconstruction_check_type").to_list())
        self.assertEqual(check_types, {"Portfolio Return", "Security Return"})
        self.assertTrue((summary.get_column("row_count") > 0).all())

    def test_malformed_reconstruction_yaml_fails_up_front(self) -> None:
        """Opted-in reconstruction YAML must include every required field."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text(
                "\n".join(
                    [
                        "portfolio_return_reconstruction:",
                        "  method: modified_dietz",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PpaError, "missing required keys"):
                PerformanceComparisonSpecification(path)


if __name__ == "__main__":
    unittest.main()
