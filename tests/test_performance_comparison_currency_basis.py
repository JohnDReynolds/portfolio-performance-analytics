"""Tests for normalized monetary-field currency-basis policy."""

from __future__ import annotations

# Python imports
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.currency_basis import (
    CURRENCY_PAIR_BASIS,
    FROM_CURRENCY_BASIS,
    PORTFOLIO_BASE_CURRENCY_BASIS,
    ROW_CURRENCY_BASIS,
    base_currency_monetary_value,
    monetary_field_currency_basis,
    normalize_currency_columns,
)
from ppar.audit.performance_comparison import return_reconstruction as _reconstruction


class TestPerformanceComparisonCurrencyBasis(unittest.TestCase):
    """Validate the normalized naming and calculation boundary."""

    def test_detailed_unqualified_monetary_fields_use_row_currency(self) -> None:
        """Detailed monetary fields without base_ use the row currency."""
        for dataset, source_column in (
            (pc_cols.HOLDINGS, pc_cols.PRICE),
            (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE),
            (pc_cols.HOLDINGS, pc_cols.ACCRUED),
            (pc_cols.TRANSACTIONS, pc_cols.AMOUNT),
        ):
            with self.subTest(dataset=dataset, source_column=source_column):
                self.assertEqual(
                    monetary_field_currency_basis(dataset, source_column),
                    ROW_CURRENCY_BASIS,
                )

    def test_currency_codes_are_normalized_at_the_source_boundary(self) -> None:
        """Currency comparisons use stripped uppercase normalized values."""
        frame = pl.DataFrame(
            {
                pc_cols.CURRENCY: [" eur "],
                pc_cols.BASE_CURRENCY: ["usd"],
                pc_cols.FROM_CURRENCY: [" gbp"],
                pc_cols.TO_CURRENCY: [" usd "],
            }
        )

        normalized = normalize_currency_columns(frame).row(0, named=True)

        self.assertEqual(normalized[pc_cols.CURRENCY], "EUR")
        self.assertEqual(normalized[pc_cols.BASE_CURRENCY], "USD")
        self.assertEqual(normalized[pc_cols.FROM_CURRENCY], "GBP")
        self.assertEqual(normalized[pc_cols.TO_CURRENCY], "USD")

    def test_explicit_and_inherent_base_fields_use_portfolio_currency(self) -> None:
        """Detailed base_ and portfolio-performance values use base currency."""
        for dataset, source_column in (
            (pc_cols.HOLDINGS, pc_cols.BASE_MARKET_VALUE),
            (pc_cols.HOLDINGS, pc_cols.BASE_ACCRUED),
            (pc_cols.TRANSACTIONS, pc_cols.BASE_AMOUNT),
            (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.BEGIN_MARKET_VALUE),
            (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.FLOW),
        ):
            with self.subTest(dataset=dataset, source_column=source_column):
                self.assertEqual(
                    monetary_field_currency_basis(dataset, source_column),
                    PORTFOLIO_BASE_CURRENCY_BASIS,
                )

    def test_fx_rate_basis_comes_from_explicit_pair(self) -> None:
        """FX rates use from/to currencies rather than a base/local prefix."""
        self.assertEqual(
            monetary_field_currency_basis(pc_cols.FX_RATES, pc_cols.FX_RATE),
            CURRENCY_PAIR_BASIS,
        )
        self.assertEqual(
            monetary_field_currency_basis(pc_cols.FX_RATES, pc_cols.LOCAL_EXPOSURE),
            FROM_CURRENCY_BASIS,
        )

    def test_foreign_local_value_never_falls_back_as_base_value(self) -> None:
        """A foreign row requires its explicit base-currency counterpart."""
        row = {
            pc_cols.CURRENCY: "EUR",
            pc_cols.BASE_CURRENCY: "USD",
            pc_cols.AMOUNT: 100.0,
            pc_cols.BASE_AMOUNT: None,
        }

        self.assertIsNone(
            base_currency_monetary_value(
                row,
                local_field=pc_cols.AMOUNT,
                base_field=pc_cols.BASE_AMOUNT,
            )
        )
        with self.assertRaisesRegex(PpaError, "transactions.base_amount"):
            # pylint: disable=protected-access
            _reconstruction._required_base_currency_value(
                row,
                local_field=pc_cols.AMOUNT,
                base_field=pc_cols.BASE_AMOUNT,
                dataset=pc_cols.TRANSACTIONS,
            )

    def test_same_currency_and_legacy_rows_can_use_unqualified_value(self) -> None:
        """Single-currency and legacy extracts retain the concise fallback."""
        for row in (
            {
                pc_cols.CURRENCY: "USD",
                pc_cols.BASE_CURRENCY: "USD",
                pc_cols.MARKET_VALUE: 100.0,
            },
            {pc_cols.MARKET_VALUE: 100.0},
        ):
            with self.subTest(row=row):
                self.assertEqual(
                    base_currency_monetary_value(
                        row,
                        local_field=pc_cols.MARKET_VALUE,
                        base_field=pc_cols.BASE_MARKET_VALUE,
                    ),
                    100.0,
                )


if __name__ == "__main__":
    unittest.main()
