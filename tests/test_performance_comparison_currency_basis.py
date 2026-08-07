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
    PORTFOLIO_BASE_CURRENCY_BASIS,
    ROW_CURRENCY_BASIS,
    base_currency_monetary_value,
    monetary_field_currency_basis,
    normalize_currency_columns,
)
from ppar.audit.performance_comparison.compare import _implied_conversion_message
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
            }
        )

        normalized = normalize_currency_columns(frame).row(0, named=True)

        self.assertEqual(normalized[pc_cols.CURRENCY], "EUR")
        self.assertEqual(normalized[pc_cols.BASE_CURRENCY], "USD")

    def test_explicit_and_inherent_base_fields_use_portfolio_currency(self) -> None:
        """Detailed base_ values use portfolio base currency."""
        for dataset, source_column in (
            (pc_cols.HOLDINGS, pc_cols.BASE_MARKET_VALUE),
            (pc_cols.HOLDINGS, pc_cols.BASE_ACCRUED),
            (pc_cols.TRANSACTIONS, pc_cols.BASE_AMOUNT),
        ):
            with self.subTest(dataset=dataset, source_column=source_column):
                self.assertEqual(
                    monetary_field_currency_basis(dataset, source_column),
                    PORTFOLIO_BASE_CURRENCY_BASIS,
                )

    def test_foreign_base_value_reports_row_level_implied_conversion(self) -> None:
        """A valid local/base pair produces concise conversion evidence."""
        row = {
            pc_cols.CURRENCY: "GBP",
            pc_cols.BASE_CURRENCY: "USD",
            f"{pc_cols.CURRENCY}_b": "GBP",
            f"{pc_cols.BASE_CURRENCY}_b": "USD",
            pc_cols.MARKET_VALUE: 13_236.02,
            f"{pc_cols.MARKET_VALUE}_b": 13_236.02,
            pc_cols.BASE_MARKET_VALUE: 16_783.28,
            f"{pc_cols.BASE_MARKET_VALUE}_b": 17_103.28,
        }

        message = _implied_conversion_message(
            row,
            pc_cols.HOLDINGS,
            pc_cols.BASE_MARKET_VALUE,
        )

        self.assertEqual(
            message,
            "Local market value remained GBP 13,236.02. The implied conversion "
            "ratio increased from 1.268001 to 1.292177 USD per GBP.",
        )

    def test_transaction_conversion_evidence_remains_on_its_source_row(self) -> None:
        """Transaction ratios use that transaction's own amount pair."""
        row = {
            pc_cols.CURRENCY: "EUR",
            pc_cols.BASE_CURRENCY: "USD",
            f"{pc_cols.CURRENCY}_b": "EUR",
            f"{pc_cols.BASE_CURRENCY}_b": "USD",
            pc_cols.AMOUNT: 100.0,
            f"{pc_cols.AMOUNT}_b": 110.0,
            pc_cols.BASE_AMOUNT: 108.0,
            f"{pc_cols.BASE_AMOUNT}_b": 121.0,
        }

        message = _implied_conversion_message(
            row,
            pc_cols.TRANSACTIONS,
            pc_cols.BASE_AMOUNT,
        )

        self.assertEqual(
            message,
            "Local amount changed from EUR 100.00 to EUR 110.00. The implied "
            "conversion ratio increased from 1.080000 to 1.100000 USD per EUR.",
        )

    def test_unsafe_value_pairs_do_not_produce_conversion_evidence(self) -> None:
        """Same-currency, zero, mismatched, and conflicting pairs fail closed."""
        base_row = {
            pc_cols.CURRENCY: "EUR",
            pc_cols.BASE_CURRENCY: "USD",
            f"{pc_cols.CURRENCY}_b": "EUR",
            f"{pc_cols.BASE_CURRENCY}_b": "USD",
            pc_cols.AMOUNT: 100.0,
            f"{pc_cols.AMOUNT}_b": 100.0,
            pc_cols.BASE_AMOUNT: 108.0,
            f"{pc_cols.BASE_AMOUNT}_b": 109.0,
        }
        unsafe_overrides = (
            {pc_cols.BASE_CURRENCY: "EUR"},
            {pc_cols.AMOUNT: 0.0},
            {f"{pc_cols.CURRENCY}_b": "GBP"},
            {pc_cols.BASE_AMOUNT: -108.0},
            {pc_cols.BASE_AMOUNT: float("inf")},
        )

        for overrides in unsafe_overrides:
            with self.subTest(overrides=overrides):
                row = {**base_row, **overrides}
                self.assertIsNone(
                    _implied_conversion_message(
                        row,
                        pc_cols.TRANSACTIONS,
                        pc_cols.BASE_AMOUNT,
                    )
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
