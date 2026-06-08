"""Tests for linking dated comparison evidence to portfolio periods."""

# Python imports
import datetime as dt
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.period_linking import (
    period_context_for_dated_evidence,
    portfolio_periods_from_snapshots,
    security_period_contexts_for_dated_evidence,
    security_periods_from_snapshots,
)


class TestPerformanceComparisonPeriodLinking(unittest.TestCase):
    """Verify dated evidence period-linking helpers."""

    def test_portfolio_periods_from_snapshots_returns_unique_sorted_periods(self) -> None:
        """Portfolio periods are deduplicated across snapshots and sorted."""
        snapshot_a = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_B", "PORT_A"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 5, 1),
                    dt.date(2025, 4, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 5, 31),
                    dt.date(2025, 4, 30),
                ],
            }
        )
        snapshot_b = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_A", "PORT_A"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 4, 1),
                    dt.date(2025, 5, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 4, 30),
                    dt.date(2025, 5, 31),
                ],
            }
        )

    def test_security_periods_from_snapshots_returns_unique_sorted_periods(self) -> None:
        """Security periods are deduplicated across snapshots and sorted."""
        snapshot_a = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_B", "PORT_A"],
                pc_cols.SECURITY_ID: ["AAPL", "AAPL"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 5, 1),
                    dt.date(2025, 4, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 5, 31),
                    dt.date(2025, 4, 30),
                ],
            }
        )
        snapshot_b = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_A", "PORT_A"],
                pc_cols.SECURITY_ID: ["AAPL", "MSFT"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 4, 1),
                    dt.date(2025, 5, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 4, 30),
                    dt.date(2025, 5, 31),
                ],
            }
        )

        periods = security_periods_from_snapshots(snapshot_a, snapshot_b)

        self.assertEqual(
            periods.to_dicts(),
            [
                {
                    pc_cols.PORTFOLIO_ID: "PORT_A",
                    pc_cols.SECURITY_ID: "AAPL",
                    pc_cols.FROM_DATE: dt.date(2025, 4, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 4, 30),
                },
                {
                    pc_cols.PORTFOLIO_ID: "PORT_A",
                    pc_cols.SECURITY_ID: "MSFT",
                    pc_cols.FROM_DATE: dt.date(2025, 5, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 5, 31),
                },
                {
                    pc_cols.PORTFOLIO_ID: "PORT_B",
                    pc_cols.SECURITY_ID: "AAPL",
                    pc_cols.FROM_DATE: dt.date(2025, 5, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 5, 31),
                },
            ],
        )

        periods = portfolio_periods_from_snapshots(snapshot_a, snapshot_b)

        self.assertEqual(
            periods.to_dicts(),
            [
                {
                    pc_cols.PORTFOLIO_ID: "PORT_A",
                    pc_cols.FROM_DATE: dt.date(2025, 4, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 4, 30),
                },
                {
                    pc_cols.PORTFOLIO_ID: "PORT_A",
                    pc_cols.FROM_DATE: dt.date(2025, 5, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 5, 31),
                },
                {
                    pc_cols.PORTFOLIO_ID: "PORT_B",
                    pc_cols.FROM_DATE: dt.date(2025, 5, 1),
                    pc_cols.THRU_DATE: dt.date(2025, 5, 31),
                },
            ],
        )

    def test_period_context_for_dated_evidence_preserves_existing_period(self) -> None:
        """Rows that already carry period context are not remapped."""
        row = {
            pc_cols.FROM_DATE: dt.date(2025, 5, 1),
            pc_cols.THRU_DATE: dt.date(2025, 5, 31),
            pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 15),
        }

        period_context = period_context_for_dated_evidence(
            row,
            pc_cols.TRANSACTIONS,
            portfolio_periods=None,
        )

        self.assertEqual(
            period_context,
            (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
        )

    def test_period_context_for_dated_evidence_links_transaction_period(self) -> None:
        """Transaction rows are linked to the portfolio period containing trade date."""
        row = {
            pc_cols.PORTFOLIO_ID: "PORT_A",
            pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
        }
        portfolio_periods = _portfolio_periods()

        period_context = period_context_for_dated_evidence(
            row,
            pc_cols.TRANSACTIONS,
            portfolio_periods,
        )

        self.assertEqual(
            period_context,
            (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
        )

    def test_period_context_for_dated_evidence_links_position_period(self) -> None:
        """Position rows are linked to the portfolio period containing position date."""
        row = {
            pc_cols.PORTFOLIO_ID: "PORT_A",
            pc_cols.POSITION_DATE: dt.date(2025, 5, 31),
        }
        portfolio_periods = _portfolio_periods()

        period_context = period_context_for_dated_evidence(
            row,
            pc_cols.POSITIONS,
            portfolio_periods,
        )

        self.assertEqual(
            period_context,
            (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
        )

    def test_period_context_for_dated_evidence_links_cash_period(self) -> None:
        """Cash rows are linked to the portfolio period containing cash date."""
        row = {
            pc_cols.PORTFOLIO_ID: "PORT_A",
            pc_cols.CASH_DATE: dt.date(2025, 5, 31),
        }
        portfolio_periods = _portfolio_periods()

        period_context = period_context_for_dated_evidence(
            row,
            pc_cols.CASH,
            portfolio_periods,
        )

        self.assertEqual(
            period_context,
            (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
        )

    def test_period_context_for_dated_evidence_prefers_narrowest_period(self) -> None:
        """A dated evidence row maps to the narrowest containing portfolio period."""
        row = {
            pc_cols.PORTFOLIO_ID: "PORT_A",
            pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
        }
        portfolio_periods = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_A", "PORT_A"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 5, 1),
                    dt.date(2025, 5, 15),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 5, 31),
                    dt.date(2025, 5, 15),
                ],
            }
        )

        period_context = period_context_for_dated_evidence(
            row,
            pc_cols.TRANSACTIONS,
            portfolio_periods,
        )

        self.assertEqual(
            period_context,
            (dt.date(2025, 5, 15), dt.date(2025, 5, 15)),
        )

    def test_period_context_for_dated_evidence_returns_empty_for_unmatched_rows(
        self,
    ) -> None:
        """Unsupported datasets and out-of-period dates return empty period context."""
        portfolio_periods = _portfolio_periods()

        unsupported_context = period_context_for_dated_evidence(
            {
                pc_cols.PORTFOLIO_ID: "PORT_A",
                pc_cols.PRICE_DATE: dt.date(2025, 5, 15),
            },
            pc_cols.PRICES,
            portfolio_periods,
        )
        out_of_period_context = period_context_for_dated_evidence(
            {
                pc_cols.PORTFOLIO_ID: "PORT_A",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 15),
            },
            pc_cols.TRANSACTIONS,
            portfolio_periods,
        )

        self.assertEqual(unsupported_context, (None, None))
        self.assertEqual(out_of_period_context, (None, None))

    def test_security_period_contexts_for_dated_evidence_links_price_periods(
        self,
    ) -> None:
        """Price rows are linked to matching security-performance periods."""
        row = {
            pc_cols.SECURITY_ID: "AAPL",
            pc_cols.PRICE_DATE: dt.date(2025, 5, 31),
        }
        security_periods = _security_periods()

        contexts = security_period_contexts_for_dated_evidence(
            row,
            pc_cols.PRICES,
            security_periods,
        )

        self.assertEqual(
            contexts,
            [
                ("PORT_A", dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
                ("PORT_B", dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
            ],
        )

    def test_security_period_contexts_for_dated_evidence_prefers_narrowest_period(
        self,
    ) -> None:
        """Price rows map to the narrowest containing period per portfolio."""
        row = {
            pc_cols.SECURITY_ID: "AAPL",
            pc_cols.PRICE_DATE: dt.date(2025, 5, 15),
        }
        security_periods = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_A", "PORT_A"],
                pc_cols.SECURITY_ID: ["AAPL", "AAPL"],
                pc_cols.FROM_DATE: [
                    dt.date(2025, 5, 1),
                    dt.date(2025, 5, 15),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2025, 5, 31),
                    dt.date(2025, 5, 15),
                ],
            }
        )

        contexts = security_period_contexts_for_dated_evidence(
            row,
            pc_cols.PRICES,
            security_periods,
        )

        self.assertEqual(
            contexts,
            [("PORT_A", dt.date(2025, 5, 15), dt.date(2025, 5, 15))],
        )

    def test_security_period_contexts_for_dated_evidence_returns_empty_for_unmatched_rows(
        self,
    ) -> None:
        """Unsupported datasets and out-of-period dates return no contexts."""
        security_periods = _security_periods()

        unsupported_contexts = security_period_contexts_for_dated_evidence(
            {
                pc_cols.SECURITY_ID: "AAPL",
                pc_cols.RATE_DATE: dt.date(2025, 5, 31),
            },
            pc_cols.FX_RATES,
            security_periods,
        )
        out_of_period_contexts = security_period_contexts_for_dated_evidence(
            {
                pc_cols.SECURITY_ID: "AAPL",
                pc_cols.PRICE_DATE: dt.date(2025, 6, 15),
            },
            pc_cols.PRICES,
            security_periods,
        )

        self.assertEqual(unsupported_contexts, [])
        self.assertEqual(out_of_period_contexts, [])


def _portfolio_periods() -> pl.DataFrame:
    """Return reusable portfolio periods for dated evidence tests."""
    return pl.DataFrame(
        {
            pc_cols.PORTFOLIO_ID: ["PORT_A"],
            pc_cols.FROM_DATE: [dt.date(2025, 5, 1)],
            pc_cols.THRU_DATE: [dt.date(2025, 5, 31)],
        }
    )


def _security_periods() -> pl.DataFrame:
    """Return reusable security periods for dated evidence tests."""
    return pl.DataFrame(
        {
            pc_cols.PORTFOLIO_ID: ["PORT_A", "PORT_B"],
            pc_cols.SECURITY_ID: ["AAPL", "AAPL"],
            pc_cols.FROM_DATE: [
                dt.date(2025, 5, 1),
                dt.date(2025, 5, 1),
            ],
            pc_cols.THRU_DATE: [
                dt.date(2025, 5, 31),
                dt.date(2025, 5, 31),
            ],
        }
    )


if __name__ == "__main__":
    unittest.main()
