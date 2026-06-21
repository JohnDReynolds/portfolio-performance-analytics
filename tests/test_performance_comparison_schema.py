"""Tests for normalized performance comparison column constants."""

# Python imports
import unittest

# Project imports
from ppar.performance_comparison import schema as pc_cols


class TestPerformanceComparisonSchema(unittest.TestCase):
    """Verify the first normalized comparison dataset column contract."""

    def test_portfolio_performance_required_columns_are_stable(self) -> None:
        """Portfolio performance requires only the top-level comparison fields."""
        self.assertEqual(pc_cols.PORTFOLIO_PERFORMANCE, "portfolio_performance")
        self.assertEqual(
            pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS,
            (
                pc_cols.PORTFOLIO_ID,
                pc_cols.FROM_DATE,
                pc_cols.THRU_DATE,
                pc_cols.PORTFOLIO_RETURN,
            ),
        )

    def test_portfolio_performance_columns_include_optional_context(self) -> None:
        """Portfolio performance schema include optional explanatory fields."""
        self.assertIn(
            pc_cols.BEGIN_MARKET_VALUE,
            pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS,
        )
        self.assertIn(pc_cols.FLOW, pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS)
        self.assertNotIn(pc_cols.WEIGHT, pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS)

    def test_optional_dataset_required_columns_are_stable(self) -> None:
        """Optional explanatory datasets have explicit minimal column contracts."""
        self.assertEqual(
            pc_cols.PRICES_REQUIRED_COLUMNS,
            (pc_cols.SECURITY_ID, pc_cols.PRICE_DATE, pc_cols.PRICE),
        )
        self.assertEqual(
            pc_cols.FX_RATES_REQUIRED_COLUMNS,
            (
                pc_cols.FROM_CURRENCY,
                pc_cols.TO_CURRENCY,
                pc_cols.RATE_DATE,
                pc_cols.FX_RATE,
            ),
        )
        self.assertEqual(
            pc_cols.TRANSACTIONS_REQUIRED_COLUMNS,
            (
                pc_cols.PORTFOLIO_ID,
                pc_cols.SECURITY_ID,
                pc_cols.TRANSACTION_DATE,
            ),
        )
        self.assertIn(
            pc_cols.TRANSACTION_CATEGORY,
            pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS,
        )
        self.assertIn(pc_cols.CASH_FLOW_SIGN, pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS)
        self.assertIn(
            pc_cols.PERFORMANCE_FLOW_SIGN,
            pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS,
        )
        self.assertIn(
            pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS,
        )
        self.assertEqual(
            pc_cols.POSITIONS_REQUIRED_COLUMNS,
            (pc_cols.PORTFOLIO_ID, pc_cols.SECURITY_ID, pc_cols.POSITION_DATE),
        )
        self.assertEqual(
            pc_cols.CASH_REQUIRED_COLUMNS,
            (pc_cols.PORTFOLIO_ID, pc_cols.CASH_DATE),
        )


if __name__ == "__main__":
    unittest.main()
