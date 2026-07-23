"""Tests for normalized performance comparison column constants."""

# Python imports
import unittest

# Project imports
from ppar.audit import schema as pc_cols


class TestAuditSchema(unittest.TestCase):
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

    def test_performance_columns_exclude_calculated_output_context(self) -> None:
        """Performance schemas retain returns and portfolio currency metadata."""
        self.assertEqual(
            pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS,
            (*pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS, pc_cols.BASE_CURRENCY),
        )
        self.assertEqual(
            pc_cols.SECURITY_PERFORMANCE_COLUMNS,
            pc_cols.SECURITY_PERFORMANCE_REQUIRED_COLUMNS,
        )

    def test_optional_dataset_required_columns_are_stable(self) -> None:
        """Optional explanatory datasets have explicit minimal column contracts."""
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
            pc_cols.HOLDINGS_REQUIRED_COLUMNS,
            (pc_cols.PORTFOLIO_ID, pc_cols.SECURITY_ID, pc_cols.HOLDING_DATE),
        )


if __name__ == "__main__":
    unittest.main()
