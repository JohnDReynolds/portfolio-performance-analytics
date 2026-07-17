"""Tests for stable Audit product vocabulary and registry metadata."""

# Python imports
import unittest

# Project imports
from ppar.audit.data_issues import (
    DATA_ISSUE_REGISTRY,
    DataIssueCategory,
    DataIssueType,
)
from ppar.audit.performance_comparison import CauseArea, explain
from ppar.audit.data_issues import checks as data_issues


class TestDataIssuesVocabulary(unittest.TestCase):
    """Verify current serialized values and complete issue metadata."""

    def test_issue_type_values_preserve_current_serialized_contract(self) -> None:
        """Every current optional and mandatory issue ID remains unchanged."""
        self.assertEqual(
            {issue_type.value for issue_type in DataIssueType},
            {
                "duplicate_transactions",
                "dividend_rate",
                "holdings_accrued_rate",
                "holdings_price_range",
                "missing_dividend",
                "pa_sa_rate",
                "portfolio_market_value_continuity",
                "security_market_value_continuity",
                "transactions_price_range",
            },
        )
        self.assertEqual(
            data_issues.ISSUE_DUPLICATE_TRANSACTIONS,
            DataIssueType.DUPLICATE_TRANSACTIONS.value,
        )
        self.assertEqual(
            data_issues.ISSUE_PORTFOLIO_MV_CONTINUITY,
            DataIssueType.PORTFOLIO_MARKET_VALUE_CONTINUITY.value,
        )

    def test_cause_area_values_preserve_current_serialized_contract(self) -> None:
        """Every current coarse cause string remains unchanged."""
        self.assertEqual(
            {cause_area.value for cause_area in CauseArea},
            {
                "security_return_or_contribution",
                "market_value_or_holding",
                "transaction_activity",
                "fx_rate",
                "portfolio_performance_input",
                "classification_or_reference",
                "unexplained",
            },
        )
        self.assertEqual(
            explain.ROOT_CAUSE_TRANSACTION_ACTIVITY,
            CauseArea.TRANSACTION_ACTIVITY.value,
        )
        self.assertEqual(
            explain.ROOT_CAUSE_UNEXPLAINED,
            CauseArea.UNEXPLAINED.value,
        )

    def test_registry_describes_every_current_issue_type(self) -> None:
        """Registry membership and required product metadata are complete."""
        self.assertEqual(set(DATA_ISSUE_REGISTRY), set(DataIssueType))
        for issue_type, definition in DATA_ISSUE_REGISTRY.items():
            with self.subTest(issue_type=issue_type):
                self.assertIsInstance(definition.category, DataIssueCategory)
                self.assertTrue(definition.required_datasets)
                self.assertTrue(definition.reviewer_meaning)
                self.assertTrue(definition.default_enabled)

        mandatory = {
            issue_type
            for issue_type, definition in DATA_ISSUE_REGISTRY.items()
            if definition.mandatory
        }
        self.assertEqual(
            mandatory,
            {
                DataIssueType.PORTFOLIO_MARKET_VALUE_CONTINUITY,
                DataIssueType.SECURITY_MARKET_VALUE_CONTINUITY,
            },
        )


if __name__ == "__main__":
    unittest.main()
