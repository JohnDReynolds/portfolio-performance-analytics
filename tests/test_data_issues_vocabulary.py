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
        """Every current issue ID remains unchanged."""
        self.assertEqual(
            {issue_type.value for issue_type in DataIssueType},
            {
                "duplicate_transactions",
                "deliver_in_original_cost_incomplete",
                "dividend_rate",
                "holdings_accrued_rate",
                "holdings_nonpositive_price",
                "holdings_price_range",
                "holdings_stale_price",
                "large_price_variation",
                "missing_dividend",
                "pa_sa_rate",
                "transaction_security_type_mismatch",
                "transactions_nonpositive_price",
                "transactions_price_range",
            },
        )
        self.assertEqual(
            data_issues.ISSUE_DUPLICATE_TRANSACTIONS,
            DataIssueType.DUPLICATE_TRANSACTIONS.value,
        )
        self.assertEqual(
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE,
            DataIssueType.DELIVER_IN_ORIGINAL_COST_INCOMPLETE.value,
        )
        self.assertEqual(
            data_issues.ISSUE_HOLDINGS_NONPOSITIVE_PRICE,
            DataIssueType.HOLDINGS_NONPOSITIVE_PRICE.value,
        )
        self.assertEqual(
            data_issues.ISSUE_HOLDINGS_STALE_PRICE,
            DataIssueType.HOLDINGS_STALE_PRICE.value,
        )
        self.assertEqual(
            data_issues.ISSUE_LARGE_PRICE_VARIATION,
            DataIssueType.LARGE_PRICE_VARIATION.value,
        )
        self.assertEqual(
            data_issues.ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH,
            DataIssueType.TRANSACTION_SECURITY_TYPE_MISMATCH.value,
        )
        self.assertEqual(
            data_issues.ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE,
            DataIssueType.TRANSACTIONS_NONPOSITIVE_PRICE.value,
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

        opt_in_checks = {
            issue_type
            for issue_type, definition in DATA_ISSUE_REGISTRY.items()
            if not definition.default_enabled
        }
        self.assertEqual(
            opt_in_checks,
            {
                DataIssueType.HOLDINGS_NONPOSITIVE_PRICE,
                DataIssueType.DELIVER_IN_ORIGINAL_COST_INCOMPLETE,
                DataIssueType.DIVIDEND_RATE,
                DataIssueType.HOLDINGS_STALE_PRICE,
                DataIssueType.LARGE_PRICE_VARIATION,
                DataIssueType.MISSING_DIVIDEND,
                DataIssueType.PA_SA_RATE,
                DataIssueType.TRANSACTION_SECURITY_TYPE_MISMATCH,
                DataIssueType.TRANSACTIONS_NONPOSITIVE_PRICE,
            },
        )
        self.assertTrue(
            DATA_ISSUE_REGISTRY[
                DataIssueType.HOLDINGS_NONPOSITIVE_PRICE
            ].requires_only_filter
        )
        deliver_in_definition = DATA_ISSUE_REGISTRY[
            DataIssueType.DELIVER_IN_ORIGINAL_COST_INCOMPLETE
        ]
        self.assertEqual(
            deliver_in_definition.category,
            DataIssueCategory.POSITION_VALUE,
        )
        self.assertTrue(deliver_in_definition.requires_only_filter)
        self.assertEqual(deliver_in_definition.required_datasets, ("transactions",))
        stale_price_definition = DATA_ISSUE_REGISTRY[
            DataIssueType.HOLDINGS_STALE_PRICE
        ]
        self.assertTrue(stale_price_definition.requires_only_filter)
        self.assertTrue(stale_price_definition.supports_minimum_calendar_days)
        self.assertEqual(
            stale_price_definition.required_datasets,
            ("holdings", "security_master"),
        )
        large_variation_definition = DATA_ISSUE_REGISTRY[
            DataIssueType.LARGE_PRICE_VARIATION
        ]
        self.assertFalse(large_variation_definition.requires_only_filter)
        self.assertEqual(
            large_variation_definition.required_datasets,
            ("portfolio_performance",),
        )
        transaction_price_definition = DATA_ISSUE_REGISTRY[
            DataIssueType.TRANSACTIONS_NONPOSITIVE_PRICE
        ]
        self.assertTrue(transaction_price_definition.requires_only_filter)
        self.assertEqual(
            transaction_price_definition.required_datasets,
            ("transactions", "security_master"),
        )
        mismatch_definition = DATA_ISSUE_REGISTRY[
            DataIssueType.TRANSACTION_SECURITY_TYPE_MISMATCH
        ]
        self.assertEqual(
            mismatch_definition.category,
            DataIssueCategory.CLASSIFICATION,
        )
        self.assertTrue(mismatch_definition.requires_only_filter)
        self.assertEqual(
            mismatch_definition.required_datasets,
            ("transactions", "security_master"),
        )


if __name__ == "__main__":
    unittest.main()
