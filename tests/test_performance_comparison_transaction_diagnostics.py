"""Tests for performance comparison transaction diagnostic helpers."""

# Python imports
import unittest

# Project imports
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import _transaction_diagnostics as tx_diagnostics
from ppar.performance_comparison.findings import (
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_ID_UNMATCHED,
    TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_SEMANTICS_SOURCE_MIXED,
    TRANSACTION_SEMANTICS_SOURCE_SOURCE,
    TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
)


class TransactionDiagnosticsTest(unittest.TestCase):
    """Test transaction diagnostic formatting and review ordering."""

    def test_transaction_matching_statuses_have_stable_review_order(self) -> None:
        """Known transaction match statuses sort in reviewer-facing order."""
        rows = [
            {TRANSACTION_MATCH_STATUS: "future_status"},
            {TRANSACTION_MATCH_STATUS: TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED},
            {TRANSACTION_MATCH_STATUS: TRANSACTION_MATCH_STATUS_ID_UNMATCHED},
            {TRANSACTION_MATCH_STATUS: TRANSACTION_MATCH_STATUS_ID_MATCH},
        ]

        sorted_statuses = [
            row[TRANSACTION_MATCH_STATUS]
            for row in sorted(
                rows,
                key=tx_diagnostics.transaction_matching_diagnostic_sort_key,
            )
        ]

        self.assertEqual(
            sorted_statuses,
            [
                TRANSACTION_MATCH_STATUS_ID_MATCH,
                TRANSACTION_MATCH_STATUS_ID_UNMATCHED,
                TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
                "future_status",
            ],
        )

    def test_transaction_matching_notes_explain_conservative_fallback(self) -> None:
        """Strict fallback status explains that edits are not inferred."""
        note = tx_diagnostics.transaction_match_review_note(
            TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED
        )

        self.assertIn("strict fallback keys", note)
        self.assertIn("rather than inferring an edit", note)

    def test_semantics_source_counts_round_trip_in_business_order(self) -> None:
        """Transaction semantics source summaries parse and reformat stably."""
        counts = tx_diagnostics.parse_transaction_semantics_sources(
            "unknown: 1, source: 2, yaml_rule: 3, mixed: 4, custom: 5"
        )

        self.assertEqual(
            tx_diagnostics.format_transaction_semantics_source_counts(counts),
            "source: 2, mixed: 4, yaml_rule: 3, unknown: 1, custom: 5",
        )

    def test_readable_semantics_sources_name_yaml_rules(self) -> None:
        """Known transaction semantics source labels are reviewer-facing."""
        expected = {
            TRANSACTION_SEMANTICS_SOURCE_SOURCE: "source",
            TRANSACTION_SEMANTICS_SOURCE_YAML_RULE: "YAML transaction_rules",
            TRANSACTION_SEMANTICS_SOURCE_MIXED: "mixed source and YAML transaction_rules",
            TRANSACTION_SEMANTICS_SOURCE_UNKNOWN: "unknown",
            None: "not provided",
        }

        for source, label in expected.items():
            with self.subTest(source=source):
                self.assertEqual(
                    tx_diagnostics.readable_transaction_semantics_source(source),
                    label,
                )

    def test_transaction_field_sort_key_puts_modeled_fields_first(self) -> None:
        """Amount, quantity, and price sort before less-modeled transaction fields."""
        fields = [
            "commission",
            pc_cols.PRICE,
            pc_cols.AMOUNT,
            pc_cols.QUANTITY,
        ]

        self.assertEqual(
            sorted(fields, key=tx_diagnostics.transaction_field_sort_key),
            [
                pc_cols.AMOUNT,
                pc_cols.QUANTITY,
                pc_cols.PRICE,
                "commission",
            ],
        )


if __name__ == "__main__":
    unittest.main()
