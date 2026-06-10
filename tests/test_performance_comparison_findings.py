"""Tests for performance comparison finding records."""

# Python imports
import unittest

# Project imports
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    DELTA_B_MINUS_A,
    EVIDENCE_ROLE,
    FINDING_COLUMNS,
    FINDING_CODE,
    MESSAGE,
    PC_PORT_RET,
    SEVERITY_MATERIAL,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    TARGET_OUTPUT,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    Finding,
    findings_to_polars,
)


class TestPerformanceComparisonFindings(unittest.TestCase):
    """Verify finding records use stable column names."""

    def test_finding_to_dict_uses_column_contract(self) -> None:
        """Finding dictionaries retain stable output keys and values."""
        finding = Finding(
            code=PC_PORT_RET,
            severity=SEVERITY_MATERIAL,
            confidence=CONFIDENCE_HIGH,
            dataset="portfolio_performance",
            evidence_role=TARGET_OUTPUT,
            snapshot_a_value=0.01,
            snapshot_b_value=0.02,
            delta_b_minus_a=0.01,
            message="Portfolio return changed.",
        )

        finding_dict = finding.to_dict()

        self.assertEqual(tuple(finding_dict), FINDING_COLUMNS)
        self.assertEqual(finding_dict[FINDING_CODE], PC_PORT_RET)
        self.assertEqual(finding_dict[EVIDENCE_ROLE], TARGET_OUTPUT)
        self.assertEqual(finding_dict[SNAPSHOT_A_VALUE], 0.01)
        self.assertEqual(finding_dict[SNAPSHOT_B_VALUE], 0.02)
        self.assertEqual(finding_dict[DELTA_B_MINUS_A], 0.01)
        self.assertIsNone(finding_dict[TRANSACTION_IMPACT_DIAGNOSTIC])
        self.assertIsNone(finding_dict[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE])
        self.assertEqual(finding_dict[MESSAGE], "Portfolio return changed.")

    def test_empty_findings_to_polars_has_stable_columns(self) -> None:
        """Empty finding output still preserves the public column contract."""
        frame = findings_to_polars([])

        self.assertEqual(frame.columns, list(FINDING_COLUMNS))
        self.assertTrue(frame.is_empty())


if __name__ == "__main__":
    unittest.main()
