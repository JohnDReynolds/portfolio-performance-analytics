"""Tests for portfolio-level performance comparison findings."""

# Python imports
from pathlib import Path
from typing import cast
import unittest

# Project imports
from ppar.performance_comparison import (
    PerformanceComparison,
    PerformanceComparisonSpecification,
    findings_to_polars,
)
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.findings import (
    DELTA_B_MINUS_A,
    FINDING_CODE,
    PC_PORT_MV,
    PC_PORT_RET,
    PC_SEC_ADD,
    PC_SEC_CONTR,
    PC_SEC_DROP,
    PC_SEC_RET,
    PC_SEC_WGT,
    PC_REF_CLASS,
    PC_REF_ID,
    SECURITY_ID,
    SOURCE_COLUMN,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)


class TestPerformanceComparison(unittest.TestCase):
    """Verify portfolio performance comparison findings."""

    def test_identical_baseline_snapshots_have_no_portfolio_findings(self) -> None:
        """The baseline fixture compares identical A/B snapshots."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_portfolio_performance()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_portfolio_return_change(self) -> None:
        """The restatement fixture reports controlled portfolio-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_portfolio_performance()
        finding_dicts = [finding.to_dict() for finding in findings]
        return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_RET
            and finding[SOURCE_COLUMN] == pc_cols.PORTFOLIO_RETURN
        ]
        end_mv_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_MV
            and finding[SOURCE_COLUMN] == pc_cols.END_MARKET_VALUE
        ]

        self.assertEqual(len(return_findings), 1)
        self.assertAlmostEqual(cast(float, return_findings[0][DELTA_B_MINUS_A]), 0.0005)
        self.assertEqual(len(end_mv_findings), 1)
        self.assertAlmostEqual(cast(float, end_mv_findings[0][DELTA_B_MINUS_A]), 500.0)

    def test_identical_baseline_snapshots_have_no_security_findings(self) -> None:
        """The baseline fixture compares identical security performance rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_performance()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_changes(self) -> None:
        """The restatement fixture reports controlled security-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_performance()
        finding_dicts = [finding.to_dict() for finding in findings]
        aapl_return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_RET
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECURITY_RETURN
        ]
        aapl_weight_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_WGT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.WEIGHT
        ]
        aapl_contribution_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_CONTR
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.CONTRIBUTION
        ]
        add_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_ADD
            and finding[SECURITY_ID] == "RESTATED_SEC"
        ]
        drop_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_DROP
            and finding[SECURITY_ID] == "PFE"
        ]

        self.assertEqual(len(aapl_return_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_return_findings[0][DELTA_B_MINUS_A]),
            0.01,
        )
        self.assertEqual(len(aapl_weight_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_weight_findings[0][DELTA_B_MINUS_A]),
            0.001,
        )
        self.assertEqual(len(aapl_contribution_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_contribution_findings[0][DELTA_B_MINUS_A]),
            0.00058425,
        )
        self.assertEqual(len(add_findings), 1)
        self.assertEqual(len(drop_findings), 1)

    def test_compare_combines_portfolio_and_security_findings(self) -> None:
        """Combined comparison returns all currently supported finding groups."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        finding_dicts = [
            finding.to_dict() for finding in PerformanceComparison(specification).compare()
        ]
        finding_codes = {finding[FINDING_CODE] for finding in finding_dicts}

        self.assertIn(PC_PORT_RET, finding_codes)
        self.assertIn(PC_SEC_RET, finding_codes)
        self.assertIn(PC_SEC_ADD, finding_codes)
        self.assertIn(PC_SEC_DROP, finding_codes)

    def test_combined_findings_convert_to_polars(self) -> None:
        """Combined findings can be converted to a stable Polars table."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)
        findings = PerformanceComparison(specification).compare()

        frame = findings_to_polars(findings)

        self.assertFalse(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(DELTA_B_MINUS_A, frame.columns)

    def test_baseline_combined_compare_has_empty_polars_output(self) -> None:
        """Identical baseline snapshots produce an empty stable finding table."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        findings = PerformanceComparison(specification).compare()

        frame = findings_to_polars(findings)

        self.assertTrue(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)

    def test_identical_baseline_snapshots_have_no_security_master_findings(self) -> None:
        """The baseline fixture compares identical security master rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_master()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_master_changes(self) -> None:
        """The restatement fixture reports controlled security master changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_master()
        finding_dicts = [finding.to_dict() for finding in findings]
        aapl_name_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_REF_ID
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECURITY_NAME
        ]
        aapl_sector_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_REF_CLASS
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECTOR
        ]
        added_reference_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == "PC-ROW-ADD"
            and finding[SECURITY_ID] == "RESTATED_SEC"
        ]

        self.assertEqual(len(aapl_name_findings), 1)
        self.assertEqual(
            aapl_name_findings[0]["snapshot_b_value"],
            "Apple Inc Restated Name",
        )
        self.assertEqual(len(aapl_sector_findings), 1)
        self.assertEqual(aapl_sector_findings[0]["snapshot_b_value"], "TECH_RESTATED")
        self.assertEqual(len(added_reference_findings), 1)


if __name__ == "__main__":
    unittest.main()
