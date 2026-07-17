"""Tests for performance comparison public runner functions."""

# Python imports
from pathlib import Path
import unittest
from unittest import mock

# Third-party imports
from polars.testing import assert_frame_equal

# Project imports
from ppar.performance_comparison import (
    compact_findings_table,
    compare_snapshots,
    summarize_findings,
)
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.findings import (
    DATASET,
    DELTA_B_MINUS_A,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    MESSAGE,
    PC_FX_RATE,
    PC_HOLD_ACCR,
    PC_HOLD_MV,
    PC_PORT_RET,
    PC_HOLD_QTY,
    PC_TXN_AMT,
    PC_TXN_PRICE,
    PC_TXN_QTY,
    PORTFOLIO_ID,
    SECURITY_ID,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SUPPRESSED,
    THRU_DATE,
)
from ppar.performance_comparison.runner import AuditComparisonViews

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_suppressed.yaml"
)
_PACKAGED_COMPARISON_PATH = Path(
    "ppar/setup_templates/axysapx_performance_comparison/"
    "axysapx_performance_comparison.yaml"
)
_COMPACT_FINDING_COLUMNS = [
    FINDING_CODE,
    DATASET,
    EVIDENCE_ROLE,
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    SOURCE_FILE,
    SOURCE_COLUMN,
    DELTA_B_MINUS_A,
    MESSAGE,
]


class TestPerformanceComparisonRunner(unittest.TestCase):
    """Verify public performance comparison runner behavior."""

    def test_compare_snapshots_returns_empty_table_for_baseline(self) -> None:
        """Identical baseline snapshots return an empty findings table."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        self.assertTrue(findings.is_empty())
        self.assertIn(FINDING_CODE, findings.columns)

    def test_compare_snapshots_returns_expected_restatement_codes(self) -> None:
        """Restatement comparison returns currently supported finding families."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        finding_codes = set(findings.get_column(FINDING_CODE).to_list())

        self.assertFalse(findings.is_empty())
        self.assertTrue(
            {
                PC_PORT_RET,
                PC_HOLD_QTY,
                PC_HOLD_ACCR,
                PC_HOLD_MV,
                PC_FX_RATE,
                PC_TXN_AMT,
                PC_TXN_QTY,
                PC_TXN_PRICE,
            }.issubset(finding_codes)
        )

    def test_audit_views_exactly_match_independent_comparisons(self) -> None:
        """Canonical shared findings preserve both independent result views."""
        views = AuditComparisonViews(_PACKAGED_COMPARISON_PATH)

        for comparison_level in ("portfolio", "security"):
            with self.subTest(comparison_level=comparison_level):
                expected = compare_snapshots(
                    _PACKAGED_COMPARISON_PATH,
                    comparison_level=comparison_level,
                )
                assert_frame_equal(views.findings(comparison_level), expected)

    def test_audit_views_are_order_independent(self) -> None:
        """Either result level can provide the canonical shared findings."""
        views = AuditComparisonViews(_PACKAGED_COMPARISON_PATH)

        security_findings = views.findings("security")
        portfolio_findings = views.findings("portfolio")

        assert_frame_equal(
            security_findings,
            compare_snapshots(
                _PACKAGED_COMPARISON_PATH,
                comparison_level="security",
            ),
        )
        assert_frame_equal(
            portfolio_findings,
            compare_snapshots(
                _PACKAGED_COMPARISON_PATH,
                comparison_level="portfolio",
            ),
        )

    def test_audit_views_preserve_level_specific_suppressions(self) -> None:
        """Each derived view applies its own suppression rules before filtering."""
        views = AuditComparisonViews(
            _SUPPRESSED_COMPARISON_PATH,
            include_suppressed=False,
        )

        for comparison_level in ("portfolio", "security"):
            with self.subTest(comparison_level=comparison_level):
                expected = compare_snapshots(
                    _SUPPRESSED_COMPARISON_PATH,
                    comparison_level=comparison_level,
                    include_suppressed=False,
                )
                assert_frame_equal(views.findings(comparison_level), expected)

    def test_audit_views_compare_shared_sources_once(self) -> None:
        """Repeated portfolio/security views reuse one shared-source comparison."""
        original = PerformanceComparison.compare_shared_sources
        with mock.patch.object(
            PerformanceComparison,
            "compare_shared_sources",
            autospec=True,
        ) as compare_shared_sources:
            compare_shared_sources.side_effect = original
            views = AuditComparisonViews(_PACKAGED_COMPARISON_PATH)

            views.findings("portfolio")
            views.findings("security")
            views.findings("portfolio")

        self.assertEqual(compare_shared_sources.call_count, 1)

    def test_summarize_findings_counts_by_code_dataset_and_suppression(self) -> None:
        """Finding summaries count rows by code, dataset, and suppression."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        summaries = summarize_findings(findings)
        by_code = summaries["by_code"]
        by_dataset = summaries["by_dataset"]
        by_suppressed = summaries["by_suppressed"]
        by_code_suppressed = summaries["by_code_suppressed"]

        self.assertIn(FINDING_CODE, by_code.columns)
        self.assertIn("count", by_code.columns)
        self.assertIn(DATASET, by_dataset.columns)
        self.assertIn("count", by_dataset.columns)
        self.assertIn(EVIDENCE_ROLE, summaries["by_evidence_role"].columns)
        self.assertIn("count", summaries["by_evidence_role"].columns)
        self.assertIn(SUPPRESSED, by_suppressed.columns)
        self.assertIn("count", by_suppressed.columns)
        self.assertEqual(by_code_suppressed.columns, [FINDING_CODE, SUPPRESSED, "count"])
        self.assertEqual(by_code.get_column("count").sum(), findings.height)
        self.assertEqual(by_dataset.get_column("count").sum(), findings.height)
        self.assertEqual(
            summaries["by_evidence_role"].get_column("count").sum(),
            findings.height,
        )
        self.assertEqual(by_suppressed.get_column("count").sum(), findings.height)
        self.assertEqual(by_code_suppressed.get_column("count").sum(), findings.height)

    def test_summarize_findings_returns_stable_empty_tables(self) -> None:
        """Empty findings produce empty summary tables with stable columns."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        summaries = summarize_findings(findings)

        self.assertTrue(summaries["by_code"].is_empty())
        self.assertTrue(summaries["by_dataset"].is_empty())
        self.assertTrue(summaries["by_evidence_role"].is_empty())
        self.assertTrue(summaries["by_suppressed"].is_empty())
        self.assertTrue(summaries["by_code_suppressed"].is_empty())
        self.assertEqual(summaries["by_code"].columns, [FINDING_CODE, "count"])
        self.assertEqual(summaries["by_dataset"].columns, [DATASET, "count"])
        self.assertEqual(
            summaries["by_evidence_role"].columns,
            [EVIDENCE_ROLE, "count"],
        )
        self.assertEqual(summaries["by_suppressed"].columns, [SUPPRESSED, "count"])
        self.assertEqual(
            summaries["by_code_suppressed"].columns,
            [FINDING_CODE, SUPPRESSED, "count"],
        )

    def test_compact_findings_table_returns_report_friendly_columns(self) -> None:
        """Compact findings table keeps the most useful reporting columns."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        compact_findings = compact_findings_table(findings)

        self.assertEqual(compact_findings.columns, _COMPACT_FINDING_COLUMNS)
        self.assertEqual(compact_findings.height, findings.height)
        self.assertIn("portperf.csv", compact_findings.get_column(SOURCE_FILE).to_list())

    def test_compact_findings_table_excludes_suppressed_by_default(self) -> None:
        """Compact findings default to active unsuppressed rows."""
        findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)

        compact_findings = compact_findings_table(findings)
        compact_with_suppressed = compact_findings_table(findings, include_suppressed=True)

        self.assertEqual(compact_findings.height, findings.height - 1)
        self.assertEqual(compact_with_suppressed.height, findings.height)

    def test_compact_findings_table_returns_stable_empty_table(self) -> None:
        """Empty compact findings retain stable report columns."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        compact_findings = compact_findings_table(findings)

        self.assertTrue(compact_findings.is_empty())
        self.assertEqual(compact_findings.columns, _COMPACT_FINDING_COLUMNS)


if __name__ == "__main__":
    unittest.main()
