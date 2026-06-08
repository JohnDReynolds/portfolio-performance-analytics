"""Tests for performance comparison Markdown reporting."""

# Python imports
from pathlib import Path
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    performance_comparison_markdown_report,
)
from ppar.performance_comparison.report import _markdown_table

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)


class TestPerformanceComparisonReport(unittest.TestCase):
    """Verify Markdown report rendering for comparison findings."""

    def test_markdown_report_summarizes_restatement_findings(self) -> None:
        """Restatement reports include run, period, cause, and evidence sections."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("# Performance Comparison Report", report)
        self.assertIn("## Run Summary", report)
        self.assertIn("- Total findings: 21", report)
        self.assertIn("- Active findings: 21", report)
        self.assertIn("## Portfolio-Period Changes", report)
        self.assertIn("| PORT_A | 2025-05-30 | 2025-05-30 | 0.0005 | 17 | no |", report)
        self.assertIn("## Cause Summary", report)
        self.assertIn("security_return_or_contribution", report)
        self.assertIn("security_contribution", report)
        self.assertIn("## Top Evidence", report)
        self.assertIn("PC-SEC-CONTR", report)
        self.assertIn("## Suppressed Findings Appendix", report)

    def test_markdown_report_limits_top_evidence_per_portfolio_period(self) -> None:
        """Top evidence limit controls displayed contribution candidate rows."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings, top_evidence_limit=2)

        top_evidence = _section(report, "## Top Evidence", "## Suppressed Findings Appendix")

        self.assertIn("PC-PORT-MV", top_evidence)
        self.assertEqual(top_evidence.count("| PORT_A | 2025-05-30 | 2025-05-30 |"), 2)
        self.assertNotIn("PC-TXN-AMT", top_evidence)

    def test_markdown_report_separates_suppressed_findings(self) -> None:
        """Suppressed findings are counted and detailed in the appendix."""
        findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("- Total findings: 21", report)
        self.assertIn("- Active findings: 20", report)
        self.assertIn("- Suppressed findings: 1", report)
        self.assertIn("| PC-SEC-RET | yes | 1 |", report)
        self.assertIn("Suppressed Finding Detail", report)

    def test_markdown_report_handles_empty_findings(self) -> None:
        """Baseline reports render stable empty-state sections."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("- Total findings: 0", report)
        self.assertIn("_No portfolio return changes._", report)
        self.assertIn("_No cause summary available._", report)
        self.assertIn("_No ranked evidence is available for portfolio return changes._", report)

    def test_markdown_table_escapes_pipe_characters(self) -> None:
        """Markdown cell values cannot accidentally split table columns."""
        table = pl.DataFrame({"message": ["a|b"]})

        rendered = _markdown_table(table, ["message"])

        self.assertIn("a\\|b", rendered)


def _section(report: str, start: str, end: str) -> str:
    """Return report text between two section markers."""
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
