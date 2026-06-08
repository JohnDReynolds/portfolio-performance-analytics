"""Tests for performance comparison Markdown reporting."""

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
    DIRECT_INPUT,
    Finding,
    TARGET_OUTPUT,
    columns as pc_cols,
    compare_snapshots,
    findings_to_polars,
    performance_comparison_markdown_report,
    write_performance_comparison_markdown_report,
)
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    PC_PORT_MV,
    PC_PORT_RET,
    SEVERITY_MATERIAL,
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
        self.assertIn("## Report Contents", report)
        self.assertIn("- Run Summary", report)
        self.assertIn("- Suppressed Findings Appendix", report)
        self.assertIn("## Run Summary", report)
        self.assertIn("- Total findings: 21", report)
        self.assertIn("- Active findings: 21", report)
        self.assertIn("## Portfolio-Period Narrative", report)
        self.assertIn(
            "PORT_A changed by 0.0005 for 2025-05-30 to 2025-05-30.",
            report,
        )
        self.assertIn(
            "The strongest currently estimated impact is "
            "security_return_or_contribution at 0.00058425",
            report,
        )
        self.assertIn("Evidence-only areas are", report)
        self.assertIn("## Review Notes", report)
        self.assertIn("return denominator", report)
        self.assertIn("transaction sign and flow semantics", report)
        self.assertIn("source-field estimates are low-confidence", report)
        self.assertIn("vendor contribution deltas are preferred", report)
        self.assertIn("No residual amount is calculated", report)
        self.assertIn("## Impact Estimate Summary", report)
        impact_summary = _section(
            report,
            "## Impact Estimate Summary",
            "## Portfolio-Period Changes",
        )
        self.assertIn("security_contribution", impact_summary)
        self.assertIn("portfolio_source_field", impact_summary)
        self.assertIn("0.00058425", impact_summary)
        self.assertIn("## Residual Status", report)
        residual_status = _section(
            report,
            "## Residual Status",
            "## Portfolio-Period Changes",
        )
        self.assertIn("Residual amounts are intentionally withheld", residual_status)
        self.assertIn("withheld", residual_status)
        self.assertIn("partial or overlapping estimates", residual_status)
        self.assertIn("## Transaction Activity", report)
        transaction_activity = _section(
            report,
            "## Transaction Activity",
            "## Portfolio-Period Changes",
        )
        self.assertIn("buy", transaction_activity)
        self.assertIn("amount, quantity, price", transaction_activity)
        self.assertIn("return denominator", transaction_activity)
        self.assertIn("transaction sign and flow semantics", transaction_activity)
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
        self.assertIn("_No portfolio return changes to narrate._", report)
        self.assertIn("_No portfolio-period review notes._", report)
        self.assertIn("_No impact estimates are currently available._", report)
        self.assertIn("_No portfolio return changes need residual review._", report)
        self.assertIn("_No changed transaction activity._", report)
        self.assertIn("_No portfolio return changes._", report)
        self.assertIn("_No cause summary available._", report)
        self.assertIn("_No ranked evidence is available for portfolio return changes._", report)

    def test_markdown_report_withholds_residual_when_no_estimates_exist(self) -> None:
        """Residual status explains changed periods with evidence but no estimates."""
        period_date = dt.date(2025, 5, 30)
        findings = findings_to_polars(
            [
                Finding(
                    code=PC_PORT_RET,
                    severity=SEVERITY_MATERIAL,
                    confidence=CONFIDENCE_HIGH,
                    dataset=pc_cols.PORTFOLIO_PERFORMANCE,
                    evidence_role=TARGET_OUTPUT,
                    portfolio_id="PORT_NO_EST",
                    from_date=period_date,
                    thru_date=period_date,
                    source_column=pc_cols.PORTFOLIO_RETURN,
                    snapshot_a_value=0.01,
                    snapshot_b_value=0.011,
                    delta_b_minus_a=0.001,
                    message="portfolio_performance 'portfolio_return' changed.",
                ),
                Finding(
                    code=PC_PORT_MV,
                    severity=SEVERITY_MATERIAL,
                    confidence=CONFIDENCE_HIGH,
                    dataset=pc_cols.PORTFOLIO_PERFORMANCE,
                    evidence_role=DIRECT_INPUT,
                    portfolio_id="PORT_NO_EST",
                    from_date=period_date,
                    thru_date=period_date,
                    source_column=pc_cols.END_MARKET_VALUE,
                    snapshot_a_value=1000000.0,
                    snapshot_b_value=1000100.0,
                    delta_b_minus_a=100.0,
                    message="portfolio_performance 'end_market_value' changed.",
                ),
            ]
        )

        report = performance_comparison_markdown_report(findings)

        self.assertIn("_No impact estimates are currently available._", report)
        residual_status = _section(
            report,
            "## Residual Status",
            "## Portfolio-Period Changes",
        )
        self.assertIn("PORT_NO_EST", residual_status)
        self.assertIn("Residual amounts are intentionally withheld", residual_status)
        self.assertIn("withheld", residual_status)
        self.assertIn("no defensible impact estimates", residual_status)
        self.assertNotIn("partial or overlapping estimates", residual_status)

    def test_markdown_table_escapes_pipe_characters(self) -> None:
        """Markdown cell values cannot accidentally split table columns."""
        table = pl.DataFrame({"message": ["a|b"]})

        rendered = _markdown_table(table, ["message"])

        self.assertIn("a\\|b", rendered)

    def test_markdown_report_contents_track_suppressed_appendix_option(self) -> None:
        """Report contents omit the appendix when the appendix is omitted."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(
            findings,
            include_suppressed_appendix=False,
        )

        contents = _section(report, "## Report Contents", "## Run Summary")

        self.assertIn("- Impact Estimate Summary", contents)
        self.assertIn("- Residual Status", contents)
        self.assertIn("- Transaction Activity", contents)
        self.assertIn("- Top Evidence", contents)
        self.assertNotIn("Suppressed Findings Appendix", contents)
        self.assertNotIn("## Suppressed Findings Appendix", report)

    def test_markdown_report_orders_impact_summary_before_detail_sections(self) -> None:
        """Impact and residual summaries appear before lower-level details."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertLess(
            report.index("## Review Notes"),
            report.index("## Impact Estimate Summary"),
        )
        self.assertLess(
            report.index("## Impact Estimate Summary"),
            report.index("## Residual Status"),
        )
        self.assertLess(
            report.index("## Residual Status"),
            report.index("## Transaction Activity"),
        )
        self.assertLess(
            report.index("## Transaction Activity"),
            report.index("## Portfolio-Period Changes"),
        )
        self.assertLess(
            report.index("## Portfolio-Period Changes"),
            report.index("## Cause Summary"),
        )

    def test_write_markdown_report_creates_parent_directory(self) -> None:
        """Markdown reports can be written as durable artifacts."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "report.md"

            written_path = write_performance_comparison_markdown_report(
                findings,
                output_path,
                title="Controlled Restatement",
                top_evidence_limit=2,
            )

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("# Controlled Restatement", report)
            self.assertIn("## Top Evidence", report)


def _section(report: str, start: str, end: str) -> str:
    """Return report text between two section markers."""
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
