"""Tests for performance comparison Markdown reporting."""

# Python imports
import datetime as dt
import json
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
    performance_comparison_html_report,
    performance_comparison_markdown_report,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
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
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)


def _write_transaction_estimate_specification(directory: Path) -> Path:
    """Write a minimal source-loaded fixture with transaction impact semantics."""
    for snapshot_name, portfolio_return, amount in (
        ("snapshot_a", "0.0100", "-100.00"),
        ("snapshot_b", "0.0110", "-110.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            f"PORT_A,2025-05-01,2025-05-31,1000.00,{portfolio_return}\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount},"
            "cash out,performance\n",
            encoding="utf-8",
        )

    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(
        "\n".join(
            [
                "snapshots:",
                "  a:",
                "    path: snapshot_a",
                "  b:",
                "    path: snapshot_b",
                "files:",
                "  portfolio_performance: portperf.csv",
                "  transactions: transactions.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return specification_path


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
            "## Impact Coverage",
        )
        self.assertIn("security_contribution", impact_summary)
        self.assertIn("portfolio_source_field", impact_summary)
        self.assertIn("0.00058425", impact_summary)
        self.assertIn("## Impact Coverage", report)
        impact_coverage = _section(
            report,
            "## Impact Coverage",
            "## Residual Status",
        )
        self.assertIn("| PORT_A | 2025-05-30 | 2025-05-30 | 0.0005 | 6 | 2 | 4 |", impact_coverage)
        self.assertIn("0.001084292504", impact_coverage)
        self.assertIn("transaction_activity", impact_coverage)
        self.assertIn("Transaction Semantics Sources", impact_coverage)
        self.assertIn("unknown: 3", impact_coverage)
        self.assertIn("return-impact method", impact_coverage)
        self.assertIn("2 cause area(s) have estimates", impact_coverage)
        self.assertIn("## Residual Status", report)
        residual_status = _section(
            report,
            "## Residual Status",
            "## Transaction Activity",
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
        self.assertIn("Transaction Semantics Sources", transaction_activity)
        self.assertIn("unknown: 3", transaction_activity)
        self.assertIn(
            "| PORT_A | AAPL | 2025-05-30 | 2025-05-30 | buy |",
            transaction_activity,
        )
        self.assertIn("amount, quantity, price", transaction_activity)
        self.assertIn("return denominator", transaction_activity)
        self.assertIn("transaction sign and flow semantics", transaction_activity)
        self.assertNotIn("portfolio period, return denominator", transaction_activity)
        self.assertIn("## Portfolio-Period Changes", report)
        self.assertIn("| PORT_A | 2025-05-30 | 2025-05-30 | 0.0005 | 17 | no |", report)
        self.assertIn("## Cause Summary", report)
        self.assertIn("security_return_or_contribution", report)
        self.assertIn("security_contribution", report)
        self.assertIn("## Top Evidence", report)
        self.assertIn("PC-SEC-CONTR", report)
        self.assertIn("## Suppressed Findings Appendix", report)

    def test_markdown_report_shows_transaction_rule_semantics(self) -> None:
        """YAML transaction rule provenance appears in reviewer-facing tables."""
        findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)

        report = performance_comparison_markdown_report(findings)
        impact_coverage = _section(
            report,
            "## Impact Coverage",
            "## Residual Status",
        )
        transaction_activity = _section(
            report,
            "## Transaction Activity",
            "## Portfolio-Period Changes",
        )
        top_evidence = _section(
            report,
            "## Top Evidence",
            "## Suppressed Findings Appendix",
        )

        self.assertIn("mixed: 3", impact_coverage)
        self.assertIn("mixed: 3", transaction_activity)
        self.assertIn("return denominator", transaction_activity)
        self.assertNotIn("transaction sign and flow semantics", transaction_activity)
        self.assertIn("Transaction Semantics Source", top_evidence)
        self.assertIn("Transaction Impact Diagnostic", top_evidence)
        self.assertIn("| mixed |  |  |  | -100 |", top_evidence)

    def test_markdown_report_shows_source_loaded_transaction_estimate(self) -> None:
        """Source transaction semantics flow through to report impact estimates."""
        with tempfile.TemporaryDirectory() as directory:
            specification_path = _write_transaction_estimate_specification(
                Path(directory)
            )

            findings = compare_snapshots(specification_path)
            report = performance_comparison_markdown_report(findings)

        impact_summary = _section(
            report,
            "## Impact Estimate Summary",
            "## Impact Coverage",
        )
        transaction_activity = _section(
            report,
            "## Transaction Activity",
            "## Portfolio-Period Changes",
        )
        top_evidence = _section(
            report,
            "## Top Evidence",
            "## Suppressed Findings Appendix",
        )

        self.assertIn("transaction_activity", impact_summary)
        self.assertIn("transaction_performance_amount", impact_summary)
        self.assertIn("-0.01", impact_summary)
        self.assertIn("performance-treated amount deltas", report)
        self.assertIn("| PORT_A | AAPL | 2025-05-01 | 2025-05-31 | buy |", transaction_activity)
        self.assertIn("source: 1", transaction_activity)
        self.assertIn("| -10 |", transaction_activity)
        self.assertIn("Transaction Semantics Source", top_evidence)
        self.assertIn("Transaction Impact Diagnostic", top_evidence)
        self.assertIn("| source |  |  |  | -10 |", top_evidence)
        self.assertIn("transaction_amount_delta_over_return_denominator", top_evidence)
        self.assertIn("source-signed transaction amount", top_evidence)
        self.assertNotIn("transaction sign and flow semantics", report)

    def test_markdown_report_shows_modified_dietz_cross_check_estimate(self) -> None:
        """Modified Dietz estimates appear only as diagnostic cross-checks."""
        with tempfile.TemporaryDirectory() as directory:
            specification_path = _write_transaction_estimate_specification(
                Path(directory)
            )
            for snapshot_name, amount in (
                ("snapshot_a", "100.00"),
                ("snapshot_b", "110.00"),
            ):
                transaction_path = Path(directory) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,DEP,1,100.00,"
                    f"{amount},cash in,external\n",
                    encoding="utf-8",
                )
            specification_path.write_text(
                specification_path.read_text(encoding="utf-8")
                + "\n"
                + "transaction_impact_methods:\n"
                + "  external_flow:\n"
                + "    method: modified_dietz\n"
                + "    flow_timing: trade_date\n"
                + "    day_count: actual_days\n"
                + "    inclusion_rule: beginning_of_day\n"
                + "    denominator_source: begin_market_value\n"
                + "    double_count_policy: cross_check_only\n",
                encoding="utf-8",
            )

            findings = compare_snapshots(specification_path)
            report = performance_comparison_markdown_report(findings)

        top_evidence = _section(
            report,
            "## Top Evidence",
            "## Suppressed Findings Appendix",
        )

        self.assertIn("modified_dietz cross-check estimate", top_evidence)
        self.assertIn("0.005483870968", top_evidence)
        self.assertIn("|  | no_estimate |", top_evidence)

    def test_html_report_summarizes_restatement_findings(self) -> None:
        """HTML reports include the same reviewer-facing sections and tables."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_html_report(
            findings,
            title="HTML <Restatement>",
            top_evidence_limit=2,
        )

        self.assertTrue(report.startswith("<!DOCTYPE html>"))
        self.assertIn("<title>HTML &lt;Restatement&gt;</title>", report)
        self.assertIn("<h1>HTML &lt;Restatement&gt;</h1>", report)
        self.assertIn('id="impact-coverage"', report)
        self.assertIn("Impact Coverage", report)
        self.assertIn("Residual Status", report)
        self.assertIn("Transaction Activity", report)
        self.assertIn("security_contribution", report)
        self.assertIn("0.001084292504", report)
        self.assertIn("PC-PORT-MV", report)
        self.assertNotIn("PC-TXN-AMT", _html_section(report, "top-evidence"))

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

    def test_html_report_handles_empty_findings(self) -> None:
        """Baseline HTML reports render stable empty-state sections."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        report = performance_comparison_html_report(findings)

        self.assertIn("No portfolio return changes to narrate.", report)
        self.assertIn("No impact estimates are currently available.", report)
        self.assertIn("No changed transaction activity.", report)
        self.assertIn("No cause summary available.", report)

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
        self.assertIn("- Impact Coverage", contents)
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
            report.index("## Impact Coverage"),
        )
        self.assertLess(
            report.index("## Impact Coverage"),
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

    def test_write_html_report_creates_parent_directory(self) -> None:
        """HTML reports can be written as durable artifacts."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "report.html"

            written_path = write_performance_comparison_html_report(
                findings,
                output_path,
                title="Controlled HTML Restatement",
                top_evidence_limit=2,
            )

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("<h1>Controlled HTML Restatement</h1>", report)
            self.assertIn("Top Evidence", report)

    def test_write_report_bundle_creates_review_artifacts(self) -> None:
        """Report bundles contain Markdown, CSV tables, and manifest metadata."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        expected_keys = {
            "report",
            "html_report",
            "findings",
            "portfolio_period_summary",
            "cause_summary",
            "impact_estimates",
            "impact_coverage",
            "residual_status",
            "transaction_activity",
            "top_evidence",
            "manifest",
        }
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            paths = write_performance_comparison_report_bundle(
                findings,
                output_directory,
                title="Bundle Restatement",
                top_evidence_limit=2,
            )

            self.assertEqual(set(paths), expected_keys)
            for path in paths.values():
                self.assertTrue(path.exists(), path)
            self.assertIn("# Bundle Restatement", paths["report"].read_text())
            self.assertIn("<h1>Bundle Restatement</h1>", paths["html_report"].read_text())

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["bundle_type"], "performance_comparison_report")
            self.assertEqual(manifest["title"], "Bundle Restatement")
            self.assertEqual(manifest["counts"]["findings"], 21)
            self.assertEqual(manifest["counts"]["active_findings"], 21)
            self.assertEqual(manifest["options"]["top_evidence_limit"], 2)
            self.assertEqual(manifest["artifacts"]["manifest"], "manifest.json")
            self.assertEqual(manifest["artifacts"]["html_report"], "report.html")
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)

            impact_coverage = pl.read_csv(paths["impact_coverage"])
            self.assertEqual(impact_coverage.height, 1)
            self.assertIn("estimated_cause_area_count", impact_coverage.columns)
            self.assertIn("transaction_semantics_sources", impact_coverage.columns)
            self.assertEqual(impact_coverage["estimated_cause_area_count"][0], 2)

            top_evidence = pl.read_csv(paths["top_evidence"])
            self.assertEqual(top_evidence.height, 2)
            self.assertIn("review_rank", top_evidence.columns)
            self.assertIn("transaction_semantics_source", top_evidence.columns)
            self.assertIn("transaction_impact_policy", top_evidence.columns)
            self.assertIn("impact_method", top_evidence.columns)
            self.assertIn("impact_message", top_evidence.columns)

    def test_write_report_bundle_preserves_empty_table_columns(self) -> None:
        """Report bundles write stable CSV headers for baseline empty tables."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(findings, directory)

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["findings"], 0)
            self.assertEqual(manifest["tables"]["impact_coverage"]["rows"], 0)
            self.assertIn(
                "portfolio_id,from_date,thru_date",
                paths["impact_coverage"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "portfolio_id,security_id,from_date",
                paths["transaction_activity"].read_text(encoding="utf-8"),
            )


def _section(report: str, start: str, end: str) -> str:
    """Return report text between two section markers."""
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _html_section(report: str, section_id: str) -> str:
    """Return one HTML section by id."""
    start = f'<section class="pc-section" id="{section_id}">'
    return report.split(start, maxsplit=1)[1].split("</section>", maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
