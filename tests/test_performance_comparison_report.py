"""Tests for performance comparison Markdown reporting."""

# Python imports
from collections.abc import Mapping
import datetime as dt
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    DIRECT_INPUT,
    Finding,
    TARGET_OUTPUT,
    columns as pc_cols,
    compare_snapshots,
    findings_to_polars,
    performance_comparison_html_report,
    performance_comparison_markdown_report,
    validate_causal_attribution_ready,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
    write_performance_comparison_review_workbook,
)
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    PC_PORT_MV,
    PC_PORT_RET,
    SEVERITY_MATERIAL,
)
from ppar.performance_comparison.report import (
    _REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    _markdown_table,
    _problem_table,
    _report_bundle_validation_issues,
    _review_dashboard_table,
    _workbook_portfolio_changes_table,
    _workbook_security_changes_table,
    _workbook_underlying_causes_table,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_MULTI_RESTATEMENT_COMPARISON_PATH = Path(
    "ppar/demo_data/axys/ppar_performance_comparison_multi_restatement.yaml"
)
_FULL_SPEC_COMPARISON_PATH = Path(
    "ppar/demo_data/axys/ppar_performance_comparison_full_spec.yaml"
)
_POLICY_GAP_DEMO_COMPARISON_PATH = Path(
    "ppar/demo_data/axys/ppar_performance_comparison_policy_gap_demo.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)
_original_import = __import__


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
                "transaction_impact_methods:",
                "  performance:",
                "    method: transaction_amount_delta_over_return_denominator",
                "    denominator_source: begin_market_value",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return specification_path


def _write_transaction_commission_review_specification(directory: Path) -> Path:
    """Write a minimal transaction commission review-only fixture."""
    for snapshot_name, commission, portfolio_return in (
        ("snapshot_a", "5.00", "0.0100"),
        ("snapshot_b", "7.50", "0.0101"),
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
            "AMOUNT,COMMISSION,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,"
            f"-100.00,{commission},cash out,performance\n",
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
                "transaction_impact_methods:",
                "  commission:",
                "    method: evidence_only",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return specification_path


def _write_position_estimate_specification(
    directory: Path,
    *,
    include_position_impact_methods: bool,
    include_accrued_impact_methods: bool = False,
) -> Path:
    """Write a minimal source-loaded fixture with position market value changes."""
    for snapshot_name, portfolio_return, market_value, accrued in (
        ("snapshot_a", "0.0100", "1000.00", "25.00"),
        ("snapshot_b", "0.0110", "1010.00", "30.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir(parents=True)
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            f"PORT_A,2025-05-01,2025-05-31,1000.00,{portfolio_return}\n",
            encoding="utf-8",
        )
        (snapshot_path / "positions.csv").write_text(
            "PORT,SEC,POSITION_DATE,QTY,MKT_VAL,ACCRUED\n"
            f"PORT_A,AAPL,2025-05-31,10,{market_value},{accrued}\n",
            encoding="utf-8",
        )

    lines = [
        "snapshots:",
        "  a:",
        "    path: snapshot_a",
        "  b:",
        "    path: snapshot_b",
        "files:",
        "  portfolio_performance: portperf.csv",
        "  positions: positions.csv",
    ]
    if include_position_impact_methods:
        lines.extend(
            [
                "position_impact_methods:",
                "  market_value:",
                "    method: market_value_delta_over_return_denominator",
                "    denominator_source: begin_market_value",
            ]
        )
    if include_accrued_impact_methods:
        if not include_position_impact_methods:
            lines.append("position_impact_methods:")
        lines.extend(
            [
                "  accrued:",
                "    method: accrued_delta_over_return_denominator",
                "    denominator_source: begin_market_value",
            ]
        )

    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return specification_path


def _write_price_estimate_specification(
    directory: Path,
    *,
    include_price_impact_methods: bool,
) -> Path:
    """Write a minimal source-loaded fixture with linked price changes."""
    for snapshot_name, portfolio_return, price in (
        ("snapshot_a", "0.0100", "100.00"),
        ("snapshot_b", "0.0110", "101.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir(parents=True)
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            f"PORT_A,2025-05-01,2025-05-31,{portfolio_return}\n",
            encoding="utf-8",
        )
        (snapshot_path / "secperf.csv").write_text(
            "PORTFOLIO_CODE,SEC,FROM_DATE,THRU_DATE,SEC_RETURN,WEIGHT\n"
            "PORT_A,AAPL,2025-05-01,2025-05-31,0.01,0.20\n",
            encoding="utf-8",
        )
        (snapshot_path / "prices.csv").write_text(
            "SEC,PRICE_DATE,PRICE\n"
            f"AAPL,2025-05-31,{price}\n",
            encoding="utf-8",
        )

    lines = [
        "snapshots:",
        "  a:",
        "    path: snapshot_a",
        "  b:",
        "    path: snapshot_b",
        "files:",
        "  portfolio_performance: portperf.csv",
        "  security_performance: secperf.csv",
        "  prices: prices.csv",
    ]
    if include_price_impact_methods:
        lines.extend(
            [
                "price_impact_methods:",
                "  price:",
                "    method: price_delta_over_snapshot_a_price_times_weight",
                "    weight_source: snapshot_a_weight",
            ]
        )

    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return specification_path


def _assert_workbook_unexplained_formula(
    test_case: unittest.TestCase,
    portfolio_changes: pl.DataFrame,
) -> None:
    """Assert workbook portfolio rows obey performance minus explained math."""
    for row in portfolio_changes.iter_rows(named=True):
        performance_change = _float_or_zero(row.get("performance_change"))
        explained_change = _float_or_zero(row.get("estimated_cause_total"))
        unexplained_change = _float_or_zero(row.get("unexplained_change"))
        test_case.assertAlmostEqual(
            performance_change - explained_change,
            unexplained_change,
            msg=f"{row.get('review_key')} unexplained change does not reconcile.",
        )


def _assert_workbook_explained_rows_reconcile(
    test_case: unittest.TestCase,
    portfolio_changes: pl.DataFrame,
    underlying_causes: pl.DataFrame,
) -> None:
    """Assert visible row-level explained changes reconcile by review key."""
    explained_by_key: dict[object, float] = {}
    for row in underlying_causes.iter_rows(named=True):
        estimated_impact = _float_or_none(row.get("estimated_impact"))
        if estimated_impact is None:
            continue
        review_key = row.get("review_key")
        explained_by_key[review_key] = explained_by_key.get(review_key, 0.0) + (
            estimated_impact
        )

    for row in portfolio_changes.iter_rows(named=True):
        review_key = row.get("review_key")
        explained_change = _float_or_zero(row.get("estimated_cause_total"))
        row_explained_change = explained_by_key.get(review_key, 0.0)
        test_case.assertAlmostEqual(
            row_explained_change,
            explained_change,
            msg=(
                f"{review_key} visible Underlying Causes rows do not match "
                "Explained Difference."
            ),
        )
        if abs(explained_change) > 0:
            test_case.assertIn(
                review_key,
                explained_by_key,
                msg=f"{review_key} has hidden Explained Difference.",
            )


def _assert_workbook_explained_row_actions(
    test_case: unittest.TestCase,
    underlying_causes: pl.DataFrame,
) -> None:
    """Assert explained workbook rows have clear YAML setup wording."""
    for row in underlying_causes.iter_rows(named=True):
        estimated_impact = _float_or_none(row.get("estimated_impact"))
        required_setup = row.get("required_yaml_setup")
        if estimated_impact is None:
            test_case.assertNotEqual(required_setup, "None")
            if row.get("dataset") == "no_underlying_cause_found":
                test_case.assertEqual(row.get("use"), "Diagnostic")
                test_case.assertEqual(row.get("impact_status"), "Review only")
                test_case.assertIn("No underlying input", str(required_setup))
                continue
            if "configured as evidence-only" in str(required_setup):
                test_case.assertEqual(row.get("impact_status"), "Review only")
                continue
            test_case.assertEqual(row.get("impact_status"), "Missing impact method")
            setup_text = str(required_setup)
            test_case.assertTrue(
                "YAML" in setup_text or "impact method" in setup_text,
                msg=f"{row.get('review_key')} has unclear setup: {setup_text}",
            )
            continue

        test_case.assertEqual(row.get("use"), "Explains Change")
        test_case.assertEqual(row.get("impact_status"), "Estimated")
        test_case.assertEqual(required_setup, "None")


def _float_or_none(value: object) -> float | None:
    """Return value as float when it is numeric and not a boolean."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _float_or_zero(value: object) -> float:
    """Return numeric value as float, treating null workbook values as zero."""
    return _float_or_none(value) or 0.0


class TestPerformanceComparisonReport(unittest.TestCase):
    """Verify Markdown report rendering for comparison findings."""

    # pylint: disable-next=too-many-statements
    def test_markdown_report_summarizes_restatement_findings(self) -> None:
        """Restatement reports include run, period, cause, and evidence sections."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("# Performance Comparison Report", report)
        self.assertIn("## Report Contents", report)
        self.assertIn("- Run Summary", report)
        self.assertIn("- Suppressed Findings Appendix", report)
        self.assertIn("## Run Summary", report)
        self.assertIn("- Total findings: 22", report)
        self.assertIn("- Active findings: 22", report)
        self.assertIn("### Reviewer Triage", report)
        self.assertIn("- Changed periods: 1", report)
        self.assertIn("- Needs-review periods: 1", report)
        self.assertIn("- Evidence-only cause areas: 4", report)
        self.assertIn("- Context evidence groups: 4", report)
        self.assertIn("- High-priority context groups: 1", report)
        self.assertIn("- Residual-withheld periods: 1", report)
        self.assertIn("PC-POS-COST", report)
        self.assertIn("## Needs Review Summary", report)
        needs_review = _section(
            report,
            "## Needs Review Summary",
            "## Portfolio-Period Narrative",
        )
        self.assertIn("needs_review", needs_review)
        self.assertIn("Review Key", needs_review)
        self.assertIn("PORT_A::2025-05-30::2025-05-30", needs_review)
        self.assertIn("Review Detail Artifacts", needs_review)
        self.assertIn("impact_coverage.csv", needs_review)
        self.assertIn("context_evidence.csv", needs_review)
        self.assertIn("4 evidence-only area(s)", needs_review)
        self.assertIn("missing inputs: return-impact method", needs_review)
        self.assertIn("high-priority context: positions/cost", needs_review)
        self.assertIn("residual withheld", needs_review)
        self.assertIn("Resolve missing impact inputs before interpreting estimates.", needs_review)
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
        self.assertIn("## Context Evidence Summary", report)
        context_summary = _section(
            report,
            "## Context Evidence Summary",
            "## Context Evidence\n",
        )
        self.assertIn("security_master", context_summary)
        self.assertIn("positions", context_summary)
        self.assertIn("cost-basis review context", context_summary)
        self.assertIn("Review Priority", context_summary)
        self.assertIn("Linked to one or more changed portfolio periods", context_summary)
        self.assertIn("## Context Evidence", report)
        context_evidence = _section(
            report,
            "## Context Evidence\n",
            "## Transaction Cross-Checks",
        )
        self.assertIn("PC-POS-COST", context_evidence)
        self.assertIn("cost-basis review context", context_evidence)
        self.assertIn("Review Priority", context_evidence)
        self.assertIn("Linked to one or more changed portfolio periods", context_evidence)
        self.assertIn("not included in return-impact estimates", context_evidence)
        self.assertIn("## Transaction Cross-Checks", report)
        impact_coverage = _section(
            report,
            "## Impact Coverage",
            "## Transaction Cross-Checks",
        )
        self.assertIn("| PORT_A | 2025-05-30 | 2025-05-30 | 0.0005 | 6 | 2 | 4 |", impact_coverage)
        self.assertIn("0.001084292504", impact_coverage)
        self.assertIn("transaction_activity", impact_coverage)
        self.assertIn("Transaction Semantics Sources", impact_coverage)
        self.assertIn("unknown: 3", impact_coverage)
        self.assertIn("return-impact method", impact_coverage)
        self.assertIn("missing_inputs", impact_coverage)
        self.assertIn("Resolve missing inputs before relying on impact totals", impact_coverage)
        self.assertIn("2 cause area(s) have estimates", impact_coverage)
        self.assertIn("## Residual Status", report)
        residual_status = _section(
            report,
            "## Residual Status",
            "## Transaction Activity",
        )
        self.assertIn("Residual amounts are intentionally withheld", residual_status)
        self.assertIn("withheld_partial_estimates", residual_status)
        self.assertIn("partial or overlapping estimates", residual_status)
        self.assertIn("do not reconcile the remaining difference as residual", residual_status)
        self.assertIn("## Transaction Activity", report)
        transaction_activity = _section(
            report,
            "## Transaction Activity",
            "## Transaction Matching Diagnostics",
        )
        self.assertIn("buy", transaction_activity)
        self.assertIn("Transaction Semantics Sources", transaction_activity)
        self.assertIn("unknown: 3", transaction_activity)
        self.assertIn(
            "| PORT_A | AAPL | 2025-05-30 | 2025-05-30 | buy |",
            transaction_activity,
        )
        self.assertIn("amount, quantity, price", transaction_activity)
        self.assertIn("transaction sign and flow semantics", transaction_activity)
        self.assertNotIn("portfolio period, return denominator", transaction_activity)
        self.assertIn("## Transaction Matching Diagnostics", report)
        transaction_matching = _section(
            report,
            "## Transaction Matching Diagnostics",
            "## Portfolio-Period Changes",
        )
        self.assertIn("transaction_id_match", transaction_matching)
        self.assertIn("Changed fields were compared", transaction_matching)
        self.assertIn("## Portfolio-Period Changes", report)
        self.assertIn("| PORT_A | 2025-05-30 | 2025-05-30 | 0.0005 | 18 | no |", report)
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
            "## Transaction Cross-Checks",
        )
        transaction_activity = _section(
            report,
            "## Transaction Activity",
            "## Transaction Matching Diagnostics",
        )
        top_evidence = _section(
            report,
            "## Top Evidence",
            "## Suppressed Findings Appendix",
        )

        self.assertIn("mixed: 3", impact_coverage)
        self.assertIn("mixed: 3", transaction_activity)
        self.assertNotIn("transaction sign and flow semantics", transaction_activity)
        self.assertIn("Transaction Semantics Source", top_evidence)
        self.assertIn("Impact Policy", top_evidence)
        self.assertIn("Transaction Impact Diagnostic", top_evidence)
        self.assertIn("mixed", top_evidence)
        self.assertIn(
            "performance:transaction_amount_delta_over_return_denominator",
            top_evidence,
        )
        self.assertIn("| -100 |", top_evidence)

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
            "## Transaction Matching Diagnostics",
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
        self.assertIn("Impact Policy", top_evidence)
        self.assertIn("Transaction Impact Diagnostic", top_evidence)
        self.assertIn("source", top_evidence)
        self.assertIn(
            "performance:transaction_amount_delta_over_return_denominator",
            top_evidence,
        )
        self.assertIn("| -10 |", top_evidence)
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
        cross_checks = _section(
            report,
            "## Transaction Cross-Checks",
            "## Flow Cross-Check Reconciliation",
        )
        reconciliation = _section(
            report,
            "## Flow Cross-Check Reconciliation",
            "## Residual Status",
        )

        self.assertIn("modified_dietz cross-check estimate", top_evidence)
        self.assertIn("0.005483870968", top_evidence)
        self.assertIn("|  | no_estimate |", top_evidence)
        self.assertIn("## Needs Review Summary", report)
        needs_review = _section(
            report,
            "## Needs Review Summary",
            "## Portfolio-Period Narrative",
        )
        self.assertIn("1 transaction cross-check(s): external_flow:modified_dietz", needs_review)
        self.assertIn("modified_dietz cross-check only", needs_review)
        self.assertIn("external_flow:modified_dietz", cross_checks)
        self.assertIn("cross_check_only", cross_checks)
        self.assertIn("0.005483870968", cross_checks)
        self.assertIn("not included in estimated impact totals", cross_checks)
        self.assertIn("missing_portfolio_flow_delta", reconciliation)
        self.assertIn("0.005483870968", reconciliation)
        residual_status = _section(
            report,
            "## Residual Status",
            "## Transaction Activity",
        )
        self.assertIn("withheld_cross_checks_only", residual_status)
        self.assertIn("transaction cross-checks only", residual_status)
        self.assertIn("review-only cross-check estimates exist", residual_status)

    def test_html_report_summarizes_restatement_findings(self) -> None:
        """HTML reports include the same reviewer-facing sections and tables."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_html_report(
            findings,
            title="HTML <Restatement>",
            top_evidence_limit=2,
        )

        self._assert_html_report_shell(report)
        self.assertIn('id="problems"', report)
        self.assertLess(
            report.index('id="problems"'),
            report.index('id="evidence-appendix"'),
        )
        self.assertIn('id="evidence-appendix"', report)
        self.assertIn('<details class="pc-detail">', report)
        self.assertIn("<summary>Portfolio-Period Narrative</summary>", report)
        self.assertIn("<summary>Impact Coverage</summary>", report)
        self.assertIn("<summary>Top Evidence</summary>", report)
        self.assertIn("<summary>Run Summary</summary>", report)
        self.assertLess(
            report.index('id="evidence-appendix"'),
            report.index('id="run-summary"'),
        )
        self.assertNotIn('id="at-a-glance"', report)
        problems = _html_section(report, "problems")
        self.assertIn("Start here", problems)
        self.assertIn(
            "1 of 1 problem(s) need review across 1 portfolio(s).",
            problems,
        )
        self.assertIn(
            'id="problems--port-a--2025-05-30--2025-05-30"',
            problems,
        )
        self.assertIn('data-dashboard-filters', problems)
        self.assertIn('data-dashboard-search', problems)
        self.assertIn('data-dashboard-status', problems)
        self.assertIn('data-dashboard-missing-only', problems)
        self.assertIn('data-dashboard-sort="portfolio"', problems)
        self.assertIn('data-dashboard-sort="return-delta"', problems)
        self.assertIn("Missing inputs only", problems)
        self.assertIn("No problem rows match the filters.", problems)
        self.assertIn('data-dashboard-row', problems)
        self.assertIn('data-review-status="needs_review"', problems)
        self.assertIn('data-missing-inputs="true"', problems)
        self.assertIn("PORT_A", problems)
        self.assertIn("PORT_A::2025-05-30::2025-05-30", report)
        self.assertIn('data-dashboard-sort="severity">Severity</button>', problems)
        self.assertIn('data-dashboard-sort="portfolio">Portfolio</button>', problems)
        self.assertIn(
            'data-dashboard-sort="return-delta">Return Delta</button>',
            problems,
        )
        self.assertIn('data-dashboard-sort="problem">Problem</button>', problems)
        self.assertIn(
            'data-dashboard-sort="action">Action Required</button>',
            problems,
        )
        self.assertIn("Return-impact estimate is blocked", problems)
        self.assertIn("Update the comparison YAML", problems)
        self.assertIn("transaction_impact_methods", problems)
        self.assertIn("transaction_impact_methods", problems)
        self.assertIn("transaction sign and flow semantics", problems)
        self.assertIn("define transaction sign and external-flow semantics", problems)
        self.assertIn("ppar can show evidence", problems)
        self.assertIn(
            'href="#impact-coverage--port-a--2025-05-30--2025-05-30"',
            problems,
        )
        self.assertNotIn('href="#context-evidence--', problems)
        self.assertNotIn('href="#transaction-activity--', problems)
        self.assertIn(
            'id="impact-coverage--port-a--2025-05-30--2025-05-30"',
            report,
        )
        self.assertIn(
            'id="context-evidence--port-a--2025-05-30--2025-05-30"',
            report,
        )
        self.assertIn(
            'id="transaction-activity--port-a--2025-05-30--2025-05-30"',
            report,
        )
        self.assertIn(
            'id="top-evidence--port-a--2025-05-30--2025-05-30"',
            report,
        )
        narrative = _html_section(report, "portfolio-period-narrative")
        self.assertIn("PORT_A changed by 0.0005", narrative)
        self.assertIn("The strongest currently estimated impact", narrative)
        self.assertIn('class="pc-dashboard-table"', problems)
        self.assertIn("querySelector(\"[data-dashboard-filters]\")", report)
        self.assertIn("querySelectorAll(\"[data-dashboard-sort]\")", report)
        self.assertIn("compareValues", report)
        self.assertIn("row.hidden = !visible", report)
        self.assertIn('id="impact-coverage"', report)
        self.assertIn('id="needs-review-summary"', report)
        self.assertIn("Impact Coverage", report)
        self.assertIn("Residual Status", report)
        self.assertIn("Context Evidence Summary", report)
        self.assertIn("Context Evidence", report)
        self.assertIn("Transaction Activity", report)
        self.assertIn('id="context-evidence-summary"', report)
        self.assertIn('id="context-evidence"', report)
        self.assertIn("Reviewer Triage", report)
        self.assertIn("Changed periods", report)
        self.assertIn("Context evidence groups", report)
        self.assertIn("High-priority context groups", report)
        self.assertIn('class="pc-card-row pc-triage-row"', report)
        self.assertIn('<p class="pc-table-meta">Rows: 1</p>', report)
        self.assertIn("pc-col-portfolio-return-delta", report)
        self.assertIn("pc-status-needs-review", report)
        self.assertIn("@media print", report)
        self.assertIn("security_contribution", report)
        self.assertIn("0.001084292504", report)
        self.assertIn("PC-PORT-MV", report)
        review_notes = _html_section(report, "review-notes")
        self.assertIn("Unless noted otherwise", review_notes)
        self.assertIn("Impact estimates are intentionally conservative", review_notes)
        self.assertNotIn("PC-TXN-AMT", _html_section(report, "top-evidence"))

    def test_review_dashboard_prioritizes_missing_inputs_across_portfolios(self) -> None:
        """Dashboard rows sort urgent multi-portfolio review periods first."""
        period_date = dt.date(2025, 5, 30)
        needs_review = pl.DataFrame(
            [
                {
                    "review_key": "PORT_LARGE::2025-05-30::2025-05-30",
                    "portfolio_id": "PORT_LARGE",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "portfolio_return_delta": 0.02,
                    "review_status": "needs_review",
                    "review_cues": "4 evidence-only area(s)",
                    "suggested_next_step": "Review evidence-only areas.",
                    "review_detail_artifacts": "impact_coverage.csv, findings.csv",
                },
                {
                    "review_key": "PORT_MISSING::2025-05-30::2025-05-30",
                    "portfolio_id": "PORT_MISSING",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "portfolio_return_delta": 0.001,
                    "review_status": "needs_review",
                    "review_cues": "missing inputs: return-impact method",
                    "suggested_next_step": "Resolve missing impact inputs.",
                    "review_detail_artifacts": "impact_coverage.csv, findings.csv",
                },
                {
                    "review_key": "PORT_MONITOR::2025-05-30::2025-05-30",
                    "portfolio_id": "PORT_MONITOR",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "portfolio_return_delta": 0.05,
                    "review_status": "monitor",
                    "review_cues": "1 low-confidence estimate(s)",
                    "suggested_next_step": "Review low-confidence estimates.",
                    "review_detail_artifacts": "impact_estimates.csv, findings.csv",
                },
            ]
        )
        coverage = pl.DataFrame(
            [
                {
                    "portfolio_id": "PORT_LARGE",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "estimated_cause_area_count": 2,
                    "evidence_only_cause_area_count": 4,
                    "missing_impact_inputs": "",
                    "impact_coverage_status": "partial_estimates",
                    "impact_coverage_review_note": "Review evidence-only areas.",
                },
                {
                    "portfolio_id": "PORT_MISSING",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "estimated_cause_area_count": 1,
                    "evidence_only_cause_area_count": 2,
                    "missing_impact_inputs": "return-impact method",
                    "impact_coverage_status": "missing_inputs",
                    "impact_coverage_review_note": "Resolve missing inputs.",
                },
                {
                    "portfolio_id": "PORT_MONITOR",
                    "from_date": period_date,
                    "thru_date": period_date,
                    "estimated_cause_area_count": 1,
                    "evidence_only_cause_area_count": 0,
                    "missing_impact_inputs": "",
                    "impact_coverage_status": "complete_estimates",
                    "impact_coverage_review_note": "All areas estimated.",
                },
            ]
        )
        with (
            mock.patch(
                "ppar.performance_comparison.report._needs_review_summary_table",
                return_value=needs_review,
            ),
            mock.patch(
                "ppar.performance_comparison.report._pc_explain."
                "portfolio_period_impact_coverage_summary",
                return_value=coverage,
            ),
        ):
            dashboard = _review_dashboard_table(pl.DataFrame())

        self.assertEqual(
            dashboard["portfolio_id"].to_list(),
            ["PORT_MISSING", "PORT_LARGE", "PORT_MONITOR"],
        )
        self.assertEqual(
            dashboard["dashboard_coverage_counts"][0],
            "1 estimated / 2 evidence-only",
        )
        self.assertEqual(
            dashboard["dashboard_main_issue"][0],
            "Missing inputs: return-impact method",
        )
        self.assertEqual(dashboard["dashboard_open_section"][0], "impact-coverage")

    def test_multi_portfolio_fixture_renders_dashboard_variety(self) -> None:
        """The demo fixture gives the HTML report multiple issue shapes."""
        findings = compare_snapshots(_MULTI_RESTATEMENT_COMPARISON_PATH)

        dashboard = _review_dashboard_table(findings)
        problems = _problem_table(findings)
        report = performance_comparison_html_report(
            findings,
            title="Multi-Portfolio HTML",
            top_evidence_limit=2,
        )

        self.assertEqual(
            dashboard["portfolio_id"].to_list(),
            ["PORT_B", "PORT_C", "PORT_A"],
        )
        self.assertEqual(
            dashboard["impact_coverage_status"].to_list(),
            ["missing_inputs", "missing_inputs", "missing_inputs"],
        )
        self.assertEqual(
            dashboard["dashboard_coverage_counts"].to_list(),
            [
                "3 estimated / 2 evidence-only",
                "1 estimated / 3 evidence-only",
                "3 estimated / 3 evidence-only",
            ],
        )
        self.assertEqual(
            problems["portfolio_id"].to_list(),
            ["PORT_B", "PORT_C", "PORT_A"],
        )
        self.assertIn("Update the comparison YAML", problems["action_required"][0])
        self.assertIn("transaction_impact_methods", problems["action_required"][0])
        problems_section = _html_section(report, "problems")
        self.assertEqual(problems_section.count('data-dashboard-row'), 3)
        self.assertIn("PORT_A", problems_section)
        self.assertIn("PORT_B", problems_section)
        self.assertIn("PORT_C", problems_section)
        self.assertIn("Return-impact estimate is blocked", problems_section)
        self.assertIn("transaction_impact_methods", problems_section)
        narrative_section = _html_section(report, "portfolio-period-narrative")
        self.assertIn("PORT_C changed by 0.0008", narrative_section)
        self.assertIn("PORT_B changed by 0.0015", narrative_section)

    def test_policy_gap_demo_fixture_surfaces_yaml_actions(self) -> None:
        """The demo policy-gap fixture produces specific YAML remediation rows."""
        findings = compare_snapshots(_POLICY_GAP_DEMO_COMPARISON_PATH)

        problems = _problem_table(findings)
        report = performance_comparison_html_report(
            findings,
            title="Policy Gap HTML",
            top_evidence_limit=2,
        )

        self.assertEqual(problems.height, 3)
        self.assertEqual(
            problems["portfolio_id"].to_list(),
            ["PORT_B", "PORT_C", "PORT_A"],
        )
        problem_rows = {
            row["portfolio_id"]: row
            for row in problems.iter_rows(named=True)
        }
        self.assertIn(
            "select the relevant `contribution_impact_methods` policy",
            problem_rows["PORT_B"]["action_required"],
        )
        self.assertIn(
            "`transaction_impact_methods`",
            problem_rows["PORT_B"]["action_required"],
        )
        self.assertIn(
            "define transaction sign and external-flow semantics",
            problem_rows["PORT_C"]["action_required"],
        )
        self.assertIn(
            "configure `transaction_impact_methods` with an explicit method",
            problem_rows["PORT_A"]["action_required"],
        )

        problems_section = _html_section(report, "problems")
        self.assertEqual(problems_section.count('data-dashboard-row'), 3)
        self.assertIn("contribution_impact_methods", problems_section)
        self.assertIn("transaction_impact_methods", problems_section)
        self.assertIn(
            "define transaction sign and external-flow semantics",
            problems_section,
        )

    def _assert_html_report_shell(self, report: str) -> None:
        """Verify stable HTML report framing and review-oriented polish."""
        self.assertTrue(report.startswith("<!DOCTYPE html>"))
        self.assertIn("<title>HTML &lt;Restatement&gt;</title>", report)
        self.assertIn("<h1>HTML &lt;Restatement&gt;</h1>", report)
        self.assertNotIn("Performance Comparison Review", report)
        self.assertNotIn("Exception Review Worksheet", report)
        self.assertNotIn('class="pc-kicker"', report)
        self.assertNotIn('class="pc-subtitle"', report)
        self.assertNotIn('class="pc-header-notes"', report)
        self.assertIn('id="problems"', report)
        self.assertIn('id="evidence-appendix"', report)
        self.assertIn("Review table with", report)
        self.assertNotIn('<ol class="pc-contents-list">', report)

    def test_markdown_report_limits_top_evidence_per_portfolio_period(self) -> None:
        """Top evidence limit controls displayed contribution candidate rows."""
        findings = compare_snapshots(_MULTI_RESTATEMENT_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings, top_evidence_limit=2)

        top_evidence = _section(report, "## Top Evidence", "## Suppressed Findings Appendix")

        self.assertIn("PC-PORT-MV", top_evidence)
        self.assertEqual(top_evidence.count("| PORT_A | 2025-05-30 | 2025-05-30 |"), 2)
        self.assertNotIn("PC-TXN-AMT", top_evidence)

    def test_markdown_report_separates_suppressed_findings(self) -> None:
        """Suppressed findings are counted and detailed in the appendix."""
        findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("- Total findings: 22", report)
        self.assertIn("- Active findings: 21", report)
        self.assertIn("- Suppressed findings: 1", report)
        self.assertIn("| PC-SEC-RET | yes | 1 |", report)
        self.assertIn("Suppressed Finding Detail", report)

    def test_markdown_report_handles_empty_findings(self) -> None:
        """Baseline reports render stable empty-state sections."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        report = performance_comparison_markdown_report(findings)

        self.assertIn("- Total findings: 0", report)
        self.assertIn("- Changed periods: 0", report)
        self.assertIn("- Needs-review periods: 0", report)
        self.assertIn("- Context evidence groups: 0", report)
        self.assertIn("_No changed portfolio periods need review._", report)
        self.assertIn("_No portfolio return changes to narrate._", report)
        self.assertIn("_No portfolio-period review notes._", report)
        self.assertIn("_No impact estimates are currently available._", report)
        self.assertIn("_No context-only evidence summary._", report)
        self.assertIn("_No context-only evidence._", report)
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
        self.assertIn("No actionable performance comparison problems.", report)
        self.assertIn("No changed portfolio periods need review.", report)
        self.assertIn("No impact estimates are currently available.", report)
        self.assertIn("No context-only evidence summary.", report)
        self.assertIn("No context-only evidence.", report)
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
        self.assertIn("withheld_no_estimates", residual_status)
        self.assertIn("no defensible impact estimates", residual_status)
        self.assertIn("would equal the whole return delta", residual_status)
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
        self.assertIn("- Needs Review Summary", contents)
        self.assertIn("- Impact Coverage", contents)
        self.assertIn("- Context Evidence Summary", contents)
        self.assertIn("- Context Evidence", contents)
        self.assertIn("- Transaction Cross-Checks", contents)
        self.assertIn("- Flow Cross-Check Reconciliation", contents)
        self.assertIn("- Residual Status", contents)
        self.assertIn("- Transaction Activity", contents)
        self.assertIn("- Transaction Matching Diagnostics", contents)
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
            report.index("## Run Summary"),
            report.index("## Needs Review Summary"),
        )
        self.assertLess(
            report.index("## Needs Review Summary"),
            report.index("## Portfolio-Period Narrative"),
        )
        self.assertLess(
            report.index("## Impact Estimate Summary"),
            report.index("## Impact Coverage"),
        )
        self.assertLess(
            report.index("## Impact Coverage"),
            report.index("## Context Evidence Summary"),
        )
        self.assertLess(
            report.index("## Context Evidence Summary"),
            report.index("## Context Evidence\n"),
        )
        self.assertLess(
            report.index("## Context Evidence\n"),
            report.index("## Transaction Cross-Checks"),
        )
        self.assertLess(
            report.index("## Transaction Cross-Checks"),
            report.index("## Flow Cross-Check Reconciliation"),
        )
        self.assertLess(
            report.index("## Flow Cross-Check Reconciliation"),
            report.index("## Residual Status"),
        )
        self.assertLess(
            report.index("## Residual Status"),
            report.index("## Transaction Activity"),
        )
        self.assertLess(
            report.index("## Transaction Activity"),
            report.index("## Transaction Matching Diagnostics"),
        )
        self.assertLess(
            report.index("## Transaction Matching Diagnostics"),
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
        expected_keys = set(_REPORT_BUNDLE_REQUIRED_ARTIFACTS)
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
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("# Bundle Restatement", readme)
            self.assertIn("## Primary Review Artifact", readme)
            self.assertIn("`report.html`: primary browser review report", readme)
            self.assertIn("## Secondary Review Views", readme)
            self.assertIn("`report.html`: browser-friendly narrative report", readme)
            self.assertIn("## Recommended Review Order", readme)
            self.assertIn("start with the Problems grid", readme)
            self.assertIn("high-priority context cues", readme)
            self.assertIn("review guidance only", readme)
            self.assertIn("## Audit/Export Files", readme)
            self.assertIn("`manifest.json`: machine-readable artifact", readme)
            self.assertIn(
                "`needs_review_summary.csv`: top triage table for changed periods",
                readme,
            )
            self.assertIn(
                "`context_evidence_summary.csv`: context-only evidence counts, "
                "reviewer priority",
                readme,
            )
            self.assertIn(
                "`context_evidence.csv`: row-level context evidence, reviewer priority",
                readme,
            )

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["bundle_type"], "performance_comparison_report")
            self.assertEqual(manifest["title"], "Bundle Restatement")
            self.assertEqual(manifest["counts"]["findings"], 22)
            self.assertEqual(manifest["counts"]["active_findings"], 22)
            self.assertEqual(manifest["options"]["top_evidence_limit"], 2)
            self.assertEqual(manifest["artifacts"]["manifest"], "manifest.json")
            self.assertEqual(manifest["artifacts"]["html_report"], "report.html")
            self.assertEqual(manifest["artifacts"]["readme"], "README.md")
            self.assertEqual(
                manifest["artifacts"]["needs_review_summary"],
                "needs_review_summary.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence"],
                "context_evidence.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence_summary"],
                "context_evidence_summary.csv",
            )
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
            self.assertEqual(manifest["tables"]["needs_review_summary"]["rows"], 1)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 4)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 4)

            needs_review = pl.read_csv(paths["needs_review_summary"])
            self.assertEqual(needs_review.height, 1)
            self.assertIn("review_key", needs_review.columns)
            self.assertIn("review_detail_artifacts", needs_review.columns)
            self.assertEqual(
                needs_review["review_key"][0],
                "PORT_A::2025-05-30::2025-05-30",
            )
            self.assertIn("review_status", needs_review.columns)
            self.assertEqual(needs_review["review_status"][0], "needs_review")
            self.assertIn(
                "high-priority context: positions/cost",
                needs_review["review_cues"][0],
            )
            self.assertIn(
                "transaction_activity.csv",
                needs_review["review_detail_artifacts"][0],
            )
            self.assertIn(
                "context_evidence.csv",
                needs_review["review_detail_artifacts"][0],
            )

            impact_coverage = pl.read_csv(paths["impact_coverage"])
            self.assertEqual(impact_coverage.height, 1)
            self.assertEqual(
                impact_coverage["review_key"][0],
                "PORT_A::2025-05-30::2025-05-30",
            )
            self.assertIn("estimated_cause_area_count", impact_coverage.columns)
            self.assertIn("transaction_semantics_sources", impact_coverage.columns)
            self.assertIn("impact_coverage_status", impact_coverage.columns)
            self.assertIn("impact_coverage_review_note", impact_coverage.columns)
            self.assertEqual(impact_coverage["estimated_cause_area_count"][0], 2)
            self.assertEqual(impact_coverage["impact_coverage_status"][0], "missing_inputs")

            context_evidence = pl.read_csv(paths["context_evidence"])
            self.assertEqual(context_evidence.height, 4)
            self.assertIn("review_key", context_evidence.columns)
            self.assertIn("context_use", context_evidence.columns)
            self.assertIn("review_priority", context_evidence.columns)
            self.assertIn("review_priority_reason", context_evidence.columns)
            self.assertIn("return_impact_treatment", context_evidence.columns)
            self.assertIn("PC-POS-COST", context_evidence["code"].to_list())
            self.assertEqual(context_evidence["review_priority"][0], "high")
            self.assertEqual(
                set(context_evidence["return_impact_treatment"]),
                {"context only; not included in return-impact estimates"},
            )

            context_evidence_summary = pl.read_csv(paths["context_evidence_summary"])
            self.assertEqual(context_evidence_summary.height, 4)
            self.assertIn("review_priority", context_evidence_summary.columns)
            self.assertIn("review_priority_reason", context_evidence_summary.columns)
            self.assertIn("finding_count", context_evidence_summary.columns)
            self.assertIn("affected_securities", context_evidence_summary.columns)
            self.assertIn("AAPL", context_evidence_summary["affected_securities"].to_list())
            self.assertEqual(context_evidence_summary["review_priority"][0], "high")

            transaction_matching = pl.read_csv(paths["transaction_matching_diagnostics"])
            self.assertEqual(transaction_matching.height, 1)
            self.assertIn("transaction_match_status", transaction_matching.columns)
            self.assertEqual(
                transaction_matching["transaction_match_status"][0],
                "transaction_id_match",
            )
            self.assertIn(
                "transaction matching status counts",
                readme,
            )

            top_evidence = pl.read_csv(paths["top_evidence"])
            self.assertEqual(top_evidence.height, 2)
            self.assertIn("review_key", top_evidence.columns)
            self.assertIn("review_rank", top_evidence.columns)
            self.assertIn("transaction_semantics_source", top_evidence.columns)
            self.assertIn("transaction_impact_policy", top_evidence.columns)
            self.assertIn("impact_method", top_evidence.columns)
            self.assertIn("impact_message", top_evidence.columns)
            self.assertEqual(_report_bundle_validation_issues(output_directory), [])

    def test_strict_causal_attribution_rejects_missing_setup(self) -> None:
        """Strict causal attribution mode fails before ambiguous reporting."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        with self.assertRaises(PpaError) as context:
            validate_causal_attribution_ready(findings)

        message = str(context.exception)
        self.assertIn("Causal attribution setup is incomplete", message)
        self.assertIn("PORT_A::2025-05-30::2025-05-30", message)
        self.assertIn("missing YAML setup", message)

    def test_workbook_tables_follow_review_accounting_invariants(self) -> None:
        """Workbook review tables stay internally consistent across demos."""
        cases = (
            ("single", _RESTATEMENT_COMPARISON_PATH, {}),
            ("transaction_rules", _RESTATEMENT_TRANSACTION_RULES_PATH, {}),
            ("multi", _MULTI_RESTATEMENT_COMPARISON_PATH, {}),
            ("full_spec", _FULL_SPEC_COMPARISON_PATH, {"require_causal_attribution": True}),
            ("policy_gap", _POLICY_GAP_DEMO_COMPARISON_PATH, {}),
        )

        for name, comparison_path, kwargs in cases:
            with self.subTest(name=name):
                findings = compare_snapshots(comparison_path, **kwargs)
                portfolio_changes = _workbook_portfolio_changes_table(findings)
                underlying_causes = _workbook_underlying_causes_table(findings)

                _assert_workbook_unexplained_formula(self, portfolio_changes)
                _assert_workbook_explained_rows_reconcile(
                    self,
                    portfolio_changes,
                    underlying_causes,
                )
                _assert_workbook_explained_row_actions(self, underlying_causes)

    def test_clean_workbook_portfolio_changes_has_no_differences_message(self) -> None:
        """Clean comparisons still give reviewers a visible workbook result."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)

        portfolio_changes = _workbook_portfolio_changes_table(findings)

        self.assertEqual(portfolio_changes.height, 1)
        self.assertEqual(
            portfolio_changes["portfolio_id"][0],
            "No portfolio performance differences found",
        )
        self.assertEqual(portfolio_changes["review_status"][0], "No differences")
        self.assertEqual(portfolio_changes["next_action"][0], "None")
        self.assertEqual(
            portfolio_changes["review_key"][0],
            "NO_PORTFOLIO_PERFORMANCE_DIFFERENCES",
        )
        _assert_workbook_unexplained_formula(self, portfolio_changes)

    def test_transaction_rules_demo_explains_transaction_amount_row(self) -> None:
        """Transaction rules demo exposes the modeled transaction amount impact."""
        plain_findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        plain_causes = _workbook_underlying_causes_table(plain_findings)
        plain_transaction_amount = plain_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "amount")
        )
        self.assertEqual(plain_transaction_amount.height, 1)
        self.assertIsNone(plain_transaction_amount["estimated_impact"][0])
        self.assertIn(
            "transaction_impact_methods.performance.method",
            plain_transaction_amount["required_yaml_setup"][0],
        )

        rules_findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)
        rules_causes = _workbook_underlying_causes_table(rules_findings)
        rules_transaction_amount = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "amount")
        )
        rules_transaction_quantity = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "quantity")
        )
        rules_transaction_price = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "price")
        )
        self.assertEqual(rules_transaction_amount.height, 1)
        self.assertAlmostEqual(
            rules_transaction_amount["estimated_impact"][0],
            -100.0 / 999915.0,
        )
        self.assertEqual(rules_transaction_amount["required_yaml_setup"][0], "None")
        self.assertIsNone(rules_transaction_quantity["estimated_impact"][0])
        self.assertEqual(
            rules_transaction_quantity["required_yaml_setup"][0],
            "None; configured as evidence-only in comparison YAML.",
        )
        self.assertEqual(rules_transaction_quantity["impact_status"][0], "Review only")
        self.assertIsNone(rules_transaction_price["estimated_impact"][0])
        self.assertEqual(
            rules_transaction_price["required_yaml_setup"][0],
            "None; configured as evidence-only in comparison YAML.",
        )
        self.assertEqual(rules_transaction_price["impact_status"][0], "Review only")

    def test_transaction_commission_policy_marks_underlying_cause_review_only(
        self,
    ) -> None:
        """Commission can appear as a review-only underlying cause."""
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_path = _write_transaction_commission_review_specification(
                Path(temp_dir)
            )

            causes = _workbook_underlying_causes_table(
                compare_snapshots(comparison_path),
                comparison_path=comparison_path,
            )

        commission = causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "commission")
        )

        self.assertEqual(commission.height, 1)
        self.assertIsNone(commission["estimated_impact"][0])
        self.assertEqual(commission["impact_status"][0], "Review only")
        self.assertEqual(
            commission["required_yaml_setup"][0],
            "None; configured as evidence-only in comparison YAML.",
        )

    def test_security_differences_roll_up_security_underlying_causes(self) -> None:
        """Security Differences shows explained amounts from security causes."""
        findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)

        security_differences = _workbook_security_changes_table(findings)
        aapl = security_differences.filter(pl.col("security_id") == "AAPL")

        self.assertEqual(aapl.height, 1)
        self.assertAlmostEqual(aapl["performance_change"][0], 0.01)
        self.assertAlmostEqual(aapl["estimated_cause_total"][0], -100.0 / 999915.0)
        self.assertAlmostEqual(
            aapl["unexplained_change"][0],
            0.01 - (-100.0 / 999915.0),
        )

    def test_security_differences_marks_periods_without_security_rows(self) -> None:
        """Portfolio periods without security differences get explicit rows."""
        findings = compare_snapshots(
            _FULL_SPEC_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        security_differences = _workbook_security_changes_table(findings)
        portfolio_differences = _workbook_portfolio_changes_table(findings)
        placeholders = security_differences.filter(
            pl.col("security_id") == "No security performance differences found"
        )
        actual_security_periods = security_differences.filter(
            pl.col("security_id") != "No security performance differences found"
        ).select(["portfolio_id", "from_date", "thru_date"])
        expected_placeholder_count = (
            portfolio_differences.select(["portfolio_id", "from_date", "thru_date"])
            .join(
                actual_security_periods,
                on=["portfolio_id", "from_date", "thru_date"],
                how="anti",
            )
            .height
        )

        self.assertEqual(placeholders.height, expected_placeholder_count)
        self.assertTrue(placeholders["performance_change"].is_null().all())
        self.assertTrue(placeholders["estimated_cause_total"].is_null().all())
        self.assertEqual(set(placeholders["review_status"].to_list()), {"No differences"})
        self.assertEqual(set(placeholders["next_action"].to_list()), {"None"})

    def test_position_impact_method_explains_market_value_row(self) -> None:
        """Position market value YAML exposes the modeled position impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_position_estimate_specification(
                Path(temp_dir) / "plain",
                include_position_impact_methods=False,
            )
            configured_path = _write_position_estimate_specification(
                Path(temp_dir) / "configured",
                include_position_impact_methods=True,
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path),
                comparison_path=configured_path,
            )

        plain_position = plain_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "market_value")
        )
        configured_position = configured_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "market_value")
        )

        self.assertEqual(plain_position.height, 1)
        self.assertIsNone(plain_position["estimated_impact"][0])
        self.assertIn(
            "position_impact_methods.market_value.method",
            plain_position["required_yaml_setup"][0],
        )
        self.assertEqual(configured_position.height, 1)
        self.assertAlmostEqual(configured_position["estimated_impact"][0], 0.01)
        self.assertEqual(configured_position["required_yaml_setup"][0], "None")

    def test_position_accrued_impact_method_explains_accrued_row(self) -> None:
        """Position accrued YAML exposes the modeled accrued-balance impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_position_estimate_specification(
                Path(temp_dir) / "plain",
                include_position_impact_methods=False,
            )
            configured_path = _write_position_estimate_specification(
                Path(temp_dir) / "configured",
                include_position_impact_methods=False,
                include_accrued_impact_methods=True,
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path),
                comparison_path=configured_path,
            )

        plain_accrued = plain_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "accrued")
        )
        configured_accrued = configured_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "accrued")
        )

        self.assertEqual(plain_accrued.height, 1)
        self.assertIsNone(plain_accrued["estimated_impact"][0])
        self.assertIn(
            "position_impact_methods.accrued.method",
            plain_accrued["required_yaml_setup"][0],
        )
        self.assertEqual(configured_accrued.height, 1)
        self.assertAlmostEqual(configured_accrued["estimated_impact"][0], 0.005)
        self.assertEqual(configured_accrued["required_yaml_setup"][0], "None")

    def test_evidence_only_impact_method_marks_row_review_only(self) -> None:
        """Evidence-only YAML removes missing-method guidance for known fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_position_estimate_specification(
                Path(temp_dir) / "plain",
                include_position_impact_methods=False,
            )
            configured_path = _write_position_estimate_specification(
                Path(temp_dir) / "configured",
                include_position_impact_methods=False,
            )
            for comparison_path in (plain_path, configured_path):
                position_path = comparison_path.parent / "snapshot_b" / "positions.csv"
                position_path.write_text(
                    position_path.read_text(encoding="utf-8").replace(
                        "PORT_A,AAPL,2025-05-31,10,1010.00,30.00",
                        "PORT_A,AAPL,2025-05-31,11,1010.00,30.00",
                    ),
                    encoding="utf-8",
                )
            configured_path.write_text(
                configured_path.read_text(encoding="utf-8")
                + "\n"
                + "position_impact_methods:\n"
                + "  quantity:\n"
                + "    method: evidence_only\n",
                encoding="utf-8",
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path),
                comparison_path=configured_path,
            )

        plain_quantity = plain_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "quantity")
        )
        configured_quantity = configured_causes.filter(
            (pl.col("dataset") == "positions")
            & (pl.col("source_column") == "quantity")
        )

        self.assertEqual(plain_quantity.height, 1)
        self.assertEqual(
            plain_quantity["required_yaml_setup"][0],
            "No supported YAML impact method exists yet for positions.quantity.",
        )
        self.assertEqual(plain_quantity["impact_status"][0], "Missing impact method")
        self.assertEqual(configured_quantity.height, 1)
        self.assertEqual(configured_quantity["estimated_impact"][0], None)
        self.assertEqual(configured_quantity["impact_status"][0], "Review only")
        self.assertEqual(
            configured_quantity["required_yaml_setup"][0],
            "None; configured as evidence-only in comparison YAML.",
        )

    def test_price_impact_method_explains_price_row(self) -> None:
        """Price YAML exposes the modeled weighted price impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_price_estimate_specification(
                Path(temp_dir) / "plain",
                include_price_impact_methods=False,
            )
            configured_path = _write_price_estimate_specification(
                Path(temp_dir) / "configured",
                include_price_impact_methods=True,
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path),
                comparison_path=configured_path,
            )

        plain_price = plain_causes.filter(
            (pl.col("dataset") == "prices")
            & (pl.col("source_column") == "price")
        )
        configured_price = configured_causes.filter(
            (pl.col("dataset") == "prices")
            & (pl.col("source_column") == "price")
        )

        self.assertEqual(plain_price.height, 1)
        self.assertIsNone(plain_price["estimated_impact"][0])
        self.assertIn(
            "price_impact_methods.price.method",
            plain_price["required_yaml_setup"][0],
        )
        self.assertEqual(configured_price.height, 1)
        self.assertAlmostEqual(configured_price["estimated_impact"][0], 0.002)
        self.assertEqual(configured_price["required_yaml_setup"][0], "None")

    def test_full_spec_workbook_marks_periods_without_underlying_causes(self) -> None:
        """Changed periods without source-data causes receive diagnostic rows."""
        findings = compare_snapshots(
            _FULL_SPEC_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        portfolio_changes = _workbook_portfolio_changes_table(findings)
        underlying_causes = _workbook_underlying_causes_table(findings)
        placeholders = underlying_causes.filter(
            pl.col("dataset") == "no_underlying_cause_found"
        )
        placeholder_keys = set(placeholders["review_key"].to_list())

        self.assertEqual(placeholders.height, 2)
        self.assertEqual(
            placeholder_keys,
            {
                "PORT_FULL_A::2025-05-01::2025-05-31",
                "PORT_FULL_C::2025-05-01::2025-05-31",
            },
        )
        self.assertTrue(
            placeholders["estimated_impact"].is_null().all(),
        )
        self.assertTrue(
            all(
                "No underlying input differences were found" in value
                for value in placeholders["required_yaml_setup"].to_list()
            )
        )
        _assert_workbook_explained_rows_reconcile(
            self,
            portfolio_changes,
            underlying_causes,
        )

    def test_write_report_bundle_can_include_review_workbook(self) -> None:
        """Report bundles can optionally include an XLSX review workbook."""
        if importlib.util.find_spec("openpyxl") is None:
            self.skipTest("Workbook checks require optional ppar[excel].")
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_MULTI_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            paths = write_performance_comparison_report_bundle(
                findings,
                output_directory,
                include_workbook=True,
                top_evidence_limit=2,
                comparison_path=_MULTI_RESTATEMENT_COMPARISON_PATH,
            )

            self.assertEqual(
                set(paths),
                {*_REPORT_BUNDLE_REQUIRED_ARTIFACTS, "review_workbook"},
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifacts"]["review_workbook"],
                "review_workbook.xlsx",
            )
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("`review_workbook.xlsx`: primary Excel review workbook", readme)
            self.assertIn("Open `review_workbook.xlsx` first", readme)
            self.assertIn(
                "1. Open `review_workbook.xlsx` and start with the Portfolio "
                "Differences sheet.",
                readme,
            )
            self.assertIn("## Primary Review Artifact", readme)
            self.assertIn("## Secondary Review Views", readme)
            self.assertIn("## Audit/Export Files", readme)
            self.assertLess(
                readme.index("`review_workbook.xlsx`: primary Excel review workbook"),
                readme.index("`report.html`: browser-friendly narrative report"),
            )

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
            )
            self.assertEqual(
                workbook.sheetnames,
                [
                    "Portfolio Differences",
                    "Security Differences",
                    "Underlying Causes",
                    "Reported Performance Checks",
                    "Context",
                    "Raw Audit Trail",
                ],
            )
            performance_change_sheet = workbook["Portfolio Differences"]
            self.assertEqual(
                [
                    performance_change_sheet.cell(row=1, column=column).value
                    for column in range(1, 7)
                ],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Performance Difference",
                    "Explained Difference",
                    "Unexplained Difference",
                ],
            )
            self.assertEqual(performance_change_sheet["G1"].value, "Status")
            self.assertEqual(performance_change_sheet["H1"].value, "Next Action")
            self.assertEqual(performance_change_sheet["I1"].value, "Review Key")
            self.assertEqual(
                [performance_change_sheet[f"I{row}"].value for row in range(2, 5)],
                [
                    "PORT_A::2025-05-30::2025-05-30",
                    "PORT_B::2025-05-30::2025-05-30",
                    "PORT_C::2025-05-30::2025-05-30",
                ],
            )
            self.assertEqual(performance_change_sheet.max_row, 4)
            self.assertEqual(performance_change_sheet["D2"].number_format, "0.######")
            self.assertEqual(performance_change_sheet["E2"].number_format, "0.######")
            self.assertEqual(performance_change_sheet["F2"].number_format, "0.######")
            self.assertIsNotNone(performance_change_sheet["A1"].comment)
            assert performance_change_sheet["A1"].comment is not None
            self.assertIn(
                "Portfolio identifier",
                performance_change_sheet["A1"].comment.text,
            )

            security_changes_sheet = workbook["Security Differences"]
            self.assertEqual(
                [
                    security_changes_sheet.cell(row=1, column=column).value
                    for column in range(1, 11)
                ],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Security",
                    "Performance Difference",
                    "Explained Difference",
                    "Unexplained Difference",
                    "Status",
                    "Next Action",
                    "Review Key",
                ],
            )
            self.assertGreater(security_changes_sheet.max_row, 2)
            self.assertEqual(security_changes_sheet["E2"].number_format, "0.######")
            self.assertEqual(
                security_changes_sheet.cell(
                    row=1,
                    column=security_changes_sheet.max_column,
                ).value,
                "Review Key",
            )

            underlying_causes_sheet = workbook["Underlying Causes"]
            self.assertEqual(
                [
                    underlying_causes_sheet.cell(row=1, column=column).value
                    for column in range(1, 12)
                ],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Dataset",
                    "Source Column",
                    "Security",
                    "Snapshot A Value",
                    "Snapshot B Value",
                    "B - A Difference",
                    "Performance Difference Explained",
                    "Required YAML Setup",
                ],
            )
            self.assertGreater(underlying_causes_sheet.max_row, 5)
            numeric_source_row = next(
                row
                for row in range(2, underlying_causes_sheet.max_row + 1)
                if isinstance(underlying_causes_sheet[f"G{row}"].value, (int, float))
                and isinstance(underlying_causes_sheet[f"H{row}"].value, (int, float))
            )
            self.assertEqual(
                underlying_causes_sheet[f"G{numeric_source_row}"].number_format,
                "0.######",
            )
            self.assertEqual(
                underlying_causes_sheet[f"H{numeric_source_row}"].number_format,
                "0.######",
            )
            numeric_explained_row = next(
                row
                for row in range(2, underlying_causes_sheet.max_row + 1)
                if isinstance(underlying_causes_sheet[f"J{row}"].value, (int, float))
            )
            self.assertEqual(
                underlying_causes_sheet[f"J{numeric_explained_row}"].number_format,
                "0.######",
            )
            portfolios = {
                str(underlying_causes_sheet[f"A{row}"].value)
                for row in range(2, underlying_causes_sheet.max_row + 1)
            }
            self.assertTrue({"PORT_A", "PORT_B", "PORT_C"}.issuperset(portfolios))
            required_setup = [
                str(underlying_causes_sheet[f"K{row}"].value)
                for row in range(2, underlying_causes_sheet.max_row + 1)
            ]
            self.assertTrue(
                any(
                    "price_impact_methods.price.method"
                    in setup
                    for setup in required_setup
                )
            )
            self.assertEqual(
                underlying_causes_sheet.cell(
                    row=1,
                    column=underlying_causes_sheet.max_column,
                ).value,
                "Review Key",
            )
            self.assertEqual(
                [
                    underlying_causes_sheet[f"L{row}"].value
                    for row in range(2, min(5, underlying_causes_sheet.max_row) + 1)
                ][0],
                "PORT_A::2025-05-30::2025-05-30",
            )

            derived_checks_sheet = workbook["Reported Performance Checks"]
            self.assertGreater(derived_checks_sheet.max_row, 1)
            self.assertEqual(
                [cell.value for cell in derived_checks_sheet[1]],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Dataset",
                    "Source Column",
                    "Security",
                    "Snapshot A Value",
                    "Snapshot B Value",
                    "B - A Difference",
                    "What Changed",
                    "Next Action",
                    "Review Key",
                ],
            )
            self.assertNotIn(
                "Performance Difference Explained",
                [cell.value for cell in derived_checks_sheet[1]],
            )
            self.assertNotIn("Code", [cell.value for cell in derived_checks_sheet[1]])
            self.assertNotIn(
                "Review Rank",
                [cell.value for cell in derived_checks_sheet[1]],
            )

            context_sheet = workbook["Context"]
            self.assertGreater(context_sheet.max_row, 1)
            self.assertEqual(
                [cell.value for cell in context_sheet[1]],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Dataset",
                    "Source Column",
                    "Security",
                    "Snapshot A Value",
                    "Snapshot B Value",
                    "B - A Difference",
                    "What Changed",
                    "Next Action",
                    "Review Key",
                ],
            )
            self.assertNotIn(
                "Performance Difference Explained",
                [cell.value for cell in context_sheet[1]],
            )
            self.assertNotIn("Code", [cell.value for cell in context_sheet[1]])
            self.assertNotIn("Review Rank", [cell.value for cell in context_sheet[1]])

            findings_sheet = workbook["Raw Audit Trail"]
            self.assertEqual(
                [findings_sheet.cell(row=1, column=column).value for column in range(1, 7)],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "Dataset",
                    "Source Column",
                    "Security",
                ],
            )
            self.assertEqual(
                findings_sheet.cell(row=1, column=findings_sheet.max_column).value,
                "Review Key",
            )
            self.assertIsNotNone(findings_sheet["A1"].comment)

    def test_write_review_workbook_requires_excel_extra(self) -> None:
        """Workbook export fails clearly when the optional dependency is absent."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        def _import_without_openpyxl(
            name: str,
            globals_: Mapping[str, object] | None = None,
            locals_: Mapping[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name.startswith("openpyxl"):
                raise ImportError(name)
            return _original_import(name, globals_, locals_, fromlist, level)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("builtins.__import__", side_effect=_import_without_openpyxl):
                with self.assertRaises(PpaError) as context:
                    write_performance_comparison_review_workbook(
                        findings,
                        Path(directory) / "review_workbook.xlsx",
                    )

        message = str(context.exception)
        self.assertIn("XLSX review workbook export requires", message)
        self.assertIn("ppar[excel]", message)

    def test_write_report_bundle_preserves_empty_table_columns(self) -> None:
        """Report bundles write stable CSV headers for baseline empty tables."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(findings, directory)

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["findings"], 0)
            self.assertEqual(manifest["tables"]["needs_review_summary"]["rows"], 0)
            self.assertEqual(manifest["tables"]["impact_coverage"]["rows"], 0)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 0)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 0)
            self.assertEqual(manifest["tables"]["transaction_cross_checks"]["rows"], 0)
            self.assertEqual(
                manifest["tables"]["transaction_matching_diagnostics"]["rows"],
                0,
            )
            self.assertEqual(
                manifest["tables"]["flow_cross_check_reconciliation"]["rows"],
                0,
            )
            self.assertIn(
                "review_key,portfolio_id,from_date,thru_date",
                paths["needs_review_summary"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "review_key,portfolio_id,from_date,thru_date",
                paths["impact_coverage"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "review_key,portfolio_id,security_id,from_date",
                paths["context_evidence"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "dataset,source_column,context_use",
                paths["context_evidence_summary"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "review_key,portfolio_id,from_date,thru_date",
                paths["transaction_cross_checks"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "review_key,portfolio_id,from_date,thru_date",
                paths["flow_cross_check_reconciliation"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "residual_review_note",
                paths["residual_status"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "review_key,portfolio_id,security_id,from_date",
                paths["transaction_activity"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "transaction_match_status,finding_count,transaction_match_review_note",
                paths["transaction_matching_diagnostics"].read_text(encoding="utf-8"),
            )

    def test_report_bundle_validation_catches_missing_artifact(self) -> None:
        """Bundle validation reports required artifact files that are absent."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(findings, directory)

            paths["needs_review_summary"].unlink()
            issues = _report_bundle_validation_issues(directory)

        self.assertIn("artifact file 'needs_review_summary.csv' is missing", issues)

    def test_report_bundle_validation_catches_missing_review_workbook(self) -> None:
        """Bundle validation reports optional workbook artifacts that are absent."""
        if importlib.util.find_spec("openpyxl") is None:
            self.skipTest("Workbook checks require optional ppar[excel].")

        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                include_workbook=True,
            )

            paths["review_workbook"].unlink()
            issues = _report_bundle_validation_issues(directory)

        self.assertIn("artifact file 'review_workbook.xlsx' is missing", issues)

    def test_report_bundle_validation_catches_invalid_review_workbook(self) -> None:
        """Bundle validation checks optional workbook sheet structure."""
        if importlib.util.find_spec("openpyxl") is None:
            self.skipTest("Workbook checks require optional ppar[excel].")
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                include_workbook=True,
            )
            workbook = openpyxl.load_workbook(paths["review_workbook"])
            del workbook["Portfolio Differences"]
            workbook.save(paths["review_workbook"])

            issues = _report_bundle_validation_issues(directory)

        self.assertIn(
            "review_workbook.xlsx is missing sheet 'Portfolio Differences'",
            issues,
        )

    def test_report_bundle_validation_catches_csv_row_count_drift(self) -> None:
        """Bundle validation compares manifest row counts to CSV row counts."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(findings, directory)
            header = paths["top_evidence"].read_text(encoding="utf-8").splitlines()[0]
            paths["top_evidence"].write_text(header + "\n", encoding="utf-8")

            issues = _report_bundle_validation_issues(directory)

        self.assertIn("table 'top_evidence' row count is 0, expected 10", issues)

    def test_report_bundle_write_fails_if_validation_fails(self) -> None:
        """Bundle writing raises if post-write validation detects corruption."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        with mock.patch(
            "ppar.performance_comparison.report._report_bundle_validation_issues",
            return_value=["simulated validation issue"],
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(PpaError, "simulated validation issue"):
                    write_performance_comparison_report_bundle(findings, directory)


def _section(report: str, start: str, end: str) -> str:
    """Return report text between two section markers."""
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _html_section(report: str, section_id: str) -> str:
    """Return one HTML section by id."""
    start = f'<section class="pc-section" id="{section_id}">'
    return report.split(start, maxsplit=1)[1].split("</section>", maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
