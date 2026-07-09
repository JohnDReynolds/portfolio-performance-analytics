"""Tests for performance comparison reporting."""

# Python imports
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest import mock

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    DIRECT_INPUT,
    Finding,
    REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    TARGET_OUTPUT,
    compare_snapshots,
    findings_to_polars,
    report_bundle_contract,
    report_bundle_validation_issues,
    write_performance_comparison_report_bundle,
    write_performance_comparison_review_workbook,
)
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    PC_PORT_MV,
    PC_PORT_RET,
    SEVERITY_MATERIAL,
)
from ppar.performance_comparison.transaction_summary import (
    transaction_semantics_summary,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_portfolio_changes_table,
    _workbook_raw_audit_trail_table,
    _workbook_security_changes_table,
    _workbook_underlying_causes_table,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement.yaml"
)
_SECURITY_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_security_restatement.yaml"
)
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_MULTI_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_multi_restatement.yaml"
)
_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/setup_templates/axysapx_performance_comparison/axysapx_performance_comparison.yaml"
)
_POLICY_GAP_DEMO_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_policy_gap_demo.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_suppressed.yaml"
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


def _write_holding_estimate_specification(
    directory: Path,
    *,
    include_holding_impact_methods: bool,
    include_accrued_impact_methods: bool = False,
) -> Path:
    """Write a minimal source-loaded fixture with holding market value changes."""
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
        (snapshot_path / "holdings.csv").write_text(
            "PORT,SEC,HOLDING_DATE,QTY,MKT_VAL,ACCRUED\n"
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
        "  holdings: holdings.csv",
    ]
    if include_holding_impact_methods:
        lines.extend(
            [
                "holding_impact_methods:",
                "  market_value:",
                "    method: market_value_delta_over_return_denominator",
                "    denominator_source: begin_market_value",
            ]
        )
    if include_accrued_impact_methods:
        if not include_holding_impact_methods:
            lines.append("holding_impact_methods:")
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


def _assert_workbook_unexplained_formula(
    test_case: unittest.TestCase,
    portfolio_changes: pl.DataFrame,
) -> None:
    """Assert workbook portfolio rows obey performance minus explained math."""
    for row in portfolio_changes.iter_rows(named=True):
        performance_change = _float_or_zero(row.get("performance_change"))
        explained_change = _float_or_zero(row.get("estimated_cause_total"))
        unexplained_change = row.get("unexplained_change")
        if row.get("review_status") == "Fully Explained":
            test_case.assertIsNone(
                unexplained_change,
                msg=f"{row.get('review_key')} should display a blank residual.",
            )
            continue
        test_case.assertAlmostEqual(
            performance_change - explained_change,
            _float_or_zero(unexplained_change),
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
                f"{review_key} visible Performance Difference Causes rows do not match "
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
    """Assert non-additive workbook rows have clear review guidance."""
    for row in underlying_causes.iter_rows(named=True):
        estimated_impact = _float_or_none(row.get("estimated_impact"))
        required_setup = row.get("review_guidance")
        if estimated_impact is None:
            test_case.assertNotEqual(required_setup, "None")
            if row.get("dataset") == "no_underlying_causes_found":
                test_case.assertEqual(row.get("use"), "Diagnostic")
                test_case.assertEqual(row.get("impact_status"), "Review only")
                test_case.assertIn("No identifiable cause", str(required_setup))
                continue
            if (
                "configured as evidence-only" in str(required_setup)
                or (
                    '"Performance Differences"."Explained Difference"'
                    in str(required_setup)
                )
                or "related performance input" in str(required_setup)
                or "changed transactions.amount" in str(required_setup)
                or "changed holdings.market_value" in str(required_setup)
                or "Input for changed" in str(required_setup)
                or "Helped explain" in str(required_setup)
                or "transactions.amount to" in str(required_setup)
                or "holdings.quantity to" in str(required_setup)
                or "split factor" in str(required_setup)
            ):
                test_case.assertEqual(row.get("impact_status"), "Review only")
                continue
            if (
                "Caused transactions.amount" in str(required_setup)
                or "Caused cash-balance" in str(required_setup)
            ):
                test_case.assertIn(
                    row.get("impact_status"),
                    {"Review only", "Missing impact method"},
                )
                continue
            if (
                "ending holdings." in str(required_setup)
                or "beginning holdings." in str(required_setup)
            ):
                test_case.assertIn(
                    row.get("impact_status"),
                    {"Review only", "Missing impact input"},
                )
                continue
            if "in Snapshot B" in str(required_setup):
                test_case.assertIn(
                    row.get("impact_status"),
                    {"Review only", "Missing impact input"},
                )
                continue
            if row.get("impact_status") == "Missing impact input":
                setup_text = str(required_setup)
                test_case.assertTrue(
                    "Configured" in setup_text
                    or "shown for review" in setup_text
                    or "Supporting detail" in setup_text
                    or "External flow" in setup_text
                    or "Add YAML configuration to count it as explained" in setup_text,
                    msg=f"{row.get('review_key')} has unclear setup: {setup_text}",
                )
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
        test_case.assertIsInstance(required_setup, str)


def _float_or_none(value: object) -> float | None:
    """Return value as float when it is numeric and not a boolean."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _float_or_zero(value: object) -> float:
    """Return numeric value as float, treating null workbook values as zero."""
    return _float_or_none(value) or 0.0


def _normalized_header(value: object) -> object:
    """Return an Excel header with intentional line breaks normalized."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _entrypoint_files(entrypoints: Mapping[str, object]) -> set[str]:
    """Return non-empty artifact filenames referenced by review entrypoints."""
    files: set[str] = set()
    for value in entrypoints.values():
        if isinstance(value, str) and value:
            files.add(value)
            continue
        if isinstance(value, list):
            files.update(item for item in value if isinstance(item, str) and item)
    return files


class TestPerformanceComparisonReport(unittest.TestCase):
    """Verify report rendering and artifact generation for comparison findings."""

    def test_report_bundle_contract_snapshot(self) -> None:
        """The public report-bundle contract exposes reviewer handoff shape."""
        self.assertEqual(
            report_bundle_contract(),
            {
                "bundle_type": "performance_comparison_report",
                "manifest_version": 1,
                "required_artifacts": [
                    "html_report",
                    "readme",
                    "manifest",
                    "review_summary",
                    "findings",
                    "needs_review_summary",
                    "portfolio_period_summary",
                    "cause_summary",
                    "impact_estimates",
                    "impact_coverage",
                    "context_evidence_summary",
                    "context_evidence",
                    "transaction_cross_checks",
                    "flow_cross_check_reconciliation",
                    "residual_status",
                    "transaction_activity",
                    "transaction_matching_diagnostics",
                    "top_evidence",
                ],
                "required_manifest_keys": [
                    "bundle_type",
                    "manifest_version",
                    "created_at",
                    "title",
                    "options",
                    "source_context",
                    "counts",
                    "transaction_semantics",
                    "artifacts",
                    "tables",
                    "review_entrypoints",
                ],
                "required_review_entrypoints": [
                    "primary_review",
                    "period_triage",
                    "formula_input_causes",
                    "supporting_context",
                    "transaction_diagnostics",
                    "audit_trail",
                    "review_handoff",
                ],
                "review_basis": "Modified Dietz evidence pack",
                "review_summary_version": 1,
                "required_review_summary_keys": [
                    "summary_version",
                    "review_basis",
                    "review_vocabulary",
                    "entrypoints",
                    "source_context",
                    "counts",
                    "transaction_semantics",
                    "artifacts",
                ],
                "review_vocabulary_keys": [
                    "formula_input",
                    "source_data",
                    "finding_level",
                    "cause_area",
                    "supporting_evidence",
                    "context_only",
                    "review_only",
                    "evidence_only",
                    "non_additive",
                    "explained_change",
                    "backlog_gate",
                ],
            },
        )

    def test_transaction_semantics_summary_preserves_native_observed_codes(self) -> None:
        """Review metadata keeps transaction-code case from the source rows."""
        frame = pl.DataFrame(
            {
                pc_cols.TRANSACTION_CODE: ["by", "BY", " Sl ", None, ""],
                pc_cols.TRANSACTION_CATEGORY: ["buy", "buy", "sell", "unknown", ""],
                pc_cols.TRANSACTION_SEMANTICS_SOURCE: [
                    "yaml_rule",
                    "source",
                    "yaml_rule",
                    "unknown",
                    "",
                ],
            }
        )

        summary = transaction_semantics_summary([frame], rule_codes={"BY"})

        self.assertEqual(summary["observed_codes"], ["BY", "Sl", "by"])
        self.assertEqual(summary["codes_without_yaml_rules"], ["Sl"])
        self.assertEqual(summary["unknown_category_count"], 1)

    def test_write_report_bundle_creates_review_artifacts(self) -> None:
        """Report bundles contain HTML, CSV tables, and manifest metadata."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        contract = report_bundle_contract()
        required_artifacts = cast(list[str], contract["required_artifacts"])
        required_manifest_keys = cast(list[str], contract["required_manifest_keys"])
        required_entrypoints = cast(list[str], contract["required_review_entrypoints"])
        required_summary_keys = cast(list[str], contract["required_review_summary_keys"])
        expected_keys = set(required_artifacts)
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            paths = write_performance_comparison_report_bundle(
                findings,
                output_directory,
                title="Bundle Restatement",
                top_evidence_limit=2,
                require_complete_yaml_setup=False,
            )

            self.assertEqual(set(paths), expected_keys)
            for path in paths.values():
                self.assertTrue(path.exists(), path)
            self.assertNotIn("report", paths)
            self.assertFalse((output_directory / "report.md").exists())
            html_report = paths["html_report"].read_text(encoding="utf-8")
            self.assertIn("<h1>Bundle Restatement</h1>", html_report)
            self.assertNotIn(
                "Browser view for reviewing this performance-comparison bundle.",
                html_report,
            )
            self.assertNotIn('class="pc-contents-list"', html_report)
            self.assertNotIn("same review model", html_report)
            self.assertNotIn("Browser review surface", html_report)
            self.assertIn("Performance Differences", html_report)
            self.assertIn("Performance Difference Causes", html_report)
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("# Bundle Restatement", readme)
            self.assertNotIn("## Primary Review Artifact", readme)
            self.assertNotIn("## Secondary Review Views", readme)
            self.assertNotIn("Open `report.xlsx` first", readme)
            self.assertNotIn("same review model in a browser", readme)
            self.assertNotIn("report.md", readme)
            self.assertIn("## Recommended Review Order", readme)
            self.assertIn("Start with Performance Differences", readme)
            self.assertIn("Use Performance Difference Causes", readme)
            self.assertIn("explain each performance period", readme)
            self.assertIn("Source Detail for audit and troubleshooting", readme)
            self.assertIn("source-data differences", readme)
            self.assertNotIn("source" + " data", readme)
            self.assertIn(
                "follow a performance period across the `supporting_files/` "
                "CSV artifacts",
                readme,
            )
            self.assertIn(
                "`supporting_files/transaction_activity.csv`, "
                "`supporting_files/transaction_cross_checks.csv`, and "
                "`supporting_files/flow_cross_check_reconciliation.csv`",
                readme,
            )
            self.assertIn(
                "supplementary transaction and external-flow diagnostics",
                readme,
            )
            self.assertIn(
                "`supporting_files/transaction_matching_diagnostics.csv` only "
                "when auditing transaction row-identity evidence",
                readme,
            )
            self.assertIn("conservative matching status", readme)
            self.assertIn("does not imply fuzzy transaction linkage", readme)
            self.assertNotIn("cross-check rows may be", readme)
            self.assertIn("`supporting_files/review_summary.json`", readme)
            self.assertIn("Modified Dietz vocabulary", readme)
            self.assertIn("`supporting_files/needs_review_summary.csv`", readme)
            self.assertIn("## Audit/Export Files", readme)
            self.assertIn(
                "`supporting_files/manifest.json`: machine-readable artifact",
                readme,
            )
            self.assertIn("source context", readme)
            self.assertIn("transaction semantics summary", readme)
            self.assertIn(
                "`supporting_files/needs_review_summary.csv`: top triage table",
                readme,
            )
            self.assertIn(
                "`supporting_files/context_evidence_summary.csv`: context-only "
                "evidence counts, reviewer priority",
                readme,
            )
            self.assertIn(
                "`supporting_files/context_evidence.csv`: row-level context evidence, "
                "reviewer priority",
                readme,
            )

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["bundle_type"], "performance_comparison_report")
            self.assertEqual(manifest["manifest_version"], contract["manifest_version"])
            self.assertEqual(manifest["title"], "Bundle Restatement")
            self.assertEqual(
                set(manifest),
                set(required_manifest_keys),
            )
            self.assertEqual(manifest["counts"]["findings"], 13)
            self.assertEqual(manifest["counts"]["active_findings"], 13)
            self.assertEqual(manifest["options"]["top_evidence_limit"], 2)
            self.assertFalse(
                manifest["options"]["include_reconstruction_diagnostics"]
            )
            self.assertEqual(
                manifest["artifacts"]["manifest"],
                "supporting_files/manifest.json",
            )
            self.assertEqual(manifest["artifacts"]["html_report"], "report.html")
            self.assertEqual(manifest["artifacts"]["readme"], "README.md")
            self.assertEqual(
                manifest["artifacts"]["review_summary"],
                "supporting_files/review_summary.json",
            )
            self.assertNotIn("report", manifest["artifacts"])
            self.assertEqual(manifest["source_context"]["comparison_path"], None)
            self.assertEqual(manifest["source_context"]["extract_contract"], None)
            self.assertIn("observed_codes", manifest["transaction_semantics"])
            self.assertIn("unknown_category_count", manifest["transaction_semantics"])
            self.assertEqual(
                manifest["transaction_semantics"]["ambiguous_context_blocked_count"],
                0,
            )
            self.assertEqual(
                manifest["review_entrypoints"]["primary_review"],
                "report.html",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["period_triage"],
                "supporting_files/needs_review_summary.csv",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["formula_input_causes"],
                "supporting_files/cause_summary.csv",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["supporting_context"],
                "supporting_files/context_evidence_summary.csv",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["transaction_diagnostics"],
                [
                    "supporting_files/transaction_activity.csv",
                    "supporting_files/transaction_cross_checks.csv",
                    "supporting_files/flow_cross_check_reconciliation.csv",
                    "supporting_files/transaction_matching_diagnostics.csv",
                ],
            )
            self.assertNotEqual(
                manifest["review_entrypoints"]["primary_review"],
                "supporting_files/transaction_matching_diagnostics.csv",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["audit_trail"],
                "supporting_files/findings.csv",
            )
            self.assertEqual(
                manifest["review_entrypoints"]["review_handoff"],
                "supporting_files/review_summary.json",
            )
            self.assertEqual(
                manifest["artifacts"]["needs_review_summary"],
                "supporting_files/needs_review_summary.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence"],
                "supporting_files/context_evidence.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence_summary"],
                "supporting_files/context_evidence_summary.csv",
            )
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
            self.assertEqual(manifest["tables"]["needs_review_summary"]["rows"], 1)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 2)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 2)

            review_summary = cast(
                dict[str, Any],
                json.loads(paths["review_summary"].read_text(encoding="utf-8")),
            )
            review_vocabulary = cast(
                Mapping[str, str],
                review_summary["review_vocabulary"],
            )
            review_vocabulary_keys = cast(
                tuple[str, ...],
                contract["review_vocabulary_keys"],
            )
            self.assertEqual(
                set(review_summary),
                set(required_summary_keys),
            )
            self.assertEqual(
                review_summary["summary_version"],
                contract["review_summary_version"],
            )
            self.assertEqual(
                review_summary["review_basis"],
                contract["review_basis"],
            )
            self.assertEqual(review_summary["entrypoints"], manifest["review_entrypoints"])
            self.assertEqual(review_summary["source_context"], manifest["source_context"])
            self.assertEqual(
                review_summary["transaction_semantics"],
                manifest["transaction_semantics"],
            )
            for vocabulary_key in review_vocabulary_keys:
                self.assertIn(vocabulary_key, review_vocabulary)
            self.assertEqual(
                set(review_vocabulary),
                set(review_vocabulary_keys),
            )
            self.assertIn(
                "Modified Dietz",
                review_vocabulary["formula_input"],
            )
            self.assertIn(
                "source-data",
                review_vocabulary["source_data"],
            )
            self.assertIn(
                "reviewer judgment",
                review_vocabulary["review_only"],
            )
            self.assertIn(
                "audit",
                review_vocabulary["evidence_only"],
            )
            self.assertIn(
                "finding-level",
                review_summary["review_vocabulary"]["finding_level"],
            )
            self.assertIn(
                "cause-area",
                review_summary["review_vocabulary"]["cause_area"],
            )
            self.assertIn(
                "review-only",
                review_summary["review_vocabulary"]["review_only"],
            )
            self.assertIn(
                "evidence-only",
                review_summary["review_vocabulary"]["evidence_only"],
            )
            self.assertIn(
                "non-additive",
                review_summary["review_vocabulary"]["non_additive"],
            )
            self.assertIn(
                "explained-change",
                review_summary["review_vocabulary"]["explained_change"],
            )
            self.assertEqual(
                set(review_summary["entrypoints"]),
                set(required_entrypoints),
            )
            artifact_files = set(manifest["artifacts"].values())
            self.assertTrue(
                _entrypoint_files(review_summary["entrypoints"]) <= artifact_files
            )
            self.assertIn(review_summary["entrypoints"]["primary_review"], readme)
            self.assertIn(review_summary["entrypoints"]["period_triage"], readme)
            self.assertIn(review_summary["entrypoints"]["review_handoff"], readme)

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
                "high-priority context: holdings/cost",
                needs_review["review_cues"][0],
            )
            self.assertIn(
                "supporting_files/transaction_activity.csv",
                needs_review["review_detail_artifacts"][0],
            )
            self.assertIn(
                "supporting_files/context_evidence.csv",
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
            self.assertEqual(impact_coverage["impact_coverage_status"][0], "partial_estimates")

            context_evidence = pl.read_csv(paths["context_evidence"])
            self.assertEqual(context_evidence.height, 2)
            self.assertIn("review_key", context_evidence.columns)
            self.assertIn("context_use", context_evidence.columns)
            self.assertIn("review_priority", context_evidence.columns)
            self.assertIn("review_priority_reason", context_evidence.columns)
            self.assertIn("return_impact_treatment", context_evidence.columns)
            self.assertIn("PC-HOLD-COST", context_evidence["code"].to_list())
            self.assertEqual(context_evidence["review_priority"][0], "high")
            self.assertEqual(
                set(context_evidence["return_impact_treatment"]),
                {"context only; not included in return-impact estimates"},
            )

            context_evidence_summary = pl.read_csv(paths["context_evidence_summary"])
            self.assertEqual(context_evidence_summary.height, 2)
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
                "matched_by_id",
            )
            self.assertIn(
                "stable transaction_id",
                transaction_matching["transaction_match_review_note"][0],
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
            self.assertEqual(report_bundle_validation_issues(output_directory), [])

    def test_write_report_bundle_manifest_includes_source_context(self) -> None:
        """Report bundle manifests summarize comparison and extract-contract context."""
        findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                comparison_path=_RESTATEMENT_TRANSACTION_RULES_PATH,
                require_complete_yaml_setup=False,
            )

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        source_context = manifest["source_context"]
        extract_contract = source_context["extract_contract"]
        semantics = manifest["transaction_semantics"]
        self.assertEqual(
            source_context["comparison_path"],
            str(_RESTATEMENT_TRANSACTION_RULES_PATH),
        )
        self.assertEqual(
            extract_contract["path"],
            "packaged:ppar.setup_templates/axysapx_performance_comparison/"
            "demo_extract_availability.yaml",
        )
        self.assertTrue(extract_contract["enforce_ambiguous_axys_flows"])
        self.assertIn(
            "source_destination_type",
            extract_contract["required_transaction_context_columns"],
        )
        self.assertIn(
            "special_security_symbol",
            extract_contract["required_transaction_context_columns"],
        )
        self.assertIn("BUY", semantics["observed_codes"])
        self.assertEqual(semantics["unknown_category_count"], 0)

    def test_write_report_bundle_requires_complete_yaml_by_default(self) -> None:
        """User-facing bundles fail when changed source fields lack YAML policy."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PpaError, "YAML setup is incomplete"):
                write_performance_comparison_report_bundle(findings, directory)

    def test_workbook_tables_follow_review_accounting_invariants(self) -> None:
        """Workbook review tables stay internally consistent across demos."""
        cases = (
            ("single", _RESTATEMENT_COMPARISON_PATH, False),
            ("transaction_rules", _RESTATEMENT_TRANSACTION_RULES_PATH, False),
            ("multi", _MULTI_RESTATEMENT_COMPARISON_PATH, False),
            ("portfolio", _PORTFOLIO_COMPARISON_PATH, True),
            ("policy_gap", _POLICY_GAP_DEMO_COMPARISON_PATH, False),
        )

        for name, comparison_path, require_causal_attribution in cases:
            with self.subTest(name=name):
                findings = compare_snapshots(
                    comparison_path,
                    require_causal_attribution=require_causal_attribution,
                )
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
        self.assertEqual(
            portfolio_changes["review_note"][0],
            "No reported portfolio return differences.",
        )
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
        self.assertEqual(
            plain_transaction_amount["review_guidance"][0],
            (
                "BUY: Caused cash-balance ending holdings.market_value "
                "to decrease by 100.00. "
                "Add YAML configuration to count it as explained."
            ),
        )

        rules_findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)
        rules_causes = _workbook_underlying_causes_table(rules_findings)
        rules_transaction_amount = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "amount")
        )
        self.assertEqual(rules_transaction_amount.height, 1)
        self.assertIsNone(rules_transaction_amount["estimated_impact"][0])
        self.assertEqual(
            rules_transaction_amount["review_guidance"][0],
            "BUY: Caused cash-balance ending holdings.market_value to decrease by 100.00.",
        )
        rules_transaction_quantity = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "quantity")
        )
        rules_transaction_price = rules_causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "price")
        )
        self.assertEqual(
            rules_transaction_quantity["review_guidance"][0],
            (
                "BUY: Caused AAPL transactions.amount to increase and "
                "AAPL holdings.quantity to increase."
            ),
        )
        self.assertEqual(
            rules_transaction_price["review_guidance"][0],
            "BUY: Caused AAPL transactions.amount to increase.",
        )
        self.assertNotIn(
            "by 0.50",
            rules_transaction_price["review_guidance"][0],
        )
        self.assertNotIn(
            "Helped explain the changed transactions.amount",
            rules_transaction_quantity["review_guidance"][0],
        )
        self.assertNotIn(
            "Helped explain the changed transactions.amount",
            rules_transaction_price["review_guidance"][0],
        )

    def test_configured_transaction_method_with_zero_denominator_needs_inputs(
        self,
    ) -> None:
        """Configured transaction methods do not masquerade as missing methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_path = _write_transaction_estimate_specification(Path(temp_dir))
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                portfolio_path = comparison_path.parent / snapshot_name / "portperf.csv"
                portfolio_path.write_text(
                    portfolio_path.read_text(encoding="utf-8").replace(
                        "1000.00",
                        "0.00",
                    ),
                    encoding="utf-8",
                )

            causes = _workbook_underlying_causes_table(
                compare_snapshots(comparison_path),
                comparison_path=comparison_path,
            )

        transaction_amount = causes.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "amount")
        )

        self.assertEqual(transaction_amount.height, 1)
        self.assertIsNone(transaction_amount["estimated_impact"][0])
        self.assertEqual(transaction_amount["impact_status"][0], "Missing impact input")
        self.assertIn(
            "Configured transaction impact method is present",
            transaction_amount["review_guidance"][0],
        )
        self.assertIn(
            "return denominator",
            transaction_amount["review_guidance"][0],
        )
        self.assertNotIn(
            "transaction_impact_methods.performance.method",
            transaction_amount["review_guidance"][0],
        )
        self.assertEqual(
            transaction_amount["review_note"][0],
            "BUY: Caused cash-balance ending holdings.market_value to decrease by 10.00.",
        )

    def test_transaction_commission_policy_marks_commission_review_only(
        self,
    ) -> None:
        """Commission appears as review-only supporting evidence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_path = _write_transaction_commission_review_specification(
                Path(temp_dir)
            )

            raw_audit_trail = _workbook_raw_audit_trail_table(
                compare_snapshots(comparison_path),
            )

        commission = raw_audit_trail.filter(
            (pl.col("dataset") == "transactions")
            & (pl.col("source_column") == "commission")
        )

        self.assertEqual(commission.height, 1)
        self.assertIsNone(commission["estimated_impact"][0])
        self.assertEqual(
            commission["review_note"][0],
            "BUY: Caused AAPL transactions.amount to increase by 2.50.",
        )

    def test_security_differences_roll_up_security_underlying_causes(self) -> None:
        """Performance Differences shows security-level performance changes."""
        findings = compare_snapshots(_SECURITY_RESTATEMENT_COMPARISON_PATH)

        security_differences = _workbook_security_changes_table(findings)
        aapl = security_differences.filter(pl.col("security_id") == "AAPL")

        self.assertEqual(aapl.height, 1)
        self.assertAlmostEqual(aapl["performance_change"][0], 0.01)
        self.assertAlmostEqual(aapl["estimated_cause_total"][0], 0.0)
        self.assertAlmostEqual(aapl["unexplained_change"][0], 0.01)

    def test_security_differences_marks_periods_without_security_rows(self) -> None:
        """Portfolio periods without security differences get explicit rows."""
        findings = compare_snapshots(_PORTFOLIO_COMPARISON_PATH)

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
        self.assertEqual(set(placeholders["review_note"].to_list()), {"None"})

    def test_holding_impact_method_explains_market_value_row(self) -> None:
        """Holding market value uses the default performance-input impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_holding_estimate_specification(
                Path(temp_dir) / "plain",
                include_holding_impact_methods=False,
            )
            configured_path = _write_holding_estimate_specification(
                Path(temp_dir) / "configured",
                include_holding_impact_methods=True,
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path),
                comparison_path=configured_path,
            )

        plain_holding = plain_causes.filter(
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "market_value")
        )
        configured_holding = configured_causes.filter(
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "market_value")
        )

        self.assertEqual(plain_holding.height, 1)
        self.assertAlmostEqual(plain_holding["estimated_impact"][0], 0.01)
        self.assertEqual(
            plain_holding["review_guidance"][0],
            "AAPL ending holdings.market_value increased by 10.00.",
        )
        self.assertEqual(configured_holding.height, 1)
        self.assertAlmostEqual(configured_holding["estimated_impact"][0], 0.01)
        self.assertEqual(
            configured_holding["review_guidance"][0],
            "AAPL ending holdings.market_value increased by 10.00.",
        )

    def test_holding_accrued_impact_method_explains_accrued_row(self) -> None:
        """Holding accrued uses the default performance-input impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_holding_estimate_specification(
                Path(temp_dir) / "plain",
                include_holding_impact_methods=False,
            )
            configured_path = _write_holding_estimate_specification(
                Path(temp_dir) / "configured",
                include_holding_impact_methods=False,
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
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "accrued")
        )
        configured_accrued = configured_causes.filter(
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "accrued")
        )

        self.assertEqual(plain_accrued.height, 1)
        self.assertAlmostEqual(plain_accrued["estimated_impact"][0], 0.005)
        self.assertEqual(
            plain_accrued["review_guidance"][0],
            "AAPL ending holdings.accrued increased by 5.00.",
        )
        self.assertEqual(configured_accrued.height, 1)
        self.assertAlmostEqual(configured_accrued["estimated_impact"][0], 0.005)
        self.assertEqual(
            configured_accrued["review_guidance"][0],
            "AAPL ending holdings.accrued increased by 5.00.",
        )

    def test_evidence_only_impact_method_marks_row_review_only(self) -> None:
        """Evidence-only YAML removes missing-method guidance for known fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plain_path = _write_holding_estimate_specification(
                Path(temp_dir) / "plain",
                include_holding_impact_methods=False,
            )
            configured_path = _write_holding_estimate_specification(
                Path(temp_dir) / "configured",
                include_holding_impact_methods=False,
            )
            for comparison_path in (plain_path, configured_path):
                holding_path = comparison_path.parent / "snapshot_b" / "holdings.csv"
                holding_path.write_text(
                    holding_path.read_text(encoding="utf-8").replace(
                        "PORT_A,AAPL,2025-05-31,10,1010.00,30.00",
                        "PORT_A,AAPL,2025-05-31,11,1010.00,30.00",
                    ),
                    encoding="utf-8",
                )
            configured_path.write_text(
                configured_path.read_text(encoding="utf-8")
                + "\n"
                + "holding_impact_methods:\n"
                + "  quantity:\n"
                + "    method: evidence_only\n",
                encoding="utf-8",
            )

            plain_causes = _workbook_underlying_causes_table(
                compare_snapshots(plain_path),
                comparison_path=plain_path,
            )
            configured_causes = _workbook_underlying_causes_table(
                compare_snapshots(configured_path)
            )

        plain_quantity = plain_causes.filter(
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "quantity")
        )
        configured_quantity = configured_causes.filter(
            (pl.col("dataset") == "holdings")
            & (pl.col("source_column") == "quantity")
        )

        self.assertEqual(plain_quantity.height, 1)
        self.assertEqual(
            plain_quantity["review_guidance"][0],
            "AAPL ending holdings.quantity increased by 1.00.",
        )
        self.assertEqual(plain_quantity["impact_status"][0], "Review only")
        self.assertEqual(configured_quantity.height, 1)
        self.assertIsNone(configured_quantity["estimated_impact"][0])
        self.assertEqual(
            configured_quantity["review_note"][0],
            "AAPL ending holdings.quantity increased by 1.00.",
        )

    def test_portfolio_workbook_links_changed_periods_to_underlying_causes(self) -> None:
        """Changed portfolio demo periods have matching cause rows."""
        findings = compare_snapshots(
            _PORTFOLIO_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        portfolio_changes = _workbook_portfolio_changes_table(findings)
        underlying_causes = _workbook_underlying_causes_table(findings)
        portfolio_keys = set(portfolio_changes["review_key"].to_list())
        cause_keys = set(underlying_causes["review_key"].to_list())
        promoted_fields = {
            (str(row["dataset"]), str(row["source_column"]))
            for row in underlying_causes.select(["dataset", "source_column"]).iter_rows(
                named=True
            )
        }

        self.assertTrue(portfolio_keys)
        self.assertTrue(portfolio_keys.issubset(cause_keys))
        self.assertTrue(
            {
                ("holdings", "market_value"),
                ("holdings", "quantity"),
                ("transactions", "amount"),
                ("transactions", "commission"),
                ("transactions", "price"),
                ("transactions", "quantity"),
            }.issubset(promoted_fields)
        )
        _assert_workbook_explained_rows_reconcile(
            self,
            portfolio_changes,
            underlying_causes,
        )

    def test_write_report_bundle_can_include_review_workbook(self) -> None:
        """Report bundles can include an XLSX review workbook."""
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
                require_complete_yaml_setup=False,
            )

            self.assertEqual(
                set(paths),
                {*REPORT_BUNDLE_REQUIRED_ARTIFACTS, "review_workbook"},
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifacts"]["review_workbook"],
                "report.xlsx",
            )
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertNotIn("Open `report.xlsx` first", readme)
            self.assertNotIn("same review model in a browser", readme)
            self.assertIn(
                "1. Open `report.xlsx` when present; use `report.html` for "
                "browser review.",
                readme,
            )
            self.assertNotIn("## Primary Review Artifact", readme)
            self.assertNotIn("## Secondary Review Views", readme)
            self.assertIn("## Audit/Export Files", readme)
            self.assertIn("explain each performance period", readme)
            self.assertIn("additively explain each performance period", readme)
            self.assertIn("Source Detail for audit and troubleshooting", readme)
            self.assertIn("complete finding-level audit trail", readme)
            self.assertIn(
                "follow a performance period across the `supporting_files/` "
                "CSV artifacts",
                readme,
            )
            self.assertIn("supplementary transaction and external-flow diagnostics", readme)
            self.assertIn(
                "`supporting_files/transaction_matching_diagnostics.csv` only "
                "when auditing transaction row-identity evidence",
                readme,
            )
            self.assertNotIn("cross-check rows may be", readme)

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
            )
            self.assertEqual(
                workbook.sheetnames,
                [
                    "Performance Differences",
                    "Performance Difference Causes",
                    "Data Audit Issues",
                    "Source Detail",
                ],
            )
            self.assertTrue(
                all(
                    workbook[sheet_name].row_dimensions[1].height is None
                    for sheet_name in workbook.sheetnames
                )
            )
            self.assertTrue(
                all(
                    workbook[sheet_name].column_dimensions[
                        workbook[sheet_name].cell(row=1, column=column).column_letter
                    ].width
                    <= 50
                    for sheet_name in workbook.sheetnames
                    for column in range(1, workbook[sheet_name].max_column + 1)
                )
            )
            performance_change_sheet = workbook["Performance Differences"]
            self.assertEqual(performance_change_sheet.column_dimensions["B"].width, 12)
            self.assertEqual(performance_change_sheet.column_dimensions["H"].width, 32)
            self.assertEqual(performance_change_sheet.column_dimensions["I"].width, 24)
            underlying_causes_sheet = workbook["Performance Difference Causes"]
            self.assertEqual(underlying_causes_sheet.column_dimensions["E"].width, 22)
            self.assertEqual(underlying_causes_sheet.column_dimensions["K"].width, 50)
            self.assertEqual(
                [
                    _normalized_header(
                        performance_change_sheet.cell(row=1, column=column).value
                    )
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
            self.assertEqual(performance_change_sheet["H1"].value, "Comments")
            self.assertEqual(
                _normalized_header(performance_change_sheet["I1"].value),
                "Review Key",
            )
            self.assertEqual(
                [performance_change_sheet[f"I{row}"].value for row in range(2, 5)],
                [
                    "PORT_A::2025-05-30::2025-05-30",
                    "PORT_B::2025-05-30::2025-05-30",
                    "PORT_C::2025-05-30::2025-05-30",
                ],
            )
            self.assertEqual(performance_change_sheet.max_row, 4)
            self.assertEqual(performance_change_sheet["D2"].number_format, "0.000000")
            self.assertEqual(performance_change_sheet["E2"].number_format, "0.000000")
            self.assertEqual(performance_change_sheet["F2"].number_format, "0.000000")
            self.assertIsNotNone(performance_change_sheet["A1"].comment)
            assert performance_change_sheet["A1"].comment is not None
            self.assertIn(
                "Portfolio identifier",
                performance_change_sheet["A1"].comment.text,
            )

            self.assertEqual(
                [
                    _normalized_header(
                        underlying_causes_sheet.cell(row=1, column=column).value
                    )
                    for column in range(1, 13)
                ],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "As Of Date",
                    "Dataset Field",
                    "Security",
                    "Snapshot A Value",
                    "Snapshot B Value",
                    "B - A Difference",
                    "Performance Difference Explained",
                    "Explanation",
                    "Review Key",
                ],
            )
            self.assertNotIn(
                "Row Type",
                [
                    _normalized_header(
                        underlying_causes_sheet.cell(row=1, column=column).value
                    )
                    for column in range(1, underlying_causes_sheet.max_column + 1)
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
                "0.000000",
            )
            self.assertEqual(
                underlying_causes_sheet[f"H{numeric_source_row}"].number_format,
                "0.000000",
            )
            numeric_explained_row = next(
                row
                for row in range(2, underlying_causes_sheet.max_row + 1)
                if isinstance(underlying_causes_sheet[f"J{row}"].value, (int, float))
            )
            self.assertEqual(
                underlying_causes_sheet[f"J{numeric_explained_row}"].number_format,
                "0.000000",
            )
            explained_cause_row = next(
                row
                for row in range(2, underlying_causes_sheet.max_row + 1)
                if underlying_causes_sheet[f"J{row}"].value not in (None, "")
            )
            self.assertEqual(
                underlying_causes_sheet[f"J{explained_cause_row}"].fill.fgColor.rgb,
                "FFFFFF00",
            )
            self.assertEqual(
                underlying_causes_sheet[f"K{explained_cause_row}"].fill.fgColor.rgb,
                "FFFFFF00",
            )
            possible_cause_rows = [
                row
                for row in range(2, underlying_causes_sheet.max_row + 1)
                if str(underlying_causes_sheet[f"K{row}"].value).startswith(
                    "Possible cause:"
                )
            ]
            if possible_cause_rows:
                possible_cause_row = possible_cause_rows[0]
                self.assertEqual(
                    underlying_causes_sheet[f"K{possible_cause_row}"].fill.fgColor.rgb,
                    "FFFFE699",
                )
                self.assertNotEqual(
                    underlying_causes_sheet[f"J{possible_cause_row}"].fill.fgColor.rgb,
                    "FFFFE699",
                )
            portfolios = {
                str(underlying_causes_sheet[f"A{row}"].value)
                for row in range(2, underlying_causes_sheet.max_row + 1)
            }
            self.assertTrue({"PORT_A", "PORT_B", "PORT_C"}.issuperset(portfolios))
            required_setup = [
                underlying_causes_sheet[f"K{row}"].value
                for row in range(2, underlying_causes_sheet.max_row + 1)
            ]
            self.assertTrue(any(setup in (None, "") for setup in required_setup))
            self.assertEqual(
                _normalized_header(
                    underlying_causes_sheet.cell(
                        row=1,
                        column=underlying_causes_sheet.max_column,
                    ).value
                ),
                "Review Key",
            )
            self.assertEqual(
                [
                    underlying_causes_sheet[f"L{row}"].value
                    for row in range(2, min(5, underlying_causes_sheet.max_row) + 1)
                ][0],
                "PORT_A::2025-05-30::2025-05-30",
            )

            findings_sheet = workbook["Source Detail"]
            self.assertEqual(
                [
                    _normalized_header(
                        findings_sheet.cell(row=1, column=column).value
                    )
                    for column in range(1, 11)
                ],
                [
                    "Portfolio",
                    "From Date",
                    "Thru Date",
                    "As Of Date",
                    "Dataset Field",
                    "Security",
                    "Snapshot A Value",
                    "Snapshot B Value",
                    "B - A Difference",
                    "Explanation",
                ],
            )
            self.assertEqual(
                _normalized_header(
                    findings_sheet.cell(row=1, column=findings_sheet.max_column).value
                ),
                "Review Key",
            )
            self.assertIsNotNone(findings_sheet["A1"].comment)

    def test_write_review_workbook_reports_missing_openpyxl(self) -> None:
        """Workbook export fails clearly when the workbook dependency is absent."""
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
                        Path(directory) / "report.xlsx",
                    )

        message = str(context.exception)
        self.assertIn("XLSX review workbook export requires", message)
        self.assertIn("openpyxl", message)

    def test_write_report_bundle_preserves_empty_table_columns(self) -> None:
        """Report bundles write stable CSV headers for baseline empty tables."""
        findings = compare_snapshots(_BASELINE_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )

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
                (
                    "transaction_match_status,finding_count,"
                    "transaction_match_confidence,transaction_match_interpretation,"
                    "transaction_match_review_note"
                ),
                paths["transaction_matching_diagnostics"].read_text(encoding="utf-8"),
            )

    def test_report_bundle_validation_catches_missing_artifact(self) -> None:
        """Bundle validation reports required artifact files that are absent."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )

            paths["needs_review_summary"].unlink()
            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            "artifact file 'supporting_files/needs_review_summary.csv' is missing",
            issues,
        )

    def test_report_bundle_validation_catches_missing_manifest_key(self) -> None:
        """Bundle validation rejects missing required top-level manifest keys."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            del manifest["manifest_version"]
            paths["manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn("manifest top-level key 'manifest_version' is missing", issues)
        self.assertIn("manifest manifest_version is unsupported", issues)

    def test_report_bundle_validation_catches_unknown_review_entrypoint(self) -> None:
        """Bundle validation rejects review entrypoints outside declared artifacts."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["review_entrypoints"]["period_triage"] = "missing.csv"
            paths["manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            (
                "manifest review entrypoint 'period_triage' points to "
                "undeclared artifact 'missing.csv'"
            ),
            issues,
        )

    def test_report_bundle_validation_catches_missing_review_entrypoint(self) -> None:
        """Bundle validation rejects drift in required review-entrypoint names."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            del manifest["review_entrypoints"]["review_handoff"]
            paths["manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn("manifest review entrypoint 'review_handoff' is missing", issues)

    def test_report_bundle_validation_catches_bad_source_context(self) -> None:
        """Bundle validation checks source-context manifest metadata shape."""
        findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                comparison_path=_RESTATEMENT_TRANSACTION_RULES_PATH,
                require_complete_yaml_setup=False,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["source_context"]["extract_contract"][
                "required_transaction_context_columns"
            ] = "source_destination_type"
            paths["manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            (
                "manifest extract_contract.required_transaction_context_columns "
                "is malformed"
            ),
            issues,
        )

    def test_report_bundle_validation_catches_bad_transaction_summary(self) -> None:
        """Bundle validation checks transaction-semantics manifest metadata."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["transaction_semantics"]["unknown_category_count"] = -1
            paths["manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            "manifest transaction_semantics.unknown_category_count is malformed",
            issues,
        )

    def test_report_bundle_validation_catches_bad_review_summary(self) -> None:
        """Bundle validation checks compact review-summary metadata."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            summary = json.loads(paths["review_summary"].read_text(encoding="utf-8"))
            summary["entrypoints"]["period_triage"] = "stale.csv"
            paths["review_summary"].write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn("review_summary entrypoints does not match manifest", issues)

    def test_report_bundle_validation_catches_missing_review_summary_key(self) -> None:
        """Bundle validation rejects drift in compact review-summary keys."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            summary = json.loads(paths["review_summary"].read_text(encoding="utf-8"))
            del summary["review_vocabulary"]
            paths["review_summary"].write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            "review_summary top-level key 'review_vocabulary' is missing",
            issues,
        )

    def test_report_bundle_validation_catches_missing_review_workbook(self) -> None:
        """Bundle validation reports workbook artifacts that are absent."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                include_workbook=True,
                require_complete_yaml_setup=False,
            )

            paths["review_workbook"].unlink()
            issues = report_bundle_validation_issues(directory)

        self.assertIn("artifact file 'report.xlsx' is missing", issues)

    def test_report_bundle_validation_catches_invalid_review_workbook(self) -> None:
        """Bundle validation checks workbook sheet structure."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                include_workbook=True,
                require_complete_yaml_setup=False,
            )
            workbook = openpyxl.load_workbook(paths["review_workbook"])
            del workbook["Performance Differences"]
            workbook.save(paths["review_workbook"])

            issues = report_bundle_validation_issues(directory)

        self.assertIn(
            "report.xlsx is missing primary sheet 'Performance Differences'",
            issues,
        )

    def test_report_bundle_validation_catches_csv_row_count_drift(self) -> None:
        """Bundle validation compares manifest row counts to CSV row counts."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                directory,
                require_complete_yaml_setup=False,
            )
            header = paths["top_evidence"].read_text(encoding="utf-8").splitlines()[0]
            paths["top_evidence"].write_text(header + "\n", encoding="utf-8")

            issues = report_bundle_validation_issues(directory)

        self.assertIn("table 'top_evidence' row count is 0, expected 10", issues)

    def test_report_bundle_write_fails_if_validation_fails(self) -> None:
        """Bundle writing raises if post-write validation detects corruption."""
        findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)

        with mock.patch(
            "ppar.performance_comparison.report._pc_bundle.report_bundle_validation_issues",
            return_value=["simulated validation issue"],
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(PpaError, "simulated validation issue"):
                    write_performance_comparison_report_bundle(
                        findings,
                        directory,
                        require_complete_yaml_setup=False,
                    )


def _section(report: str, start: str, end: str) -> str:
    """Return report text between two section markers."""
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
