"""Validate performance comparison scenario coverage."""

# Python imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import compare_snapshots, summarize_findings
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import field_roles as _field_roles
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import schema as _pc_cols
from ppar.performance_comparison.backlog_gates import (
    CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES,
    SHORT_SIDE_BACKLOG_TRANSACTION_CODES,
)
from ppar.performance_comparison.config_validation import validate_config
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.transactions import TransactionsLoader
from ppar.performance_comparison.report import (
    _context_evidence_table,
    _residual_status_table,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_portfolio_changes_table,
    _workbook_raw_audit_trail_table,
    _workbook_underlying_causes_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEMO_DIRECTORY = (
    _REPO_ROOT / "ppar" / "demos" / "data" / "axysapx_performance_comparison"
)
_DEFAULT_SCENARIO_DIRECTORY = _REPO_ROOT / "tests" / "data" / "axys" / "validation"
_DEFAULT_SNAPSHOT_DIRECTORY = _REPO_ROOT / "tests" / "data" / "axys" / "snapshots"
_DEFAULT_SITE_VARIANTS_DIRECTORY = (
    _REPO_ROOT / "tests" / "data" / "axys" / "site_variants"
)
_TRANSACTION_SEMANTICS_MATRIX_PATH = (
    _REPO_ROOT
    / "docs"
    / "axys-apx-reference"
    / "contracts"
    / "transaction_semantics_matrix.yaml"
)
_BASELINE_YAML = "ppar_performance_comparison.yaml"
_RESTATEMENT_YAML = "ppar_performance_comparison_restatement.yaml"
_RESTATEMENT_TRANSACTION_RULES_YAML = (
    "ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_MULTI_YAML = "ppar_performance_comparison_multi_restatement.yaml"
_PACKAGED_DEMO_YAML = "axysapx_performance_comparison.yaml"
_SITE_VARIANT_YAML = "ppar_performance_comparison.yaml"
_MODIFIED_DIETZ_YAML = "ppar_performance_comparison_modified_dietz.yaml"
_POLICY_GAP_YAML = "ppar_performance_comparison_policy_gap_demo.yaml"
_SUPPRESSED_YAML = "ppar_performance_comparison_suppressed.yaml"


@dataclass(frozen=True)
class _ScenarioCheck:
    """One scenario validation result."""

    name: str
    passed: bool
    detail: str


def main(argv: list[str] | None = None) -> int:
    """Validate scenario fixtures against the documented matrix.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means the covered scenarios still pass; ``1``
        means one or more matrix expectations drifted.
    """
    args = _argument_parser().parse_args(argv)
    checks = _validate_demo_matrix(
        args.scenario_directory,
        args.snapshot_directory,
        args.demo_directory,
        args.site_variants_directory,
    )
    failures = [check for check in checks if not check.passed]

    if not failures:
        print(
            f"Demo matrix validation passed: {len(checks)} scenario(s) checked "
            f"under {args.scenario_directory}"
        )
        print(
            "Demo matrix coverage includes ambiguous-flow context variants, "
            "code-only guard, reviewed local opt-out checks, and "
            "review-only action quarantine, plus capital-return and short-side "
            "candidate gates."
        )
        for check in checks:
            print(f"- {check.name}: {check.detail}")
        return 0

    print(
        f"Demo matrix validation failed: {len(failures)} of {len(checks)} "
        f"scenario(s) failed under {args.scenario_directory}",
        file=sys.stderr,
    )
    for check in failures:
        print(f"- {check.name}: {check.detail}", file=sys.stderr)
    return 1


def _validate_demo_matrix(
    scenario_directory: Path,
    snapshot_directory: Path,
    demo_directory: Path,
    site_variants_directory: Path | None = None,
) -> list[_ScenarioCheck]:
    """Return scenario validation checks for the Axys fixture directories.

    Args:
        scenario_directory: Directory containing test-only validation YAML files.
        snapshot_directory: Directory containing test-only Axys CSV snapshots.
        demo_directory: Directory containing packaged user-facing demo data.
        site_variants_directory: Directory containing site-shape fixtures.

    Returns:
        One result per covered scenario in the demo matrix.

    Raises:
        FileNotFoundError: If one of the required YAML fixtures is missing.
    """
    if site_variants_directory is None:
        site_variants_directory = _DEFAULT_SITE_VARIANTS_DIRECTORY
    baseline_findings = compare_snapshots(scenario_directory / _BASELINE_YAML)
    restatement_findings = compare_snapshots(scenario_directory / _RESTATEMENT_YAML)
    transaction_rules_findings = compare_snapshots(
        scenario_directory / _RESTATEMENT_TRANSACTION_RULES_YAML
    )
    multi_findings = compare_snapshots(scenario_directory / _MULTI_YAML)
    portfolio_findings = compare_snapshots(
        demo_directory / _PACKAGED_DEMO_YAML,
        require_causal_attribution=True,
        comparison_level="portfolio",
    )
    security_findings = compare_snapshots(
        demo_directory / _PACKAGED_DEMO_YAML,
        comparison_level="security",
    )
    modified_dietz_findings = compare_snapshots(
        scenario_directory / _MODIFIED_DIETZ_YAML
    )
    policy_gap_findings = compare_snapshots(scenario_directory / _POLICY_GAP_YAML)
    suppressed_findings = compare_snapshots(scenario_directory / _SUPPRESSED_YAML)
    suppressed_active_findings = compare_snapshots(
        scenario_directory / _SUPPRESSED_YAML,
        include_suppressed=False,
    )

    baseline_portfolio_changes = _workbook_portfolio_changes_table(baseline_findings)
    restatement_causes = _workbook_underlying_causes_table(restatement_findings)
    restatement_raw_audit_trail = _workbook_raw_audit_trail_table(restatement_findings)
    transaction_rules_causes = _workbook_underlying_causes_table(transaction_rules_findings)
    multi_causes = _workbook_underlying_causes_table(multi_findings)
    policy_gap_causes = _workbook_underlying_causes_table(policy_gap_findings)
    context_evidence = _context_evidence_table(multi_findings)
    modified_dietz_cross_checks = (
        _pc_explain.portfolio_period_transaction_cross_checks(modified_dietz_findings)
    )
    residual_status = _residual_status_table(multi_findings)
    suppressed_summary = summarize_findings(suppressed_findings)["by_suppressed"]

    checks: list[_ScenarioCheck] = []
    checks.extend(
        _baseline_and_attribution_checks(
            baseline_portfolio_changes,
            restatement_causes,
            restatement_raw_audit_trail,
            transaction_rules_causes,
            policy_gap_causes,
            context_evidence,
            snapshot_directory,
            multi_findings,
            modified_dietz_cross_checks,
            portfolio_findings,
            security_findings,
            suppressed_findings,
            suppressed_active_findings,
            suppressed_summary,
            residual_status,
        )
    )
    checks.extend(_site_variant_checks(site_variants_directory))
    checks.extend(_backlog_gate_checks())
    return checks


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate performance comparison scenario coverage.",
    )
    parser.add_argument(
        "--scenario-directory",
        type=Path,
        default=_DEFAULT_SCENARIO_DIRECTORY,
        help="Directory containing test-only scenario YAML files.",
    )
    parser.add_argument(
        "--demo-directory",
        type=Path,
        default=_DEFAULT_DEMO_DIRECTORY,
        help="Directory containing packaged user-facing Axys/APX demo data.",
    )
    parser.add_argument(
        "--snapshot-directory",
        type=Path,
        default=_DEFAULT_SNAPSHOT_DIRECTORY,
        help="Directory containing test-only Axys CSV snapshots.",
    )
    parser.add_argument(
        "--site-variants-directory",
        type=Path,
        default=_DEFAULT_SITE_VARIANTS_DIRECTORY,
        help="Directory containing test-only site-shape fixtures.",
    )
    return parser


def _baseline_and_attribution_checks(
    baseline_portfolio_changes: pl.DataFrame,
    restatement_causes: pl.DataFrame,
    restatement_raw_audit_trail: pl.DataFrame,
    transaction_rules_causes: pl.DataFrame,
    policy_gap_causes: pl.DataFrame,
    context_evidence: pl.DataFrame,
    snapshot_directory: Path,
    multi_findings: pl.DataFrame,
    modified_dietz_cross_checks: pl.DataFrame,
    portfolio_findings: pl.DataFrame,
    security_findings: pl.DataFrame,
    suppressed_findings: pl.DataFrame,
    suppressed_active_findings: pl.DataFrame,
    suppressed_summary: pl.DataFrame,
    residual_status: pl.DataFrame,
) -> list[_ScenarioCheck]:
    """Return baseline, attribution, and report-surface matrix checks."""
    return [
        _check_no_portfolio_differences(baseline_portfolio_changes),
        _check_workbook_column(
            "Missing transaction method",
            policy_gap_causes,
            "review_guidance",
            "transaction_impact_methods",
        ),
        _check_workbook_column(
            "Missing transaction rules",
            policy_gap_causes,
            "review_guidance",
            "transaction_rules",
        ),
        _check_transaction_rows_visible(restatement_causes, restatement_raw_audit_trail),
        _check_transaction_rules_explain_amount(transaction_rules_causes),
        _check_non_empty_table(
            "Context-only evidence",
            context_evidence,
            "context evidence row(s) remain available",
        ),
        _check_large_clean_background(snapshot_directory, multi_findings),
        _check_modified_dietz_cross_check(modified_dietz_cross_checks),
        _check_portfolio_strict_attribution(portfolio_findings),
        _check_security_attribution(security_findings),
        _check_suppressed_findings(
            suppressed_findings,
            suppressed_active_findings,
            suppressed_summary,
        ),
        _check_residual_withheld(residual_status),
    ]


def _site_variant_checks(site_variants_directory: Path) -> list[_ScenarioCheck]:
    """Return site-shape, opt-out, and review-only quarantine checks."""
    return [
        _check_ambiguous_flow_context_variants(site_variants_directory),
        _check_code_only_failure_guard(site_variants_directory),
        _check_reviewed_local_opt_out(site_variants_directory),
        _check_review_only_action_quarantine(site_variants_directory),
    ]


def _backlog_gate_checks() -> list[_ScenarioCheck]:
    """Return explicit backlog-gate matrix checks."""
    return [_check_capital_return_and_short_side_backlog_gates()]


def _check_no_portfolio_differences(portfolio_changes: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the baseline fixture produces no portfolio differences."""
    portfolio_ids = [
        str(value) for value in portfolio_changes.get_column("portfolio_id").to_list()
    ]
    if "No portfolio performance differences found" in portfolio_ids:
        return _ScenarioCheck(
            "Clean/no issue",
            True,
            "baseline produced no portfolio differences",
        )
    return _ScenarioCheck(
        "Clean/no issue",
        False,
        f"baseline produced {portfolio_changes.height} portfolio-difference row(s)",
    )


def _check_workbook_column(
    name: str,
    table: pl.DataFrame,
    column: str,
    expected_text: str,
) -> _ScenarioCheck:
    """Return whether any workbook row contains expected text in a column."""
    values = [str(value) for value in table.get_column(column).to_list()]
    if any(expected_text in value for value in values):
        return _ScenarioCheck(name, True, f"found `{expected_text}` in `{column}`")
    return _ScenarioCheck(name, False, f"missing `{expected_text}` in `{column}`")


def _check_transaction_rows_visible(
    causes: pl.DataFrame,
    raw_audit_trail: pl.DataFrame,
) -> _ScenarioCheck:
    """Return whether the single-restatement workbook table shows transactions."""
    transaction_rows = pl.concat(
        [
            causes.filter(pl.col("dataset") == "transactions"),
            raw_audit_trail.filter(pl.col("dataset") == "transactions"),
        ],
        how="diagonal_relaxed",
    )
    expected_columns = {"amount", "quantity", "price"}
    actual_columns = set(transaction_rows.get_column("source_column").to_list())
    if expected_columns.issubset(actual_columns):
        return _ScenarioCheck(
            "Single-restatement transaction rows",
            True,
            "transaction amount, quantity, and price rows are workbook-visible",
        )
    missing_columns = sorted(expected_columns - actual_columns)
    return _ScenarioCheck(
        "Single-restatement transaction rows",
        False,
        f"missing transaction source column(s): {', '.join(missing_columns)}",
    )


def _check_transaction_rules_explain_amount(causes: pl.DataFrame) -> _ScenarioCheck:
    """Return whether transaction-rules YAML surfaces transaction amount evidence."""
    amount_rows = causes.filter(
        (pl.col("dataset") == "transactions")
        & (pl.col("source_column") == "amount")
        & pl.col("estimated_impact").is_null()
        & pl.col("review_guidance").str.contains("Caused")
    )
    if amount_rows.height == 1:
        return _ScenarioCheck(
            "Transaction rules amount explanation",
            True,
            "transaction amount row is visible with review guidance",
        )
    return _ScenarioCheck(
        "Transaction rules amount explanation",
        False,
        "transaction amount row is not visible with review guidance",
    )


def _check_non_empty_table(
    name: str,
    table: pl.DataFrame,
    detail: str,
) -> _ScenarioCheck:
    """Return whether a supporting table has rows."""
    if not table.is_empty():
        return _ScenarioCheck(name, True, f"{table.height} {detail}")
    return _ScenarioCheck(name, False, "expected at least one supporting row")


def _check_large_clean_background(
    snapshot_directory: Path,
    findings: pl.DataFrame,
) -> _ScenarioCheck:
    """Return whether the multi fixture includes clean multi-period scale data."""
    name = "Large multi-period clean background"
    try:
        snapshot_a_periods = _large_background_period_count(
            snapshot_directory / "axys_a" / "portperf.csv"
        )
        snapshot_b_periods = _large_background_period_count(
            snapshot_directory / "axys_b_multi_restatement" / "portperf.csv"
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        return _ScenarioCheck(name, False, f"could not read PORT_LARGE rows: {error}")

    large_findings = findings.filter(pl.col(_pc_findings.PORTFOLIO_ID) == "PORT_LARGE")
    if snapshot_a_periods < 40 or snapshot_b_periods < 40:
        return _ScenarioCheck(
            name,
            False,
            (
                "expected at least 40 PORT_LARGE periods in each snapshot; "
                f"found {snapshot_a_periods} and {snapshot_b_periods}"
            ),
        )
    if not large_findings.is_empty():
        return _ScenarioCheck(
            name,
            False,
            f"PORT_LARGE produced {large_findings.height} unexpected finding row(s)",
        )
    return _ScenarioCheck(
        name,
        True,
        (
            f"PORT_LARGE has {snapshot_a_periods} clean period(s) in snapshot A "
            f"and {snapshot_b_periods} in snapshot B"
        ),
    )


def _large_background_period_count(portfolio_performance_path: Path) -> int:
    """Return unique PORT_LARGE period count from a demo portfolio file."""
    table = pl.read_csv(portfolio_performance_path)
    large_rows = table.filter(pl.col("PORTFOLIO_CODE") == "PORT_LARGE")
    if large_rows.is_empty():
        return 0
    return large_rows.select(["FROM_DATE", "THRU_DATE"]).unique().height


def _check_modified_dietz_cross_check(cross_checks: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the Modified Dietz demo produces a cross-check row."""
    name = "Modified Dietz external-flow cross-check"
    if cross_checks.is_empty():
        return _ScenarioCheck(name, False, "transaction cross-check table is empty")

    policies = [
        str(value)
        for value in cross_checks.get_column("transaction_impact_policies").to_list()
    ]
    diagnostics = [
        str(value)
        for value in cross_checks.get_column("transaction_impact_diagnostics").to_list()
    ]
    estimates = [
        float(value)
        for value in cross_checks.get_column("cross_check_estimate_total").to_list()
    ]
    if (
        any("external_flow:modified_dietz" in policy for policy in policies)
        and any("modified_dietz cross-check estimate" in item for item in diagnostics)
        and any(abs(estimate) > 0 for estimate in estimates)
    ):
        return _ScenarioCheck(name, True, "Modified Dietz cross-check row is available")
    return _ScenarioCheck(
        name,
        False,
        "missing modified_dietz policy, diagnostic, or nonzero estimate",
    )


def _check_portfolio_strict_attribution(findings: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the portfolio fixture exercises field-role attribution."""
    name = "Portfolio field-role specifications"
    expected_policy_prefixes = {
        "holding_market_value:market_value_delta_over_return_denominator",
        "holding_accrued:accrued_delta_over_return_denominator",
        "price_weighted:price_delta_over_snapshot_a_price_times_weight",
    }
    impact_policies = {
        str(value)
        for value in findings.get_column("impact_policy").drop_nulls().to_list()
    }
    missing = sorted(
        expected
        for expected in expected_policy_prefixes
        if not any(policy.startswith(expected) for policy in impact_policies)
    )
    if missing:
        return _ScenarioCheck(
            name,
            False,
            (
                "full fixture is missing default field-role policy value(s): "
                + ", ".join(missing)
            ),
        )
    transaction_policies = {
        str(value)
        for value in findings.get_column("transaction_impact_policy").drop_nulls().to_list()
    }
    if not any(
        policy.startswith("performance:transaction_amount_delta_over_return_denominator")
        for policy in transaction_policies
    ):
        return _ScenarioCheck(
            name,
            False,
            "portfolio fixture is missing transaction performance amount policy",
        )
    causes = _workbook_underlying_causes_table(findings)
    promoted_fields = {
        (str(row["dataset"]), str(row["source_column"]))
        for row in causes.select(["dataset", "source_column"]).iter_rows(named=True)
    }
    expected_promoted_fields = {
        ("holdings", "market_value"),
        ("holdings", "quantity"),
        ("transactions", "amount"),
        ("transactions", "commission"),
        ("transactions", "price"),
        ("transactions", "quantity"),
    }
    missing_promoted_fields = sorted(expected_promoted_fields - promoted_fields)
    if missing_promoted_fields:
        return _ScenarioCheck(
            name,
            False,
            "portfolio fixture is missing promoted evidence-only field(s): "
            + ", ".join(
                f"{dataset}.{column}" for dataset, column in missing_promoted_fields
            ),
        )
    raw_audit_trail = _workbook_raw_audit_trail_table(findings)
    raw_fields = {
        (str(row["dataset"]), str(row["source_column"]))
        for row in raw_audit_trail.select(["dataset", "source_column"]).iter_rows(
            named=True
        )
    }
    expected_raw_fields = {
        ("holdings", "cost"),
    }
    missing_raw_fields = sorted(expected_raw_fields - raw_fields)
    if missing_raw_fields:
        return _ScenarioCheck(
            name,
            False,
            "portfolio fixture is missing raw audit field(s): "
            + ", ".join(f"{dataset}.{column}" for dataset, column in missing_raw_fields),
        )
    return _ScenarioCheck(
        name,
        True,
        "field roles cover additive causes plus intentional context-only examples",
    )


def _check_security_attribution(findings: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the security fixture exercises security review."""
    name = "Security field-role specifications"
    check_fields = _reported_performance_component_fields(findings)
    expected_check_fields = {
        ("security_performance", "security_return"),
        ("security_performance", "contribution"),
    }
    if not expected_check_fields.issubset(check_fields):
        return _ScenarioCheck(
            name,
            False,
            "security fixture is missing reported performance check rows",
        )
    return _ScenarioCheck(
        name,
        True,
        "security review covered reported return and contribution diagnostics",
    )


def _reported_performance_component_fields(findings: pl.DataFrame) -> set[tuple[str, str]]:
    """Return reported-performance component fields present in findings."""
    fields: set[tuple[str, str]] = set()
    for row in findings.select(["dataset", "source_column"]).iter_rows(named=True):
        dataset = row["dataset"]
        source_column = row["source_column"]
        if _field_roles.is_reported_performance_component(dataset, source_column):
            fields.add((str(dataset), str(source_column)))
    return fields


def _check_suppressed_findings(
    all_findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    suppressed_summary: pl.DataFrame,
) -> _ScenarioCheck:
    """Return whether the suppression fixture hides a known active finding."""
    suppressed_count = _suppressed_count(suppressed_summary)
    active_delta = all_findings.height - active_findings.height
    if suppressed_count > 0 and active_delta == suppressed_count:
        return _ScenarioCheck(
            "Suppressed finding",
            True,
            f"{suppressed_count} suppressed finding(s) remain audit-visible",
        )
    return _ScenarioCheck(
        "Suppressed finding",
        False,
        (
            f"suppressed count {suppressed_count} did not match active "
            f"finding delta {active_delta}"
        ),
    )


def _suppressed_count(suppressed_summary: pl.DataFrame) -> int:
    """Return the count of suppressed rows from a summary table."""
    if suppressed_summary.is_empty():
        return 0
    rows = suppressed_summary.filter(pl.col(_pc_findings.SUPPRESSED))
    if rows.is_empty():
        return 0
    return int(rows.get_column("count").sum())


def _check_residual_withheld(residual_status: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the multi fixture still demonstrates withheld residuals."""
    if residual_status.is_empty():
        return _ScenarioCheck("Residual withheld", False, "residual table is empty")

    statuses = [
        str(value)
        for value in residual_status.get_column("residual_status").to_list()
    ]
    if any(status.startswith("withheld") for status in statuses):
        return _ScenarioCheck(
            "Residual withheld",
            True,
            "found at least one withheld residual status",
        )
    return _ScenarioCheck("Residual withheld", False, "no withheld residual status found")


def _check_ambiguous_flow_context_variants(site_directory: Path) -> _ScenarioCheck:
    """Return whether IMEX context fixtures cover ambiguous flow variants."""
    frame = _site_variant_transactions(site_directory, "imex_context")
    actual = {
        (
            str(row[_pc_cols.TRANSACTION_CODE]),
            str(row[_pc_cols.TRANSACTION_CATEGORY]),
            str(row[_pc_cols.PERFORMANCE_FLOW_SIGN]),
        )
        for row in frame.iter_rows(named=True)
    }
    expected = {
        ("li", "external_flow", "external"),
        ("li", "transfer", "neutral"),
        ("lo", "external_flow", "external"),
        ("lo", "transfer", "neutral"),
        ("dp", "fee_expense", "performance"),
        ("dp", "transfer", "neutral"),
        ("wd", "external_flow", "external"),
        ("wd", "transfer", "neutral"),
    }
    missing = sorted(expected - actual)
    if missing:
        return _ScenarioCheck(
            "Ambiguous flow context variants",
            False,
            "missing context variant(s): " + ", ".join(map(str, missing)),
        )
    return _ScenarioCheck(
        "Ambiguous flow context variants",
        True,
        "li/lo/dp/wd external, fee, and neutral variants are covered",
    )


def _check_code_only_failure_guard(site_directory: Path) -> _ScenarioCheck:
    """Return whether code-only ambiguous rows still fail before classification."""
    specification = PerformanceComparisonSpecification(
        site_directory / "imex_code_only" / _SITE_VARIANT_YAML
    )
    try:
        TransactionsLoader(specification).load("a")
    except PpaError as error:
        message = str(error)
        if (
            "ambiguous Axys/APX transaction codes DP, LI, LO, WD" in message
            and "IMEX transaction code alone is not enough" in message
        ):
            return _ScenarioCheck(
                "Code-only failure guard",
                True,
                "code-only li/lo/dp/wd rows fail before broad YAML classification",
            )
        return _ScenarioCheck(
            "Code-only failure guard",
            False,
            f"unexpected failure message: {message}",
        )
    return _ScenarioCheck(
        "Code-only failure guard",
        False,
        "code-only ambiguous rows loaded unexpectedly",
    )


def _check_reviewed_local_opt_out(site_directory: Path) -> _ScenarioCheck:
    """Return whether the reviewed local opt-out boundary remains explicit."""
    yaml_path = site_directory / "local_opt_out" / _SITE_VARIANT_YAML
    summary = validate_config(yaml_path, require_complete_yaml_setup=False)
    if summary["enforce_ambiguous_axys_flows"] is not False:
        return _ScenarioCheck(
            "Reviewed local opt-out",
            False,
            "local opt-out fixture did not disable ambiguous-flow enforcement",
        )
    frame = _site_variant_transactions(site_directory, "local_opt_out")
    sources = set(frame.get_column(_pc_cols.TRANSACTION_SEMANTICS_SOURCE).to_list())
    if sources == {"yaml_rule"}:
        return _ScenarioCheck(
            "Reviewed local opt-out",
            True,
            "code-only ambiguous rows classify only under explicit local opt-out",
        )
    return _ScenarioCheck(
        "Reviewed local opt-out",
        False,
        f"unexpected transaction semantics source(s): {sorted(sources)}",
    )


def _check_review_only_action_quarantine(site_directory: Path) -> _ScenarioCheck:
    """Return whether review-only action fixtures stay neutral."""
    frame = _site_variant_transactions(site_directory, "review_only_actions")
    actual = {
        (
            str(row[_pc_cols.TRANSACTION_CODE]),
            str(row[_pc_cols.TRANSACTION_CATEGORY]),
            str(row[_pc_cols.PERFORMANCE_FLOW_SIGN]),
            str(row[_pc_cols.TRANSACTION_SEMANTICS_SOURCE]),
        )
        for row in frame.iter_rows(named=True)
    }
    expected = {
        ("CXL", "transfer", "neutral", "source"),
        ("REV", "transfer", "neutral", "source"),
        (";", "corporate_action", "neutral", "source"),
    }
    missing = sorted(expected - actual)
    if missing:
        return _ScenarioCheck(
            "Review-only action quarantine",
            False,
            "missing neutral review-only row(s): " + ", ".join(map(str, missing)),
        )
    return _ScenarioCheck(
        "Review-only action quarantine",
        True,
        "correction/reversal and synthetic corporate-action rows stay neutral",
    )


def _check_capital_return_and_short_side_backlog_gates() -> _ScenarioCheck:
    """Return whether high-risk transaction families remain test-only candidates."""
    matrix = yaml.safe_load(_TRANSACTION_SEMANTICS_MATRIX_PATH.read_text())
    rows = matrix["rows"]
    expected = {
        code: (
            "Test-only site variant",
            "Code-only treatment remains unknown",
        )
        for code in CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES
    }
    expected.update(
        {
            code: (
                "Test-only site variant",
                "Code-only treatment remains unknown",
            )
            for code in SHORT_SIDE_BACKLOG_TRANSACTION_CODES
        }
    )

    failures = []
    for code, required_fragments in sorted(expected.items()):
        row = rows[code]
        coverage_notes = str(row["coverage_notes"])
        if row["coverage_status"] != "partial":
            failures.append(f"{code} status={row['coverage_status']}")
        missing_fragments = [
            fragment
            for fragment in required_fragments
            if fragment not in coverage_notes
        ]
        if missing_fragments:
            failures.append(f"{code} missing {', '.join(missing_fragments)}")
        if not row["fixtures"]:
            failures.append(f"{code} missing test-only fixture")

    if failures:
        return _ScenarioCheck(
            "Capital-return and short-side candidate gates",
            False,
            "; ".join(failures),
        )
    return _ScenarioCheck(
        "Capital-return and short-side candidate gates",
        True,
        "rc/pd and ss/cs remain test-only until site policy and evidence are present",
    )


def _site_variant_transactions(site_directory: Path, variant_name: str) -> pl.DataFrame:
    """Return snapshot A transactions for one site-variant fixture."""
    specification = PerformanceComparisonSpecification(
        site_directory / variant_name / _SITE_VARIANT_YAML
    )
    frame = TransactionsLoader(specification).load("a")
    if frame is None:
        raise AssertionError(f"{variant_name} transaction fixture did not load")
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
