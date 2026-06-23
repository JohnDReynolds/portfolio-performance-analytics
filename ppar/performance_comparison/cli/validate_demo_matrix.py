"""Validate packaged performance comparison demo scenario coverage."""

# Python imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import compare_snapshots, summarize_findings
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison.report import (
    _context_evidence_table,
    _residual_status_table,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_portfolio_changes_table,
    _workbook_underlying_causes_table,
)

_DEFAULT_DEMO_DIRECTORY = Path(__file__).resolve().parents[2] / "demos" / "data" / "axys"
_BASELINE_YAML = "ppar_performance_comparison.yaml"
_RESTATEMENT_YAML = "ppar_performance_comparison_restatement.yaml"
_RESTATEMENT_TRANSACTION_RULES_YAML = (
    "ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_MULTI_YAML = "ppar_performance_comparison_multi_restatement.yaml"
_FULL_SPEC_YAML = "ppar_performance_comparison_full_spec.yaml"
_SECURITY_FULL_SPEC_YAML = "ppar_performance_comparison_security_full_spec.yaml"
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
    """Validate packaged demo fixtures against the documented scenario matrix.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means the covered scenarios still pass; ``1``
        means one or more matrix expectations drifted.
    """
    args = _argument_parser().parse_args(argv)
    checks = _validate_demo_matrix(args.demo_directory)
    failures = [check for check in checks if not check.passed]

    if not failures:
        print(
            f"Demo matrix validation passed: {len(checks)} scenario(s) checked "
            f"under {args.demo_directory}"
        )
        for check in checks:
            print(f"- {check.name}: {check.detail}")
        return 0

    print(
        f"Demo matrix validation failed: {len(failures)} of {len(checks)} "
        f"scenario(s) failed under {args.demo_directory}",
        file=sys.stderr,
    )
    for check in failures:
        print(f"- {check.name}: {check.detail}", file=sys.stderr)
    return 1


def _validate_demo_matrix(demo_directory: Path) -> list[_ScenarioCheck]:
    """Return scenario validation checks for the packaged Axys demo directory.

    Args:
        demo_directory: Directory containing the packaged Axys demo YAML files.

    Returns:
        One result per covered scenario in the demo matrix.

    Raises:
        FileNotFoundError: If one of the required YAML fixtures is missing.
    """
    baseline_findings = compare_snapshots(demo_directory / _BASELINE_YAML)
    restatement_findings = compare_snapshots(demo_directory / _RESTATEMENT_YAML)
    transaction_rules_findings = compare_snapshots(
        demo_directory / _RESTATEMENT_TRANSACTION_RULES_YAML
    )
    multi_findings = compare_snapshots(demo_directory / _MULTI_YAML)
    full_spec_findings = compare_snapshots(
        demo_directory / _FULL_SPEC_YAML,
        require_causal_attribution=True,
    )
    security_full_spec_findings = compare_snapshots(
        demo_directory / _SECURITY_FULL_SPEC_YAML,
    )
    modified_dietz_findings = compare_snapshots(demo_directory / _MODIFIED_DIETZ_YAML)
    policy_gap_findings = compare_snapshots(demo_directory / _POLICY_GAP_YAML)
    suppressed_findings = compare_snapshots(demo_directory / _SUPPRESSED_YAML)
    suppressed_active_findings = compare_snapshots(
        demo_directory / _SUPPRESSED_YAML,
        include_suppressed=False,
    )

    baseline_portfolio_changes = _workbook_portfolio_changes_table(baseline_findings)
    restatement_causes = _workbook_underlying_causes_table(restatement_findings)
    transaction_rules_causes = _workbook_underlying_causes_table(transaction_rules_findings)
    multi_causes = _workbook_underlying_causes_table(multi_findings)
    policy_gap_causes = _workbook_underlying_causes_table(policy_gap_findings)
    context_evidence = _context_evidence_table(multi_findings)
    modified_dietz_cross_checks = (
        _pc_explain.portfolio_period_transaction_cross_checks(modified_dietz_findings)
    )
    residual_status = _residual_status_table(multi_findings)
    suppressed_summary = summarize_findings(suppressed_findings)["by_suppressed"]

    return [
        _check_no_portfolio_differences(baseline_portfolio_changes),
        _check_workbook_column(
            "Missing price impact method",
            policy_gap_causes,
            "required_yaml_setup",
            "price_impact_methods",
        ),
        _check_workbook_column(
            "Missing transaction method",
            policy_gap_causes,
            "required_yaml_setup",
            "transaction_impact_methods",
        ),
        _check_workbook_column(
            "Missing transaction rules",
            policy_gap_causes,
            "required_yaml_setup",
            "transaction_rules",
        ),
        _check_workbook_column(
            "Multi-portfolio missing specifications",
            multi_causes,
            "required_yaml_setup",
            "position_impact_methods",
        ),
        _check_transaction_rows_visible(restatement_causes),
        _check_transaction_rules_explain_amount(transaction_rules_causes),
        _check_non_empty_table(
            "Context-only evidence",
            context_evidence,
            "context evidence row(s) remain available",
        ),
        _check_large_clean_background(demo_directory, multi_findings),
        _check_modified_dietz_cross_check(modified_dietz_cross_checks),
        _check_full_spec_strict_attribution(full_spec_findings),
        _check_security_full_spec_attribution(security_full_spec_findings),
        _check_suppressed_findings(
            suppressed_findings,
            suppressed_active_findings,
            suppressed_summary,
        ),
        _check_residual_withheld(residual_status),
    ]


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate packaged performance comparison demo scenario coverage.",
    )
    parser.add_argument(
        "--demo-directory",
        type=Path,
        default=_DEFAULT_DEMO_DIRECTORY,
        help="Directory containing packaged Axys demo YAML files.",
    )
    return parser


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


def _check_transaction_rows_visible(causes: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the single-restatement workbook table shows transactions."""
    transaction_rows = causes.filter(pl.col("dataset") == "transactions")
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
    """Return whether transaction-rules YAML explains transaction amount deltas."""
    amount_rows = causes.filter(
        (pl.col("dataset") == "transactions")
        & (pl.col("source_column") == "amount")
        & pl.col("estimated_impact").is_not_null()
    )
    if amount_rows.height == 1:
        return _ScenarioCheck(
            "Transaction rules amount explanation",
            True,
            "transaction amount row has a performance explanation",
        )
    return _ScenarioCheck(
        "Transaction rules amount explanation",
        False,
        "transaction amount row does not have a performance explanation",
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
    demo_directory: Path,
    findings: pl.DataFrame,
) -> _ScenarioCheck:
    """Return whether the multi fixture includes clean multi-period scale data."""
    name = "Large multi-period clean background"
    try:
        snapshot_a_periods = _large_background_period_count(
            demo_directory / "axys_a" / "portperf.csv"
        )
        snapshot_b_periods = _large_background_period_count(
            demo_directory / "axys_b_multi_restatement" / "portperf.csv"
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


def _check_full_spec_strict_attribution(findings: pl.DataFrame) -> _ScenarioCheck:
    """Return whether full-spec YAML exercises every strict attribution basis."""
    name = "Full YAML specifications"
    expected_policy_prefixes = {
        "portfolio_source_field:source_field_delta_over_begin_market_value",
        "position_accrued:accrued_delta_over_return_denominator",
        "position_market_value:market_value_delta_over_return_denominator",
        "position_quantity:quantity_delta_times_snapshot_a_unit_market_value",
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
            f"strict fixture is missing impact policy value(s): {', '.join(missing)}",
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
            "strict fixture is missing transaction performance amount policy",
        )
    causes = _workbook_underlying_causes_table(findings)
    evidence_only_rows = causes.filter(
        (pl.col("dataset") == "positions")
        & (pl.col("source_column") == "cost")
        & (pl.col("impact_status") == "Review only")
    )
    if evidence_only_rows.is_empty():
        return _ScenarioCheck(
            name,
            False,
            "strict fixture is missing explicit evidence-only position row",
        )
    return _ScenarioCheck(
        name,
        True,
        "strict attribution accepted full YAML and covered all impact bases",
    )


def _check_security_full_spec_attribution(findings: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the security full-spec fixture exercises security impact."""
    name = "Security full YAML specifications"
    security_policies = {
        str(value)
        for value in findings.filter(pl.col("dataset") == "security_performance")
        .get_column("impact_policy")
        .drop_nulls()
        .to_list()
    }
    has_weighted_return = any(
        policy.startswith("security_return:security_return_delta_times_weight")
        for policy in security_policies
    )
    has_vendor_contribution = any(
        policy.startswith("security_contribution:vendor_contribution_delta")
        for policy in security_policies
    )
    if not has_weighted_return or not has_vendor_contribution:
        return _ScenarioCheck(
            name,
            False,
            "security fixture is missing security return or contribution policy",
        )
    return _ScenarioCheck(
        name,
        True,
        "security attribution covered security-period return and contribution policies",
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
