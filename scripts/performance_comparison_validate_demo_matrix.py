"""Validate packaged performance comparison demo scenario coverage."""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Third-party imports
import polars as pl  # noqa: E402

# Project imports
from ppar.performance_comparison import compare_snapshots, summarize_findings  # noqa: E402
from ppar.performance_comparison import explain as _pc_explain  # noqa: E402
from ppar.performance_comparison import findings as _pc_findings  # noqa: E402
from ppar.performance_comparison.report import (  # noqa: E402
    _context_evidence_table,
    _problem_table,
    _residual_status_table,
)

_DEFAULT_DEMO_DIRECTORY = _REPO_ROOT / "ppar" / "demo_data" / "axys"
_BASELINE_YAML = "ppar_performance_comparison.yaml"
_MULTI_YAML = "ppar_performance_comparison_multi_restatement.yaml"
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
    multi_findings = compare_snapshots(demo_directory / _MULTI_YAML)
    modified_dietz_findings = compare_snapshots(demo_directory / _MODIFIED_DIETZ_YAML)
    policy_gap_findings = compare_snapshots(demo_directory / _POLICY_GAP_YAML)
    suppressed_findings = compare_snapshots(demo_directory / _SUPPRESSED_YAML)
    suppressed_active_findings = compare_snapshots(
        demo_directory / _SUPPRESSED_YAML,
        include_suppressed=False,
    )

    baseline_problems = _problem_table(baseline_findings)
    multi_problems = _problem_table(multi_findings)
    policy_gap_problems = _problem_table(policy_gap_findings)
    context_evidence = _context_evidence_table(multi_findings)
    modified_dietz_cross_checks = (
        _pc_explain.portfolio_period_transaction_cross_checks(modified_dietz_findings)
    )
    residual_status = _residual_status_table(multi_findings)
    suppressed_summary = summarize_findings(suppressed_findings)["by_suppressed"]

    return [
        _check_no_problems(baseline_problems),
        _check_problem_action(
            "Missing contribution policy",
            policy_gap_problems,
            "contribution_impact_methods",
        ),
        _check_problem_action(
            "Missing transaction method",
            policy_gap_problems,
            "transaction_impact_methods",
        ),
        _check_problem_action(
            "Missing denominator",
            policy_gap_problems,
            "denominator_source",
        ),
        _check_problem_action(
            "Missing transaction sign/flow semantics",
            policy_gap_problems,
            "transaction sign and external-flow semantics",
        ),
        _check_problem_text(
            "Low-confidence estimate",
            multi_problems,
            "low-confidence screening estimate",
        ),
        _check_non_empty_table(
            "Context-only evidence",
            context_evidence,
            "context evidence row(s) remain available",
        ),
        _check_modified_dietz_cross_check(modified_dietz_cross_checks),
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


def _check_no_problems(problems: pl.DataFrame) -> _ScenarioCheck:
    """Return whether the baseline fixture produces no Problems-grid rows."""
    if problems.is_empty():
        return _ScenarioCheck("Clean/no issue", True, "baseline produced no problems")
    return _ScenarioCheck(
        "Clean/no issue",
        False,
        f"baseline produced {problems.height} problem row(s)",
    )


def _check_problem_action(
    name: str,
    problems: pl.DataFrame,
    expected_text: str,
) -> _ScenarioCheck:
    """Return whether a Problems-grid action contains expected text."""
    return _check_problem_column(name, problems, "action_required", expected_text)


def _check_problem_text(
    name: str,
    problems: pl.DataFrame,
    expected_text: str,
) -> _ScenarioCheck:
    """Return whether a Problems-grid problem statement contains expected text."""
    return _check_problem_column(name, problems, "problem", expected_text)


def _check_problem_column(
    name: str,
    problems: pl.DataFrame,
    column: str,
    expected_text: str,
) -> _ScenarioCheck:
    """Return whether any Problems-grid row contains expected text in a column."""
    values = [str(value) for value in problems.get_column(column).to_list()]
    if any(expected_text in value for value in values):
        return _ScenarioCheck(name, True, f"found `{expected_text}` in `{column}`")
    return _ScenarioCheck(name, False, f"missing `{expected_text}` in `{column}`")


def _check_non_empty_table(
    name: str,
    table: pl.DataFrame,
    detail: str,
) -> _ScenarioCheck:
    """Return whether a supporting table has rows."""
    if not table.is_empty():
        return _ScenarioCheck(name, True, f"{table.height} {detail}")
    return _ScenarioCheck(name, False, "expected at least one supporting row")


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
