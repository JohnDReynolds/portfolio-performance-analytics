"""Audit packaged performance-comparison demo data for accounting consistency."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

import pandas as pd

from ppar.performance_comparison import compare_snapshots
from ppar.performance_comparison.workbook_tables import _workbook_portfolio_changes_table


_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_DEFAULT_COMPARISON_PATH: Final = (
    _REPO_ROOT / "ppar" / "demos" / "data" / "axys" / "ppar_performance_comparison_portfolio.yaml"
)
_INTENTIONAL_PORTFOLIO_RESIDUALS: Final = {
    ("BALANCED", "2026-05-01", "2026-05-29", "Partly Explained"): (
        "Intentional partial example: beginning-value and ending-value changes "
        "both foot, but the selected report causes leave a denominator-effect "
        "residual for review."
    ),
    ("INCOME", "2026-04-01", "2026-04-30", "Unexplained"): (
        "Intentional vendor/methodology residual used to demonstrate unresolved review."
    ),
}
_MONEY_TOLERANCE: Final = 0.025
_RETURN_TOLERANCE: Final = 0.000001


@dataclass(frozen=True)
class AuditIssue:
    """One demo-data audit issue.

    Attributes:
        check: Name of the consistency check that failed.
        snapshot: Snapshot label such as ``a`` or ``b`` when applicable.
        portfolio: Portfolio code for the affected row.
        from_date: Period start date, if applicable.
        thru_date: Period end date, if applicable.
        detail: Human-readable explanation of the issue.
    """

    check: str
    detail: str
    snapshot: str | None = None
    portfolio: str | None = None
    from_date: str | None = None
    thru_date: str | None = None


def audit_demo_data(comparison_path: Path = _DEFAULT_COMPARISON_PATH) -> list[AuditIssue]:
    """Return accounting and visible-residual issues for packaged demo data.

    Args:
        comparison_path: Portfolio performance-comparison YAML to audit.

    Returns:
        Audit issues. An empty list means the currently enforced accounting
        relationships and review-status guardrails passed.

    Notes:
        This audit intentionally avoids security-level return reconstruction
        residuals because security-level flow reconstruction is still a roadmap
        item. It checks relationships that are already definitive for the demo:
        portperf arithmetic, secperf-to-portperf rollups, and visible portfolio
        residuals.
    """
    comparison_path = comparison_path.resolve()
    demo_root = comparison_path.parent
    issues: list[AuditIssue] = []
    for snapshot in ("a", "b"):
        snapshot_path = demo_root / f"axys_full_spec_{snapshot}"
        issues.extend(_audit_portfolio_performance_arithmetic(snapshot, snapshot_path))
        issues.extend(_audit_security_to_portfolio_rollup(snapshot, snapshot_path))
    issues.extend(_audit_visible_portfolio_residuals(comparison_path))
    return issues


def _audit_portfolio_performance_arithmetic(
    snapshot: str,
    snapshot_path: Path,
) -> list[AuditIssue]:
    """Return issues where portfolio rows do not foot internally."""
    portperf = pd.read_csv(snapshot_path / "portperf.csv")
    expected_end = (
        portperf["BEGIN_MV"]
        + portperf["FLOW"]
        + portperf["INCOME"]
        + portperf["GAIN_LOSS"]
    )
    delta = portperf["END_MV"] - expected_end
    issues: list[AuditIssue] = []
    for row in portperf.loc[delta.abs() > _MONEY_TOLERANCE].itertuples(index=False):
        issues.append(
            AuditIssue(
                check="portperf_arithmetic",
                snapshot=snapshot,
                portfolio=str(row.PORTFOLIO_CODE),
                from_date=str(row.FROM_DATE),
                thru_date=str(row.THRU_DATE),
                detail=(
                    "END_MV does not equal BEGIN_MV + FLOW + INCOME + GAIN_LOSS."
                ),
            )
        )
    return issues


def _audit_security_to_portfolio_rollup(
    snapshot: str,
    snapshot_path: Path,
) -> list[AuditIssue]:
    """Return issues where security rows do not roll up to portfolio rows."""
    portperf = pd.read_csv(snapshot_path / "portperf.csv")
    secperf = pd.read_csv(snapshot_path / "secperf.csv")
    rollup = secperf.groupby(
        ["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"],
        as_index=False,
    )[["BEGIN_MV", "END_MV", "INCOME", "GAIN_LOSS"]].sum()
    merged = portperf.merge(
        rollup,
        on=["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"],
        suffixes=("_portfolio", "_security"),
        how="outer",
    )
    issues: list[AuditIssue] = []
    for value_column in ("BEGIN_MV", "END_MV", "INCOME", "GAIN_LOSS"):
        delta = (
            merged[f"{value_column}_portfolio"]
            - merged[f"{value_column}_security"]
        )
        for row in merged.loc[delta.abs() > _MONEY_TOLERANCE].itertuples(index=False):
            issues.append(
                AuditIssue(
                    check=f"secperf_to_portperf_{value_column.lower()}",
                    snapshot=snapshot,
                    portfolio=str(row.PORTFOLIO_CODE),
                    from_date=str(row.FROM_DATE),
                    thru_date=str(row.THRU_DATE),
                    detail=f"Security {value_column} does not roll up to portperf.",
                )
            )
    return issues


def _audit_visible_portfolio_residuals(comparison_path: Path) -> list[AuditIssue]:
    """Return issues for unintended visible portfolio residuals."""
    findings = compare_snapshots(comparison_path)
    portfolio_rows = _workbook_portfolio_changes_table(
        findings,
        comparison_path=comparison_path,
    )
    issues: list[AuditIssue] = []
    for row in portfolio_rows.iter_rows(named=True):
        status = str(row["review_status"])
        if status == "Fully Explained":
            unexplained = float(row["unexplained_change"] or 0.0)
            if abs(unexplained) > _RETURN_TOLERANCE:
                issues.append(
                    _portfolio_residual_issue(row, "Fully explained row has residual.")
                )
            continue
        key = (
            str(row["portfolio_id"]),
            row["from_date"].isoformat(),
            row["thru_date"].isoformat(),
            status,
        )
        if key not in _INTENTIONAL_PORTFOLIO_RESIDUALS:
            issues.append(
                _portfolio_residual_issue(
                    row,
                    "Non-fully-explained portfolio period is not intentional.",
                )
            )
    return issues


def _portfolio_residual_issue(row: dict[str, object], detail: str) -> AuditIssue:
    """Return an audit issue for one portfolio-period residual row."""
    return AuditIssue(
        check="visible_portfolio_residual",
        portfolio=str(row["portfolio_id"]),
        from_date=row["from_date"].isoformat(),
        thru_date=row["thru_date"].isoformat(),
        detail=(
            f"{detail} Status={row['review_status']}; "
            f"unexplained={row['unexplained_change']}."
        ),
    )


def main() -> None:
    """Run the demo-data audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_COMPARISON_PATH,
        help="Portfolio performance-comparison YAML to audit.",
    )
    args = parser.parse_args()
    issues = audit_demo_data(args.comparison_path)
    print(json.dumps([asdict(issue) for issue in issues], indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
