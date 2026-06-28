"""Audit or rebuild derived performance-comparison demo CSV files.

The packaged performance-comparison demos keep user-visible operational inputs
in ``holdings.csv`` and ``transactions.csv``. The ``secperf.csv`` and
``portperf.csv`` files are derived review targets. This script keeps the derived
performance files internally aligned by:

1. deriving ``secperf.csv`` from holdings and security-level transactions;
2. deriving ``portperf.csv`` from holdings and portfolio-level transactions; and
3. reporting whether the checked-in files already match those derived values.

By default, the script audits without writing. Pass ``--write`` to update the
packaged demo files.
"""

from __future__ import annotations

# Python imports
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

# Third-party imports
import pandas as pd

# Project imports
from ppar.performance_comparison import compare_snapshots
from ppar.performance_comparison.methods import ReturnReconstructionMethod
from ppar.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.performance_comparison.specification import (
    PerformanceComparisonSpecification,
    PortfolioReturnReconstruction,
    SecurityReturnReconstruction,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_portfolio_changes_table,
    _workbook_security_changes_table,
)


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_AXYS_DIRECTORY: Final = _REPO_ROOT / "ppar" / "demos" / "data" / "axys"
_DEFAULT_COMPARISON_PATH: Final = (
    _DEFAULT_AXYS_DIRECTORY / "ppar_performance_comparison.yaml"
)
_SNAPSHOT_DIRECTORIES: Final = ("axys_full_spec_a", "axys_full_spec_b")
_PERIOD_KEY: Final = ["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"]
_SECURITY_PERIOD_KEY: Final = [*_PERIOD_KEY, "SECURITY_ID"]
_PORTPERF_COLUMNS: Final = [
    "END_MV",
    "FLOW",
    "INCOME",
    "GAIN_LOSS",
    "PORTFOLIO_CODE",
    "PORTFOLIO_NAME",
    "PERIOD_ID",
    "FROM_DATE",
    "THRU_DATE",
    "BEGIN_MV",
    "PORT_RETURN",
]
_SECPERF_NUMERIC_COLUMNS: Final = [
    "END_MV",
    "INCOME",
    "GAIN_LOSS",
    "BEGIN_WEIGHT",
    "BEGIN_MV",
    "SEC_RETURN",
    "CONTRIBUTION",
]
_PORTPERF_NUMERIC_COLUMNS: Final = [
    "END_MV",
    "FLOW",
    "INCOME",
    "GAIN_LOSS",
    "BEGIN_MV",
    "PORT_RETURN",
]
_CHECK_TOLERANCE: Final = 0.000000001
_RETURN_TOLERANCE: Final = 0.000001
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
_SECURITY_FLOW_CODES: Final = {"BUY", "SELL"}
_INCOME_CODES: Final = {"DIV", "INT", "FEE"}
_PORTFOLIO_EXTERNAL_FLOW_CODES: Final = {"DEP", "WD"}


@dataclass(frozen=True)
class AuditIssue:
    """One packaged demo-data audit issue.

    Attributes:
        check: Name of the consistency check that failed.
        detail: Human-readable explanation of the issue.
        snapshot: Snapshot label such as ``axys_full_spec_a`` when applicable.
        portfolio: Portfolio code for the affected row.
        from_date: Period start date, if applicable.
        thru_date: Period end date, if applicable.
    """

    check: str
    detail: str
    snapshot: str | None = None
    portfolio: str | None = None
    from_date: str | None = None
    thru_date: str | None = None


def main() -> int:
    """Audit or rewrite packaged performance-comparison demo performance files."""
    args = _parse_args()
    summary = rebuild_demo_performance_files(
        args.axys_directory,
        comparison_path=args.comparison_path,
        write=args.write,
    )
    audit_issues = audit_demo_data(
        axys_directory=args.axys_directory,
        comparison_path=args.comparison_path,
    )
    summary["audit_issues"] = [asdict(issue) for issue in audit_issues]
    print(json.dumps(summary, indent=2))

    if args.write:
        return 0
    if any(snapshot["has_drift"] for snapshot in summary["snapshots"]) or audit_issues:
        return 1
    return 0


def audit_demo_data(
    *,
    axys_directory: Path = _DEFAULT_AXYS_DIRECTORY,
    comparison_path: Path = _DEFAULT_COMPARISON_PATH,
) -> list[AuditIssue]:
    """Return packaged demo-data audit issues.

    Args:
        axys_directory: Directory containing the packaged Axys snapshots.
        comparison_path: Portfolio comparison YAML used for visible residual
            guardrails.

    Returns:
        Audit issues. An empty list means the checked-in derived performance
        files have no drift and visible portfolio residuals are intentional.
    """
    issues: list[AuditIssue] = []
    rebuild_summary = rebuild_demo_performance_files(axys_directory, write=False)
    for snapshot in rebuild_summary["snapshots"]:
        if snapshot["has_drift"]:
            issues.append(
                AuditIssue(
                    check="derived_performance_drift",
                    snapshot=str(snapshot["snapshot"]),
                    detail=(
                        "Derived secperf.csv or portperf.csv no longer matches "
                        "the rebuild script. Run this script with --write."
                    ),
                )
            )
    issues.extend(_audit_visible_portfolio_residuals(comparison_path))
    issues.extend(_audit_visible_security_residuals(comparison_path))
    return issues


def rebuild_demo_performance_files(
    axys_directory: Path,
    *,
    comparison_path: Path = _DEFAULT_COMPARISON_PATH,
    write: bool = False,
) -> dict[str, object]:
    """Return audit summary, optionally rewriting derived performance files.

    Args:
        axys_directory: Directory containing ``axys_full_spec_a`` and
            ``axys_full_spec_b``.
        comparison_path: Shared comparison YAML with reconstruction rules.
        write: Whether to write rebuilt ``secperf.csv`` and ``portperf.csv``.

    Returns:
        JSON-serializable audit summary with one entry per snapshot.
    """
    specification = PerformanceComparisonSpecification(comparison_path)
    portfolio_reconstruction = specification.portfolio_return_reconstruction
    security_reconstruction = specification.security_return_reconstruction
    if portfolio_reconstruction is None or security_reconstruction is None:
        raise ValueError("Demo rebuild requires portfolio and security reconstruction YAML.")

    snapshots: list[dict[str, object]] = []
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        snapshot_directory = axys_directory / snapshot_name
        current_secperf = pd.read_csv(snapshot_directory / "secperf.csv")
        current_portperf = pd.read_csv(snapshot_directory / "portperf.csv")
        holdings = pd.read_csv(snapshot_directory / "holdings.csv")
        transactions = pd.read_csv(snapshot_directory / "transactions.csv")

        rebuilt_secperf = _rebuild_security_performance(
            current_secperf,
            holdings,
            transactions,
            security_reconstruction,
        )
        rebuilt_portperf = _rebuild_portfolio_performance(
            current_portperf,
            holdings,
            transactions,
            portfolio_reconstruction,
        )
        secperf_delta = _max_numeric_delta(
            current_secperf,
            rebuilt_secperf,
            _SECPERF_NUMERIC_COLUMNS,
        )
        portperf_delta = _max_numeric_delta(
            current_portperf,
            rebuilt_portperf,
            _PORTPERF_NUMERIC_COLUMNS,
        )
        has_drift = (
            secperf_delta > _CHECK_TOLERANCE or portperf_delta > _CHECK_TOLERANCE
        )
        if write:
            rebuilt_secperf.to_csv(snapshot_directory / "secperf.csv", index=False)
            rebuilt_portperf.to_csv(snapshot_directory / "portperf.csv", index=False)

        snapshots.append(
            {
                "snapshot": snapshot_name,
                "secperf_rows": int(rebuilt_secperf.shape[0]),
                "portperf_rows": int(rebuilt_portperf.shape[0]),
                "max_secperf_numeric_delta": secperf_delta,
                "max_portperf_numeric_delta": portperf_delta,
                "has_drift": has_drift,
                "written": write,
            }
        )

    return {
        "axys_directory": str(axys_directory),
        "mode": "write" if write else "check",
        "snapshots": snapshots,
    }


def _rebuild_security_performance(
    secperf: pd.DataFrame,
    holdings: pd.DataFrame,
    transactions: pd.DataFrame,
    reconstruction: SecurityReturnReconstruction,
) -> pd.DataFrame:
    """Return security rows derived from holdings and security transactions."""
    holding_values = _holding_values(holdings)
    transaction_rows = _prepared_transactions(transactions)
    rows: list[dict[str, object]] = []
    for row in secperf.itertuples(index=False):
        begin_date = _begin_holding_date_or_none(
            holding_values,
            row.PORTFOLIO_CODE,
            pd.Timestamp(row.FROM_DATE),
        )
        end_date = pd.Timestamp(row.THRU_DATE)
        begin_value = (
            float(row.BEGIN_MV)
            if begin_date is None
            else _security_holding_value(
                holding_values,
                row.PORTFOLIO_CODE,
                row.SECURITY_ID,
                begin_date,
            )
        )
        end_value = _security_holding_value(
            holding_values,
            row.PORTFOLIO_CODE,
            row.SECURITY_ID,
            end_date,
        )
        net_flow, weighted_flow = _security_flows(
            transaction_rows,
            row.PORTFOLIO_CODE,
            row.SECURITY_ID,
            pd.Timestamp(row.FROM_DATE),
            end_date,
            reconstruction,
        )
        income = _security_income(
            transaction_rows,
            row.PORTFOLIO_CODE,
            row.SECURITY_ID,
            pd.Timestamp(row.FROM_DATE),
            end_date,
            reconstruction,
        )
        gain_loss = end_value - begin_value - net_flow
        denominator = begin_value + weighted_flow
        sec_return = (gain_loss + income) / denominator if denominator else 0.0
        rebuilt_row = row._asdict()
        rebuilt_row.update(
            {
                "END_MV": round(end_value, 2),
                "INCOME": round(income, 2),
                "GAIN_LOSS": round(gain_loss, 2),
                "BEGIN_MV": round(begin_value, 2),
                "SEC_RETURN": round(sec_return, 10),
            }
        )
        rows.append(rebuilt_row)

    rebuilt = pd.DataFrame(rows)
    period_begin_market_value = rebuilt.groupby(_PERIOD_KEY)["BEGIN_MV"].transform("sum")
    rebuilt["BEGIN_WEIGHT"] = (rebuilt["BEGIN_MV"] / period_begin_market_value).round(10)
    rebuilt["CONTRIBUTION"] = (rebuilt["BEGIN_WEIGHT"] * rebuilt["SEC_RETURN"]).round(10)
    return rebuilt[secperf.columns]


def _rebuild_portfolio_performance(
    portperf: pd.DataFrame,
    holdings: pd.DataFrame,
    transactions: pd.DataFrame,
    reconstruction: PortfolioReturnReconstruction,
) -> pd.DataFrame:
    """Return portfolio rows derived from holdings and external-flow transactions."""
    holding_values = _holding_values(holdings)
    transaction_rows = _prepared_transactions(transactions)
    rows: list[dict[str, object]] = []
    for row in portperf.itertuples(index=False):
        begin_date = _begin_holding_date_or_none(
            holding_values,
            row.PORTFOLIO_CODE,
            pd.Timestamp(row.FROM_DATE),
        )
        end_date = pd.Timestamp(row.THRU_DATE)
        begin_value = (
            float(row.BEGIN_MV)
            if begin_date is None
            else _portfolio_holding_value(
                holding_values,
                row.PORTFOLIO_CODE,
                begin_date,
            )
        )
        end_value = _portfolio_holding_value(
            holding_values,
            row.PORTFOLIO_CODE,
            end_date,
        )
        flow, weighted_flow = _portfolio_flows(
            transaction_rows,
            row.PORTFOLIO_CODE,
            pd.Timestamp(row.FROM_DATE),
            end_date,
            reconstruction,
        )
        income = _portfolio_income(
            transaction_rows,
            row.PORTFOLIO_CODE,
            pd.Timestamp(row.FROM_DATE),
            end_date,
            reconstruction,
        )
        numerator = end_value - begin_value - flow
        denominator = begin_value + weighted_flow
        rebuilt_row = row._asdict()
        rebuilt_row.update(
            {
                "END_MV": round(end_value, 2),
                "FLOW": round(flow, 2),
                "INCOME": round(income, 2),
                "GAIN_LOSS": round(numerator - income, 2),
                "BEGIN_MV": round(begin_value, 2),
                "PORT_RETURN": round(numerator / denominator if denominator else 0.0, 10),
            }
        )
        rows.append(rebuilt_row)
    return pd.DataFrame(rows)[_PORTPERF_COLUMNS]


def _holding_values(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings with normalized dates and total holding value."""
    values = holdings.copy()
    values["HOLDING_DATE"] = pd.to_datetime(values["HOLDING_DATE"])
    values["HOLDING_VALUE"] = values["MKT_VAL"].astype(float) + values["ACCRUED"].astype(float)
    return values


def _prepared_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with normalized dates and transaction codes."""
    prepared = transactions.copy()
    prepared["TRANSACTION_DATE"] = pd.to_datetime(prepared["TRANSACTION_DATE"])
    prepared["TRAN"] = prepared["TRAN"].astype(str).str.upper()
    prepared["AMOUNT"] = prepared["AMOUNT"].astype(float)
    return prepared


def _begin_holding_date_or_none(
    holdings: pd.DataFrame,
    portfolio_code: str,
    from_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Return the latest holding date before a period start, if available."""
    dates = holdings.loc[
        (holdings["PORT"].eq(portfolio_code)) & (holdings["HOLDING_DATE"] < from_date),
        "HOLDING_DATE",
    ]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def _portfolio_holding_value(
    holdings: pd.DataFrame,
    portfolio_code: str,
    holding_date: pd.Timestamp,
) -> float:
    """Return total portfolio holding value for one date."""
    rows = holdings[
        holdings["PORT"].eq(portfolio_code) & holdings["HOLDING_DATE"].eq(holding_date)
    ]
    if rows.empty:
        raise ValueError(f"Missing portfolio holdings for {portfolio_code} on {holding_date}.")
    return float(rows["HOLDING_VALUE"].sum())


def _security_holding_value(
    holdings: pd.DataFrame,
    portfolio_code: str,
    security_id: str,
    holding_date: pd.Timestamp,
) -> float:
    """Return one security holding value for one date."""
    rows = holdings[
        holdings["PORT"].eq(portfolio_code)
        & holdings["SEC"].eq(security_id)
        & holdings["HOLDING_DATE"].eq(holding_date)
    ]
    if rows.empty:
        raise ValueError(
            f"Missing holding for {portfolio_code}/{security_id} on {holding_date}."
        )
    return float(rows["HOLDING_VALUE"].sum())


def _security_flows(
    transactions: pd.DataFrame,
    portfolio_code: str,
    security_id: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    reconstruction: SecurityReturnReconstruction,
) -> tuple[float, float]:
    """Return net and weighted security-level flows for buy/sell rows."""
    rows = _period_transactions(transactions, portfolio_code, from_date, thru_date)
    rows = rows[rows["SEC"].eq(security_id) & rows["TRAN"].isin(_SECURITY_FLOW_CODES)]
    net_flow = 0.0
    weighted_flow = 0.0
    for row in rows.itertuples(index=False):
        security_flow = -float(row.AMOUNT)
        weight = _flow_weight(reconstruction, from_date, thru_date, row.TRANSACTION_DATE)
        net_flow += security_flow
        weighted_flow += security_flow * weight
    return net_flow, weighted_flow


def _security_income(
    transactions: pd.DataFrame,
    portfolio_code: str,
    security_id: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    reconstruction: SecurityReturnReconstruction,
) -> float:
    """Return security income/expense amount for one period."""
    rows = _period_transactions(transactions, portfolio_code, from_date, thru_date)
    rows = rows[
        rows["SEC"].eq(security_id)
        & rows["TRAN"].isin(set(reconstruction.income_categories) | _INCOME_CODES)
    ]
    return float(rows["AMOUNT"].sum())


def _portfolio_flows(
    transactions: pd.DataFrame,
    portfolio_code: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    reconstruction: PortfolioReturnReconstruction,
) -> tuple[float, float]:
    """Return net and weighted portfolio-level external flows."""
    rows = _period_transactions(transactions, portfolio_code, from_date, thru_date)
    rows = rows[
        rows["TRAN"].isin(set(reconstruction.flow_categories) | _PORTFOLIO_EXTERNAL_FLOW_CODES)
    ]
    net_flow = 0.0
    weighted_flow = 0.0
    for row in rows.itertuples(index=False):
        flow = float(row.AMOUNT)
        weight = _flow_weight(reconstruction, from_date, thru_date, row.TRANSACTION_DATE)
        net_flow += flow
        weighted_flow += flow * weight
    return net_flow, weighted_flow


def _portfolio_income(
    transactions: pd.DataFrame,
    portfolio_code: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    reconstruction: PortfolioReturnReconstruction,
) -> float:
    """Return portfolio income/expense amount for one period."""
    rows = _period_transactions(transactions, portfolio_code, from_date, thru_date)
    rows = rows[rows["TRAN"].isin(set(reconstruction.income_categories) | _INCOME_CODES)]
    return float(rows["AMOUNT"].sum())


def _period_transactions(
    transactions: pd.DataFrame,
    portfolio_code: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return transactions for one portfolio period."""
    return transactions[
        transactions["PORT"].eq(portfolio_code)
        & transactions["TRANSACTION_DATE"].between(from_date, thru_date)
    ]


def _flow_weight(
    reconstruction: PortfolioReturnReconstruction | SecurityReturnReconstruction,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    flow_date: pd.Timestamp,
) -> float:
    """Return the configured Dietz flow weight for one transaction."""
    if reconstruction.method == ReturnReconstructionMethod.SIMPLE_DIETZ.value:
        return 0.0
    if reconstruction.method == ReturnReconstructionMethod.MODIFIED_SIMPLE_DIETZ.value:
        return 0.5
    if reconstruction.inclusion_rule is None:
        raise ValueError("modified_dietz reconstruction requires inclusion_rule.")
    return modified_dietz_flow_weight(
        from_date=from_date.date(),
        thru_date=thru_date.date(),
        flow_date=flow_date.date(),
        inclusion_rule=reconstruction.inclusion_rule,
    )


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


def _audit_visible_security_residuals(comparison_path: Path) -> list[AuditIssue]:
    """Return issues for unintended visible security residuals."""
    findings = compare_snapshots(comparison_path, comparison_level="security")
    security_rows = _workbook_security_changes_table(
        findings,
        comparison_path=comparison_path,
        comparison_level="security",
    )
    issues: list[AuditIssue] = []
    for row in security_rows.iter_rows(named=True):
        status = str(row["review_status"])
        unexplained = float(row["unexplained_change"] or 0.0)
        if status == "Fully Explained":
            if abs(unexplained) > _RETURN_TOLERANCE:
                issues.append(
                    _security_residual_issue(row, "Fully explained row has residual.")
                )
            continue
        if abs(unexplained) > _RETURN_TOLERANCE:
            issues.append(
                _security_residual_issue(
                    row,
                    "Non-fully-explained security period is not intentional.",
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


def _security_residual_issue(row: dict[str, object], detail: str) -> AuditIssue:
    """Return an audit issue for one security-period residual row."""
    return AuditIssue(
        check="visible_security_residual",
        portfolio=str(row["portfolio_id"]),
        from_date=row["from_date"].isoformat(),
        thru_date=row["thru_date"].isoformat(),
        detail=(
            f"{detail} Security={row['security_id']}; "
            f"Status={row['review_status']}; unexplained={row['unexplained_change']}."
        ),
    )


def _max_numeric_delta(
    current: pd.DataFrame,
    rebuilt: pd.DataFrame,
    numeric_columns: list[str],
) -> float:
    """Return the maximum absolute numeric difference between aligned frames."""
    if current.shape != rebuilt.shape:
        return float("inf")

    max_delta = 0.0
    for column in numeric_columns:
        current_values = pd.to_numeric(current[column], errors="coerce")
        rebuilt_values = pd.to_numeric(rebuilt[column], errors="coerce")
        column_delta = (current_values - rebuilt_values).abs().max()
        if pd.notna(column_delta):
            max_delta = max(max_delta, float(column_delta))
    return max_delta


def _parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit or rebuild derived portperf/secperf files for the packaged "
            "performance-comparison demos."
        )
    )
    parser.add_argument(
        "--axys-directory",
        type=Path,
        default=_DEFAULT_AXYS_DIRECTORY,
        help="Directory containing axys_full_spec_a and axys_full_spec_b.",
    )
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=_DEFAULT_COMPARISON_PATH,
        help="Portfolio comparison YAML used for visible residual guardrails.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite secperf.csv and portperf.csv instead of audit-only mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
