"""Audit or rebuild derived performance-comparison demo CSV files.

The packaged performance-comparison demos keep user-visible operational inputs
in ``holdings.csv`` and ``transactions.csv``. The ``secperf.csv`` and
``portperf.csv`` files are derived review targets. This script keeps the derived
performance files internally aligned by:

1. deriving ``secperf.csv`` from holdings and security-level transactions;
2. deriving ``portperf.csv`` from holdings and portfolio-level transactions; and
3. deriving snapshot B ``transactions.csv`` from snapshot A transactions plus
   explicit transaction scenarios;
4. deriving snapshot B ``holdings.csv`` from snapshot A holdings plus
   transaction-derived and explicit holding scenarios; and
5. reporting whether the checked-in files already match those derived values.

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
_DEFAULT_HOLDING_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "performance_comparison_holding_scenarios.csv"
)
_DEFAULT_TRANSACTION_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "performance_comparison_transaction_scenarios.csv"
)
_SNAPSHOT_DIRECTORIES: Final = ("axys_full_spec_a", "axys_full_spec_b")
_BASE_SNAPSHOT_DIRECTORY: Final = "axys_full_spec_a"
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
_HOLDINGS_NUMERIC_COLUMNS: Final = ["QTY", "PRICE", "MKT_VAL", "COST", "ACCRUED"]
_HOLDING_SCENARIO_COLUMNS: Final = [
    "snapshot",
    "scenario_type",
    "PORT",
    "SEC",
    "HOLDING_DATE",
    "QTY_delta",
    "PRICE_delta",
    "MKT_VAL_delta",
    "COST_delta",
    "ACCRUED_delta",
    "scenario",
]
_HOLDING_SCENARIO_KEY: Final = [
    "snapshot",
    "scenario_type",
    "PORT",
    "SEC",
    "HOLDING_DATE",
    "scenario",
]
_HOLDING_SCENARIO_TYPES: Final = {
    "valuation_mark",
    "cash_balance_correction",
    "quantity_valuation_correction",
    "accrual_correction",
    "cost_only_correction",
}
_TRANSACTION_NUMERIC_COLUMNS: Final = ["QTY", "PRICE", "AMOUNT", "COMMISSION"]
_TRANSACTION_SCENARIO_COLUMNS: Final = [
    "snapshot",
    "TRANSACTION_ID",
    "QTY_delta",
    "PRICE_delta",
    "AMOUNT_delta",
    "COMMISSION_delta",
    "scenario",
]
_TRANSACTION_SCENARIO_KEY: Final = ["snapshot", "TRANSACTION_ID", "scenario"]
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
_SECURITY_FLOW_CODES: Final = {"by", "sl"}
_INCOME_CODES: Final = {"dv", "in", "dp"}
_PORTFOLIO_EXTERNAL_FLOW_CODES: Final = {"wd"}
_TRANSACTION_HOLDING_EFFECT_CODES: Final = {
    "by",
    "sl",
    "wd",
    "dv",
    "in",
    "dp",
}
_CASH_SECURITY_ID: Final = "CASH_USD"
_EXPECTED_SCENARIO_COVERAGE: Final = {
    "axys_full_spec_b": {
        "transaction_scenarios_by_type": {
            ";": 1,
            "by": 1,
            "dp": 1,
            "dv": 1,
            "in": 1,
            "sl": 1,
            "wd": 1,
        },
        "transaction_derived_holdings_by_type": {
            "by": 2,
            "dp": 1,
            "dv": 1,
            "in": 1,
            "sl": 2,
            "wd": 1,
        },
        "holding_scenarios_by_type": {
            "accrual_correction": 1,
            "cash_balance_correction": 1,
            "cost_only_correction": 1,
            "quantity_valuation_correction": 1,
            "valuation_mark": 3,
        },
    }
}


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


@dataclass(frozen=True)
class HoldingScenarioAdjustment:
    """One intentional holding adjustment used to derive a demo snapshot.

    Attributes:
        snapshot: Snapshot directory receiving the adjustment.
        portfolio: Portfolio code.
        security: Security identifier.
        holding_date: Holding date to adjust.
        scenario_type: Scenario category used for validation and audit.
        deltas: Numeric changes keyed by packaged holding column name.
        scenario: Human-readable scenario description.
    """

    snapshot: str
    portfolio: str
    security: str
    holding_date: str
    scenario_type: str
    deltas: dict[str, float]
    scenario: str


@dataclass(frozen=True)
class HoldingScenarioSet:
    """Validated holding adjustments for deriving demo snapshot holdings.

    Attributes:
        adjustments: Scenario adjustment rows in deterministic file order.
        source_path: CSV file used to load the adjustments.
    """

    adjustments: tuple[HoldingScenarioAdjustment, ...]
    source_path: Path

    def for_snapshot(self, snapshot: str) -> tuple[HoldingScenarioAdjustment, ...]:
        """Return scenario adjustments for one snapshot in file order."""
        return tuple(
            adjustment
            for adjustment in self.adjustments
            if adjustment.snapshot == snapshot
        )


@dataclass(frozen=True)
class TransactionScenarioAdjustment:
    """One intentional transaction adjustment used to derive a demo snapshot.

    Attributes:
        snapshot: Snapshot directory receiving the adjustment.
        transaction_id: Transaction row identifier.
        deltas: Numeric changes keyed by packaged transaction column name.
        scenario: Human-readable scenario description.
    """

    snapshot: str
    transaction_id: str
    deltas: dict[str, float]
    scenario: str


@dataclass(frozen=True)
class TransactionScenarioSet:
    """Validated transaction adjustments for deriving demo snapshot transactions.

    Attributes:
        adjustments: Scenario adjustment rows in deterministic file order.
        source_path: CSV file used to load the adjustments.
    """

    adjustments: tuple[TransactionScenarioAdjustment, ...]
    source_path: Path

    def for_snapshot(self, snapshot: str) -> tuple[TransactionScenarioAdjustment, ...]:
        """Return transaction adjustments for one snapshot in file order."""
        return tuple(
            adjustment
            for adjustment in self.adjustments
            if adjustment.snapshot == snapshot
        )


def main() -> int:
    """Audit or rewrite packaged performance-comparison demo performance files."""
    args = _parse_args()
    summary = rebuild_demo_performance_files(
        args.axys_directory,
        comparison_path=args.comparison_path,
        holding_scenarios_path=args.holding_scenarios_path,
        transaction_scenarios_path=args.transaction_scenarios_path,
        write=args.write,
    )
    audit_issues = audit_demo_data(
        axys_directory=args.axys_directory,
        comparison_path=args.comparison_path,
        holding_scenarios_path=args.holding_scenarios_path,
        transaction_scenarios_path=args.transaction_scenarios_path,
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
    holding_scenarios_path: Path = _DEFAULT_HOLDING_SCENARIOS_PATH,
    transaction_scenarios_path: Path = _DEFAULT_TRANSACTION_SCENARIOS_PATH,
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
    rebuild_summary = rebuild_demo_performance_files(
        axys_directory,
        comparison_path=comparison_path,
        holding_scenarios_path=holding_scenarios_path,
        transaction_scenarios_path=transaction_scenarios_path,
        write=False,
    )
    for snapshot in rebuild_summary["snapshots"]:
        if snapshot["has_transaction_drift"]:
            issues.append(
                AuditIssue(
                    check="derived_transaction_drift",
                    snapshot=str(snapshot["snapshot"]),
                    detail=(
                        "Derived transactions.csv no longer matches the "
                        "transaction scenario file. Update the scenario file "
                        "or run this script with --write after reviewing the change."
                    ),
                )
            )
        if snapshot["has_performance_drift"]:
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
        if snapshot["has_holdings_drift"]:
            issues.append(
                AuditIssue(
                    check="derived_holdings_drift",
                    snapshot=str(snapshot["snapshot"]),
                    detail=(
                        "Derived holdings.csv no longer matches the scenario "
                        "adjustment file. Update the scenario file or run this "
                        "script with --write after reviewing the change."
                    ),
                )
            )
    issues.extend(_audit_visible_portfolio_residuals(comparison_path))
    issues.extend(_audit_visible_security_residuals(comparison_path))
    issues.extend(_audit_scenario_coverage(rebuild_summary))
    return issues


def rebuild_demo_performance_files(
    axys_directory: Path,
    *,
    comparison_path: Path = _DEFAULT_COMPARISON_PATH,
    holding_scenarios_path: Path = _DEFAULT_HOLDING_SCENARIOS_PATH,
    transaction_scenarios_path: Path = _DEFAULT_TRANSACTION_SCENARIOS_PATH,
    write: bool = False,
) -> dict[str, object]:
    """Return audit summary, optionally rewriting derived performance files.

    Args:
        axys_directory: Directory containing ``axys_full_spec_a`` and
            ``axys_full_spec_b``.
        comparison_path: Shared comparison YAML with reconstruction rules.
        holding_scenarios_path: CSV containing intentional holding adjustments.
        transaction_scenarios_path: CSV containing intentional transaction
            adjustments.
        write: Whether to write rebuilt ``transactions.csv``, ``holdings.csv``,
            ``secperf.csv``, and ``portperf.csv``.

    Returns:
        JSON-serializable audit summary with one entry per snapshot.
    """
    specification = PerformanceComparisonSpecification(comparison_path)
    portfolio_reconstruction = specification.portfolio_return_reconstruction
    security_reconstruction = specification.security_return_reconstruction
    if portfolio_reconstruction is None or security_reconstruction is None:
        raise ValueError("Demo rebuild requires portfolio and security reconstruction YAML.")

    snapshots: list[dict[str, object]] = []
    base_snapshot_directory = axys_directory / _BASE_SNAPSHOT_DIRECTORY
    base_holdings = pd.read_csv(base_snapshot_directory / "holdings.csv")
    base_transactions = pd.read_csv(base_snapshot_directory / "transactions.csv")
    holding_scenarios = _load_holding_scenarios(holding_scenarios_path)
    transaction_scenarios = _load_transaction_scenarios(transaction_scenarios_path)
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        snapshot_directory = axys_directory / snapshot_name
        current_secperf = pd.read_csv(snapshot_directory / "secperf.csv")
        current_portperf = pd.read_csv(snapshot_directory / "portperf.csv")
        holdings = pd.read_csv(snapshot_directory / "holdings.csv")
        current_transactions = pd.read_csv(snapshot_directory / "transactions.csv")
        rebuilt_transactions = _rebuild_transactions(
            snapshot_name,
            current_transactions=current_transactions,
            base_transactions=base_transactions,
            transaction_scenarios=transaction_scenarios,
        )
        transaction_adjustments = _transaction_derived_holding_adjustments(
            snapshot_name,
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=current_portperf,
        )
        rebuilt_holdings = _rebuild_holdings(
            snapshot_name,
            current_holdings=holdings,
            base_holdings=base_holdings,
            transaction_adjustments=transaction_adjustments,
            holding_scenarios=holding_scenarios,
        )

        rebuilt_secperf = _rebuild_security_performance(
            current_secperf,
            rebuilt_holdings,
            rebuilt_transactions,
            security_reconstruction,
        )
        rebuilt_portperf = _rebuild_portfolio_performance(
            current_portperf,
            rebuilt_holdings,
            rebuilt_transactions,
            portfolio_reconstruction,
        )
        transaction_delta = _max_numeric_delta(
            current_transactions,
            rebuilt_transactions,
            _TRANSACTION_NUMERIC_COLUMNS,
        )
        has_transaction_field_drift = _has_non_numeric_delta(
            current_transactions,
            rebuilt_transactions,
            _TRANSACTION_NUMERIC_COLUMNS,
        )
        holdings_delta = _max_numeric_delta(
            holdings,
            rebuilt_holdings,
            _HOLDINGS_NUMERIC_COLUMNS,
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
        has_performance_drift = (
            secperf_delta > _CHECK_TOLERANCE or portperf_delta > _CHECK_TOLERANCE
        )
        has_transaction_drift = (
            transaction_delta > _CHECK_TOLERANCE or has_transaction_field_drift
        )
        has_holdings_drift = holdings_delta > _CHECK_TOLERANCE
        if write:
            rebuilt_transactions.to_csv(snapshot_directory / "transactions.csv", index=False)
            rebuilt_holdings.to_csv(snapshot_directory / "holdings.csv", index=False)
            rebuilt_secperf.to_csv(snapshot_directory / "secperf.csv", index=False)
            rebuilt_portperf.to_csv(snapshot_directory / "portperf.csv", index=False)

        snapshots.append(
            {
                "snapshot": snapshot_name,
                "transaction_scenario_rows": len(
                    transaction_scenarios.for_snapshot(snapshot_name)
                ),
                "transaction_scenarios_by_type": _transaction_scenario_type_counts(
                    snapshot_name,
                    base_transactions=base_transactions,
                    transaction_scenarios=transaction_scenarios,
                ),
                "transaction_derived_holding_rows": len(transaction_adjustments),
                "transaction_derived_holdings_by_type": (
                    _transaction_derived_holding_type_counts(transaction_adjustments)
                ),
                "holding_scenario_rows": len(
                    holding_scenarios.for_snapshot(snapshot_name)
                ),
                "holding_scenarios_by_type": _holding_scenario_type_counts(
                    holding_scenarios.for_snapshot(snapshot_name)
                ),
                "transaction_rows": int(rebuilt_transactions.shape[0]),
                "holdings_rows": int(rebuilt_holdings.shape[0]),
                "secperf_rows": int(rebuilt_secperf.shape[0]),
                "portperf_rows": int(rebuilt_portperf.shape[0]),
                "max_transaction_numeric_delta": transaction_delta,
                "max_holdings_numeric_delta": holdings_delta,
                "max_secperf_numeric_delta": secperf_delta,
                "max_portperf_numeric_delta": portperf_delta,
                "has_transaction_field_drift": has_transaction_field_drift,
                "has_transaction_drift": has_transaction_drift,
                "has_holdings_drift": has_holdings_drift,
                "has_performance_drift": has_performance_drift,
                "has_drift": (
                    has_transaction_drift or has_holdings_drift or has_performance_drift
                ),
                "written": write,
            }
        )

    return {
        "axys_directory": str(axys_directory),
        "mode": "write" if write else "check",
        "snapshots": snapshots,
    }


def _rebuild_holdings(
    snapshot_name: str,
    *,
    current_holdings: pd.DataFrame,
    base_holdings: pd.DataFrame,
    transaction_adjustments: tuple[HoldingScenarioAdjustment, ...],
    holding_scenarios: HoldingScenarioSet,
) -> pd.DataFrame:
    """Return holdings derived from the base snapshot and scenario adjustments.

    Args:
        snapshot_name: Snapshot directory name being rebuilt.
        current_holdings: Checked-in holdings for the snapshot. Used for column
            order and as the source for the base snapshot.
        base_holdings: Snapshot A holdings used as the starting point for
            scenario-derived snapshots.
        transaction_adjustments: Holding adjustments derived from changed
            transactions.
        holding_scenarios: Validated explicit holding adjustment rows.

    Returns:
        Holdings with the same columns as ``current_holdings``.

    Raises:
        ValueError: If a scenario adjustment references a missing holding row.
    """
    if snapshot_name == _BASE_SNAPSHOT_DIRECTORY:
        return current_holdings.copy()

    rebuilt = base_holdings.copy(deep=True)
    adjustments = [
        *transaction_adjustments,
        *holding_scenarios.for_snapshot(snapshot_name),
    ]
    for scenario in adjustments:
        mask = (
            rebuilt["PORT"].eq(scenario.portfolio)
            & rebuilt["SEC"].eq(scenario.security)
            & rebuilt["HOLDING_DATE"].eq(scenario.holding_date)
        )
        if int(mask.sum()) != 1:
            raise ValueError(
                "Holding scenario must match exactly one row: "
                f"{scenario.portfolio}/{scenario.security}/{scenario.holding_date}."
            )
        for column, delta in scenario.deltas.items():
            if delta:
                rebuilt.loc[mask, column] = rebuilt.loc[mask, column].astype(float) + delta
    return _rounded_holdings(rebuilt[current_holdings.columns])


def _transaction_scenario_type_counts(
    snapshot_name: str,
    *,
    base_transactions: pd.DataFrame,
    transaction_scenarios: TransactionScenarioSet,
) -> dict[str, int]:
    """Return transaction scenario counts grouped by transaction code."""
    transaction_codes = dict(
        zip(
            base_transactions["TRANSACTION_ID"].astype(str),
            base_transactions["TRAN"].astype(str),
            strict=True,
        )
    )
    counts: dict[str, int] = {}
    for scenario in transaction_scenarios.for_snapshot(snapshot_name):
        transaction_code = transaction_codes.get(scenario.transaction_id)
        if transaction_code is None:
            raise ValueError(
                "Transaction scenario must match one base transaction before "
                f"it can be summarized: {scenario.transaction_id}."
            )
        counts[transaction_code] = counts.get(transaction_code, 0) + 1
    return dict(sorted(counts.items()))


def _transaction_derived_holding_type_counts(
    transaction_adjustments: tuple[HoldingScenarioAdjustment, ...],
) -> dict[str, int]:
    """Return transaction-derived holding counts grouped by transaction code."""
    counts: dict[str, int] = {}
    for adjustment in transaction_adjustments:
        transaction_code = adjustment.scenario.split(" ", maxsplit=2)[1]
        counts[transaction_code] = counts.get(transaction_code, 0) + 1
    return dict(sorted(counts.items()))


def _holding_scenario_type_counts(
    holding_scenarios: tuple[HoldingScenarioAdjustment, ...],
) -> dict[str, int]:
    """Return residual holding scenario counts grouped by scenario type."""
    counts: dict[str, int] = {}
    for scenario in holding_scenarios:
        counts[scenario.scenario_type] = counts.get(scenario.scenario_type, 0) + 1
    return dict(sorted(counts.items()))


def _rebuild_transactions(
    snapshot_name: str,
    *,
    current_transactions: pd.DataFrame,
    base_transactions: pd.DataFrame,
    transaction_scenarios: TransactionScenarioSet,
) -> pd.DataFrame:
    """Return transactions derived from the base snapshot and transaction scenarios.

    Args:
        snapshot_name: Snapshot directory name being rebuilt.
        current_transactions: Checked-in transactions for the snapshot. Used for
            column order and as the source for the base snapshot.
        base_transactions: Snapshot A transactions used as the starting point.
        transaction_scenarios: Validated explicit transaction adjustment rows.

    Returns:
        Transactions with the same columns as ``current_transactions``.

    Raises:
        ValueError: If a scenario adjustment references a missing transaction
            row.
    """
    if snapshot_name == _BASE_SNAPSHOT_DIRECTORY:
        return current_transactions.copy()

    rebuilt = base_transactions.copy(deep=True)
    for scenario in transaction_scenarios.for_snapshot(snapshot_name):
        mask = rebuilt["TRANSACTION_ID"].eq(scenario.transaction_id)
        if int(mask.sum()) != 1:
            raise ValueError(
                "Transaction scenario must match exactly one row: "
                f"{scenario.transaction_id}."
            )
        for column, delta in scenario.deltas.items():
            if delta:
                rebuilt.loc[mask, column] = rebuilt.loc[mask, column].astype(float) + delta
    return _rounded_transactions(rebuilt[current_transactions.columns])


def _rounded_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions rounded to the packaged Axys fixture precision."""
    rounded = transactions.copy()
    rounded["QTY"] = rounded["QTY"].astype(float).round(4)
    rounded["PRICE"] = rounded["PRICE"].astype(float).round(4)
    rounded["AMOUNT"] = rounded["AMOUNT"].astype(float).round(2)
    rounded["COMMISSION"] = rounded["COMMISSION"].astype(float).round(2)
    return rounded


def _transaction_derived_holding_adjustments(
    snapshot_name: str,
    *,
    base_holdings: pd.DataFrame,
    base_transactions: pd.DataFrame,
    current_transactions: pd.DataFrame,
    periods: pd.DataFrame,
) -> tuple[HoldingScenarioAdjustment, ...]:
    """Return holding adjustments implied by changed transaction rows.

    Notes:
        These rules intentionally cover only simple demo scenarios. They are not
        a full accounting engine. Buy/sell rows update the traded security and
        the cash balance. Cash-like income, fee, deposit, and withdrawal rows
        update only the cash balance.
    """
    if snapshot_name == _BASE_SNAPSHOT_DIRECTORY:
        return ()

    base_prepared = _prepared_transactions(base_transactions)
    current_prepared = _prepared_transactions(current_transactions)
    transaction_diffs = _changed_transaction_rows(base_prepared, current_prepared)
    holdings = _holding_values(base_holdings)
    adjustments: list[HoldingScenarioAdjustment] = []
    for row in transaction_diffs.itertuples(index=False):
        transaction_code = str(row.TRAN)
        if transaction_code not in _TRANSACTION_HOLDING_EFFECT_CODES:
            continue
        holding_date = _period_end_for_transaction(periods, row.PORT, row.TRANSACTION_DATE)
        if transaction_code == "by":
            adjustments.append(
                _security_trade_adjustment(
                    snapshot_name,
                    holdings=holdings,
                    transaction_code=transaction_code,
                    portfolio=str(row.PORT),
                    security=str(row.SEC),
                    holding_date=holding_date,
                    quantity_delta=float(row.QTY_delta),
                    scenario=f"{row.TRANSACTION_ID} by transaction changes ending holding.",
                )
            )
            adjustments.append(
                _cash_adjustment(
                    snapshot_name,
                    portfolio=str(row.PORT),
                    holding_date=holding_date,
                    cash_delta=float(row.AMOUNT_delta),
                    scenario=f"{row.TRANSACTION_ID} by transaction changes cash balance.",
                )
            )
        elif transaction_code == "sl":
            adjustments.append(
                _security_trade_adjustment(
                    snapshot_name,
                    holdings=holdings,
                    transaction_code=transaction_code,
                    portfolio=str(row.PORT),
                    security=str(row.SEC),
                    holding_date=holding_date,
                    quantity_delta=-float(row.QTY_delta),
                    scenario=f"{row.TRANSACTION_ID} sl transaction changes ending holding.",
                )
            )
            adjustments.append(
                _cash_adjustment(
                    snapshot_name,
                    portfolio=str(row.PORT),
                    holding_date=holding_date,
                    cash_delta=float(row.AMOUNT_delta),
                    scenario=f"{row.TRANSACTION_ID} sl transaction changes cash balance.",
                )
            )
        else:
            adjustments.append(
                _cash_adjustment(
                    snapshot_name,
                    portfolio=str(row.PORT),
                    holding_date=holding_date,
                    cash_delta=float(row.AMOUNT_delta),
                    scenario=(
                        f"{row.TRANSACTION_ID} {transaction_code} transaction "
                        "changes cash balance."
                    ),
                )
            )
    return tuple(adjustments)


def _changed_transaction_rows(
    base_transactions: pd.DataFrame,
    current_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Return transaction rows whose simple accounting fields changed."""
    compare_columns = [
        "TRANSACTION_ID",
        "PORT",
        "TRANSACTION_DATE",
        "SEC",
        "TRAN",
        "QTY",
        "PRICE",
        "AMOUNT",
        "COMMISSION",
    ]
    base = base_transactions[compare_columns]
    current = current_transactions[compare_columns]
    merged = base.merge(
        current,
        on="TRANSACTION_ID",
        how="outer",
        suffixes=("_base", "_current"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        unmatched = merged.loc[
            ~merged["_merge"].eq("both"),
            ["TRANSACTION_ID", "_merge"],
        ].to_dict("records")
        raise ValueError(f"Transaction scenarios must not add or remove rows: {unmatched}.")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        base_port = str(row.PORT_base)
        current_port = str(row.PORT_current)
        base_security = str(row.SEC_base)
        current_security = str(row.SEC_current)
        base_code = str(row.TRAN_base)
        current_code = str(row.TRAN_current)
        if (
            base_port != current_port
            or base_security != current_security
            or base_code != current_code
            or pd.Timestamp(row.TRANSACTION_DATE_base)
            != pd.Timestamp(row.TRANSACTION_DATE_current)
        ):
            raise ValueError(
                "Transaction scenario rows may change numeric fields only: "
                f"{row.TRANSACTION_ID}."
            )
        deltas = {
            column: float(getattr(row, f"{column}_current"))
            - float(getattr(row, f"{column}_base"))
            for column in ("QTY", "PRICE", "AMOUNT", "COMMISSION")
        }
        if not any(deltas.values()):
            continue
        rows.append(
            {
                "TRANSACTION_ID": str(row.TRANSACTION_ID),
                "PORT": base_port,
                "TRANSACTION_DATE": pd.Timestamp(row.TRANSACTION_DATE_base),
                "SEC": base_security,
                "TRAN": base_code,
                **{f"{column}_delta": delta for column, delta in deltas.items()},
            }
        )
    return pd.DataFrame(rows)


def _period_end_for_transaction(
    periods: pd.DataFrame,
    portfolio_code: str,
    transaction_date: pd.Timestamp,
) -> str:
    """Return inclusive period end date for one transaction row."""
    period_rows = periods[
        periods["PORTFOLIO_CODE"].eq(portfolio_code)
        & pd.to_datetime(periods["FROM_DATE"]).le(transaction_date)
        & pd.to_datetime(periods["THRU_DATE"]).ge(transaction_date)
    ]
    if period_rows.shape[0] != 1:
        raise ValueError(
            "Transaction must map to exactly one inclusive performance period: "
            f"{portfolio_code}/{transaction_date.date()}."
        )
    return pd.Timestamp(period_rows.iloc[0]["THRU_DATE"]).date().isoformat()


def _security_trade_adjustment(
    snapshot: str,
    *,
    holdings: pd.DataFrame,
    transaction_code: str,
    portfolio: str,
    security: str,
    holding_date: str,
    quantity_delta: float,
    scenario: str,
) -> HoldingScenarioAdjustment:
    """Return the holding impact for a changed buy/sell quantity."""
    rows = holdings[
        holdings["PORT"].eq(portfolio)
        & holdings["SEC"].eq(security)
        & holdings["HOLDING_DATE"].eq(pd.Timestamp(holding_date))
    ]
    if rows.shape[0] != 1:
        raise ValueError(
            "Transaction-derived security adjustment must match one holding: "
            f"{portfolio}/{security}/{holding_date}."
        )
    price = float(rows.iloc[0]["PRICE"])
    quantity = float(rows.iloc[0]["QTY"])
    cost = float(rows.iloc[0]["COST"])
    cost_per_share = cost / quantity if quantity else 0.0
    market_value_delta = quantity_delta * price
    cost_delta = market_value_delta
    if transaction_code == "sl":
        cost_delta = quantity_delta * cost_per_share
    return HoldingScenarioAdjustment(
        snapshot=snapshot,
        portfolio=portfolio,
        security=security,
        holding_date=holding_date,
        scenario_type="transaction_derived",
        deltas={
            "QTY": quantity_delta,
            "PRICE": 0.0,
            "MKT_VAL": market_value_delta,
            "COST": cost_delta,
            "ACCRUED": 0.0,
        },
        scenario=scenario,
    )


def _cash_adjustment(
    snapshot: str,
    *,
    portfolio: str,
    holding_date: str,
    cash_delta: float,
    scenario: str,
) -> HoldingScenarioAdjustment:
    """Return the cash holding impact for a changed cash-affecting transaction."""
    return HoldingScenarioAdjustment(
        snapshot=snapshot,
        portfolio=portfolio,
        security=_CASH_SECURITY_ID,
        holding_date=holding_date,
        scenario_type="transaction_derived",
        deltas={
            "QTY": cash_delta,
            "PRICE": 0.0,
            "MKT_VAL": cash_delta,
            "COST": cash_delta,
            "ACCRUED": 0.0,
        },
        scenario=scenario,
    )


def _load_holding_scenarios(path: Path) -> HoldingScenarioSet:
    """Return validated holding scenario adjustments.

    Args:
        path: CSV file containing one explicit holding adjustment per row.

    Returns:
        Validated scenario adjustments in file order.

    Raises:
        ValueError: If the CSV shape, snapshot names, keys, or numeric deltas are
            invalid.
    """
    scenarios = pd.read_csv(path)
    missing_columns = [
        column for column in _HOLDING_SCENARIO_COLUMNS if column not in scenarios.columns
    ]
    extra_columns = [
        column for column in scenarios.columns if column not in _HOLDING_SCENARIO_COLUMNS
    ]
    if missing_columns or extra_columns:
        raise ValueError(
            "Holding scenario CSV columns must exactly match "
            f"{_HOLDING_SCENARIO_COLUMNS}. "
            f"Missing={missing_columns}; extra={extra_columns}."
        )

    key_nulls = scenarios[_HOLDING_SCENARIO_KEY].isna().any(axis=1)
    if bool(key_nulls.any()):
        raise ValueError("Holding scenario rows must not have blank key values.")
    duplicate_keys = scenarios.duplicated(_HOLDING_SCENARIO_KEY, keep=False)
    if bool(duplicate_keys.any()):
        duplicates = scenarios.loc[duplicate_keys, _HOLDING_SCENARIO_KEY].to_dict("records")
        raise ValueError(f"Duplicate holding scenario keys are not allowed: {duplicates}.")

    supported_snapshots = set(_SNAPSHOT_DIRECTORIES) - {_BASE_SNAPSHOT_DIRECTORY}
    unknown_snapshots = sorted(set(scenarios["snapshot"]) - supported_snapshots)
    if unknown_snapshots:
        raise ValueError(
            "Holding scenarios may only target derived snapshots. "
            f"Unsupported snapshots={unknown_snapshots}."
        )
    unknown_types = sorted(set(scenarios["scenario_type"]) - _HOLDING_SCENARIO_TYPES)
    if unknown_types:
        raise ValueError(
            "Holding scenario types are not supported: "
            f"{unknown_types}. Supported={sorted(_HOLDING_SCENARIO_TYPES)}."
        )

    delta_columns = [f"{column}_delta" for column in _HOLDINGS_NUMERIC_COLUMNS]
    converted_deltas = scenarios[delta_columns].apply(pd.to_numeric, errors="coerce")
    if bool(converted_deltas.isna().any().any()):
        raise ValueError("Holding scenario delta columns must be numeric.")

    adjustments: list[HoldingScenarioAdjustment] = []
    for row_index, row in scenarios.iterrows():
        deltas = {
            column: float(converted_deltas.loc[row_index, f"{column}_delta"])
            for column in _HOLDINGS_NUMERIC_COLUMNS
        }
        if not any(deltas.values()):
            raise ValueError(
                "Holding scenario rows must change at least one numeric value: "
                f"{row['PORT']}/{row['SEC']}/{row['HOLDING_DATE']}."
            )
        scenario_type = str(row["scenario_type"])
        _validate_holding_scenario_deltas(
            scenario_type=scenario_type,
            portfolio=str(row["PORT"]),
            security=str(row["SEC"]),
            holding_date=str(row["HOLDING_DATE"]),
            deltas=deltas,
        )
        adjustments.append(
            HoldingScenarioAdjustment(
                snapshot=str(row["snapshot"]),
                portfolio=str(row["PORT"]),
                security=str(row["SEC"]),
                holding_date=str(row["HOLDING_DATE"]),
                scenario_type=scenario_type,
                deltas=deltas,
                scenario=str(row["scenario"]),
            )
        )
    return HoldingScenarioSet(tuple(adjustments), path)


def _validate_holding_scenario_deltas(
    *,
    scenario_type: str,
    portfolio: str,
    security: str,
    holding_date: str,
    deltas: dict[str, float],
) -> None:
    """Validate one explicit holding scenario's changed fields.

    Notes:
        These rules intentionally describe the remaining residual holding
        scenarios. Transaction-derived rows are validated by transaction rules
        before this layer.
    """
    changed_columns = {column for column, delta in deltas.items() if delta}
    allowed_columns_by_type = {
        "valuation_mark": {"PRICE", "MKT_VAL"},
        "cash_balance_correction": {"QTY", "MKT_VAL", "COST"},
        "quantity_valuation_correction": {"QTY", "MKT_VAL", "COST"},
        "accrual_correction": {"QTY", "MKT_VAL", "COST", "ACCRUED"},
        "cost_only_correction": {"COST"},
    }
    allowed_columns = allowed_columns_by_type[scenario_type]
    if not changed_columns.issubset(allowed_columns):
        raise ValueError(
            "Holding scenario changed fields do not match scenario_type: "
            f"{portfolio}/{security}/{holding_date}; type={scenario_type}; "
            f"changed={sorted(changed_columns)}; allowed={sorted(allowed_columns)}."
        )


def _load_transaction_scenarios(path: Path) -> TransactionScenarioSet:
    """Return validated transaction scenario adjustments.

    Args:
        path: CSV file containing one explicit transaction adjustment per row.

    Returns:
        Validated scenario adjustments in file order.

    Raises:
        ValueError: If the CSV shape, snapshot names, keys, or numeric deltas are
            invalid.
    """
    scenarios = pd.read_csv(path)
    missing_columns = [
        column
        for column in _TRANSACTION_SCENARIO_COLUMNS
        if column not in scenarios.columns
    ]
    extra_columns = [
        column
        for column in scenarios.columns
        if column not in _TRANSACTION_SCENARIO_COLUMNS
    ]
    if missing_columns or extra_columns:
        raise ValueError(
            "Transaction scenario CSV columns must exactly match "
            f"{_TRANSACTION_SCENARIO_COLUMNS}. "
            f"Missing={missing_columns}; extra={extra_columns}."
        )

    key_nulls = scenarios[_TRANSACTION_SCENARIO_KEY].isna().any(axis=1)
    if bool(key_nulls.any()):
        raise ValueError("Transaction scenario rows must not have blank key values.")
    duplicate_keys = scenarios.duplicated(_TRANSACTION_SCENARIO_KEY, keep=False)
    if bool(duplicate_keys.any()):
        duplicates = scenarios.loc[
            duplicate_keys, _TRANSACTION_SCENARIO_KEY
        ].to_dict("records")
        raise ValueError(f"Duplicate transaction scenario keys are not allowed: {duplicates}.")

    supported_snapshots = set(_SNAPSHOT_DIRECTORIES) - {_BASE_SNAPSHOT_DIRECTORY}
    unknown_snapshots = sorted(set(scenarios["snapshot"]) - supported_snapshots)
    if unknown_snapshots:
        raise ValueError(
            "Transaction scenarios may only target derived snapshots. "
            f"Unsupported snapshots={unknown_snapshots}."
        )

    delta_columns = [f"{column}_delta" for column in _TRANSACTION_NUMERIC_COLUMNS]
    converted_deltas = scenarios[delta_columns].apply(pd.to_numeric, errors="coerce")
    if bool(converted_deltas.isna().any().any()):
        raise ValueError("Transaction scenario delta columns must be numeric.")

    adjustments: list[TransactionScenarioAdjustment] = []
    for row_index, row in scenarios.iterrows():
        deltas = {
            column: float(converted_deltas.loc[row_index, f"{column}_delta"])
            for column in _TRANSACTION_NUMERIC_COLUMNS
        }
        if not any(deltas.values()):
            raise ValueError(
                "Transaction scenario rows must change at least one numeric value: "
                f"{row['TRANSACTION_ID']}."
            )
        adjustments.append(
            TransactionScenarioAdjustment(
                snapshot=str(row["snapshot"]),
                transaction_id=str(row["TRANSACTION_ID"]),
                deltas=deltas,
                scenario=str(row["scenario"]),
            )
        )
    return TransactionScenarioSet(tuple(adjustments), path)


def _rounded_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings rounded to the packaged Axys fixture precision."""
    rounded = holdings.copy()
    rounded["QTY"] = rounded["QTY"].astype(float).round(4)
    rounded["PRICE"] = rounded["PRICE"].astype(float).round(4)
    for column in ("MKT_VAL", "COST", "ACCRUED"):
        rounded[column] = rounded[column].astype(float).round(2)
    return rounded


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
    prepared["TRAN"] = prepared["TRAN"].astype(str)
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


def _audit_scenario_coverage(rebuild_summary: dict[str, object]) -> list[AuditIssue]:
    """Return issues when packaged demo scenario coverage changes unexpectedly."""
    issues: list[AuditIssue] = []
    snapshots = rebuild_summary.get("snapshots", [])
    if not isinstance(snapshots, list):
        return [
            AuditIssue(
                check="scenario_coverage",
                detail="Rebuild summary did not include snapshot scenario coverage.",
            )
        ]

    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            issues.append(
                AuditIssue(
                    check="scenario_coverage",
                    detail="Rebuild summary snapshot coverage was not a dictionary.",
                )
            )
            continue
        snapshot_name = str(snapshot.get("snapshot"))
        expected = _EXPECTED_SCENARIO_COVERAGE.get(snapshot_name)
        if expected is None:
            continue
        for field_name, expected_counts in expected.items():
            actual_counts = snapshot.get(field_name)
            if actual_counts != expected_counts:
                issues.append(
                    AuditIssue(
                        check="scenario_coverage",
                        snapshot=snapshot_name,
                        detail=(
                            f"Unexpected {field_name}. "
                            f"Expected={expected_counts}; actual={actual_counts}."
                        ),
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


def _has_non_numeric_delta(
    current: pd.DataFrame,
    rebuilt: pd.DataFrame,
    numeric_columns: list[str],
) -> bool:
    """Return whether nonnumeric columns differ between aligned frames."""
    if current.shape != rebuilt.shape:
        return True

    numeric_column_set = set(numeric_columns)
    for column in current.columns:
        if column in numeric_column_set:
            continue
        current_values = current[column].fillna("").astype(str).reset_index(drop=True)
        rebuilt_values = rebuilt[column].fillna("").astype(str).reset_index(drop=True)
        if not current_values.equals(rebuilt_values):
            return True
    return False


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
        "--holding-scenarios-path",
        type=Path,
        default=_DEFAULT_HOLDING_SCENARIOS_PATH,
        help=(
            "CSV file containing explicit scenario adjustments used to derive "
            "snapshot B holdings from snapshot A holdings."
        ),
    )
    parser.add_argument(
        "--transaction-scenarios-path",
        type=Path,
        default=_DEFAULT_TRANSACTION_SCENARIOS_PATH,
        help=(
            "CSV file containing explicit scenario adjustments used to derive "
            "snapshot B transactions from snapshot A transactions."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Rewrite transactions.csv, holdings.csv, secperf.csv, and portperf.csv "
            "instead of audit-only mode."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
