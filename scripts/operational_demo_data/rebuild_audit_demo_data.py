"""Audit or rebuild derived Audit demo CSV files.

The packaged Audit demo keeps user-visible operational inputs
in ``holdings.csv`` and ``transactions.csv``. The ``secperf.csv`` and
``portperf.csv`` files are derived review targets. This script keeps the derived
performance files internally aligned by:

1. deriving ``secperf.csv`` from holdings and security-level transactions;
2. deriving ``portperf.csv`` from holdings and portfolio-level transactions; and
3. deriving snapshot B ``transactions.csv`` from snapshot A transactions plus
   explicit transaction scenarios that adjust, insert, or delete base rows;
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
from functools import cache
import json
from pathlib import Path
from typing import Any, Final, Mapping

# Third-party imports
import pandas as pd
import yaml

# Project imports
from ppar.audit import compare_snapshots
from ppar.audit.performance_comparison.methods import ReturnReconstructionMethod
from ppar.audit.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.audit.specification import (
    AuditSpecification,
    PortfolioReturnReconstruction,
    SecurityReturnReconstruction,
)
from ppar.audit.workbook_tables import (
    _workbook_portfolio_changes_table,
    _workbook_security_changes_table,
    _workbook_underlying_causes_table,
)
from ppar.audit.data_issues.checks import data_issues_table


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_AXYS_APX_DIRECTORY: Final = (
    _REPO_ROOT / "ppar" / "setup_templates" / "axys_apx_audit"
)
_DEFAULT_COMPARISON_PATH: Final = _DEFAULT_AXYS_APX_DIRECTORY / "axys_apx_audit.yaml"
_DEFAULT_HOLDING_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "audit_holding_scenarios.csv"
)
_DEFAULT_TRANSACTION_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "audit_transaction_scenarios.csv"
)
_DEFAULT_SCENARIO_CALENDAR_PATH: Final = (
    Path(__file__).resolve().parent / "audit_scenario_calendar.csv"
)
_DEFAULT_SCENARIO_INVENTORY_PATH: Final = (
    Path(__file__).resolve().parent / "audit_scenario_inventory.csv"
)
_DEFAULT_PERIOD_SPLIT_PLAN_PATH: Final = (
    Path(__file__).resolve().parent / "audit_period_split_plan.csv"
)
_DEFAULT_TRANSACTION_POLICY_PATH: Final = (
    Path(__file__).resolve().parent / "audit_demo_transaction_policy.yaml"
)
_SNAPSHOT_DIRECTORIES: Final = ("snapshot_a", "snapshot_b")
_BASE_SNAPSHOT_DIRECTORY: Final = "snapshot_a"
_COMMON_AXYS_HEADERS_TO_INTERNAL: Final[dict[str, dict[str, str]]] = {
    "holdings": {
        "Portfolio Code": "PORT",
        "Security Symbol": "SEC",
        "Holding Date": "HOLDING_DATE",
        "Currency Code": "CURRENCY",
        "Base Currency": "BASE_CURRENCY",
        "Quantity": "QTY",
        "Price": "PRICE",
        "Market Value": "MKT_VAL",
        "Base Market Value": "BASE_MKT_VAL",
        "Accrued Income": "ACCRUED",
    },
    "portfolio_performance": {
        "Portfolio Code": "PORTFOLIO_CODE",
        "From Date": "FROM_DATE",
        "Thru Date": "THRU_DATE",
        "Portfolio Return": "PORT_RETURN",
        "Base Currency": "BASE_CURRENCY",
    },
    "security_performance": {
        "Portfolio Code": "PORTFOLIO_CODE",
        "Security Symbol": "SECURITY_ID",
        "From Date": "FROM_DATE",
        "Thru Date": "THRU_DATE",
        "Security Return": "SEC_RETURN",
    },
    "transactions": {
        "Portfolio Code": "PORT",
        "Transaction Date": "TRANSACTION_DATE",
        "Settlement Date": "SETTLE_DATE",
        "Security Symbol": "SEC",
        "Transaction Code": "TRAN",
        "Transaction Security Type": "SEC_TYPE",
        "Source/Destination Type": "SRC_DEST_TYPE",
        "Source/Destination Symbol": "SRC_DEST_SYMBOL",
        "Special Security Type": "SPECIAL_SEC_TYPE",
        "Special Security Symbol": "SPECIAL_SEC_SYMBOL",
        "Currency Code": "CURRENCY",
        "Base Currency": "BASE_CURRENCY",
        "Quantity": "QTY",
        "Price": "PRICE",
        "Amount": "AMOUNT",
        "Base Amount": "BASE_AMOUNT",
        "Commission": "COMMISSION",
    },
}
_PERIOD_KEY: Final = ["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"]
_SECURITY_PERIOD_KEY: Final = [*_PERIOD_KEY, "SECURITY_ID"]
_PORTPERF_COLUMNS: Final = [
    "PORTFOLIO_CODE",
    "FROM_DATE",
    "THRU_DATE",
    "PORT_RETURN",
    "BASE_CURRENCY",
]
_SECPERF_COLUMNS: Final = [
    "PORTFOLIO_CODE",
    "SECURITY_ID",
    "FROM_DATE",
    "THRU_DATE",
    "SEC_RETURN",
]
_SECPERF_NUMERIC_COLUMNS: Final = ["SEC_RETURN"]
_PORTPERF_NUMERIC_COLUMNS: Final = ["PORT_RETURN"]
_PACKAGED_HOLDINGS_NUMERIC_COLUMNS: Final = [
    "QTY",
    "PRICE",
    "MKT_VAL",
    "BASE_MKT_VAL",
    "ACCRUED",
]
_INTERNAL_HOLDINGS_NUMERIC_COLUMNS: Final = [
    "QTY",
    "PRICE",
    "MKT_VAL",
    "BASE_MKT_VAL",
    "COST",
    "ACCRUED",
]
_PACKAGED_HOLDINGS_COLUMNS: Final = [
    "PORT",
    "SEC",
    "HOLDING_DATE",
    "ACCRUED",
    "BASE_CURRENCY",
    "BASE_MKT_VAL",
    "CURRENCY",
    "MKT_VAL",
    "PRICE",
    "QTY",
]
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
    "data_issues_holdings_accrued_rate",
}
_TRANSACTION_NUMERIC_COLUMNS: Final = [
    "QTY",
    "PRICE",
    "AMOUNT",
    "BASE_AMOUNT",
    "COMMISSION",
]
_TRANSACTION_ID_COLUMN: Final = "TRANSACTION_ID"
_TRANSACTION_SOURCE_COLUMNS: Final = [
    "PORT",
    "TRANSACTION_DATE",
    "SETTLE_DATE",
    "SEC",
    "TRAN",
    "SEC_TYPE",
    "SRC_DEST_TYPE",
    "SRC_DEST_SYMBOL",
    "SPECIAL_SEC_TYPE",
    "SPECIAL_SEC_SYMBOL",
    "CURRENCY",
    "BASE_CURRENCY",
]
_PACKAGED_TRANSACTION_COLUMNS: Final = [
    "PORT",
    "SEC",
    "TRANSACTION_DATE",
    "AMOUNT",
    "BASE_AMOUNT",
    "BASE_CURRENCY",
    "COMMISSION",
    "CURRENCY",
    "PRICE",
    "QTY",
    "SETTLE_DATE",
    "SRC_DEST_SYMBOL",
    "SRC_DEST_TYPE",
    "SPECIAL_SEC_SYMBOL",
    "SPECIAL_SEC_TYPE",
    "TRAN",
    "SEC_TYPE",
]
_TRANSACTION_SCENARIO_COLUMNS: Final = [
    "snapshot",
    "action",
    _TRANSACTION_ID_COLUMN,
    *_TRANSACTION_SOURCE_COLUMNS,
    *_TRANSACTION_NUMERIC_COLUMNS,
    "QTY_delta",
    "PRICE_delta",
    "AMOUNT_delta",
    "BASE_AMOUNT_delta",
    "COMMISSION_delta",
    "scenario",
]
_TRANSACTION_SCENARIO_KEY: Final = ["snapshot", "action", "TRANSACTION_ID", "scenario"]
_TRANSACTION_SCENARIO_ACTIONS: Final = {"adjust", "delete", "insert"}
_SCENARIO_CALENDAR_COLUMNS: Final = [
    "scenario_key",
    "scenario_source",
    "portfolio",
    "from_date",
    "thru_date",
    "scenario_family",
    "primary_security",
    "current_expected_difference_rows",
    "future_max_expected_differences",
    "notes",
]
_SCENARIO_CALENDAR_KEY: Final = "scenario_key"
_SCENARIO_CALENDAR_SOURCES: Final = {"holding", "multicurrency", "transaction"}
_SCENARIO_CALENDAR_NUMERIC_COLUMNS: Final = [
    "current_expected_difference_rows",
    "future_max_expected_differences",
]
_SCENARIO_INVENTORY_COLUMNS: Final = [
    "scenario_key",
    "protection_reason",
    "economic_meaning",
    "scenario_source",
    "portfolio",
    "source_from_date",
    "source_thru_date",
    "story_from_date",
    "story_thru_date",
    "scenario_family",
    "primary_security",
    "expected_report_disposition",
    "expected_period_status",
    "independent_change_id",
    "source_period_independent_changes",
    "carry_forward_status",
]
_SCENARIO_INVENTORY_NUMERIC_COLUMNS: Final = [
    "source_period_independent_changes",
]
_SCENARIO_REPORT_DISPOSITIONS: Final = {
    "counted_cause",
    "data_issues_issue",
    "fixture_only_context",
    "review_evidence",
}
_SCENARIO_PERIOD_STATUSES: Final = {
    "Fully Explained",
    "No Performance Difference",
    "Partly Explained",
    "Unexplained",
}
_SCENARIO_CARRY_FORWARD_STATUSES: Final = {
    "carry_forward_effect",
    "originating_change",
}
_SCENARIO_DATA_ISSUE_TYPES: Final = {
    "data_issues_dividend_rate": "dividend_rate",
    "data_issues_holdings_accrued_rate": "holdings_accrued_rate",
    "data_issues_pa_sa_rate": "pa_sa_rate",
}
_SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES: Final = 2
_SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS: Final = 2
_BALANCED_APRIL_PERIOD: Final = ("2026-04-01", "2026-04-30")
_BALANCED_APRIL_SUBPERIODS: Final = (
    ("2026-04-01", "2026-04-10"),
    ("2026-04-11", "2026-04-16"),
    ("2026-04-17", "2026-04-24"),
    ("2026-04-25", "2026-04-30"),
)
_CONTRIBUTION_DEMO_PORTFOLIO: Final = "BALANCED_CONTRIBUTION"
_CONTRIBUTION_DEMO_BASELINE_PERIOD: Final = ("2026-02-01", "2026-02-28")
_CONTRIBUTION_DEMO_PERIOD: Final = ("2026-03-01", "2026-03-31")
_CONTRIBUTION_DEMO_HOLDINGS: Final = (
    ("2026-02-28", 100_000.0),
    ("2026-03-31", 101_000.0),
)
_RETIRED_DEMO_PORTFOLIOS: Final = {"BALANCED_AAPL"}
_MULTICURRENCY_SCENARIO_CALENDAR_KEYS: Final = {
    "multicurrency:BALANCED:SAP.DE:2026-04-15:EUR dividend correction": 1,
    "multicurrency:BALANCED:SHEL.L:2026-04-30:GBP FX correction": 1,
}
_PERIOD_SPLIT_PLAN_COLUMNS: Final = [
    "scenario_key",
    "portfolio",
    "current_from_date",
    "current_thru_date",
    "planned_from_date",
    "planned_thru_date",
    "planned_difference_rows",
    "notes",
]
_PERIOD_SPLIT_PLAN_NUMERIC_COLUMNS: Final = ["planned_difference_rows"]
_CHECK_TOLERANCE: Final = 0.000000001
_RETURN_TOLERANCE: Final = 0.000001
_INTENTIONAL_PORTFOLIO_RESIDUALS: Final = {
    ("BALANCED", "2026-05-09", "2026-05-14", "Partly Explained"): (
        "Intentional partial example: beginning-value and ending-value changes "
        "both foot, but the selected report causes leave a denominator-effect "
        "residual for review."
    ),
    ("INCOME", "2026-04-01", "2026-04-30", "Unexplained"): (
        "Intentional vendor/methodology residual used to demonstrate unresolved review."
    ),
    ("INCOME", "2026-05-09", "2026-05-14", "Partly Explained"): (
        "Intentional late-dividend timing example: the corrected AAPL dividend "
        "explains the income change, while cash timing leaves a small residual."
    ),
    ("INCOME", "2026-05-15", "2026-05-15", "Partly Explained"): (
        "Intentional carry-forward example from the corrected AAPL dividend "
        "and same-day 91282Y2Y1 interest/accrual correction."
    ),
}
_INTENTIONAL_PORTFOLIO_RETURN_RESIDUALS: Final = {
    ("BALANCED", "2026-05-09", "2026-05-14"): 0.0002,
    ("INCOME", "2026-04-01", "2026-04-30"): 0.00035,
}
_INTENTIONAL_SECURITY_RESIDUALS: Final = {
    ("BALANCED", "MSFT", "2026-05-09", "2026-05-14", "Partly Explained"): (
        "Intentional partial security example: a holding correction explains "
        "part of the reported security-return change, with the remainder left "
        "as a methodology/source-data residual."
    ),
    ("BALANCED", "JPM", "2026-05-09", "2026-05-14", "Unexplained"): (
        "Intentional possible-cause example: a code-only rc transaction remains "
        "neutral review evidence and is not counted as explained performance."
    ),
    ("INCOME", "91282Y5Y1", "2026-04-01", "2026-04-30", "Unexplained"): (
        "Intentional unexplained security example: reported security return "
        "changed while the visible source-data change is cost-only context."
    ),
    ("INCOME", "CASHUSD", "2026-05-09", "2026-05-14", "Unexplained"): (
        "Intentional cash-side residual from the corrected AAPL dividend timing."
    ),
    ("INCOME", "91282Y2Y1", "2026-05-09", "2026-05-14", "Partly Explained"): (
        "Intentional carry-forward residual after splitting the May income period."
    ),
    ("INCOME", "AAPL", "2026-05-15", "2026-05-15", "Unexplained"): (
        "Intentional one-day security residual from the dividend timing split."
    ),
    ("INCOME", "91282Y2Y1", "2026-05-15", "2026-05-15", "Partly Explained"): (
        "Intentional 91282Y2Y1 residual from the one-day interest/accrual example."
    ),
}
_INTENTIONAL_SECURITY_RETURN_RESIDUALS: Final = {
    ("BALANCED", "JPM", "2026-05-09", "2026-05-14"): 0.0056216158,
    ("BALANCED", "MSFT", "2026-05-09", "2026-05-14"): 0.002,
    ("INCOME", "91282Y5Y1", "2026-04-01", "2026-04-30"): 0.004,
}
_CASH_SECURITY_IDS: Final = {
    "USD": "CASHUSD",
    "EUR": "CASHEUR",
    "GBP": "CASHGBP",
}
_DEMO_CONTEXT_COLUMNS: Final[Mapping[str, str]] = {
    "security_id": "SEC",
    "transaction_security_type": "SEC_TYPE",
    "source_destination_type": "SRC_DEST_TYPE",
    "source_destination_symbol": "SRC_DEST_SYMBOL",
    "special_security_type": "SPECIAL_SEC_TYPE",
    "special_security_symbol": "SPECIAL_SEC_SYMBOL",
}
_DEMO_HOLDING_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "cash",
        "principal_and_cash",
        "security_quantity",
        "security_trade_and_cash",
    }
)
_DEMO_COST_BASIS_METHODS: Final[frozenset[str]] = frozenset(
    {"market_value", "proportional_existing"}
)
_CASH_SECURITY_ID: Final = _CASH_SECURITY_IDS["USD"]
_EXPECTED_SCENARIO_COVERAGE: Final = {
    "snapshot_b": {
        "transaction_scenarios_by_type": {
            "ai": 1,
            "by": 3,
            "dp": 2,
            "dv": 4,
            "in": 1,
            "li": 1,
            "lo": 1,
            "pa": 2,
            "pd": 1,
            "rc": 2,
            "sa": 1,
            "sl": 2,
            "ss": 1,
            "cs": 1,
            "ti": 1,
            "wd": 1,
        },
        "transaction_derived_holdings_by_type": {
            "ai": 1,
            "by": 14,
            "dp": 5,
            "dv": 13,
            "in": 3,
            "li": 1,
            "lo": 1,
            "pa": 3,
            "pd": 4,
            "rc": 2,
            "sa": 1,
            "sl": 4,
            "ss": 1,
            "cs": 1,
            "ti": 1,
            "wd": 1,
        },
        "holding_scenarios_by_type": {
            "accrual_correction": 1,
            "cash_balance_correction": 1,
            "cost_only_correction": 1,
            "quantity_valuation_correction": 2,
            "valuation_mark": 3,
            "data_issues_holdings_accrued_rate": 1,
        },
    }
}


@dataclass(frozen=True)
class AuditIssue:
    """One packaged demo-data audit issue.

    Attributes:
        check: Name of the consistency check that failed.
        detail: Human-readable explanation of the issue.
        snapshot: Snapshot label such as ``snapshot_a`` when applicable.
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
            adjustment for adjustment in self.adjustments if adjustment.snapshot == snapshot
        )


@dataclass(frozen=True)
class TransactionScenarioAdjustment:
    """One intentional transaction adjustment used to derive a demo snapshot.

    Attributes:
        snapshot: Snapshot directory receiving the adjustment.
        action: Whether the scenario adjusts a base row or inserts a new row.
        transaction_id: Transaction row identifier.
        values: Full transaction values for inserted rows.
        deltas: Numeric changes keyed by packaged transaction column name.
        scenario: Human-readable scenario description.
    """

    snapshot: str
    action: str
    transaction_id: str
    values: dict[str, object]
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
            adjustment for adjustment in self.adjustments if adjustment.snapshot == snapshot
        )


def main() -> int:
    """Audit or rewrite packaged Audit demo performance files."""
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
        scenario_calendar_path=args.scenario_calendar_path,
        scenario_inventory_path=args.scenario_inventory_path,
        period_split_plan_path=args.period_split_plan_path,
    )
    scenario_calendar = _load_scenario_calendar(args.scenario_calendar_path)
    scenario_inventory = _load_scenario_inventory(args.scenario_inventory_path)
    period_split_plan = _load_period_split_plan(args.period_split_plan_path)
    summary["scenario_calendar_density"] = _scenario_calendar_density(
        scenario_calendar,
    )
    summary["scenario_readability_matrix"] = _scenario_readability_matrix(
        scenario_calendar,
    )
    summary["scenario_isolation_matrix"] = _scenario_isolation_matrix(
        scenario_inventory,
    )
    summary["scenario_period_split_plan"] = _scenario_period_split_plan_summary(
        period_split_plan,
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
    axys_directory: Path = _DEFAULT_AXYS_APX_DIRECTORY,
    comparison_path: Path = _DEFAULT_COMPARISON_PATH,
    holding_scenarios_path: Path = _DEFAULT_HOLDING_SCENARIOS_PATH,
    transaction_scenarios_path: Path = _DEFAULT_TRANSACTION_SCENARIOS_PATH,
    scenario_calendar_path: Path = _DEFAULT_SCENARIO_CALENDAR_PATH,
    scenario_inventory_path: Path = _DEFAULT_SCENARIO_INVENTORY_PATH,
    period_split_plan_path: Path = _DEFAULT_PERIOD_SPLIT_PLAN_PATH,
) -> list[AuditIssue]:
    """Return packaged demo-data audit issues.

    Args:
        axys_directory: Directory containing the packaged Axys/APX snapshots.
        comparison_path: Portfolio comparison YAML used for visible residual
            guardrails.
        holding_scenarios_path: CSV containing intentional holding
            adjustments.
        transaction_scenarios_path: CSV containing intentional transaction
            adjustments.
        scenario_calendar_path: CSV mapping intentional scenario rows to the
            demo periods they are meant to explain.
        scenario_inventory_path: CSV protecting every intentional named
            scenario from silent removal or replacement.
        period_split_plan_path: CSV mapping crowded period scenario rows to
            proposed intra-month periods.

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
    issues.extend(
        _audit_scenario_calendar(
            calendar=_load_scenario_calendar(scenario_calendar_path),
            holding_scenarios=_load_holding_scenarios(holding_scenarios_path),
            transaction_scenarios=_load_transaction_scenarios(transaction_scenarios_path),
            axys_directory=axys_directory,
        )
    )
    scenario_calendar = _load_scenario_calendar(scenario_calendar_path)
    scenario_inventory = _load_scenario_inventory(scenario_inventory_path)
    issues.extend(
        _audit_protected_scenario_inventory(
            inventory=scenario_inventory,
            calendar=scenario_calendar,
            comparison_path=comparison_path,
            holding_scenarios=_load_holding_scenarios(holding_scenarios_path),
            transaction_scenarios=_load_transaction_scenarios(
                transaction_scenarios_path
            ),
            axys_directory=axys_directory,
        )
    )
    issues.extend(
        _audit_generated_causal_story_coverage(
            comparison_path=comparison_path,
            calendar=scenario_calendar,
        )
    )
    issues.extend(
        _audit_period_split_plan(
            plan=_load_period_split_plan(period_split_plan_path),
            calendar=scenario_calendar,
        )
    )
    return issues


def _audit_protected_scenario_inventory(
    *,
    inventory: pd.DataFrame,
    calendar: pd.DataFrame,
    comparison_path: Path = _DEFAULT_COMPARISON_PATH,
    holding_scenarios: HoldingScenarioSet | None = None,
    transaction_scenarios: TransactionScenarioSet | None = None,
    axys_directory: Path = _DEFAULT_AXYS_APX_DIRECTORY,
) -> list[AuditIssue]:
    """Return issues when a protected demo scenario changes meaning or outcome.

    The independent inventory is deliberately stricter than the operational
    calendar. It protects each scenario's source period, report period,
    economic identity, report disposition, and carry-forward treatment so a
    fixture edit cannot silently rewrite the demo's review story.
    """
    protected_keys = set(inventory["scenario_key"].astype(str))
    calendar_keys = set(calendar["scenario_key"].astype(str))
    issues = [
        AuditIssue(
            check="protected_scenario_inventory",
            detail=f"Protected demo scenario disappeared: {scenario_key}.",
        )
        for scenario_key in sorted(protected_keys - calendar_keys)
    ]
    issues.extend(
        AuditIssue(
            check="protected_scenario_inventory",
            detail=(
                "Demo scenario is not protected by the independent inventory: "
                f"{scenario_key}."
            ),
        )
        for scenario_key in sorted(calendar_keys - protected_keys)
    )
    if issues:
        return issues

    calendar_by_key = calendar.set_index("scenario_key", drop=False)
    contract_columns = {
        "scenario_source": "scenario_source",
        "portfolio": "portfolio",
        "story_from_date": "from_date",
        "story_thru_date": "thru_date",
        "scenario_family": "scenario_family",
        "primary_security": "primary_security",
    }
    for row in inventory.itertuples(index=False):
        calendar_row = calendar_by_key.loc[str(row.scenario_key)]
        for inventory_column, calendar_column in contract_columns.items():
            expected = str(getattr(row, inventory_column))
            actual = str(calendar_row[calendar_column])
            if expected != actual:
                issues.append(
                    AuditIssue(
                        check="protected_scenario_semantics",
                        portfolio=str(row.portfolio),
                        from_date=str(row.story_from_date),
                        thru_date=str(row.story_thru_date),
                        detail=(
                            "Protected scenario meaning changed without an explicit "
                            f"inventory update: {row.scenario_key}; "
                            f"{inventory_column} expected={expected}; actual={actual}."
                        ),
                    )
                )

    issues.extend(_audit_scenario_independent_change_contract(inventory))
    issues.extend(
        _audit_scenario_source_period_contract(
            inventory=inventory,
            holding_scenarios=(
                holding_scenarios
                or _load_holding_scenarios(_DEFAULT_HOLDING_SCENARIOS_PATH)
            ),
            transaction_scenarios=(
                transaction_scenarios
                or _load_transaction_scenarios(_DEFAULT_TRANSACTION_SCENARIOS_PATH)
            ),
            axys_directory=axys_directory,
        )
    )
    issues.extend(
        _audit_scenario_report_contract(
            inventory=inventory,
            comparison_path=comparison_path,
        )
    )
    return issues


def _audit_scenario_independent_change_contract(
    inventory: pd.DataFrame,
) -> list[AuditIssue]:
    """Return issues when a source period exceeds its economic-change contract."""
    issues: list[AuditIssue] = []
    period_columns = ["portfolio", "source_from_date", "source_thru_date"]
    for period_key, period_rows in inventory.groupby(period_columns, sort=True):
        portfolio, from_date, thru_date = (str(value) for value in period_key)
        actual_count = int(period_rows["independent_change_id"].nunique())
        declared_counts = set(period_rows["source_period_independent_changes"].astype(int))
        if declared_counts != {actual_count}:
            issues.append(
                AuditIssue(
                    check="scenario_independent_change_count",
                    portfolio=portfolio,
                    from_date=from_date,
                    thru_date=thru_date,
                    detail=(
                        "Source-period independent-change count does not match "
                        f"the protected scenario IDs: declared={sorted(declared_counts)}; "
                        f"actual={actual_count}."
                    ),
                )
            )
        if actual_count > _SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES:
            issues.append(
                AuditIssue(
                    check="scenario_independent_change_budget",
                    portfolio=portfolio,
                    from_date=from_date,
                    thru_date=thru_date,
                    detail=(
                        "Source period exceeds the protected independent-change "
                        f"budget: actual={actual_count}; "
                        f"maximum={_SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES}."
                    ),
                )
            )
    return issues


def _audit_scenario_source_period_contract(
    *,
    inventory: pd.DataFrame,
    holding_scenarios: HoldingScenarioSet,
    transaction_scenarios: TransactionScenarioSet,
    axys_directory: Path,
) -> list[AuditIssue]:
    """Return issues when fixture input dates drift outside protected periods."""
    actual_inputs = _scenario_source_inputs(
        holding_scenarios=holding_scenarios,
        transaction_scenarios=transaction_scenarios,
        axys_directory=axys_directory,
    )
    issues: list[AuditIssue] = []
    for row in inventory.itertuples(index=False):
        source_input = actual_inputs.get(str(row.scenario_key))
        if source_input is None:
            issues.append(
                AuditIssue(
                    check="scenario_source_period",
                    detail=f"Protected scenario has no fixture source input: {row.scenario_key}.",
                )
            )
            continue
        actual_portfolio, input_date = source_input
        actual_period = _portfolio_period_containing_date(
            axys_directory=axys_directory,
            portfolio=actual_portfolio,
            input_date=input_date,
        )
        expected_period = (
            str(row.portfolio),
            str(row.source_from_date),
            str(row.source_thru_date),
        )
        if actual_period != expected_period:
            issues.append(
                AuditIssue(
                    check="scenario_source_period",
                    portfolio=str(row.portfolio),
                    from_date=str(row.source_from_date),
                    thru_date=str(row.source_thru_date),
                    detail=(
                        "Protected scenario source period changed: "
                        f"{row.scenario_key}; expected={expected_period}; "
                        f"actual={actual_period}; input_date={input_date}."
                    ),
                )
            )
    return issues


def _scenario_source_inputs(
    *,
    holding_scenarios: HoldingScenarioSet,
    transaction_scenarios: TransactionScenarioSet,
    axys_directory: Path,
) -> dict[str, tuple[str, str]]:
    """Return each named scenario's actual portfolio and source-input date."""
    source_inputs: dict[str, tuple[str, str]] = {}
    base_transactions = _read_packaged_transactions(
        axys_directory / _BASE_SNAPSHOT_DIRECTORY / "transactions.csv"
    ).set_index("TRANSACTION_ID", drop=False)
    for adjustment in transaction_scenarios.adjustments:
        if adjustment.action == "insert":
            portfolio = str(adjustment.values["PORT"])
            input_date = str(adjustment.values["TRANSACTION_DATE"])
        else:
            base_row = base_transactions.loc[adjustment.transaction_id]
            portfolio = str(base_row["PORT"])
            input_date = pd.Timestamp(str(base_row["TRANSACTION_DATE"])).date().isoformat()
        source_inputs[_transaction_scenario_calendar_key(adjustment)] = (
            portfolio,
            pd.Timestamp(input_date).date().isoformat(),
        )
    for adjustment in holding_scenarios.adjustments:
        source_inputs[_holding_scenario_calendar_key(adjustment)] = (
            adjustment.portfolio,
            adjustment.holding_date,
        )
    source_inputs.update(
        {
            "multicurrency:BALANCED:SAP.DE:2026-04-15:EUR dividend correction": (
                "BALANCED",
                "2026-04-15",
            ),
            "multicurrency:BALANCED:SHEL.L:2026-04-30:GBP FX correction": (
                "BALANCED",
                "2026-04-30",
            ),
        }
    )
    return source_inputs


def _portfolio_period_containing_date(
    *,
    axys_directory: Path,
    portfolio: str,
    input_date: str,
) -> tuple[str, str, str] | None:
    """Return the unique packaged portfolio period containing an input date."""
    date_value = pd.Timestamp(input_date)
    matches: set[tuple[str, str, str]] = set()
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        portperf = _read_packaged_axys_frame(
            axys_directory / snapshot_name / "portperf.csv",
            "portfolio_performance",
        )
        portfolio_rows = portperf[portperf["PORTFOLIO_CODE"].astype(str).eq(portfolio)]
        for row in portfolio_rows.itertuples(index=False):
            if (
                pd.Timestamp(str(row.FROM_DATE))
                <= date_value
                <= pd.Timestamp(str(row.THRU_DATE))
            ):
                matches.add((portfolio, str(row.FROM_DATE), str(row.THRU_DATE)))
    if len(matches) != 1:
        return None
    return matches.pop()


def _audit_scenario_report_contract(
    *,
    inventory: pd.DataFrame,
    comparison_path: Path,
) -> list[AuditIssue]:
    """Return issues when a protected scenario's reviewer-visible result drifts."""
    findings = compare_snapshots(
        comparison_path,
        require_causal_attribution=True,
        comparison_level="portfolio",
    )
    portfolio_changes = _workbook_portfolio_changes_table(
        findings,
        comparison_path=comparison_path,
    )
    causes = _workbook_underlying_causes_table(
        findings,
        comparison_path=comparison_path,
    )
    data_issues = data_issues_table(comparison_path)
    statuses = {
        (
            str(row["portfolio_id"]),
            row["from_date"].isoformat(),
            row["thru_date"].isoformat(),
        ): str(row["review_status"])
        for row in portfolio_changes.iter_rows(named=True)
    }
    finding_rows = list(findings.iter_rows(named=True))
    cause_rows = list(causes.iter_rows(named=True))
    data_issues_rows = list(data_issues.iter_rows(named=True))
    issues: list[AuditIssue] = []
    for row in inventory.itertuples(index=False):
        period_key = (
            str(row.portfolio),
            str(row.story_from_date),
            str(row.story_thru_date),
        )
        actual_status = statuses.get(period_key, "No Performance Difference")
        if actual_status != str(row.expected_period_status):
            issues.append(
                AuditIssue(
                    check="scenario_report_status",
                    portfolio=period_key[0],
                    from_date=period_key[1],
                    thru_date=period_key[2],
                    detail=(
                        "Protected scenario report status changed: "
                        f"{row.scenario_key}; expected={row.expected_period_status}; "
                        f"actual={actual_status}."
                    ),
                )
            )

        matching_findings = [
            finding
            for finding in finding_rows
            if _report_row_matches_scenario(finding, row, date_field="input_date")
        ]
        matching_causes = [
            cause
            for cause in cause_rows
            if _report_row_matches_scenario(cause, row, date_field="as_of_date")
        ]
        if str(row.expected_report_disposition) == "fixture_only_context":
            if matching_findings or matching_causes:
                issues.append(
                    _scenario_disposition_issue(
                        row,
                        "Fixture-only context unexpectedly entered report evidence.",
                    )
                )
            continue
        if str(row.expected_report_disposition) == "data_issues_issue":
            expected_issue_type = _SCENARIO_DATA_ISSUE_TYPES.get(
                str(row.scenario_family)
            )
            matching_data_issues = [
                issue
                for issue in data_issues_rows
                if str(issue.get("portfolio_id") or "") == str(row.portfolio)
                and _security_symbol_from_ppar_id(issue.get("security_id"))
                == str(row.primary_security)
                and str(issue.get("issue_type") or "") == expected_issue_type
                and _date_is_within(
                    issue.get("as_of_date"),
                    str(row.source_from_date),
                    str(row.source_thru_date),
                )
            ]
            if expected_issue_type is None:
                issues.append(
                    _scenario_disposition_issue(
                        row,
                        "Data Issues scenario family has no protected issue type.",
                    )
                )
            elif not matching_data_issues:
                issues.append(_scenario_disposition_issue(row, "Data Issues issue is absent."))
            continue

        if not matching_findings and not matching_causes:
            issues.append(
                _scenario_disposition_issue(
                    row,
                    "Primary-security evidence is absent from the report tables.",
                )
            )
            continue
        if str(row.expected_report_disposition) == "counted_cause":
            counted_causes = [
                cause
                for cause in cause_rows
                if _report_row_period_key(cause) == period_key
                and str(cause.get("safety_disposition") or "") == "counted_cause"
            ]
            if not counted_causes:
                issues.append(
                    _scenario_disposition_issue(
                        row,
                        "Story period has no counted Performance Difference Cause.",
                    )
                )
        if str(row.carry_forward_status) == "carry_forward_effect":
            carried_causes = [
                cause
                for cause in matching_causes
                if cause.get("as_of_date") is not None
                and str(cause["as_of_date"]) < str(row.story_from_date)
            ]
            if not carried_causes:
                issues.append(
                    _scenario_disposition_issue(
                        row,
                        "Carry-forward effect is not visible as a beginning-period cause.",
                    )
                )
    return issues


def _report_row_matches_scenario(
    report_row: dict[str, object],
    scenario_row: Any,
    *,
    date_field: str,
) -> bool:
    """Return whether a finding or cause is evidence for one scenario story."""
    return (
        _report_row_period_key(report_row)
        == (
            str(scenario_row.portfolio),
            str(scenario_row.story_from_date),
            str(scenario_row.story_thru_date),
        )
        and _security_symbol_from_ppar_id(report_row.get("security_id"))
        == str(scenario_row.primary_security)
        and (
            report_row.get(date_field) is None
            or _date_is_within(
                report_row.get(date_field),
                str(scenario_row.source_from_date),
                str(scenario_row.story_thru_date),
            )
        )
    )


def _report_row_period_key(report_row: dict[str, object]) -> tuple[str, str, str]:
    """Return a normalized portfolio-period key for one report row."""
    return (
        str(report_row.get("portfolio_id") or ""),
        str(report_row.get("from_date") or ""),
        str(report_row.get("thru_date") or ""),
    )


def _security_symbol_from_ppar_id(value: object) -> str:
    """Return the source symbol from a type-first PPAR security identifier."""
    security_id = str(value or "")
    demo_security_types = {"caus", "cseu", "csgb", "csus", "fius"}
    security_type = security_id[:4]
    if security_type not in demo_security_types:
        return security_id
    symbol = security_id[4:]
    return symbol[1:] if symbol.startswith("_") else symbol


def _date_is_within(value: object, from_date: str, thru_date: str) -> bool:
    """Return whether a date-like value is inside an inclusive ISO-date range."""
    if value is None:
        return False
    date_text = pd.Timestamp(str(value)).date().isoformat()
    return from_date <= date_text <= thru_date


def _scenario_disposition_issue(row: Any, detail: str) -> AuditIssue:
    """Return a consistently keyed scenario report-disposition issue."""
    return AuditIssue(
        check="scenario_report_disposition",
        portfolio=str(row.portfolio),
        from_date=str(row.story_from_date),
        thru_date=str(row.story_thru_date),
        detail=f"Protected scenario disposition changed: {row.scenario_key}; {detail}",
    )


def _audit_generated_causal_story_coverage(
    *,
    comparison_path: Path,
    calendar: pd.DataFrame,
) -> list[AuditIssue]:
    """Return issues for report-visible causal securities absent from the calendar.

    This closes the gap that allowed hard-coded multi-currency scenarios to
    bypass the hand-maintained scenario files and overcrowd a demo period.
    """
    findings = compare_snapshots(
        comparison_path,
        require_causal_attribution=True,
        comparison_level="portfolio",
    )
    causes = _workbook_underlying_causes_table(
        findings,
        comparison_path=comparison_path,
    )
    expected_by_period: dict[tuple[str, str, str], set[str]] = {}
    for row in calendar.itertuples(index=False):
        if str(row.scenario_source) != "multicurrency":
            continue
        period_key = (str(row.portfolio), str(row.from_date), str(row.thru_date))
        expected_by_period.setdefault(period_key, set()).add(str(row.primary_security))

    story_securities: dict[tuple[str, str, str], set[str]] = {}
    explained_cash: dict[tuple[str, str, str], set[str]] = {}
    for row in causes.iter_rows(named=True):
        security = _security_symbol_from_ppar_id(row.get("security_id"))
        if not security:
            continue
        period_key = (
            str(row["portfolio_id"]),
            row["from_date"].isoformat(),
            row["thru_date"].isoformat(),
        )
        dataset = str(row.get("dataset") or "")
        estimated_impact = row.get("estimated_impact")
        is_cash = security.startswith("CASH")
        if dataset in {"fx_rates", "transactions"} and not is_cash:
            story_securities.setdefault(period_key, set()).add(security)
        elif estimated_impact is not None and not is_cash:
            story_securities.setdefault(period_key, set()).add(security)
        elif estimated_impact is not None and is_cash:
            explained_cash.setdefault(period_key, set()).add(security)

    issues: list[AuditIssue] = []
    for period_key in sorted(expected_by_period):
        actual_securities = story_securities.get(period_key) or explained_cash.get(
            period_key, set()
        )
        unexpected = actual_securities - expected_by_period.get(period_key, set())
        if not unexpected:
            continue
        portfolio, from_date, thru_date = period_key
        issues.append(
            AuditIssue(
                check="generated_causal_story_coverage",
                portfolio=portfolio,
                from_date=from_date,
                thru_date=thru_date,
                detail=(
                    "Generated report contains causal securities missing from "
                    f"the scenario calendar: {', '.join(sorted(unexpected))}."
                ),
            )
        )
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
        axys_directory: Directory containing ``snapshot_a`` and
            ``snapshot_b``.
        comparison_path: Shared comparison YAML with reconstruction rules.
        holding_scenarios_path: CSV containing intentional holding adjustments.
        transaction_scenarios_path: CSV containing intentional transaction
            adjustments.
        write: Whether to write rebuilt ``transactions.csv``, ``holdings.csv``,
            ``secperf.csv``, and ``portperf.csv``.

    Returns:
        JSON-serializable audit summary with one entry per snapshot.
    """
    specification = AuditSpecification(
        comparison_path,
        comparison_level="portfolio",
    )
    portfolio_reconstruction = specification.portfolio_return_reconstruction
    security_reconstruction = specification.security_return_reconstruction
    if portfolio_reconstruction is None or security_reconstruction is None:
        raise ValueError("Demo rebuild requires portfolio and security reconstruction YAML.")

    snapshots: list[dict[str, object]] = []
    base_snapshot_directory = axys_directory / _BASE_SNAPSHOT_DIRECTORY
    base_holdings = _with_internal_cost(
        _with_multicurrency_holdings(
            _read_packaged_axys_frame(
                base_snapshot_directory / "holdings.csv",
                "holdings",
            )
        )
    )
    base_transactions = _read_packaged_transactions(base_snapshot_directory / "transactions.csv")
    holding_scenarios = _load_holding_scenarios(holding_scenarios_path)
    transaction_scenarios = _load_transaction_scenarios(transaction_scenarios_path)
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        snapshot_directory = axys_directory / snapshot_name
        current_secperf = _with_multicurrency_performance_rows(
            _read_packaged_axys_frame(
                snapshot_directory / "secperf.csv",
                "security_performance",
            ),
            security_level=True,
        )
        current_portperf = _with_multicurrency_performance_rows(
            _read_packaged_axys_frame(
                snapshot_directory / "portperf.csv",
                "portfolio_performance",
            ),
            security_level=False,
        )
        holdings = _with_internal_cost(
            _with_multicurrency_holdings(
                _read_packaged_axys_frame(
                    snapshot_directory / "holdings.csv",
                    "holdings",
                )
            )
        )
        current_transactions = _read_packaged_transactions(snapshot_directory / "transactions.csv")
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

        rebuilt_portperf = _rebuild_portfolio_performance(
            snapshot_name,
            current_portperf,
            rebuilt_holdings,
            rebuilt_transactions,
            portfolio_reconstruction,
        )
        rebuilt_secperf = _rebuild_security_performance(
            snapshot_name,
            current_secperf,
            rebuilt_holdings,
            rebuilt_transactions,
            security_reconstruction,
        )
        current_packaged_transactions = current_transactions[_PACKAGED_TRANSACTION_COLUMNS]
        rebuilt_packaged_transactions = rebuilt_transactions[_PACKAGED_TRANSACTION_COLUMNS]
        transaction_delta = _max_numeric_delta(
            current_packaged_transactions,
            rebuilt_packaged_transactions,
            _TRANSACTION_NUMERIC_COLUMNS,
        )
        has_transaction_field_drift = _has_non_numeric_delta(
            current_packaged_transactions,
            rebuilt_packaged_transactions,
            _TRANSACTION_NUMERIC_COLUMNS,
        )
        holdings_delta = _max_numeric_delta(
            holdings,
            rebuilt_holdings,
            _PACKAGED_HOLDINGS_NUMERIC_COLUMNS,
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
        has_transaction_drift = transaction_delta > _CHECK_TOLERANCE or has_transaction_field_drift
        has_holdings_drift = holdings_delta > _CHECK_TOLERANCE
        if write:
            _write_packaged_transactions(
                rebuilt_transactions,
                snapshot_directory / "transactions.csv",
            )
            _write_packaged_axys_frame(
                _packaged_holdings(rebuilt_holdings),
                snapshot_directory / "holdings.csv",
                "holdings",
            )
            _write_packaged_axys_frame(
                rebuilt_secperf[_SECPERF_COLUMNS],
                snapshot_directory / "secperf.csv",
                "security_performance",
            )
            _write_packaged_axys_frame(
                rebuilt_portperf,
                snapshot_directory / "portperf.csv",
                "portfolio_performance",
            )

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
                "holding_scenario_rows": len(holding_scenarios.for_snapshot(snapshot_name)),
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
    fx_restatement = (
        rebuilt["PORT"].eq("BALANCED")
        & rebuilt["SEC"].eq("SHEL.L")
        & rebuilt["HOLDING_DATE"].astype(str).eq("2026-04-30")
    )
    if int(fx_restatement.sum()) != 1:
        raise ValueError("Multi-currency FX restatement must match one holding row.")
    rebuilt.loc[fx_restatement, "BASE_MKT_VAL"] = (
        rebuilt.loc[fx_restatement, "BASE_MKT_VAL"].astype(float) + 320.0
    )
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
        if scenario.action == "insert":
            transaction_code = str(scenario.values["TRAN"])
        else:
            transaction_code = transaction_codes.get(scenario.transaction_id)
        if not transaction_code:
            raise ValueError(
                "Transaction scenario must match one base transaction before "
                f"it can be summarized: {scenario.transaction_id}."
            )
        counts[transaction_code] = counts.get(transaction_code, 0) + 1
    return dict(sorted(counts.items()))


def _read_packaged_axys_frame(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read one common-caption Axys/APX fixture into rebuild-internal columns."""
    frame = pd.read_csv(path)
    if "Security Type" in frame.columns:
        frame = frame.drop(columns="Security Type")
    renamed = frame.rename(columns=_COMMON_AXYS_HEADERS_TO_INTERNAL[dataset_name])
    if dataset_name == "portfolio_performance":
        return renamed[_PORTPERF_COLUMNS]
    if dataset_name == "security_performance":
        return renamed[_SECPERF_COLUMNS]
    return renamed


def _security_types_for_symbols(path: Path, symbols: pd.Series) -> pd.Series:
    """Return reviewed security-master types for source security symbols."""
    reference = pd.read_csv(path.parent / "secmast.csv", dtype=str)
    type_by_symbol = dict(
        zip(
            reference["Security Symbol"],
            reference["Security Type"],
            strict=True,
        )
    )
    security_types = symbols.astype(str).map(type_by_symbol)
    if security_types.isna().any():
        missing_symbols = sorted(symbols.loc[security_types.isna()].astype(str).unique())
        raise ValueError(f"Security master is missing types for: {missing_symbols}")
    return security_types


def _write_packaged_axys_frame(
    frame: pd.DataFrame,
    path: Path,
    dataset_name: str,
) -> None:
    """Write one rebuild-internal frame with common Axys/APX captions."""
    output = frame.rename(
        columns={
            internal: common
            for common, internal in _COMMON_AXYS_HEADERS_TO_INTERNAL[dataset_name].items()
        }
    )
    symbol_column = (
        "Security Symbol"
        if dataset_name in {"holdings", "security_performance", "transactions"}
        else None
    )
    if symbol_column is not None:
        security_types = _security_types_for_symbols(path, output[symbol_column])
        output.insert(
            output.columns.get_loc(symbol_column) + 1,
            "Security Type",
            security_types,
        )
    output.to_csv(path, index=False)


def _read_packaged_transactions(path: Path) -> pd.DataFrame:
    """Return packaged transactions with internal scenario IDs restored.

    The user-facing Axys/APX demo intentionally omits ``TRANSACTION_ID`` because a
    durable native transaction identifier is not proven as typical Axys/APX output.
    The rebuild scenario CSV still uses deterministic IDs as internal fixture
    handles so the demo derivation remains auditable.
    """
    return _with_internal_transaction_ids(
        _with_demo_transactions(_read_packaged_axys_frame(path, "transactions"))
    )


def _with_multicurrency_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return demo holdings with explicit local and base-currency values."""
    rows = holdings.loc[~holdings["PORT"].isin(_RETIRED_DEMO_PORTFOLIOS)].copy()
    if "CURRENCY" not in rows.columns:
        rows["CURRENCY"] = "USD"
    if "BASE_CURRENCY" not in rows.columns:
        rows["BASE_CURRENCY"] = "USD"
    if "BASE_MKT_VAL" not in rows.columns:
        rows["BASE_MKT_VAL"] = rows["MKT_VAL"]

    currencies = {
        "SAP.DE": "EUR",
        "CASHEUR": "EUR",
        "SHEL.L": "GBP",
        "CASHGBP": "GBP",
    }
    rows.loc[rows["SEC"].isin(currencies), "CURRENCY"] = rows.loc[
        rows["SEC"].isin(currencies), "SEC"
    ].map(currencies)
    balanced_dates = sorted(rows.loc[rows["PORT"].eq("BALANCED"), "HOLDING_DATE"].unique())
    existing_keys = set(zip(rows["PORT"], rows["SEC"], rows["HOLDING_DATE"], strict=True))
    additions: list[dict[str, object]] = []
    for date_index, holding_date in enumerate(balanced_dates):
        eur_rate = 1.08 + 0.002 * date_index
        gbp_rate = 1.26 + 0.002 * date_index
        local_values = {
            "SAP.DE": (200.0, 100.0, 20_000.0, "EUR", eur_rate),
            "CASHEUR": (5_000.0, 1.0, 5_000.0, "EUR", eur_rate),
            "SHEL.L": (480.0, 25.0, 12_000.0, "GBP", gbp_rate),
            "CASHGBP": (4_000.0, 1.0, 4_000.0, "GBP", gbp_rate),
        }
        for security, (quantity, price, market_value, currency, rate) in local_values.items():
            if ("BALANCED", security, holding_date) in existing_keys:
                continue
            additions.append(
                {
                    "PORT": "BALANCED",
                    "SEC": security,
                    "HOLDING_DATE": holding_date,
                    "CURRENCY": currency,
                    "BASE_CURRENCY": "USD",
                    "QTY": quantity,
                    "PRICE": price,
                    "MKT_VAL": market_value,
                    "BASE_MKT_VAL": round(market_value * rate, 2),
                    "ACCRUED": 0.0,
                }
            )
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return _with_contribution_demo_holdings(_with_balanced_april_holding_dates(rows))


def _with_contribution_demo_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings with an isolated March contribution demonstration."""
    rows = holdings.copy()
    existing_keys = set(zip(rows["PORT"], rows["SEC"], rows["HOLDING_DATE"], strict=True))
    additions: list[dict[str, object]] = []
    for holding_date, market_value in _CONTRIBUTION_DEMO_HOLDINGS:
        key = (_CONTRIBUTION_DEMO_PORTFOLIO, "CASHUSD", holding_date)
        if key in existing_keys:
            continue
        additions.append(
            {
                "PORT": _CONTRIBUTION_DEMO_PORTFOLIO,
                "SEC": "CASHUSD",
                "HOLDING_DATE": holding_date,
                "CURRENCY": "USD",
                "BASE_CURRENCY": "USD",
                "QTY": market_value,
                "PRICE": 1.0,
                "MKT_VAL": market_value,
                "BASE_MKT_VAL": market_value,
                "ACCRUED": 0.0,
            }
        )
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def _with_balanced_april_holding_dates(holdings: pd.DataFrame) -> pd.DataFrame:
    """Clone BALANCED month-end holdings onto the new April subperiod ends."""
    rows = holdings.copy()
    source_date = _BALANCED_APRIL_PERIOD[1]
    source_rows = rows.loc[
        rows["PORT"].eq("BALANCED") & rows["HOLDING_DATE"].astype(str).eq(source_date)
    ]
    if source_rows.empty:
        return rows
    existing_dates = set(rows.loc[rows["PORT"].eq("BALANCED"), "HOLDING_DATE"].astype(str))
    additions: list[pd.DataFrame] = []
    for _, thru_date in _BALANCED_APRIL_SUBPERIODS[:-1]:
        if thru_date in existing_dates:
            continue
        addition = source_rows.copy()
        addition["HOLDING_DATE"] = thru_date
        additions.append(addition)
    if additions:
        rows = pd.concat([rows, *additions], ignore_index=True)
    return rows


def _with_demo_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with baseline semantics and foreign-currency examples."""
    rows = transactions.copy()
    if "CURRENCY" not in rows.columns:
        rows["CURRENCY"] = "USD"
    if "BASE_CURRENCY" not in rows.columns:
        rows["BASE_CURRENCY"] = "USD"
    if "BASE_AMOUNT" not in rows.columns:
        rows["BASE_AMOUNT"] = rows["AMOUNT"]
    examples = [
        {
            "PORT": "BALANCED",
            "TRANSACTION_DATE": "2026-01-12",
            "SETTLE_DATE": "2026-01-14",
            "SEC": "SAP.DE",
            "TRAN": "by",
            "SEC_TYPE": "cseu",
            "SRC_DEST_TYPE": "$cash",
            "SRC_DEST_SYMBOL": "CASHEUR",
            "SPECIAL_SEC_TYPE": "",
            "SPECIAL_SEC_SYMBOL": "",
            "CURRENCY": "EUR",
            "BASE_CURRENCY": "USD",
            "QTY": 20.0,
            "PRICE": 100.0,
            "AMOUNT": -2_000.0,
            "BASE_AMOUNT": -2_180.0,
            "COMMISSION": 0.0,
        },
        {
            "PORT": "BALANCED",
            "TRANSACTION_DATE": "2026-02-16",
            "SETTLE_DATE": "2026-02-18",
            "SEC": "SHEL.L",
            "TRAN": "sl",
            "SEC_TYPE": "csgb",
            "SRC_DEST_TYPE": "$cash",
            "SRC_DEST_SYMBOL": "CASHGBP",
            "SPECIAL_SEC_TYPE": "",
            "SPECIAL_SEC_SYMBOL": "",
            "CURRENCY": "GBP",
            "BASE_CURRENCY": "USD",
            "QTY": 40.0,
            "PRICE": 25.0,
            "AMOUNT": 1_000.0,
            "BASE_AMOUNT": 1_270.0,
            "COMMISSION": 0.0,
        },
        {
            "PORT": "BALANCED",
            "TRANSACTION_DATE": "2026-04-15",
            "SETTLE_DATE": "2026-04-15",
            "SEC": "SAP.DE",
            "TRAN": "dv",
            "SEC_TYPE": "cseu",
            "SRC_DEST_TYPE": "$income",
            "SRC_DEST_SYMBOL": "CASHEUR",
            "SPECIAL_SEC_TYPE": "",
            "SPECIAL_SEC_SYMBOL": "",
            "CURRENCY": "EUR",
            "BASE_CURRENCY": "USD",
            "QTY": 0.0,
            "PRICE": 0.0,
            "AMOUNT": 120.0,
            "BASE_AMOUNT": 129.6,
            "COMMISSION": 0.0,
        },
        {
            "PORT": "INCOME",
            "TRANSACTION_DATE": "2026-01-22",
            "SETTLE_DATE": "2026-01-22",
            "SEC": "MARGIN_USD",
            "TRAN": "ai",
            "SEC_TYPE": "caus",
            "SRC_DEST_TYPE": "$pth",
            "SRC_DEST_SYMBOL": "$cash",
            "SPECIAL_SEC_TYPE": "caus",
            "SPECIAL_SEC_SYMBOL": "margin",
            "CURRENCY": "USD",
            "BASE_CURRENCY": "USD",
            "QTY": 0.0,
            "PRICE": 0.0,
            "AMOUNT": -18.75,
            "BASE_AMOUNT": -18.75,
            "COMMISSION": 0.0,
        },
        {
            "PORT": "BALANCED",
            "TRANSACTION_DATE": "2026-03-20",
            "SETTLE_DATE": "2026-03-20",
            "SEC": "JPM",
            "TRAN": "ti",
            "SEC_TYPE": "csus",
            "SRC_DEST_TYPE": "$pty",
            "SRC_DEST_SYMBOL": "external_delivery",
            "SPECIAL_SEC_TYPE": "",
            "SPECIAL_SEC_SYMBOL": "",
            "CURRENCY": "USD",
            "BASE_CURRENCY": "USD",
            "QTY": 5.0,
            "PRICE": 294.16,
            "AMOUNT": 1_470.80,
            "BASE_AMOUNT": 1_470.80,
            "COMMISSION": 0.0,
        },
        {
            "PORT": "BALANCED",
            "TRANSACTION_DATE": "2026-04-06",
            "SETTLE_DATE": "2026-04-30",
            "SEC": "JPM",
            "TRAN": "dp",
            "SEC_TYPE": "csus",
            "SRC_DEST_TYPE": "$pty",
            "SRC_DEST_SYMBOL": "$cash",
            "SPECIAL_SEC_TYPE": "epus",
            "SPECIAL_SEC_SYMBOL": "with",
            "CURRENCY": "USD",
            "BASE_CURRENCY": "USD",
            "QTY": 0.0,
            "PRICE": 0.0,
            "AMOUNT": -70.48,
            "BASE_AMOUNT": -70.48,
            "COMMISSION": 0.0,
        },
    ]
    keys = set(
        zip(
            rows["PORT"],
            rows["TRANSACTION_DATE"].astype(str),
            rows["SEC"],
            rows["TRAN"],
            strict=True,
        )
    )
    additions = [
        row
        for row in examples
        if (row["PORT"], row["TRANSACTION_DATE"], row["SEC"], row["TRAN"]) not in keys
    ]
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def _with_multicurrency_performance_rows(
    performance: pd.DataFrame,
    *,
    security_level: bool,
) -> pd.DataFrame:
    """Return performance targets with required multi-currency demo rows."""
    retained_performance = performance.loc[
        ~performance["PORTFOLIO_CODE"].isin(_RETIRED_DEMO_PORTFOLIOS)
    ].copy()
    rows = _with_contribution_demo_performance_rows(
        _with_balanced_april_performance_periods(retained_performance),
        security_level=security_level,
    )
    if not security_level:
        rows["BASE_CURRENCY"] = "USD"
        return rows

    multicurrency_securities = ("SAP.DE", "CASHEUR", "SHEL.L", "CASHGBP")
    balanced_periods = rows.loc[
        rows["PORTFOLIO_CODE"].eq("BALANCED"), ["FROM_DATE", "THRU_DATE"]
    ].drop_duplicates()
    existing_keys = set(
        zip(
            rows["PORTFOLIO_CODE"],
            rows["SECURITY_ID"],
            rows["FROM_DATE"],
            rows["THRU_DATE"],
            strict=True,
        )
    )
    additions: list[dict[str, object]] = []
    for period in balanced_periods.itertuples(index=False):
        for security in multicurrency_securities:
            key = ("BALANCED", security, period.FROM_DATE, period.THRU_DATE)
            if key in existing_keys:
                continue
            additions.append(
                {
                    "PORTFOLIO_CODE": "BALANCED",
                    "SECURITY_ID": security,
                    "FROM_DATE": period.FROM_DATE,
                    "THRU_DATE": period.THRU_DATE,
                    **{column: 0.0 for column in _SECPERF_NUMERIC_COLUMNS},
                }
            )
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def _with_contribution_demo_performance_rows(
    performance: pd.DataFrame,
    *,
    security_level: bool,
) -> pd.DataFrame:
    """Return performance targets with one self-contained contribution period."""
    rows = performance.copy()
    additions: list[dict[str, object]] = []
    for from_date, thru_date in (
        _CONTRIBUTION_DEMO_BASELINE_PERIOD,
        _CONTRIBUTION_DEMO_PERIOD,
    ):
        portfolio_mask = (
            rows["PORTFOLIO_CODE"].eq(_CONTRIBUTION_DEMO_PORTFOLIO)
            & rows["FROM_DATE"].astype(str).eq(from_date)
            & rows["THRU_DATE"].astype(str).eq(thru_date)
        )
        if security_level:
            portfolio_mask &= rows["SECURITY_ID"].eq("CASHUSD")
        if bool(portfolio_mask.any()):
            continue
        addition: dict[str, object] = {
            "PORTFOLIO_CODE": _CONTRIBUTION_DEMO_PORTFOLIO,
            "FROM_DATE": from_date,
            "THRU_DATE": thru_date,
            **{
                column: 0.0
                for column in (
                    _SECPERF_NUMERIC_COLUMNS
                    if security_level
                    else _PORTPERF_NUMERIC_COLUMNS
                )
            },
        }
        if security_level:
            addition["SECURITY_ID"] = "CASHUSD"
        if "BASE_CURRENCY" in rows.columns:
            addition["BASE_CURRENCY"] = "USD"
        additions.append(addition)
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def _with_balanced_april_performance_periods(performance: pd.DataFrame) -> pd.DataFrame:
    """Replace the crowded BALANCED April period with four readable periods."""
    rows = performance.copy()
    from_date, thru_date = _BALANCED_APRIL_PERIOD
    source_mask = (
        rows["PORTFOLIO_CODE"].eq("BALANCED")
        & rows["FROM_DATE"].astype(str).eq(from_date)
        & rows["THRU_DATE"].astype(str).eq(thru_date)
    )
    source_rows = rows.loc[source_mask]
    if source_rows.empty:
        return rows
    rows = rows.loc[~source_mask].copy()
    additions: list[pd.DataFrame] = []
    for subperiod_from, subperiod_thru in _BALANCED_APRIL_SUBPERIODS:
        addition = source_rows.copy()
        addition["FROM_DATE"] = subperiod_from
        addition["THRU_DATE"] = subperiod_thru
        additions.append(addition)
    return pd.concat([rows, *additions], ignore_index=True)


def _with_internal_cost(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings with internal best-efforts cost available for rebuild math.

    The packaged Axys/APX demo intentionally omits ``COST`` from user-facing
    ``holdings.csv`` files because cost is not a Modified Dietz input. Rebuild
    tooling may still use a private best-efforts cost value while constructing
    realistic fixtures, so missing cost falls back to market value.
    """
    if "COST" in holdings.columns:
        return holdings
    internal = holdings.copy()
    internal["COST"] = internal["MKT_VAL"]
    return internal


def _packaged_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return the public holdings columns written to packaged demo CSV files."""
    return holdings[_PACKAGED_HOLDINGS_COLUMNS]


def _write_packaged_transactions(transactions: pd.DataFrame, path: Path) -> None:
    """Write user-facing transactions without internal scenario IDs."""
    _write_packaged_axys_frame(
        transactions[_PACKAGED_TRANSACTION_COLUMNS],
        path,
        "transactions",
    )


def _with_internal_transaction_ids(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with deterministic internal scenario IDs."""
    if _TRANSACTION_ID_COLUMN in transactions.columns:
        return transactions.copy()

    missing_columns = [
        column for column in _PACKAGED_TRANSACTION_COLUMNS if column not in transactions.columns
    ]
    if missing_columns:
        raise ValueError(
            "transactions.csv is missing columns required to derive internal "
            f"scenario IDs: {missing_columns}."
        )

    rows = transactions.copy()
    rows.insert(0, _TRANSACTION_ID_COLUMN, _derived_transaction_ids(rows))
    scenario_ids = {
        ("ALPHA", "2026-01-20", "CASHUSD", "wd"): "ALPHA0203",
        ("ALPHA", "2026-04-06", "JPM", "dv"): "ALPHA0503",
        ("BALANCED", "2026-01-15", "MSFT", "sl"): "BALANCED0203",
        ("BALANCED", "2026-04-06", "JPM", "dv"): "BALANCED0502",
        ("INCOME", "2026-01-20", "CASHUSD", "dp"): "INCOME0203",
        ("INCOME", "2026-05-15", "91282Y2Y1", "in"): "INCOME0603",
        ("INCOME", "2026-05-23", "AAPL", "dv"): "INCOME0604",
        (
            _CONTRIBUTION_DEMO_PORTFOLIO,
            "2026-03-20",
            "CASHUSD",
            "li",
        ): "BALANCED0403",
        ("BALANCED", "2026-01-12", "SAP.DE", "by"): "MC_BAL_EUR_BUY",
        ("BALANCED", "2026-02-16", "SHEL.L", "sl"): "MC_BAL_GBP_SELL",
        ("BALANCED", "2026-04-15", "SAP.DE", "dv"): "MC_BAL_EUR_DIV",
        ("BALANCED", "2026-04-06", "JPM", "dp"): "BALANCED_JPM_WHT",
        ("BALANCED", "2026-03-20", "JPM", "ti"): "BALANCED_TI_20260320",
        ("INCOME", "2026-01-22", "MARGIN_USD", "ai"): "INCOME_AI_20260122",
    }
    transaction_keys = list(
        zip(
            rows["PORT"],
            rows["TRANSACTION_DATE"].astype(str),
            rows["SEC"],
            rows["TRAN"],
            strict=True,
        )
    )
    rows[_TRANSACTION_ID_COLUMN] = [
        scenario_ids.get(key, identifier)
        for key, identifier in zip(
            transaction_keys,
            rows[_TRANSACTION_ID_COLUMN],
            strict=True,
        )
    ]
    return rows


def _derived_transaction_ids(transactions: pd.DataFrame) -> list[str]:
    """Return stable fixture IDs from portfolio, transaction month, and row order."""
    period_index_by_portfolio: dict[str, dict[object, int]] = {}
    row_count_by_period: dict[tuple[str, object], int] = {}
    identifiers: list[str] = []
    for row in transactions.itertuples(index=False):
        portfolio = str(getattr(row, "PORT"))
        transaction_month = pd.Timestamp(getattr(row, "TRANSACTION_DATE")).to_period("M")
        portfolio_periods = period_index_by_portfolio.setdefault(portfolio, {})
        if transaction_month not in portfolio_periods:
            portfolio_periods[transaction_month] = len(portfolio_periods) + 1
        period_index = portfolio_periods[transaction_month]
        row_count_key = (portfolio, transaction_month)
        row_count_by_period[row_count_key] = row_count_by_period.get(row_count_key, 0) + 1
        identifiers.append(
            f"{portfolio}{period_index:02d}{row_count_by_period[row_count_key]:02d}"
        )
    return identifiers


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
        if scenario.action == "insert":
            rebuilt = pd.concat(
                [rebuilt, pd.DataFrame([scenario.values])],
                ignore_index=True,
            )
            continue
        mask = rebuilt["TRANSACTION_ID"].eq(scenario.transaction_id)
        if int(mask.sum()) != 1:
            raise ValueError(
                "Transaction scenario must match exactly one row: " f"{scenario.transaction_id}."
            )
        if scenario.action == "delete":
            rebuilt = rebuilt.loc[~mask].copy()
            continue
        for column, delta in scenario.deltas.items():
            if delta:
                rebuilt.loc[mask, column] = rebuilt.loc[mask, column].astype(float) + delta
    eur_dividend = (
        rebuilt["PORT"].eq("BALANCED")
        & rebuilt["TRANSACTION_DATE"].astype(str).eq("2026-04-15")
        & rebuilt["SEC"].eq("SAP.DE")
        & rebuilt["TRAN"].eq("dv")
    )
    if int(eur_dividend.sum()) != 1:
        raise ValueError("Multi-currency demo dividend must match exactly one row.")
    rebuilt.loc[eur_dividend, "AMOUNT"] = 150.0
    rebuilt.loc[eur_dividend, "BASE_AMOUNT"] = 162.0
    return _rounded_transactions(rebuilt[current_transactions.columns])


def _rounded_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions rounded to the packaged Axys/APX fixture precision."""
    rounded = transactions.copy()
    rounded["QTY"] = rounded["QTY"].astype(float).round(4)
    rounded["PRICE"] = rounded["PRICE"].astype(float).round(4)
    rounded["AMOUNT"] = rounded["AMOUNT"].astype(float).round(2)
    rounded["BASE_AMOUNT"] = rounded["BASE_AMOUNT"].astype(float).round(2)
    rounded["COMMISSION"] = rounded["COMMISSION"].astype(float).round(2)
    return rounded


@cache
def _demo_transaction_policy() -> Mapping[str, object]:
    """Return the validated executable policy for demo transaction effects."""
    try:
        values = yaml.safe_load(
            _DEFAULT_TRANSACTION_POLICY_PATH.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Unable to load {_DEFAULT_TRANSACTION_POLICY_PATH}: {error}"
        ) from error
    if not isinstance(values, dict):
        raise ValueError(f"{_DEFAULT_TRANSACTION_POLICY_PATH}: root must be a mapping.")
    if values.get("schema_version") != 1:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: schema_version must be 1."
        )
    unknown_root_keys = set(values) - {
        "schema_version",
        "holding_effects",
        "reconstruction_roles",
    }
    if unknown_root_keys:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: unsupported root keys "
            f"{sorted(unknown_root_keys)}."
        )
    holding_effects = values.get("holding_effects")
    reconstruction_roles = values.get("reconstruction_roles")
    if not isinstance(holding_effects, dict) or not holding_effects:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: holding_effects must be nonempty."
        )
    if not isinstance(reconstruction_roles, dict) or not reconstruction_roles:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: reconstruction_roles must be nonempty."
        )
    for transaction_code, raw_rule in holding_effects.items():
        _validated_demo_rule(transaction_code, raw_rule, require_effect=True)
    _validated_demo_reconstruction_roles(reconstruction_roles)
    return values


def _validated_demo_rule(
    transaction_code: object,
    raw_rule: object,
    *,
    require_effect: bool,
) -> Mapping[str, object]:
    """Return one validated demo-policy rule."""
    if not isinstance(transaction_code, str) or not transaction_code:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: transaction codes must be strings."
        )
    if not isinstance(raw_rule, dict):
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
            "must be a mapping."
        )
    allowed_keys = (
        {"when", "effect", "quantity_multiplier", "cost_basis"}
        if require_effect
        else {"when"}
    )
    unknown_keys = set(raw_rule) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
            f"has unsupported keys {sorted(unknown_keys)}."
        )
    when = raw_rule.get("when", {})
    if not isinstance(when, dict) or any(
        field not in _DEMO_CONTEXT_COLUMNS
        or not isinstance(expected, str)
        or not expected
        for field, expected in when.items()
    ):
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
            "has an invalid when mapping."
        )
    if require_effect:
        effect = raw_rule.get("effect")
        if effect not in _DEMO_HOLDING_EFFECTS:
            raise ValueError(
                f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
                f"effect must be one of {sorted(_DEMO_HOLDING_EFFECTS)}."
            )
        multiplier = raw_rule.get("quantity_multiplier", 1)
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
            raise ValueError(
                f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
                "quantity_multiplier must be numeric."
            )
        cost_basis = raw_rule.get("cost_basis", "market_value")
        if cost_basis not in _DEMO_COST_BASIS_METHODS:
            raise ValueError(
                f"{_DEFAULT_TRANSACTION_POLICY_PATH}: rule {transaction_code!r} "
                f"cost_basis must be one of {sorted(_DEMO_COST_BASIS_METHODS)}."
            )
    return raw_rule


def _validated_demo_reconstruction_roles(roles: Mapping[object, object]) -> None:
    """Validate demo return-reconstruction role definitions."""
    unknown_roles = set(roles) - {
        "security_flow",
        "income",
        "portfolio_external_flow",
    }
    if unknown_roles:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: unsupported reconstruction "
            f"roles {sorted(unknown_roles)}."
        )
    for role_name in ("security_flow", "income"):
        codes = roles.get(role_name)
        if not isinstance(codes, list) or any(
            not isinstance(code, str) or not code for code in codes
        ):
            raise ValueError(
                f"{_DEFAULT_TRANSACTION_POLICY_PATH}: {role_name} must be a "
                "list of transaction codes."
            )
    external_rules = roles.get("portfolio_external_flow")
    if not isinstance(external_rules, dict) or not external_rules:
        raise ValueError(
            f"{_DEFAULT_TRANSACTION_POLICY_PATH}: portfolio_external_flow "
            "must be nonempty."
        )
    for transaction_code, raw_rule in external_rules.items():
        _validated_demo_rule(transaction_code, raw_rule, require_effect=False)


def _demo_holding_effect(row: object) -> tuple[str, float, str] | None:
    """Return the configured generic holding effect for one raw transaction."""
    effects = _demo_transaction_policy()["holding_effects"]
    assert isinstance(effects, dict)
    raw_rule = effects.get(_row_string(row, "TRAN").lower())
    if not isinstance(raw_rule, dict) or not _demo_rule_matches(row, raw_rule):
        return None
    return (
        str(raw_rule["effect"]),
        float(raw_rule.get("quantity_multiplier", 1)),
        str(raw_rule.get("cost_basis", "market_value")),
    )


def _demo_reconstruction_codes(role_name: str) -> frozenset[str]:
    """Return configured transaction codes for one static reconstruction role."""
    roles = _demo_transaction_policy()["reconstruction_roles"]
    assert isinstance(roles, dict)
    raw_codes = roles[role_name]
    assert isinstance(raw_codes, list)
    return frozenset(str(code).lower() for code in raw_codes)


def _demo_external_flow_rule_matches(row: object) -> bool:
    """Return whether the configured contextual external-flow rule matches."""
    roles = _demo_transaction_policy()["reconstruction_roles"]
    assert isinstance(roles, dict)
    raw_rules = roles["portfolio_external_flow"]
    assert isinstance(raw_rules, dict)
    raw_rule = raw_rules.get(_row_string(row, "TRAN").lower())
    return isinstance(raw_rule, dict) and _demo_rule_matches(row, raw_rule)


def _demo_rule_matches(row: object, rule: Mapping[str, object]) -> bool:
    """Return whether one generic demo-policy condition matches a raw row."""
    when = rule.get("when", {})
    assert isinstance(when, dict)
    return all(
        _row_string(row, _DEMO_CONTEXT_COLUMNS[field]).strip().casefold()
        == str(expected).strip().casefold()
        for field, expected in when.items()
    )


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
        update only the cash balance. Fixed-income accrued-interest adjuncts
        update cash without changing principal quantity or accrued holdings.
        Corporate-action quantity corrections update only the affected security
        holding.
    """
    if snapshot_name == _BASE_SNAPSHOT_DIRECTORY:
        return ()

    base_prepared = _prepared_transactions(base_transactions)
    current_prepared = _prepared_transactions(current_transactions)
    # Portfolio return reconstruction uses base-currency transaction amounts,
    # while currency-specific cash holdings roll forward in their local currency.
    # Restore the preserved local amounts only for deriving holding adjustments.
    base_prepared["AMOUNT"] = base_prepared["LOCAL_AMOUNT"]
    current_prepared["AMOUNT"] = current_prepared["LOCAL_AMOUNT"]
    transaction_diffs = _changed_transaction_rows(base_prepared, current_prepared)
    holdings = _holding_values(_with_internal_cost(base_holdings))
    adjustments: list[HoldingScenarioAdjustment] = []
    for row in transaction_diffs.itertuples(index=False):
        transaction_code = str(row.TRAN)
        configured_effect = _demo_holding_effect(row)
        if configured_effect is None:
            continue
        effect, quantity_multiplier, cost_basis_method = configured_effect
        holding_dates = _holding_dates_for_transaction_effect(
            periods,
            str(row.PORT),
            row.TRANSACTION_DATE,
        )
        for holding_date in holding_dates:
            if effect == "security_trade_and_cash":
                adjustments.append(
                    _security_trade_adjustment(
                        snapshot_name,
                        holdings=holdings,
                        cost_basis_method=cost_basis_method,
                        portfolio=str(row.PORT),
                        security=str(row.SEC),
                        holding_date=holding_date,
                        quantity_delta=quantity_multiplier * float(row.QTY_delta),
                        scenario=(
                            f"{row.TRANSACTION_ID} {transaction_code} transaction "
                            "changes ending holding."
                        ),
                    )
                )
                adjustments.append(
                    _cash_adjustment(
                        snapshot_name,
                        portfolio=str(row.PORT),
                        holding_date=holding_date,
                        cash_delta=float(row.AMOUNT_delta),
                        base_cash_delta=float(row.BASE_AMOUNT_delta),
                        cash_security=_cash_security_for_transaction(row),
                        scenario=(
                            f"{row.TRANSACTION_ID} {transaction_code} transaction "
                            "changes cash balance."
                        ),
                    )
                )
            elif effect == "security_quantity":
                adjustments.append(
                    _security_trade_adjustment(
                        snapshot_name,
                        holdings=holdings,
                        cost_basis_method=cost_basis_method,
                        portfolio=str(row.PORT),
                        security=str(row.SEC),
                        holding_date=holding_date,
                        quantity_delta=quantity_multiplier * float(row.QTY_delta),
                        scenario=(
                            f"{row.TRANSACTION_ID} {transaction_code} transaction "
                            "changes ending holding."
                        ),
                    )
                )
            elif effect == "principal_and_cash":
                principal_delta = -float(row.AMOUNT_delta)
                adjustments.append(
                    _principal_paydown_adjustment(
                        snapshot_name,
                        holdings=holdings,
                        portfolio=str(row.PORT),
                        security=str(row.SEC),
                        holding_date=holding_date,
                        principal_delta=principal_delta,
                        scenario=(
                            f"{row.TRANSACTION_ID} {transaction_code} transaction "
                            "changes ending holding."
                        ),
                    )
                )
                adjustments.append(
                    _cash_adjustment(
                        snapshot_name,
                        portfolio=str(row.PORT),
                        holding_date=holding_date,
                        cash_delta=float(row.AMOUNT_delta),
                        base_cash_delta=float(row.BASE_AMOUNT_delta),
                        cash_security=_cash_security_for_transaction(row),
                        scenario=(
                            f"{row.TRANSACTION_ID} {transaction_code} transaction "
                            "changes cash balance."
                        ),
                    )
                )
            elif effect == "cash":
                adjustments.append(
                    _cash_adjustment(
                        snapshot_name,
                        portfolio=str(row.PORT),
                        holding_date=holding_date,
                        cash_delta=float(row.AMOUNT_delta),
                        base_cash_delta=float(row.BASE_AMOUNT_delta),
                        cash_security=_cash_security_for_transaction(row),
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
        "BASE_AMOUNT",
        "COMMISSION",
    ]
    context_columns = [
        column
        for column in (
            "SEC_TYPE",
            "SRC_DEST_TYPE",
            "SRC_DEST_SYMBOL",
            "SPECIAL_SEC_TYPE",
            "SPECIAL_SEC_SYMBOL",
            "CURRENCY",
            "BASE_CURRENCY",
        )
        if column in base_transactions.columns and column in current_transactions.columns
    ]
    compare_columns.extend(context_columns)
    base = base_transactions[compare_columns]
    current = current_transactions[compare_columns]
    merged = base.merge(
        current,
        on="TRANSACTION_ID",
        how="outer",
        suffixes=("_base", "_current"),
        indicator=True,
    )
    merged = merged.rename(columns={"_merge": "MERGE_STATUS"})

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        merge_status = str(row.MERGE_STATUS)
        if merge_status == "left_only":
            context_values = {
                column: _row_string(row, f"{column}_base") for column in context_columns
            }
            rows.append(
                {
                    "TRANSACTION_ID": str(row.TRANSACTION_ID),
                    "PORT": str(row.PORT_base),
                    "TRANSACTION_DATE": pd.Timestamp(row.TRANSACTION_DATE_base),
                    "SEC": str(row.SEC_base),
                    "TRAN": str(row.TRAN_base),
                    **context_values,
                    **{
                        f"{column}_delta": -float(getattr(row, f"{column}_base"))
                        for column in _TRANSACTION_NUMERIC_COLUMNS
                    },
                }
            )
            continue
        if merge_status == "right_only":
            context_values = {
                column: _row_string(row, f"{column}_current") for column in context_columns
            }
            rows.append(
                {
                    "TRANSACTION_ID": str(row.TRANSACTION_ID),
                    "PORT": str(row.PORT_current),
                    "TRANSACTION_DATE": pd.Timestamp(row.TRANSACTION_DATE_current),
                    "SEC": str(row.SEC_current),
                    "TRAN": str(row.TRAN_current),
                    **context_values,
                    **{
                        f"{column}_delta": float(getattr(row, f"{column}_current"))
                        for column in _TRANSACTION_NUMERIC_COLUMNS
                    },
                }
            )
            continue

        base_port = str(row.PORT_base)
        current_port = str(row.PORT_current)
        base_security = str(row.SEC_base)
        current_security = str(row.SEC_current)
        base_code = str(row.TRAN_base)
        current_code = str(row.TRAN_current)
        context_values: dict[str, str] = {}
        for column in context_columns:
            base_value = _row_string(row, f"{column}_base")
            current_value = _row_string(row, f"{column}_current")
            if base_value != current_value:
                raise ValueError(
                    "Transaction scenario rows may change numeric fields only: "
                    f"{row.TRANSACTION_ID}."
                )
            context_values[column] = base_value
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
            for column in _TRANSACTION_NUMERIC_COLUMNS
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
                **context_values,
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


def _holding_dates_for_transaction_effect(
    periods: pd.DataFrame,
    portfolio_code: str,
    transaction_date: pd.Timestamp,
) -> tuple[str, ...]:
    """Return same-month holding dates affected by one transaction change.

    Notes:
        The operational demo rebuild is intentionally narrow. A transaction
        affects the period-end holding date that contains it plus any later
        holding dates in the same calendar month. This lets intra-month demo
        periods carry a buy/sell/cash effect forward without changing the
        legacy month-to-month fixture model.
    """
    primary_holding_date = _period_end_for_transaction(
        periods,
        portfolio_code,
        transaction_date,
    )
    primary_timestamp = pd.Timestamp(primary_holding_date)
    transaction_month = transaction_date.to_period("M")
    period_rows = periods[
        periods["PORTFOLIO_CODE"].eq(portfolio_code)
        & (pd.to_datetime(periods["THRU_DATE"]).dt.to_period("M") == transaction_month)
        & pd.to_datetime(periods["THRU_DATE"]).ge(primary_timestamp)
    ].copy()
    holding_dates = sorted(
        {pd.Timestamp(thru_date).date().isoformat() for thru_date in period_rows["THRU_DATE"]}
    )
    if primary_holding_date not in holding_dates:
        raise ValueError(
            "Transaction effect did not include its primary holding date: "
            f"{portfolio_code}/{transaction_date.date()}."
        )
    return tuple(holding_dates)


def _security_trade_adjustment(
    snapshot: str,
    *,
    holdings: pd.DataFrame,
    cost_basis_method: str,
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
    accrued = float(rows.iloc[0]["ACCRUED"])
    cost_per_share = cost / quantity if quantity else 0.0
    accrued_per_share = accrued / quantity if quantity else 0.0
    market_value_delta = quantity_delta * price
    cost_delta = market_value_delta
    if cost_basis_method == "proportional_existing":
        cost_delta = quantity_delta * cost_per_share
    accrued_delta = quantity_delta * accrued_per_share
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
            "BASE_MKT_VAL": market_value_delta,
            "COST": cost_delta,
            "ACCRUED": accrued_delta,
        },
        scenario=scenario,
    )


def _principal_paydown_adjustment(
    snapshot: str,
    *,
    holdings: pd.DataFrame,
    portfolio: str,
    security: str,
    holding_date: str,
    principal_delta: float,
    scenario: str,
) -> HoldingScenarioAdjustment:
    """Return the holding impact for a principal paydown with unchanged quantity."""
    rows = holdings[
        holdings["PORT"].eq(portfolio)
        & holdings["SEC"].eq(security)
        & holdings["HOLDING_DATE"].eq(pd.Timestamp(holding_date))
    ]
    if rows.shape[0] != 1:
        raise ValueError(
            "Transaction-derived principal adjustment must match one holding: "
            f"{portfolio}/{security}/{holding_date}."
        )
    return HoldingScenarioAdjustment(
        snapshot=snapshot,
        portfolio=portfolio,
        security=security,
        holding_date=holding_date,
        scenario_type="transaction_derived",
        deltas={
            "QTY": 0.0,
            "PRICE": 0.0,
            "MKT_VAL": principal_delta,
            "BASE_MKT_VAL": principal_delta,
            "COST": principal_delta,
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
    base_cash_delta: float,
    cash_security: str,
    scenario: str,
) -> HoldingScenarioAdjustment:
    """Return the cash holding impact for a changed cash-affecting transaction."""
    return HoldingScenarioAdjustment(
        snapshot=snapshot,
        portfolio=portfolio,
        security=cash_security,
        holding_date=holding_date,
        scenario_type="transaction_derived",
        deltas={
            "QTY": cash_delta,
            "PRICE": 0.0,
            "MKT_VAL": cash_delta,
            "BASE_MKT_VAL": base_cash_delta,
            "COST": cash_delta,
            "ACCRUED": 0.0,
        },
        scenario=scenario,
    )


def _cash_security_for_transaction(row: object) -> str:
    """Return the currency-specific cash security for a transaction diff row."""
    source_symbol = _row_string(row, "SRC_DEST_SYMBOL")
    if source_symbol in _CASH_SECURITY_IDS.values():
        return source_symbol
    currency = _row_string(row, "CURRENCY") or "USD"
    return _CASH_SECURITY_IDS.get(currency, "CASHUSD")


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
    scenarios = pd.read_csv(path, keep_default_na=False)
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
    scenarios["BASE_MKT_VAL_delta"] = scenarios["MKT_VAL_delta"]

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

    delta_columns = [f"{column}_delta" for column in _INTERNAL_HOLDINGS_NUMERIC_COLUMNS]
    converted_deltas = scenarios[delta_columns].apply(pd.to_numeric, errors="coerce")
    if bool(converted_deltas.isna().any().any()):
        raise ValueError("Holding scenario delta columns must be numeric.")

    adjustments: list[HoldingScenarioAdjustment] = []
    for row_index, row in scenarios.iterrows():
        deltas = {
            column: float(converted_deltas.loc[row_index, f"{column}_delta"])
            for column in _INTERNAL_HOLDINGS_NUMERIC_COLUMNS
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
        "valuation_mark": {"PRICE", "MKT_VAL", "BASE_MKT_VAL"},
        "cash_balance_correction": {"QTY", "MKT_VAL", "BASE_MKT_VAL", "COST"},
        "quantity_valuation_correction": {
            "QTY",
            "MKT_VAL",
            "BASE_MKT_VAL",
            "COST",
        },
        "accrual_correction": {
            "QTY",
            "MKT_VAL",
            "BASE_MKT_VAL",
            "COST",
            "ACCRUED",
        },
        "cost_only_correction": {"COST"},
        "data_issues_holdings_accrued_rate": {"ACCRUED"},
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
    scenarios = pd.read_csv(path, keep_default_na=False)
    legacy_columns = [
        column
        for column in _TRANSACTION_SCENARIO_COLUMNS
        if column not in {"CURRENCY", "BASE_CURRENCY", "BASE_AMOUNT", "BASE_AMOUNT_delta"}
    ]
    if tuple(scenarios.columns) not in {
        tuple(_TRANSACTION_SCENARIO_COLUMNS),
        tuple(legacy_columns),
    }:
        raise ValueError(
            "Transaction scenario CSV columns must exactly match either the "
            f"current or legacy fixture columns. Actual={list(scenarios.columns)}."
        )
    if "CURRENCY" not in scenarios.columns:
        scenarios["CURRENCY"] = "USD"
    else:
        scenarios["CURRENCY"] = scenarios["CURRENCY"].replace("", "USD")
    if "BASE_CURRENCY" not in scenarios.columns:
        scenarios["BASE_CURRENCY"] = "USD"
    else:
        scenarios["BASE_CURRENCY"] = scenarios["BASE_CURRENCY"].replace("", "USD")
    if "BASE_AMOUNT" not in scenarios.columns:
        scenarios["BASE_AMOUNT"] = scenarios["AMOUNT"]
    else:
        scenarios["BASE_AMOUNT"] = scenarios["BASE_AMOUNT"].where(
            scenarios["BASE_AMOUNT"].astype(str).str.strip().ne(""),
            scenarios["AMOUNT"],
        )
    if "BASE_AMOUNT_delta" not in scenarios.columns:
        scenarios["BASE_AMOUNT_delta"] = scenarios["AMOUNT_delta"]
    else:
        scenarios["BASE_AMOUNT_delta"] = scenarios["BASE_AMOUNT_delta"].where(
            scenarios["BASE_AMOUNT_delta"].astype(str).str.strip().ne(""),
            scenarios["AMOUNT_delta"],
        )
    missing_columns = [
        column for column in _TRANSACTION_SCENARIO_COLUMNS if column not in scenarios.columns
    ]
    extra_columns = [
        column for column in scenarios.columns if column not in _TRANSACTION_SCENARIO_COLUMNS
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
        duplicates = scenarios.loc[duplicate_keys, _TRANSACTION_SCENARIO_KEY].to_dict("records")
        raise ValueError(f"Duplicate transaction scenario keys are not allowed: {duplicates}.")

    supported_snapshots = set(_SNAPSHOT_DIRECTORIES) - {_BASE_SNAPSHOT_DIRECTORY}
    unknown_snapshots = sorted(set(scenarios["snapshot"]) - supported_snapshots)
    if unknown_snapshots:
        raise ValueError(
            "Transaction scenarios may only target derived snapshots. "
            f"Unsupported snapshots={unknown_snapshots}."
        )
    unknown_actions = sorted(set(scenarios["action"]) - _TRANSACTION_SCENARIO_ACTIONS)
    if unknown_actions:
        raise ValueError(
            "Transaction scenario action must be one of "
            f"{sorted(_TRANSACTION_SCENARIO_ACTIONS)}. "
            f"Unsupported actions={unknown_actions}."
        )

    delta_columns = [f"{column}_delta" for column in _TRANSACTION_NUMERIC_COLUMNS]
    converted_deltas = scenarios[delta_columns].apply(pd.to_numeric, errors="coerce")
    if bool(converted_deltas.isna().any().any()):
        raise ValueError("Transaction scenario delta columns must be numeric.")
    converted_values = scenarios[_TRANSACTION_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    insert_mask = scenarios["action"].eq("insert")
    if bool(converted_values.loc[insert_mask].isna().any().any()):
        raise ValueError("Inserted transaction scenario numeric values must be numeric.")

    adjustments: list[TransactionScenarioAdjustment] = []
    for row_index, row in scenarios.iterrows():
        action = str(row["action"])
        deltas = {
            column: float(converted_deltas.loc[row_index, f"{column}_delta"])
            for column in _TRANSACTION_NUMERIC_COLUMNS
        }
        if action == "adjust" and not any(deltas.values()):
            raise ValueError(
                "Transaction scenario rows must change at least one numeric value: "
                f"{row['TRANSACTION_ID']}."
            )
        values: dict[str, object] = {}
        if action == "insert":
            required_insert_columns = [
                "PORT",
                "TRANSACTION_DATE",
                "SETTLE_DATE",
                "SEC",
                "TRAN",
                "SEC_TYPE",
            ]
            blank_columns = [
                column for column in required_insert_columns if not str(row[column]).strip()
            ]
            if blank_columns:
                raise ValueError(
                    "Inserted transaction scenarios require transaction values: "
                    f"{row['TRANSACTION_ID']}; missing={blank_columns}."
                )
            values = {column: str(row[column]) for column in _TRANSACTION_SOURCE_COLUMNS}
            values["TRANSACTION_ID"] = str(row["TRANSACTION_ID"])
            values.update(
                {
                    column: float(converted_values.loc[row_index, column])
                    for column in _TRANSACTION_NUMERIC_COLUMNS
                }
            )
        adjustments.append(
            TransactionScenarioAdjustment(
                snapshot=str(row["snapshot"]),
                action=action,
                transaction_id=str(row["TRANSACTION_ID"]),
                values=values,
                deltas=deltas,
                scenario=str(row["scenario"]),
            )
        )
    return TransactionScenarioSet(tuple(adjustments), path)


def _load_scenario_calendar(path: Path) -> pd.DataFrame:
    """Return the validated scenario calendar used by demo simplification work.

    Args:
        path: CSV mapping each intentional scenario row to a portfolio period
            and short scenario family.

    Returns:
        Validated scenario calendar rows in file order.

    Raises:
        ValueError: If the CSV has unexpected columns, duplicate keys, unknown
            sources, invalid dates, or invalid row-count expectations.
    """
    calendar = pd.read_csv(path, keep_default_na=False)
    missing_columns = [
        column for column in _SCENARIO_CALENDAR_COLUMNS if column not in calendar.columns
    ]
    extra_columns = [
        column for column in calendar.columns if column not in _SCENARIO_CALENDAR_COLUMNS
    ]
    if missing_columns or extra_columns:
        raise ValueError(
            "Scenario calendar CSV columns must exactly match "
            f"{_SCENARIO_CALENDAR_COLUMNS}. "
            f"Missing={missing_columns}; extra={extra_columns}."
        )

    required_text_columns = [
        column
        for column in _SCENARIO_CALENDAR_COLUMNS
        if column not in _SCENARIO_CALENDAR_NUMERIC_COLUMNS
    ]
    blank_rows = calendar[required_text_columns].map(lambda value: not str(value).strip())
    if bool(blank_rows.any().any()):
        raise ValueError("Scenario calendar rows must not have blank key text values.")

    duplicate_keys = calendar.duplicated([_SCENARIO_CALENDAR_KEY], keep=False)
    if bool(duplicate_keys.any()):
        duplicates = calendar.loc[
            duplicate_keys,
            [_SCENARIO_CALENDAR_KEY],
        ].to_dict("records")
        raise ValueError(f"Duplicate scenario calendar keys are not allowed: {duplicates}.")

    unknown_sources = sorted(set(calendar["scenario_source"]) - _SCENARIO_CALENDAR_SOURCES)
    if unknown_sources:
        raise ValueError(
            "Scenario calendar source must be one of "
            f"{sorted(_SCENARIO_CALENDAR_SOURCES)}. "
            f"Unsupported sources={unknown_sources}."
        )

    for date_column in ("from_date", "thru_date"):
        parsed_dates = pd.to_datetime(calendar[date_column], errors="coerce")
        if bool(parsed_dates.isna().any()):
            raise ValueError(f"Scenario calendar {date_column} values must be dates.")

    converted_counts = calendar[_SCENARIO_CALENDAR_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if bool(converted_counts.isna().any().any()):
        raise ValueError("Scenario calendar expected row counts must be numeric.")
    calendar = calendar.copy()
    for column in _SCENARIO_CALENDAR_NUMERIC_COLUMNS:
        calendar[column] = converted_counts[column].astype(int)
    invalid_counts = (calendar["current_expected_difference_rows"] <= 0) | (
        calendar["future_max_expected_differences"] < calendar["current_expected_difference_rows"]
    )
    if bool(invalid_counts.any()):
        raise ValueError(
            "Scenario calendar future row limits must be positive and at least "
            "the current expected row count."
        )
    return calendar


def _load_scenario_inventory(path: Path) -> pd.DataFrame:
    """Return the independent semantic contract for named demo scenarios.

    Args:
        path: CSV containing every protected scenario and its expected meaning.

    Returns:
        Validated protected scenario rows in file order.

    Raises:
        ValueError: If columns, values, dates, counts, or keys are invalid.
    """
    inventory = pd.read_csv(path, keep_default_na=False)
    if list(inventory.columns) != _SCENARIO_INVENTORY_COLUMNS:
        raise ValueError(
            "Scenario inventory CSV columns must exactly match "
            f"{_SCENARIO_INVENTORY_COLUMNS}; actual={list(inventory.columns)}."
        )
    text_columns = [
        column
        for column in _SCENARIO_INVENTORY_COLUMNS
        if column not in _SCENARIO_INVENTORY_NUMERIC_COLUMNS
    ]
    blank_rows = inventory[text_columns].map(
        lambda value: not str(value).strip()
    )
    if bool(blank_rows.any().any()):
        raise ValueError("Scenario inventory text values must not be blank.")
    duplicate_keys = inventory.duplicated(["scenario_key"], keep=False)
    if bool(duplicate_keys.any()):
        duplicates = inventory.loc[duplicate_keys, ["scenario_key"]].to_dict("records")
        raise ValueError(f"Duplicate protected scenario keys are not allowed: {duplicates}.")

    inventory = inventory.copy()
    date_columns = [
        "source_from_date",
        "source_thru_date",
        "story_from_date",
        "story_thru_date",
    ]
    for column in date_columns:
        parsed_dates = pd.to_datetime(inventory[column], errors="coerce")
        if bool(parsed_dates.isna().any()):
            raise ValueError(f"Scenario inventory {column} values must be dates.")
        canonical_dates = parsed_dates.dt.strftime("%Y-%m-%d")
        if not canonical_dates.equals(inventory[column].astype(str)):
            raise ValueError(
                f"Scenario inventory {column} values must use YYYY-MM-DD format."
            )

    converted_counts = pd.to_numeric(
        inventory["source_period_independent_changes"],
        errors="coerce",
    )
    if bool(converted_counts.isna().any()) or not bool(
        converted_counts.mod(1).eq(0).all()
    ):
        raise ValueError("Scenario independent-change counts must be whole numbers.")
    inventory["source_period_independent_changes"] = converted_counts.astype(int)
    invalid_counts = (
        inventory["source_period_independent_changes"].le(0)
        | inventory["source_period_independent_changes"].gt(
            _SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES
        )
    )
    if bool(invalid_counts.any()):
        raise ValueError(
            "Scenario independent-change counts must be between 1 and "
            f"{_SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES}."
        )

    _validate_scenario_inventory_enum(
        inventory,
        "expected_report_disposition",
        _SCENARIO_REPORT_DISPOSITIONS,
    )
    _validate_scenario_inventory_enum(
        inventory,
        "expected_period_status",
        _SCENARIO_PERIOD_STATUSES,
    )
    _validate_scenario_inventory_enum(
        inventory,
        "carry_forward_status",
        _SCENARIO_CARRY_FORWARD_STATUSES,
    )
    invalid_source_periods = inventory["source_from_date"].gt(
        inventory["source_thru_date"]
    )
    invalid_story_periods = inventory["story_from_date"].gt(
        inventory["story_thru_date"]
    )
    if bool(invalid_source_periods.any()) or bool(invalid_story_periods.any()):
        raise ValueError("Scenario inventory periods must not end before they begin.")

    same_period = (
        inventory["source_from_date"].eq(inventory["story_from_date"])
        & inventory["source_thru_date"].eq(inventory["story_thru_date"])
    )
    originating = inventory["carry_forward_status"].eq("originating_change")
    if not bool(same_period.eq(originating).all()):
        raise ValueError(
            "Originating scenarios must use the same source and story period; "
            "different periods must be declared carry_forward_effect."
        )
    carry_rows = inventory["carry_forward_status"].eq("carry_forward_effect")
    if bool(
        inventory.loc[carry_rows, "source_thru_date"]
        .ge(inventory.loc[carry_rows, "story_from_date"])
        .any()
    ):
        raise ValueError(
            "Carry-forward source periods must end before their story period begins."
        )
    return inventory


def _validate_scenario_inventory_enum(
    inventory: pd.DataFrame,
    column: str,
    allowed_values: set[str],
) -> None:
    """Raise when a scenario-contract column contains unsupported values."""
    unknown_values = sorted(set(inventory[column].astype(str)) - allowed_values)
    if unknown_values:
        raise ValueError(
            f"Scenario inventory {column} values must be one of "
            f"{sorted(allowed_values)}; unsupported={unknown_values}."
        )


def _load_period_split_plan(path: Path) -> pd.DataFrame:
    """Return the validated intra-month period split backlog.

    Args:
        path: CSV mapping any crowded calendar scenario rows to proposed
            shorter periods. An empty file means the current demo has no known
            crowded-period backlog.

    Returns:
        Validated split-backlog rows in file order.

    Raises:
        ValueError: If columns, dates, or planned row counts are invalid.
    """
    plan = pd.read_csv(path, keep_default_na=False)
    missing_columns = [
        column for column in _PERIOD_SPLIT_PLAN_COLUMNS if column not in plan.columns
    ]
    extra_columns = [column for column in plan.columns if column not in _PERIOD_SPLIT_PLAN_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            "Period split plan CSV columns must exactly match "
            f"{_PERIOD_SPLIT_PLAN_COLUMNS}. "
            f"Missing={missing_columns}; extra={extra_columns}."
        )

    required_text_columns = [
        column
        for column in _PERIOD_SPLIT_PLAN_COLUMNS
        if column not in _PERIOD_SPLIT_PLAN_NUMERIC_COLUMNS
    ]
    blank_rows = plan[required_text_columns].map(lambda value: not str(value).strip())
    if bool(blank_rows.any().any()):
        raise ValueError("Period split plan rows must not have blank key text values.")
    duplicate_keys = plan.duplicated(["scenario_key"], keep=False)
    if bool(duplicate_keys.any()):
        duplicates = plan.loc[duplicate_keys, ["scenario_key"]].to_dict("records")
        raise ValueError(f"Duplicate period split plan keys are not allowed: {duplicates}.")

    for date_column in (
        "current_from_date",
        "current_thru_date",
        "planned_from_date",
        "planned_thru_date",
    ):
        parsed_dates = pd.to_datetime(plan[date_column], errors="coerce")
        if bool(parsed_dates.isna().any()):
            raise ValueError(f"Period split plan {date_column} values must be dates.")

    converted_counts = plan[_PERIOD_SPLIT_PLAN_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if bool(converted_counts.isna().any().any()):
        raise ValueError("Period split plan expected row counts must be numeric.")
    plan = plan.copy()
    for column in _PERIOD_SPLIT_PLAN_NUMERIC_COLUMNS:
        plan[column] = converted_counts[column].astype(int)
    if bool((plan["planned_difference_rows"] <= 0).any()):
        raise ValueError("Period split plan row counts must be positive.")
    return plan


def _rounded_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings rounded to the packaged Axys/APX fixture precision."""
    rounded = holdings.copy()
    rounded["QTY"] = rounded["QTY"].astype(float).round(4)
    rounded["PRICE"] = rounded["PRICE"].astype(float).round(4)
    for column in ("MKT_VAL", "BASE_MKT_VAL", "COST", "ACCRUED"):
        rounded[column] = rounded[column].astype(float).round(2)
    return rounded


def _rebuild_security_performance(
    snapshot_name: str,
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
        if begin_date is None:
            rows.append(row._asdict())
            continue
        begin_value = _security_holding_value(
            holding_values,
            row.PORTFOLIO_CODE,
            row.SECURITY_ID,
            begin_date,
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
        rebuilt_row["SEC_RETURN"] = round(sec_return, 10)
        rows.append(rebuilt_row)

    rebuilt = pd.DataFrame(rows)
    if snapshot_name != _BASE_SNAPSHOT_DIRECTORY:
        rebuilt = _with_intentional_security_return_residuals(rebuilt)
    return rebuilt[secperf.columns]


def _rebuild_portfolio_performance(
    snapshot_name: str,
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
        if begin_date is None:
            rows.append(row._asdict())
            continue
        begin_value = _portfolio_holding_value(
            holding_values,
            row.PORTFOLIO_CODE,
            begin_date,
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
        numerator = end_value - begin_value - flow
        denominator = begin_value + weighted_flow
        rebuilt_row = row._asdict()
        rebuilt_row["PORT_RETURN"] = round(
            numerator / denominator if denominator else 0.0,
            10,
        )
        rows.append(rebuilt_row)
    rebuilt = pd.DataFrame(rows)
    if snapshot_name != _BASE_SNAPSHOT_DIRECTORY:
        rebuilt = _with_intentional_portfolio_return_residuals(rebuilt)
    return rebuilt[_PORTPERF_COLUMNS]


def _with_intentional_portfolio_return_residuals(portperf: pd.DataFrame) -> pd.DataFrame:
    """Return portfolio performance with explicit reported-return residuals."""
    adjusted = portperf.copy()
    for (
        portfolio,
        from_date,
        thru_date,
    ), residual in _INTENTIONAL_PORTFOLIO_RETURN_RESIDUALS.items():
        mask = (
            adjusted["PORTFOLIO_CODE"].eq(portfolio)
            & adjusted["FROM_DATE"].eq(from_date)
            & adjusted["THRU_DATE"].eq(thru_date)
        )
        if int(mask.sum()) != 1:
            raise ValueError(
                "Intentional portfolio residual must match one portperf row: "
                f"{portfolio}/{from_date}/{thru_date}."
            )
        adjusted.loc[mask, "PORT_RETURN"] = (
            adjusted.loc[mask, "PORT_RETURN"].astype(float) + residual
        ).round(10)
    return adjusted


def _with_intentional_security_return_residuals(secperf: pd.DataFrame) -> pd.DataFrame:
    """Return security performance with explicit reported-return residuals."""
    adjusted = secperf.copy()
    for (
        portfolio,
        security,
        from_date,
        thru_date,
    ), residual in _INTENTIONAL_SECURITY_RETURN_RESIDUALS.items():
        mask = (
            adjusted["PORTFOLIO_CODE"].eq(portfolio)
            & adjusted["SECURITY_ID"].eq(security)
            & adjusted["FROM_DATE"].eq(from_date)
            & adjusted["THRU_DATE"].eq(thru_date)
        )
        if int(mask.sum()) != 1:
            raise ValueError(
                "Intentional security residual must match one secperf row: "
                f"{portfolio}/{security}/{from_date}/{thru_date}."
            )
        adjusted.loc[mask, "SEC_RETURN"] = (
            adjusted.loc[mask, "SEC_RETURN"].astype(float) + residual
        ).round(10)
    return adjusted


def _holding_values(holdings: pd.DataFrame) -> pd.DataFrame:
    """Return holdings with normalized dates and total holding value."""
    values = holdings.copy()
    values["HOLDING_DATE"] = pd.to_datetime(values["HOLDING_DATE"])
    values["HOLDING_VALUE"] = values["BASE_MKT_VAL"].astype(float) + values["ACCRUED"].astype(
        float
    )
    return values


def _prepared_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with normalized dates and transaction codes."""
    prepared = transactions.copy()
    prepared["TRANSACTION_DATE"] = pd.to_datetime(prepared["TRANSACTION_DATE"])
    prepared["TRAN"] = prepared["TRAN"].astype(str)
    prepared["LOCAL_AMOUNT"] = prepared["AMOUNT"].astype(float)
    prepared["AMOUNT"] = prepared["BASE_AMOUNT"].astype(float)
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
        raise ValueError(f"Missing holding for {portfolio_code}/{security_id} on {holding_date}.")
    return float(rows["HOLDING_VALUE"].sum())


def _security_flows(
    transactions: pd.DataFrame,
    portfolio_code: str,
    security_id: str,
    from_date: pd.Timestamp,
    thru_date: pd.Timestamp,
    reconstruction: SecurityReturnReconstruction,
) -> tuple[float, float]:
    """Return net and weighted configured security-level flows."""
    rows = _period_transactions(transactions, portfolio_code, from_date, thru_date)
    rows = rows[
        rows["SEC"].eq(security_id)
        & rows["TRAN"].isin(_demo_reconstruction_codes("security_flow"))
    ]
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
        & rows["TRAN"].isin(
            set(reconstruction.income_categories)
            | _demo_reconstruction_codes("income")
        )
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
    net_flow = 0.0
    weighted_flow = 0.0
    for row in rows.itertuples(index=False):
        if not _is_portfolio_external_flow(row, reconstruction):
            continue
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
    rows = rows[
        rows["TRAN"].isin(
            set(reconstruction.income_categories)
            | _demo_reconstruction_codes("income")
        )
    ]
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


def _is_portfolio_external_flow(
    row: object,
    reconstruction: PortfolioReturnReconstruction,
) -> bool:
    """Return whether one raw Axys/APX demo transaction is a portfolio external flow."""
    transaction_code = _row_string(row, "TRAN").lower()
    if transaction_code in set(reconstruction.flow_categories):
        return True
    return _demo_external_flow_rule_matches(row)


def _row_string(row: object, field: str) -> str:
    """Return a string field value from a pandas namedtuple row."""
    value = getattr(row, field, "")
    if pd.isna(value):
        return ""
    return str(value).strip()


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
    findings = compare_snapshots(
        comparison_path,
        comparison_level="portfolio",
    )
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
                issues.append(_portfolio_residual_issue(row, "Fully explained row has residual."))
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
                issues.append(_security_residual_issue(row, "Fully explained row has residual."))
            continue
        key = (
            str(row["portfolio_id"]),
            _security_symbol_from_ppar_id(row["security_id"]),
            row["from_date"].isoformat(),
            row["thru_date"].isoformat(),
            status,
        )
        if key not in _INTENTIONAL_SECURITY_RESIDUALS:
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


def _audit_scenario_calendar(
    *,
    calendar: pd.DataFrame,
    holding_scenarios: HoldingScenarioSet,
    transaction_scenarios: TransactionScenarioSet,
    axys_directory: Path,
) -> list[AuditIssue]:
    """Return issues when scenario rows are missing calendar period guardrails.

    The calendar is a planning/audit layer for the Axys/APX demo simplification.
    It does not drive rebuild behavior yet; it records where each intentional
    source-data difference currently appears and which periods should be split
    in later phases.
    """
    issues: list[AuditIssue] = []
    expected_counts: dict[str, int] = {}
    for adjustment in transaction_scenarios.adjustments:
        key = _transaction_scenario_calendar_key(adjustment)
        expected_counts[key] = expected_counts.get(key, 0) + 1
    for adjustment in holding_scenarios.adjustments:
        key = _holding_scenario_calendar_key(adjustment)
        expected_counts[key] = expected_counts.get(key, 0) + 1
    expected_counts.update(_MULTICURRENCY_SCENARIO_CALENDAR_KEYS)

    calendar_counts = {
        str(row.scenario_key): int(row.current_expected_difference_rows)
        for row in calendar.itertuples(index=False)
    }
    for key in sorted(set(expected_counts) - set(calendar_counts)):
        issues.append(
            AuditIssue(
                check="scenario_calendar",
                detail=f"Scenario row is missing from the simplification calendar: {key}.",
            )
        )
    for key in sorted(set(calendar_counts) - set(expected_counts)):
        issues.append(
            AuditIssue(
                check="scenario_calendar",
                detail=f"Scenario calendar key no longer matches a scenario row: {key}.",
            )
        )
    for key in sorted(set(expected_counts) & set(calendar_counts)):
        if expected_counts[key] != calendar_counts[key]:
            issues.append(
                AuditIssue(
                    check="scenario_calendar",
                    detail=(
                        "Scenario calendar expected row count changed: "
                        f"{key}; expected={expected_counts[key]}; "
                        f"calendar={calendar_counts[key]}."
                    ),
                )
            )

    period_keys = _demo_portfolio_period_keys(axys_directory)
    for row in calendar.itertuples(index=False):
        period_key = (str(row.portfolio), str(row.from_date), str(row.thru_date))
        if period_key not in period_keys:
            issues.append(
                AuditIssue(
                    check="scenario_calendar",
                    portfolio=str(row.portfolio),
                    from_date=str(row.from_date),
                    thru_date=str(row.thru_date),
                    detail=(
                        "Scenario calendar references a portfolio period that "
                        f"is not in the packaged demo: {row.scenario_key}."
                    ),
                )
            )
    return issues


def _scenario_calendar_density(calendar: pd.DataFrame) -> list[dict[str, object]]:
    """Return current scenario-row density by portfolio period.

    The target is intentionally tracked as a planning metric. The current
    packaged demo is expected to stay within the target; future scenario
    additions can use this density view to spot periods that need another
    intra-month split.
    """
    density_rows: list[dict[str, object]] = []
    period_columns = ["portfolio", "from_date", "thru_date"]
    grouped = calendar.groupby(period_columns, sort=True, dropna=False)
    for period_key, period_rows in grouped:
        portfolio, from_date, thru_date = (str(value) for value in period_key)
        current_difference_rows = int(period_rows["current_expected_difference_rows"].sum())
        scenario_families = sorted(set(period_rows["scenario_family"].astype(str)))
        primary_securities = sorted(set(period_rows["primary_security"].astype(str)))
        density_rows.append(
            {
                "portfolio": portfolio,
                "from_date": from_date,
                "thru_date": thru_date,
                "current_difference_rows": current_difference_rows,
                "target_max_difference_rows": (_SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS),
                "needs_intra_month_split": (
                    current_difference_rows > _SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS
                ),
                "scenario_families": scenario_families,
                "primary_securities": primary_securities,
            }
        )
    return density_rows


def _scenario_readability_matrix(calendar: pd.DataFrame) -> list[dict[str, object]]:
    """Return a compact scenario story matrix by portfolio period.

    The scenario calendar is the source-of-truth audit layer for the packaged
    Axys/APX demo. This matrix turns the same rows into a reviewer-facing
    planning summary so future demo scenarios can be added without crowding a
    period beyond the intended one-or-two-difference story.
    """
    matrix_rows: list[dict[str, object]] = []
    period_columns = ["portfolio", "from_date", "thru_date"]
    grouped = calendar.groupby(period_columns, sort=True, dropna=False)
    for period_key, period_rows in grouped:
        portfolio, from_date, thru_date = (str(value) for value in period_key)
        difference_rows = int(period_rows["current_expected_difference_rows"].sum())
        scenario_keys = sorted(set(period_rows["scenario_key"].astype(str)))
        scenario_families = sorted(set(period_rows["scenario_family"].astype(str)))
        primary_securities = sorted(set(period_rows["primary_security"].astype(str)))
        scenario_notes = sorted(set(period_rows["notes"].astype(str)))
        matrix_rows.append(
            {
                "portfolio": portfolio,
                "from_date": from_date,
                "thru_date": thru_date,
                "expected_difference_rows": difference_rows,
                "target_max_difference_rows": (_SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS),
                "within_target": (difference_rows <= _SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS),
                "scenario_families": scenario_families,
                "primary_securities": primary_securities,
                "scenario_keys": scenario_keys,
                "scenario_notes": scenario_notes,
            }
        )
    return matrix_rows


def _scenario_isolation_matrix(inventory: pd.DataFrame) -> list[dict[str, object]]:
    """Return protected independent-change and carry-forward counts by period.

    Physical fixture rows are intentionally not counted here. Paired accounting
    legs may share one economic-change ID, while carry-forward effects remain
    visible in their later story period without being misclassified as a new
    source change there.
    """
    rows: list[dict[str, object]] = []
    source_period_keys = {
        (str(row.portfolio), str(row.source_from_date), str(row.source_thru_date))
        for row in inventory.itertuples(index=False)
    }
    story_period_keys = {
        (str(row.portfolio), str(row.story_from_date), str(row.story_thru_date))
        for row in inventory.itertuples(index=False)
    }
    for portfolio, from_date, thru_date in sorted(source_period_keys | story_period_keys):
        source_rows = inventory[
            inventory["portfolio"].astype(str).eq(portfolio)
            & inventory["source_from_date"].astype(str).eq(from_date)
            & inventory["source_thru_date"].astype(str).eq(thru_date)
        ]
        carry_rows = inventory[
            inventory["portfolio"].astype(str).eq(portfolio)
            & inventory["story_from_date"].astype(str).eq(from_date)
            & inventory["story_thru_date"].astype(str).eq(thru_date)
            & inventory["carry_forward_status"].eq("carry_forward_effect")
        ]
        independent_ids = sorted(
            set(source_rows["independent_change_id"].astype(str))
        )
        source_scenario_keys = sorted(set(source_rows["scenario_key"].astype(str)))
        carry_forward_keys = sorted(
            set(carry_rows["scenario_key"].astype(str))
        )
        rows.append(
            {
                "portfolio": portfolio,
                "source_from_date": from_date,
                "source_thru_date": thru_date,
                "independent_change_count": len(independent_ids),
                "maximum_independent_changes": (
                    _SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES
                ),
                "within_budget": (
                    len(independent_ids)
                    <= _SCENARIO_PERIOD_MAX_INDEPENDENT_CHANGES
                ),
                "independent_change_ids": independent_ids,
                "source_scenario_keys": source_scenario_keys,
                "visible_carry_forward_scenario_keys": carry_forward_keys,
            }
        )
    return rows


def _audit_period_split_plan(
    *,
    plan: pd.DataFrame,
    calendar: pd.DataFrame,
) -> list[AuditIssue]:
    """Return issues for an invalid intra-month period split backlog."""
    issues: list[AuditIssue] = []
    calendar_by_key = {str(row.scenario_key): row for row in calendar.itertuples(index=False)}
    planned_keys = set(plan["scenario_key"].astype(str))
    for key in sorted(planned_keys - set(calendar_by_key)):
        issues.append(
            AuditIssue(
                check="period_split_plan",
                detail=f"Period split plan key is not in the scenario calendar: {key}.",
            )
        )

    crowded_keys = _crowded_scenario_calendar_keys(calendar)
    for key in sorted(crowded_keys - planned_keys):
        issues.append(
            AuditIssue(
                check="period_split_plan",
                detail=f"Crowded-period scenario is missing from split plan: {key}.",
            )
        )

    for row in plan.itertuples(index=False):
        calendar_row = calendar_by_key.get(str(row.scenario_key))
        if calendar_row is None:
            continue
        current_key = (
            str(row.portfolio),
            str(row.current_from_date),
            str(row.current_thru_date),
        )
        calendar_key = (
            str(calendar_row.portfolio),
            str(calendar_row.from_date),
            str(calendar_row.thru_date),
        )
        if current_key != calendar_key:
            issues.append(
                AuditIssue(
                    check="period_split_plan",
                    portfolio=str(row.portfolio),
                    from_date=str(row.current_from_date),
                    thru_date=str(row.current_thru_date),
                    detail=(
                        "Period split plan current period does not match "
                        f"scenario calendar: {row.scenario_key}."
                    ),
                )
            )
        current_from = pd.Timestamp(row.current_from_date)
        current_thru = pd.Timestamp(row.current_thru_date)
        planned_from = pd.Timestamp(row.planned_from_date)
        planned_thru = pd.Timestamp(row.planned_thru_date)
        if planned_from > planned_thru:
            issues.append(
                AuditIssue(
                    check="period_split_plan",
                    portfolio=str(row.portfolio),
                    from_date=str(row.planned_from_date),
                    thru_date=str(row.planned_thru_date),
                    detail=f"Planned period starts after it ends: {row.scenario_key}.",
                )
            )
        if planned_from < current_from or planned_thru > current_thru:
            issues.append(
                AuditIssue(
                    check="period_split_plan",
                    portfolio=str(row.portfolio),
                    from_date=str(row.planned_from_date),
                    thru_date=str(row.planned_thru_date),
                    detail=(
                        "Planned period must stay inside current period: " f"{row.scenario_key}."
                    ),
                )
            )

    for row in _scenario_period_split_plan_summary(plan):
        if int(row["planned_difference_rows"]) > _SCENARIO_PERIOD_TARGET_MAX_DIFFERENCE_ROWS:
            issues.append(
                AuditIssue(
                    check="period_split_plan",
                    portfolio=str(row["portfolio"]),
                    from_date=str(row["planned_from_date"]),
                    thru_date=str(row["planned_thru_date"]),
                    detail=(
                        "Planned period exceeds target difference-row density: "
                        f"{row['planned_difference_rows']}."
                    ),
                )
            )
    issues.extend(_audit_period_split_plan_overlaps(plan))
    return issues


def _audit_period_split_plan_overlaps(plan: pd.DataFrame) -> list[AuditIssue]:
    """Return issues for overlapping planned periods inside a current period."""
    issues: list[AuditIssue] = []
    period_columns = [
        "portfolio",
        "current_from_date",
        "current_thru_date",
        "planned_from_date",
        "planned_thru_date",
    ]
    planned_periods = plan[period_columns].drop_duplicates()
    current_period_columns = ["portfolio", "current_from_date", "current_thru_date"]
    grouped = planned_periods.groupby(current_period_columns, sort=True, dropna=False)
    for current_period, current_rows in grouped:
        portfolio, current_from_date, current_thru_date = (str(value) for value in current_period)
        sorted_rows = current_rows.assign(
            _planned_from=pd.to_datetime(current_rows["planned_from_date"]),
            _planned_thru=pd.to_datetime(current_rows["planned_thru_date"]),
        ).sort_values(["_planned_from", "_planned_thru"])
        previous_thru: pd.Timestamp | None = None
        previous_period = ""
        for row in sorted_rows.itertuples(index=False):
            planned_from = pd.Timestamp(row.planned_from_date)
            planned_thru = pd.Timestamp(row.planned_thru_date)
            current_period_label = f"{row.planned_from_date}/{row.planned_thru_date}"
            if previous_thru is not None and planned_from <= previous_thru:
                issues.append(
                    AuditIssue(
                        check="period_split_plan",
                        portfolio=portfolio,
                        from_date=str(current_from_date),
                        thru_date=str(current_thru_date),
                        detail=(
                            "Planned periods overlap inside the same current "
                            f"period: {previous_period} and {current_period_label}."
                        ),
                    )
                )
            previous_thru = planned_thru
            previous_period = current_period_label
    return issues


def _crowded_scenario_calendar_keys(calendar: pd.DataFrame) -> set[str]:
    """Return scenario keys from periods above the simplification target."""
    crowded_periods = {
        (row["portfolio"], row["from_date"], row["thru_date"])
        for row in _scenario_calendar_density(calendar)
        if row["needs_intra_month_split"]
    }
    return {
        str(row.scenario_key)
        for row in calendar.itertuples(index=False)
        if (str(row.portfolio), str(row.from_date), str(row.thru_date)) in crowded_periods
    }


def _scenario_period_split_plan_summary(plan: pd.DataFrame) -> list[dict[str, object]]:
    """Return planned scenario-row density by proposed intra-month period."""
    summary_rows: list[dict[str, object]] = []
    period_columns = ["portfolio", "planned_from_date", "planned_thru_date"]
    grouped = plan.groupby(period_columns, sort=True, dropna=False)
    for period_key, period_rows in grouped:
        portfolio, planned_from_date, planned_thru_date = (str(value) for value in period_key)
        summary_rows.append(
            {
                "portfolio": portfolio,
                "planned_from_date": planned_from_date,
                "planned_thru_date": planned_thru_date,
                "planned_difference_rows": int(period_rows["planned_difference_rows"].sum()),
                "scenario_keys": sorted(set(period_rows["scenario_key"].astype(str))),
            }
        )
    return summary_rows


def _transaction_scenario_calendar_key(
    adjustment: TransactionScenarioAdjustment,
) -> str:
    """Return the calendar key for one transaction scenario row."""
    return f"transaction:{adjustment.transaction_id}:{adjustment.scenario}"


def _holding_scenario_calendar_key(adjustment: HoldingScenarioAdjustment) -> str:
    """Return the calendar key for one holding scenario row."""
    return (
        f"holding:{adjustment.portfolio}:{adjustment.security}:"
        f"{adjustment.holding_date}:{adjustment.scenario}"
    )


def _demo_portfolio_period_keys(axys_directory: Path) -> set[tuple[str, str, str]]:
    """Return all portfolio-period keys present in packaged demo portperf files."""
    period_keys: set[tuple[str, str, str]] = set()
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        portperf = _read_packaged_axys_frame(
            axys_directory / snapshot_name / "portperf.csv",
            "portfolio_performance",
        )
        for row in portperf.itertuples(index=False):
            period_keys.add(
                (
                    str(row.PORTFOLIO_CODE),
                    str(row.FROM_DATE),
                    str(row.THRU_DATE),
                )
            )
    return period_keys


def _portfolio_residual_issue(row: dict[str, object], detail: str) -> AuditIssue:
    """Return an audit issue for one portfolio-period residual row."""
    return AuditIssue(
        check="visible_portfolio_residual",
        portfolio=str(row["portfolio_id"]),
        from_date=row["from_date"].isoformat(),
        thru_date=row["thru_date"].isoformat(),
        detail=(
            f"{detail} Status={row['review_status']}; " f"unexplained={row['unexplained_change']}."
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
            "Audit demos."
        )
    )
    parser.add_argument(
        "--axys-directory",
        type=Path,
        default=_DEFAULT_AXYS_APX_DIRECTORY,
        help="Directory containing snapshot_a and snapshot_b.",
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
        "--scenario-calendar-path",
        type=Path,
        default=_DEFAULT_SCENARIO_CALENDAR_PATH,
        help=(
            "CSV file mapping intentional scenario rows to the demo periods "
            "they are meant to explain."
        ),
    )
    parser.add_argument(
        "--scenario-inventory-path",
        type=Path,
        default=_DEFAULT_SCENARIO_INVENTORY_PATH,
        help=(
            "Independent CSV inventory that prevents intentional named demo "
            "scenarios from being silently removed."
        ),
    )
    parser.add_argument(
        "--period-split-plan-path",
        type=Path,
        default=_DEFAULT_PERIOD_SPLIT_PLAN_PATH,
        help=(
            "CSV file mapping any crowded current periods to proposed shorter "
            "intra-month periods. An empty file means no split backlog remains."
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
