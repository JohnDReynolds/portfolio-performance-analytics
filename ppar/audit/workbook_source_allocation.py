"""Allocate reconstruction formula effects to Audit workbook source rows."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime as _dt

# Project imports
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_formula_rows as formula_rows
from ppar.audit import workbook_layout as layout
from ppar.audit import workbook_rows as rows
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings
from ppar.audit.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.audit.rendering import format_value
from ppar.audit.specification import SECURITY_COMPARISON_LEVEL
from ppar.audit.transactions import (
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
)

CASH_BALANCE_SECURITY_ID = "_workbook_cash_balance_security_id"
FX_RATE_BASE_VALUE_CHANGE = "_workbook_fx_rate_base_value_change"
FX_RATE_SUPPORTS_BASE_INPUT = "_workbook_fx_rate_supports_base_input"
FX_RATE_TARGET_FIELD = "_workbook_fx_rate_target_field"

__all__ = [
    "CASH_BALANCE_SECURITY_ID",
    "FX_RATE_BASE_VALUE_CHANGE",
    "FX_RATE_SUPPORTS_BASE_INPUT",
    "FX_RATE_TARGET_FIELD",
    "allocate_formula_sources",
    "cash_security_matches",
    "fx_source_identity",
    "fx_support_rows",
    "source_flow_weight",
    "source_row_key",
    "with_cash_balance_security",
]

_UNEXPLAINED_TOLERANCE = 0.0000005


@dataclass(frozen=True)
class _FormulaSourceIndex:
    """Source rows indexed by the Modified Dietz component they support.

    Attributes:
        value_rows: Eligible holding values keyed by formula owner and date.
        flow_rows: Eligible transaction flows keyed by owner and period.
        income_rows: Eligible transaction income or expense rows keyed by owner
            and period.
    """

    value_rows: dict[tuple[object, ...], list[Mapping[str, object]]]
    flow_rows: dict[tuple[object, ...], list[Mapping[str, object]]]
    income_rows: dict[tuple[object, ...], list[Mapping[str, object]]]


def allocate_formula_sources(
    source_rows: Sequence[Mapping[str, object]],
    reconstruction_formula_rows: Sequence[Mapping[str, object]],
    *,
    matched_cash_securities: Mapping[tuple[object, ...], object],
    comparison_level: str,
) -> tuple[list[dict[str, object]], list[Mapping[str, object]]]:
    """Return allocated source effects and necessarily visible formula rows.

    Args:
        source_rows: Ranked source-data evidence rows.
        reconstruction_formula_rows: Modified Dietz component rows to allocate.
        matched_cash_securities: Transaction source rows mapped to changed cash
            holdings.
        comparison_level: Portfolio or security review grain.

    Returns:
        Allocated source rows followed by formula rows without an allocatable
        source basis.

    Notes:
        Candidate source rows and their bases are derived once per Modified
        Dietz formula input. Inputs without an allocatable source row remain in
        the second result so they cannot disappear from the cause sheet.
    """
    allocated_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    unallocated: list[Mapping[str, object]] = []
    source_index = _formula_source_index(
        source_rows,
        comparison_level=comparison_level,
    )
    for formula_row in reconstruction_formula_rows:
        candidate_rows = _formula_source_candidates(
            source_index,
            formula_row,
            comparison_level=comparison_level,
        )
        bases = [_formula_source_basis(row, formula_row) for row in candidate_rows]
        total_basis = sum(bases)
        if not candidate_rows or abs(total_basis) <= _UNEXPLAINED_TOLERANCE:
            unallocated.append(formula_row)
            continue

        estimated_impact = rows.number_or_none(formula_row.get(layout.ESTIMATED_IMPACT))
        if estimated_impact is None:
            continue
        for source_row, basis in zip(candidate_rows, bases, strict=True):
            attributed_row = _source_attributed_row(
                source_row,
                formula_row,
                estimated_impact * basis / total_basis,
                comparison_level=comparison_level,
            )
            _attach_cash_balance_security(
                attributed_row,
                matched_cash_securities,
                comparison_level=comparison_level,
            )
            key = source_row_key(attributed_row, comparison_level)
            existing_row = allocated_by_key.get(key)
            if existing_row is None:
                allocated_by_key[key] = attributed_row
                continue
            existing_impact = rows.number_or_none(
                existing_row.get(explain.ESTIMATED_RETURN_IMPACT)
            )
            additional_impact = rows.number_or_none(
                attributed_row.get(explain.ESTIMATED_RETURN_IMPACT)
            )
            existing_row[explain.ESTIMATED_RETURN_IMPACT] = (existing_impact or 0.0) + (
                additional_impact or 0.0
            )
            existing_components = format_value(
                existing_row.get(formula_rows.RECONSTRUCTION_COMPONENTS)
            )
            additional_components = format_value(
                attributed_row.get(formula_rows.RECONSTRUCTION_COMPONENTS)
            )
            existing_row[formula_rows.RECONSTRUCTION_COMPONENTS] = "|".join(
                sorted(
                    {
                        component
                        for component in (
                            *existing_components.split("|"),
                            *additional_components.split("|"),
                        )
                        if component
                    }
                )
            )
    return list(allocated_by_key.values()), unallocated


def cash_security_matches(
    source_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> dict[tuple[object, ...], object]:
    """Return transaction source-row keys mapped to matching cash securities.

    Args:
        source_rows: Ranked source-data evidence rows.
        comparison_level: Portfolio or security review grain.

    Returns:
        Source-row keys mapped to uniquely matched cash security identifiers.
    """
    matches: dict[tuple[object, ...], object] = {}
    cash_holdings_by_period = _cash_holdings_by_period(source_rows)
    for source_row in source_rows:
        if (
            source_row.get(findings.DATASET) != audit_schema.TRANSACTIONS
            or source_row.get(findings.SOURCE_COLUMN) != audit_schema.AMOUNT
        ):
            continue
        cash_security_id = _matching_cash_security_id(
            source_row,
            cash_holdings_by_period.get(_cash_transaction_key(source_row), ()),
        )
        if cash_security_id:
            matches[source_row_key(source_row, comparison_level)] = cash_security_id
    return matches


def with_cash_balance_security(
    row: Mapping[str, object],
    matched_cash_securities: Mapping[tuple[object, ...], object],
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return a row with its matched cash security attached when available.

    Args:
        row: Source-data evidence row.
        matched_cash_securities: Source rows mapped to changed cash holdings.
        comparison_level: Portfolio or security review grain.

    Returns:
        A copied row with cash-security metadata when a unique match exists.
    """
    row_dict = dict(row)
    _attach_cash_balance_security(
        row_dict,
        matched_cash_securities,
        comparison_level=comparison_level,
    )
    return row_dict


def source_row_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return a stable key for one workbook source-data row."""
    return (
        *rows.primary_review_period_key(row, comparison_level),
        row.get(findings.DATASET),
        row.get(findings.SOURCE_COLUMN),
        row.get(findings.SECURITY_ID),
        rows.evidence_as_of_date(row),
        row.get(findings.TRANSACTION_CATEGORY),
        row.get(findings.SNAPSHOT_A_VALUE),
        row.get(findings.SNAPSHOT_B_VALUE),
        row.get(findings.DELTA_B_MINUS_A),
    )


def fx_support_rows(
    source_rows: Sequence[Mapping[str, object]],
    attributed_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> list[dict[str, object]]:
    """Return changed FX rates linked to counted base-currency inputs.

    Args:
        source_rows: Ranked source-data evidence rows.
        attributed_rows: Formula effects allocated to source rows.
        comparison_level: Portfolio or security review grain.

    Returns:
        Non-additive FX support rows linked to unique base-currency inputs.
    """
    support_rows: list[dict[str, object]] = []
    for source_row in source_rows:
        if not (
            source_row.get(findings.DATASET) == audit_schema.FX_RATES
            and source_row.get(findings.SOURCE_COLUMN) == audit_schema.FX_RATE
        ):
            continue
        rate_delta = rows.number_or_none(source_row.get(findings.DELTA_B_MINUS_A))
        local_exposure = rows.number_or_none(source_row.get(findings.IMPACT_INPUT_VALUE))
        if rate_delta is None or local_exposure is None:
            continue
        base_value_delta = rate_delta * local_exposure
        matching_rows = [
            candidate
            for candidate in attributed_rows
            if candidate.get(findings.PORTFOLIO_ID)
            == source_row.get(findings.PORTFOLIO_ID)
            and rows.evidence_as_of_date(candidate) == rows.evidence_as_of_date(source_row)
            and candidate.get(findings.DATASET)
            in {audit_schema.HOLDINGS, audit_schema.TRANSACTIONS}
            and candidate.get(findings.SOURCE_COLUMN)
            in {audit_schema.BASE_MARKET_VALUE, audit_schema.BASE_AMOUNT}
            and abs(
                (rows.number_or_none(candidate.get(findings.DELTA_B_MINUS_A)) or 0.0)
                - base_value_delta
            )
            <= 0.005
        ]
        targets_by_period: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
        for target in matching_rows:
            targets_by_period.setdefault(
                rows.primary_review_period_key(target, comparison_level),
                [],
            ).append(target)
        for targets in targets_by_period.values():
            if len(targets) == 1:
                support_rows.append(_fx_support_row(source_row, targets[0]))
    return support_rows


def fx_source_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return a period-independent identity for one changed FX-rate row."""
    if not (
        row.get(findings.DATASET) == audit_schema.FX_RATES
        and row.get(findings.SOURCE_COLUMN) == audit_schema.FX_RATE
    ):
        return ()
    return (
        row.get(findings.PORTFOLIO_ID),
        rows.evidence_as_of_date(row),
        row.get(findings.SNAPSHOT_A_VALUE),
        row.get(findings.SNAPSHOT_B_VALUE),
    )


def source_flow_weight(row: Mapping[str, object]) -> float:
    """Return Modified Dietz flow weight for a transaction source row."""
    from_date = row.get(findings.FROM_DATE)
    thru_date = row.get(findings.THRU_DATE)
    flow_date = rows.evidence_as_of_date(row)
    if not isinstance(from_date, _dt.date) or not isinstance(thru_date, _dt.date):
        return 1.0
    if not isinstance(flow_date, _dt.date):
        return 1.0
    try:
        return modified_dietz_flow_weight(
            from_date=from_date,
            thru_date=thru_date,
            flow_date=flow_date,
            inclusion_rule="beginning_of_day",
        )
    except ValueError:
        return 1.0


def _formula_source_index(
    source_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> _FormulaSourceIndex:
    """Index source rows once by their eligible formula component."""
    value_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    flow_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    income_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    flow_categories = (
        {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
        if comparison_level == SECURITY_COMPARISON_LEVEL
        else {TRANSACTION_CATEGORY_EXTERNAL_FLOW}
    )
    for source_row in source_rows:
        owner = _formula_owner_key(source_row, comparison_level)
        dataset = source_row.get(findings.DATASET)
        source_column = source_row.get(findings.SOURCE_COLUMN)
        if (
            dataset == audit_schema.HOLDINGS
            and source_column
            in {
                audit_schema.MARKET_VALUE,
                audit_schema.BASE_MARKET_VALUE,
                audit_schema.ACCRUED,
                audit_schema.BASE_ACCRUED,
            }
            and not rows.has_evidence_only_policy(source_row)
        ):
            value_key = (*owner, rows.evidence_as_of_date(source_row))
            value_rows.setdefault(value_key, []).append(source_row)
        if (
            dataset != audit_schema.TRANSACTIONS
            or not _is_effective_transaction_amount(source_row)
        ):
            continue
        period_key = (
            *owner,
            source_row.get(findings.FROM_DATE),
            source_row.get(findings.THRU_DATE),
        )
        transaction_category = source_row.get(findings.TRANSACTION_CATEGORY)
        if transaction_category in flow_categories:
            flow_rows.setdefault(period_key, []).append(source_row)
        if transaction_category in {
            TRANSACTION_CATEGORY_FEE_EXPENSE,
            TRANSACTION_CATEGORY_INCOME,
        }:
            income_rows.setdefault(period_key, []).append(source_row)
    return _FormulaSourceIndex(value_rows, flow_rows, income_rows)


def _formula_owner_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the ownership key used to connect source and formula rows."""
    portfolio_id = row.get(findings.PORTFOLIO_ID)
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return portfolio_id, row.get(findings.SECURITY_ID)
    return (portfolio_id,)


def _formula_source_candidates(
    source_index: _FormulaSourceIndex,
    formula_row: Mapping[str, object],
    *,
    comparison_level: str,
) -> Sequence[Mapping[str, object]]:
    """Return source rows that make up one reconstruction formula row."""
    formula_field = formula_row.get(findings.SOURCE_COLUMN)
    owner = _formula_owner_key(formula_row, comparison_level)
    if formula_field in {
        formula_rows.BEGINNING_VALUE_FIELD,
        formula_rows.ENDING_VALUE_FIELD,
    }:
        return source_index.value_rows.get(
            (*owner, formula_row.get(layout.AS_OF_DATE)),
            (),
        )
    period_key = (
        *owner,
        formula_row.get(findings.FROM_DATE),
        formula_row.get(findings.THRU_DATE),
    )
    if formula_field in {
        formula_rows.NET_FLOW_FIELD,
        formula_rows.WEIGHTED_FLOW_FIELD,
    }:
        return source_index.flow_rows.get(period_key, ())
    if formula_field == formula_rows.INCOME_FIELD:
        return source_index.income_rows.get(period_key, ())
    return ()


def _is_effective_transaction_amount(row: Mapping[str, object]) -> bool:
    """Return whether a row is the transaction amount used in base returns."""
    source_column = row.get(findings.SOURCE_COLUMN)
    if source_column == audit_schema.BASE_AMOUNT:
        return True
    return source_column == audit_schema.AMOUNT and not rows.has_evidence_only_policy(row)


def _source_attributed_row(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
    estimated_impact: float,
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return a source row cloned into the formula period with allocated impact."""
    row_dict = dict(source_row)
    row_dict[findings.FROM_DATE] = formula_row.get(findings.FROM_DATE)
    row_dict[findings.THRU_DATE] = formula_row.get(findings.THRU_DATE)
    row_dict[layout.REVIEW_KEY] = formula_row.get(layout.REVIEW_KEY)
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        row_dict[findings.SECURITY_ID] = formula_row.get(findings.SECURITY_ID)
    row_dict[explain.ESTIMATED_RETURN_IMPACT] = estimated_impact
    row_dict[explain.IMPACT_BASIS] = "source_row_reconstruction"
    row_dict[explain.IMPACT_METHOD] = "return_reconstruction_source_allocation"
    row_dict[formula_rows.RECONSTRUCTION_COMPONENTS] = formula_row.get(
        findings.SOURCE_COLUMN
    )
    if (
        row_dict.get(findings.DATASET) == audit_schema.TRANSACTIONS
        and row_dict.get(findings.SOURCE_COLUMN) == audit_schema.AMOUNT
    ):
        row_dict[rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW] = True
        row_dict["_workbook_reconstruction_comparison_level"] = comparison_level
    return row_dict


def _attach_cash_balance_security(
    attributed_row: dict[str, object],
    matched_cash_securities: Mapping[tuple[object, ...], object],
    *,
    comparison_level: str,
) -> None:
    """Attach the changed cash holding security when one row is identifiable."""
    if (
        attributed_row.get(findings.DATASET) != audit_schema.TRANSACTIONS
        or attributed_row.get(findings.SOURCE_COLUMN) != audit_schema.AMOUNT
    ):
        return
    cash_security_id = matched_cash_securities.get(
        source_row_key(attributed_row, comparison_level)
    )
    if cash_security_id:
        attributed_row[CASH_BALANCE_SECURITY_ID] = cash_security_id


def _cash_holdings_by_period(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    """Index eligible cash holding rows once for transaction matching."""
    holdings_by_period: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for source_row in source_rows:
        if (
            source_row.get(findings.DATASET) != audit_schema.HOLDINGS
            or source_row.get(findings.SOURCE_COLUMN) != audit_schema.MARKET_VALUE
            or not _is_cash_security(source_row.get(findings.SECURITY_ID))
        ):
            continue
        holdings_by_period.setdefault(_cash_period_key(source_row), []).append(source_row)
    return holdings_by_period


def _cash_period_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the ownership, period, and as-of key for cash matching."""
    return (
        row.get(findings.PORTFOLIO_ID),
        row.get(findings.FROM_DATE),
        row.get(findings.THRU_DATE),
        rows.evidence_as_of_date(row),
    )


def _cash_transaction_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the cash-holding lookup key for a transaction source row."""
    return (
        row.get(findings.PORTFOLIO_ID),
        row.get(findings.FROM_DATE),
        row.get(findings.THRU_DATE),
        row.get(findings.THRU_DATE),
    )


def _matching_cash_security_id(
    transaction_row: Mapping[str, object],
    cash_holding_rows: Sequence[Mapping[str, object]],
) -> object | None:
    """Return the matching cash holding security for a transaction amount row."""
    transaction_delta = rows.number_or_none(transaction_row.get(findings.DELTA_B_MINUS_A))
    if transaction_delta is None:
        return None
    matches = [
        row
        for row in cash_holding_rows
        if _same_amount(
            rows.number_or_none(row.get(findings.DELTA_B_MINUS_A)),
            transaction_delta,
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0].get(findings.SECURITY_ID)


def _is_cash_security(security_id: object) -> bool:
    """Return whether an identifier appears to be a cash holding."""
    return format_value(security_id).upper().startswith("CASH")


def _same_amount(first_value: float | None, second_value: float | None) -> bool:
    """Return whether two source amounts are effectively the same amount."""
    if first_value is None or second_value is None:
        return False
    return abs(first_value - second_value) <= 0.005


def _formula_source_basis(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
) -> float:
    """Return source-row basis used to allocate one formula impact."""
    formula_field = formula_row.get(findings.SOURCE_COLUMN)
    delta = rows.number_or_none(source_row.get(findings.DELTA_B_MINUS_A)) or 0.0
    if formula_field == formula_rows.WEIGHTED_FLOW_FIELD:
        return delta * source_flow_weight(source_row)
    return delta


def _fx_support_row(
    row: Mapping[str, object],
    target: Mapping[str, object],
) -> dict[str, object]:
    """Return a non-additive FX row linked to its counted base-currency input."""
    row_dict = rows.non_additive_row(row)
    row_dict[findings.FROM_DATE] = target.get(findings.FROM_DATE)
    row_dict[findings.THRU_DATE] = target.get(findings.THRU_DATE)
    row_dict[layout.REVIEW_KEY] = target.get(layout.REVIEW_KEY)
    row_dict[FX_RATE_SUPPORTS_BASE_INPUT] = True
    row_dict[FX_RATE_TARGET_FIELD] = (
        f"{target.get(findings.DATASET)}.{target.get(findings.SOURCE_COLUMN)}"
    )
    row_dict[findings.SECURITY_ID] = target.get(findings.SECURITY_ID)
    row_dict[FX_RATE_BASE_VALUE_CHANGE] = target.get(findings.DELTA_B_MINUS_A)
    return row_dict
