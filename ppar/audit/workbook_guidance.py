"""Build deterministic reviewer guidance for Audit workbook evidence rows."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import datetime as _dt
from functools import cache
from pathlib import Path

# Project imports
import ppar.utilities as util
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_rows as rows
from ppar.audit import workbook_source_allocation as source_allocation
from ppar.audit.extract_contract import transaction_semantics_exact_case
from ppar.audit.performance_comparison import findings
from ppar.audit.rendering import format_value
from ppar.audit.specification import (
    AuditSpecification,
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.audit.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
)
from ppar.audit.transaction_policy import (
    transaction_boundary_codes,
    transaction_code_matching_key,
)

IMPACT_STATUS_ESTIMATED = "Estimated"
IMPACT_STATUS_MISSING_METHOD = "Missing impact method"
IMPACT_STATUS_MISSING_INPUT = "Missing impact input"
IMPACT_STATUS_REVIEW_ONLY = "Review only"

__all__ = [
    "IMPACT_STATUS_ESTIMATED",
    "IMPACT_STATUS_MISSING_INPUT",
    "IMPACT_STATUS_MISSING_METHOD",
    "IMPACT_STATUS_REVIEW_ONLY",
    "dataset_field",
    "explanation_contract_issues",
    "possible_cause_field_name",
    "possible_cause_row_comment",
    "possible_cause_summary",
    "review_guidance",
    "review_note",
]

_POSSIBLE_CAUSE_CONFIGURATION_NOTE = "Add YAML configuration to count it as explained."
_REVIEW_ONLY_EVIDENCE_NOTE = "Review-only evidence."
_POSSIBLE_CAUSE_FIELDS = {
    (audit_schema.HOLDINGS, audit_schema.MARKET_VALUE),
    (audit_schema.TRANSACTIONS, audit_schema.AMOUNT),
}


def review_note(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
    *,
    comparison_path: util.PathLike | None = None,
) -> str:
    """Return one reviewer-facing note for a changed workbook row.

    Args:
        row: Finding or intermediate workbook evidence row.
        estimated_impact: Additive estimated return impact, when available.
        row_use: Reviewer-facing use classification.
        impact_status: Reviewer-facing impact treatment status.
        comparison_path: Optional comparison YAML path used to apply its
            transaction case-matching contract.

    Returns:
        Concise reviewer note for the cause table.
    """
    if estimated_impact is not None:
        return ""

    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if dataset in {
        audit_schema.PORTFOLIO_PERFORMANCE,
        audit_schema.SECURITY_PERFORMANCE,
    }:
        return _performance_dataset_review_note(source_column)
    review_only_holding_explanation = _review_only_holding_explanation(
        row,
        dataset,
        source_column,
        impact_status,
    )
    source_explanation = (
        review_only_holding_explanation
        if review_only_holding_explanation
        else _source_row_explanation(
            row,
            dataset,
            source_column,
            exact_case=_comparison_uses_exact_transaction_case(comparison_path),
        )
    )
    if source_explanation:
        return source_explanation
    if rows.has_evidence_only_policy(row):
        return _REVIEW_ONLY_EVIDENCE_NOTE
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return "Supporting evidence for Modified Dietz flow rows; not counted separately."
    if row.get(rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return "Supporting evidence for Modified Dietz flow rows; not counted separately."
    if row.get(rows.UNSELECTED_RELATED_ESTIMATE):
        return "Review this input component; a related performance input is selected."
    if impact_status == IMPACT_STATUS_MISSING_INPUT:
        return (
            "Review inputs needed by the configured YAML method; no "
            "estimate is available for this row."
        )
    if impact_status == IMPACT_STATUS_MISSING_METHOD:
        return _missing_impact_method_action(dataset, source_column)
    if row_use == rows.USE_REVIEW_CONTEXT:
        return "Review context; not included in explained performance difference."
    dataset_actions = {
        audit_schema.TRANSACTIONS: _review_change_action("transaction", source_column),
        audit_schema.HOLDINGS: _review_change_action("holding", source_column),
    }
    return dataset_actions.get(
        dataset,
        _review_change_action("input", source_column),
    )


def review_guidance(
    row: Mapping[str, object],
    estimated_impact: float | None,
    *,
    comparison_path: util.PathLike | None,
    impact_status: str,
    row_kind: str,
) -> str:
    """Return deterministic guidance that begins with the source change.

    Args:
        row: Finding or intermediate workbook evidence row.
        estimated_impact: Additive estimated return impact, when available.
        comparison_path: Optional comparison YAML path used in setup guidance.
        impact_status: Reviewer-facing impact treatment status.
        row_kind: Internal evidence-row classification.

    Returns:
        Reviewer guidance normalized to the explanation contract.
    """
    guidance = _review_guidance(
        row,
        estimated_impact,
        comparison_path=comparison_path,
        impact_status=impact_status,
        row_kind=row_kind,
    )
    return _with_source_change_lead(row, guidance)


def _review_guidance(
    row: Mapping[str, object],
    estimated_impact: float | None,
    *,
    comparison_path: util.PathLike | None,
    impact_status: str,
    row_kind: str,
) -> str:
    """Return the role-specific part of one explanation.

    Args:
        row: Finding or intermediate workbook evidence row.
        estimated_impact: Additive estimated return impact, when available.
        comparison_path: Optional comparison YAML path used in setup guidance.
        impact_status: Reviewer-facing impact treatment status.
        row_kind: Internal evidence-row classification.

    Returns:
        Role-specific reviewer guidance for the cause table.
    """
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    exact_case = _comparison_uses_exact_transaction_case(comparison_path)
    if estimated_impact is not None:
        if row.get(rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
            return _transaction_reconstruction_flow_guidance(
                row,
                exact_case=exact_case,
                estimated_impact=estimated_impact,
            )
        if dataset == audit_schema.HOLDINGS and source_column in {
            audit_schema.ACCRUED,
            audit_schema.BASE_ACCRUED,
            audit_schema.MARKET_VALUE,
            audit_schema.BASE_MARKET_VALUE,
            audit_schema.PRICE,
            audit_schema.QUANTITY,
        }:
            return _holding_source_explanation(row, source_column)
        if dataset == audit_schema.TRANSACTIONS:
            return _transaction_component_explanation(
                row,
                source_column,
                exact_case=exact_case,
                estimated_impact=estimated_impact,
            )
        return ""

    if dataset == audit_schema.TRANSACTIONS and source_column in {
        audit_schema.COMMISSION,
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _transaction_component_explanation(
            row,
            source_column,
            exact_case=exact_case,
        )
    if row.get(rows.POSSIBLE_CAUSE_ROW):
        return _possible_cause_review_guidance(
            row,
            dataset,
            source_column,
            exact_case=exact_case,
        )
    if rows.has_additive_policy(row) and impact_status == IMPACT_STATUS_MISSING_INPUT:
        return _missing_impact_input_setup(dataset, source_column)
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return _transaction_cash_balance_explanation(row)
    if row.get(rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return _transaction_reconstruction_flow_guidance(
            row,
            exact_case=exact_case,
        )
    if row.get(rows.TRANSACTION_FLOW_SUPPORTS_HOLDING):
        security_id = format_value(row.get(findings.SECURITY_ID))
        if security_id:
            return (
                f"{security_id} transaction activity changed. The security holding "
                "value row shows the counted effect."
            )
        return "Transaction activity changed. The holding value row shows the counted effect."
    if row.get(rows.SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _split_factor_explanation(row)
    if row.get(source_allocation.FX_RATE_SUPPORTS_BASE_INPUT):
        return _fx_rate_support_explanation(row)
    if row.get(rows.UNSELECTED_RELATED_ESTIMATE):
        return _related_input_guidance(
            row,
            dataset,
            source_column,
            exact_case=exact_case,
        )
    review_only_holding_explanation = _review_only_holding_explanation(
        row,
        dataset,
        source_column,
        impact_status,
    )
    if review_only_holding_explanation or rows.has_evidence_only_policy(row):
        return review_only_holding_explanation or _REVIEW_ONLY_EVIDENCE_NOTE
    if row_kind in {
        rows.ROW_KIND_CONTEXT,
        rows.ROW_KIND_REPORTED_DIAGNOSTIC,
        rows.ROW_KIND_DIAGNOSTIC,
    }:
        return "Review context; not an underlying input difference."

    dataset_column = _dataset_column_label(dataset, source_column)
    yaml_path = _yaml_path_label(comparison_path)
    if dataset == audit_schema.HOLDINGS and source_column in {
        audit_schema.ACCRUED,
        audit_schema.BASE_ACCRUED,
        audit_schema.MARKET_VALUE,
        audit_schema.BASE_MARKET_VALUE,
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _holding_source_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        if source_column == audit_schema.AMOUNT:
            return _source_row_explanation(
                row,
                dataset,
                source_column,
                exact_case=exact_case,
            )
        return f"No supported YAML impact method exists yet for {dataset_column}."
    if rows.has_additive_policy(row):
        return _missing_impact_input_setup(dataset, source_column)
    if dataset == audit_schema.HOLDINGS:
        if source_column not in {
            audit_schema.MARKET_VALUE,
            audit_schema.ACCRUED,
            audit_schema.BASE_ACCRUED,
            audit_schema.QUANTITY,
        }:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        if source_column in {audit_schema.ACCRUED, audit_schema.BASE_ACCRUED}:
            return (
                "Specify the YAML holding_impact_methods.accrued.method and "
                "holding_impact_methods.accrued.denominator_source in "
                f"{yaml_path}."
            )
        if source_column == audit_schema.QUANTITY:
            return (
                "Specify the YAML holding_impact_methods.quantity.method and "
                "holding_impact_methods.quantity.denominator_source in "
                f"{yaml_path}."
            )
        return (
            "Specify the YAML holding_impact_methods.market_value.method and "
            "holding_impact_methods.market_value.denominator_source in "
            f"{yaml_path}."
        )
    if dataset == audit_schema.FX_RATES:
        if source_column != audit_schema.FX_RATE:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return f"Specify the YAML fx_rate_impact_methods.fx_rate.method in {yaml_path}."
    return f"No supported YAML impact method exists yet for {dataset_column}."


def dataset_field(row: Mapping[str, object]) -> str:
    """Return a compact ``dataset.field`` label for a workbook row."""
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if dataset and source_column:
        return f"{dataset}.{source_column}"
    if dataset:
        return dataset
    return source_column


def explanation_contract_issues(
    row: Mapping[str, object],
    explanation: str,
    *,
    impact_status: str,
    comparison_path: util.PathLike | None,
) -> tuple[str, ...]:
    """Return semantic consistency issues for one source-row explanation.

    Args:
        row: Finding or intermediate workbook evidence row.
        explanation: Rendered reviewer-facing explanation.
        impact_status: Reviewer-facing impact treatment status.
        comparison_path: Optional comparison YAML path used to apply its
            transaction case-matching contract.

    Returns:
        Deterministically ordered contract violations. An empty tuple means
        that the explanation states the source change and every required
        downstream performance field consistently.
    """
    issues: list[str] = []
    exact_case = _comparison_uses_exact_transaction_case(comparison_path)
    expected_lead = _source_change_sentence(row)
    if expected_lead and not explanation.startswith(expected_lead):
        issues.append(f"must begin with {expected_lead!r}")

    dataset = format_value(row.get(findings.DATASET))
    transaction_code = format_value(row.get(findings.TRANSACTION_CODE)).strip()
    if dataset == audit_schema.TRANSACTIONS and transaction_code:
        expected_prefix = f"{transaction_code}:"
        if not explanation.startswith(expected_prefix):
            issues.append(f"must begin with transaction code {expected_prefix!r}")

    for performance_reference in _required_performance_references(
        row,
        impact_status=impact_status,
        exact_case=exact_case,
    ):
        if performance_reference not in explanation:
            issues.append(
                "must name affected performance field "
                f"{performance_reference!r}"
            )

    expected_currency_field = _expected_currency_value_field(row)
    if expected_currency_field == "holdings.base_market_value":
        wrong_phrases = (
            "through holdings.market_value",
            "ending holdings.market_value",
        )
        if any(phrase in explanation for phrase in wrong_phrases):
            issues.append("uses local-currency language for a base-currency effect")
    elif expected_currency_field == "holdings.market_value":
        wrong_phrases = (
            "through holdings.base_market_value",
            "ending holdings.base_market_value",
        )
        if any(phrase in explanation for phrase in wrong_phrases):
            issues.append("uses base-currency language for a local-currency effect")

    if explanation and not explanation.endswith("."):
        issues.append("must end with a period")
    if explanation == _REVIEW_ONLY_EVIDENCE_NOTE and expected_lead:
        issues.append("must state the source change instead of generic review-only text")
    if any(token in explanation.lower() for token in (" none", " nan")):
        issues.append("contains a missing-value token")
    return tuple(issues)


def _with_source_change_lead(
    row: Mapping[str, object],
    guidance: str,
) -> str:
    """Ensure source evidence begins by stating its exact changed field."""
    source_change = _source_change_sentence(row)
    if not source_change or guidance.startswith(source_change):
        return guidance
    if not guidance:
        return source_change
    return f"{source_change} {guidance}"


def _source_change_sentence(row: Mapping[str, object]) -> str:
    """Return the canonical first sentence for a changed source row."""
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if _source_change_value(row) is None:
        return ""
    if dataset == audit_schema.HOLDINGS:
        return _holding_source_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        return _transaction_change_sentence(row, source_column)
    if dataset == audit_schema.FX_RATES and source_column == audit_schema.FX_RATE:
        return _fx_rate_change_sentence(row)
    if dataset == audit_schema.SPLITS and source_column == audit_schema.SPLIT_FACTOR:
        return _split_factor_change_sentence(row)
    return ""


def _source_change_value(row: Mapping[str, object]) -> float | None:
    """Return a numeric delta, deriving it from snapshot values when needed."""
    change_value = rows.number_or_none(rows.row_change_value(row))
    if change_value is not None:
        return change_value
    snapshot_a = rows.number_or_none(row.get(findings.SNAPSHOT_A_VALUE))
    snapshot_b = rows.number_or_none(row.get(findings.SNAPSHOT_B_VALUE))
    if snapshot_a is None or snapshot_b is None:
        return None
    return snapshot_b - snapshot_a


def _transaction_change_sentence(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return the canonical source-change sentence for a transaction row."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = rows.row_change_value(row)
    return (
        f"{_transaction_code_prefix(row)}{security_prefix}"
        f"transactions.{source_column} "
        f"{rows.increased_or_decreased(change_value)} by "
        f"{_field_change_text(change_value, source_column)}."
    )


def _fx_rate_change_sentence(row: Mapping[str, object]) -> str:
    """Return the canonical source-change sentence for an FX-rate row."""
    snapshot_a = rows.number_or_none(row.get(findings.SNAPSHOT_A_VALUE))
    snapshot_b = rows.number_or_none(row.get(findings.SNAPSHOT_B_VALUE))
    from_currency = format_value(row.get(findings.FROM_CURRENCY))
    to_currency = format_value(row.get(findings.TO_CURRENCY))
    pair_prefix = (
        f"{from_currency}-to-{to_currency} "
        if from_currency and to_currency
        else ""
    )
    change_value = _source_change_value(row)
    value_detail = ""
    if snapshot_a is not None and snapshot_b is not None:
        quote_suffix = (
            f" {to_currency} per {from_currency}"
            if from_currency and to_currency
            else ""
        )
        value_detail = f", from {snapshot_a:g} to {snapshot_b:g}{quote_suffix}"
    return (
        f"{pair_prefix}fx_rates.fx_rate "
        f"{rows.increased_or_decreased(change_value)} by "
        f"{rows.change_amount_text(change_value)}{value_detail}."
    )


def _split_factor_change_sentence(row: Mapping[str, object]) -> str:
    """Return the canonical source-change sentence for a split row."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = rows.row_change_value(row)
    return (
        f"{security_prefix}splits.split_factor "
        f"{rows.increased_or_decreased(change_value)} by "
        f"{rows.change_amount_text(change_value)}."
    )


def _required_performance_references(
    row: Mapping[str, object],
    *,
    impact_status: str,
    exact_case: bool,
) -> tuple[str, ...]:
    """Return downstream performance references required by the row's role."""
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if dataset == audit_schema.HOLDINGS:
        if source_column in {audit_schema.PRICE, audit_schema.QUANTITY}:
            return (_holding_performance_input(row, source_column),)
        if (
            source_column == audit_schema.BASE_MARKET_VALUE
            and impact_status == IMPACT_STATUS_REVIEW_ONLY
        ):
            return ("holdings.market_value",)
        return ()
    if dataset == audit_schema.FX_RATES and source_column == audit_schema.FX_RATE:
        if not row.get(source_allocation.FX_RATE_SUPPORTS_BASE_INPUT):
            return ()
        target_field = format_value(row.get(source_allocation.FX_RATE_TARGET_FIELD))
        return (target_field,) if target_field else ()
    if dataset == audit_schema.SPLITS and source_column == audit_schema.SPLIT_FACTOR:
        if row.get(rows.SPLIT_FACTOR_SUPPORTS_HOLDING):
            return ("holdings.market_value",)
        return ()
    if dataset != audit_schema.TRANSACTIONS:
        return ()
    if source_column in {audit_schema.COMMISSION, audit_schema.PRICE}:
        return ("transactions.amount",)
    if source_column == audit_schema.QUANTITY:
        references = ["transactions.amount"]
        if _transaction_quantity_affects_holding_value(row, exact_case=exact_case):
            references.append("holdings.market_value")
        return tuple(references)
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return (_transaction_cash_balance_field(row),)
    if impact_status == IMPACT_STATUS_ESTIMATED:
        category = row.get(findings.TRANSACTION_CATEGORY)
        if category in {
            TRANSACTION_CATEGORY_BUY,
            TRANSACTION_CATEGORY_SELL,
            TRANSACTION_CATEGORY_EXTERNAL_FLOW,
        }:
            return ("weighted external flow",)
        if category in {
            TRANSACTION_CATEGORY_FEE_EXPENSE,
            TRANSACTION_CATEGORY_INCOME,
        }:
            return ("income",)
    if row.get(rows.TRANSACTION_FLOW_SUPPORTS_HOLDING):
        return ("holdings.market_value",)
    return ()


def _expected_currency_value_field(row: Mapping[str, object]) -> str:
    """Return the local/base value field implied by one supporting row."""
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if dataset == audit_schema.HOLDINGS and source_column in {
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _holding_performance_input(row, source_column)
    if dataset == audit_schema.TRANSACTIONS and row.get(
        rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION
    ):
        return _transaction_cash_balance_field(row)
    if dataset == audit_schema.FX_RATES:
        return format_value(row.get(source_allocation.FX_RATE_TARGET_FIELD))
    return ""


def possible_cause_field_name(row: Mapping[str, object]) -> str:
    """Return ``dataset.field`` text for a supported possible-cause row."""
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if (dataset, source_column) not in _POSSIBLE_CAUSE_FIELDS:
        return ""
    return f"{dataset}.{source_column}"


def possible_cause_row_comment(row: Mapping[str, object]) -> str:
    """Return a row-specific possible-cause sentence fragment."""
    field_name = possible_cause_field_name(row)
    if not field_name:
        return ""

    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = rows.row_change_value(row)
    change_direction = rows.increased_or_decreased(change_value)
    change_amount = rows.change_amount_text(change_value)
    input_date = format_value(row.get(findings.INPUT_DATE))
    if input_date:
        return (
            f"{security_prefix}{field_name} {change_direction} by "
            f"{change_amount} on {input_date}."
        )
    return f"{security_prefix}{field_name} {change_direction} by {change_amount}."


def possible_cause_summary(comments: Sequence[str]) -> str:
    """Return summary-sheet possible-cause wording for unresolved periods."""
    if not comments:
        return ""
    if len(comments) == 1:
        return f"Possible cause: {comments[0]} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"
    return f"Possible causes: {' '.join(comments)} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"


def _source_row_explanation(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
    *,
    exact_case: bool,
) -> str:
    """Return source-data explanation text for a recognized source shape."""
    if dataset == audit_schema.HOLDINGS:
        return _holding_detail_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        if source_column == audit_schema.AMOUNT:
            if row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
                return _portfolio_external_flow_transaction_explanation(row)
            return _transaction_cash_balance_explanation(row)
        return _transaction_component_explanation(
            row,
            source_column,
            exact_case=exact_case,
        )
    if dataset == audit_schema.SPLITS and source_column == audit_schema.SPLIT_FACTOR:
        return _split_factor_explanation(row)
    return ""


def _performance_dataset_review_note(source_column: str) -> str:
    """Return review guidance for reported performance-extract rows."""
    if source_column in {audit_schema.PORTFOLIO_RETURN, audit_schema.SECURITY_RETURN}:
        return "Reported return residual; no supported source-data row explains this difference."
    return "Unsupported performance-extract field; review the source-data contract."


def _related_input_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
    *,
    exact_case: bool,
) -> str:
    """Return guidance for an input component's related performance field."""
    if dataset == audit_schema.HOLDINGS and source_column in {
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _review_only_holding_explanation(
            row,
            dataset,
            source_column,
            IMPACT_STATUS_REVIEW_ONLY,
        )
    if dataset == audit_schema.HOLDINGS and source_column in {
        audit_schema.ACCRUED,
        audit_schema.BASE_ACCRUED,
        audit_schema.MARKET_VALUE,
        audit_schema.BASE_MARKET_VALUE,
        audit_schema.PRICE,
    }:
        return _holding_source_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        return _transaction_component_explanation(
            row,
            source_column,
            exact_case=exact_case,
        )
    return "Review-only supporting evidence for the related counted row."


def _transaction_reconstruction_flow_guidance(
    row: Mapping[str, object],
    *,
    exact_case: bool,
    estimated_impact: float | None = None,
) -> str:
    """Return guidance for transaction rows absorbed by reconstruction formulas."""
    comparison_level = row.get("_workbook_reconstruction_comparison_level")
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _transaction_component_explanation(
            row,
            audit_schema.AMOUNT,
            exact_case=exact_case,
            estimated_impact=estimated_impact,
        )
    if row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
        return _portfolio_external_flow_transaction_explanation(row)
    return _transaction_cash_balance_explanation(row)


def _possible_cause_review_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
    *,
    exact_case: bool,
) -> str:
    """Return concise guidance for evidence that may explain a residual."""
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        explanation = _transaction_cash_balance_explanation(row)
    elif dataset == audit_schema.TRANSACTIONS and source_column == audit_schema.AMOUNT:
        explanation = _transaction_amount_possible_cause_explanation(row)
    else:
        explanation = _source_row_explanation(
            row,
            dataset,
            source_column,
            exact_case=exact_case,
        )
    if not explanation:
        explanation = possible_cause_row_comment(row)
    return f"{explanation} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"


def _transaction_amount_possible_cause_explanation(
    row: Mapping[str, object],
) -> str:
    """Return compact possible-cause wording for transaction amount changes."""
    return _transaction_change_sentence(row, audit_schema.AMOUNT)


def _portfolio_external_flow_transaction_explanation(
    row: Mapping[str, object],
) -> str:
    """Return source-data wording for a portfolio external-flow transaction."""
    flow_delta = rows.row_change_value(row)
    weighted_flow_delta = (rows.number_or_none(flow_delta) or 0.0) * (
        source_allocation.source_flow_weight(row)
    )
    return (
        f"{_transaction_change_sentence(row, audit_schema.AMOUNT)} "
        "This affects the performance calculation through weighted external flow, "
        f"which {rows.increased_or_decreased(weighted_flow_delta)} by "
        f"{rows.change_amount_text(weighted_flow_delta)}."
    )


def _transaction_cash_balance_explanation(row: Mapping[str, object]) -> str:
    """Return source-data wording for a transaction's ending cash-balance effect."""
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    return (
        f"{_transaction_change_sentence(row, source_column)} "
        "This affects the performance calculation through cash-balance ending "
        f"{_transaction_cash_balance_field(row)}."
    )


def _transaction_cash_balance_field(row: Mapping[str, object]) -> str:
    """Return the cash value field matching a transaction's source basis."""
    if format_value(row.get(findings.SOURCE_COLUMN)) == audit_schema.BASE_AMOUNT:
        return "holdings.base_market_value"
    return "holdings.market_value"


def _holding_detail_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return plain-language explanation for a holding source row."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    timing_label = _holding_timing_label(row)
    holdings_label = f"{timing_label} holdings" if timing_label else "holdings"
    change_value = rows.row_change_value(row)
    change_text = _field_change_text(change_value, source_column)
    return (
        f"{security_prefix}{holdings_label}.{source_column} "
        f"{rows.increased_or_decreased(change_value)} by {change_text}."
    )


def _holding_source_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return a holding change plus its downstream value field when needed."""
    detail = _holding_detail_explanation(row, source_column)
    if source_column not in {audit_schema.PRICE, audit_schema.QUANTITY}:
        return detail
    return (
        f"{detail} This affects the performance calculation through "
        f"{_holding_performance_input(row, source_column)}."
    )


def _review_only_holding_explanation(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
    impact_status: str,
) -> str:
    """Return specific guidance for review-only holding input components."""
    if dataset != audit_schema.HOLDINGS or impact_status != IMPACT_STATUS_REVIEW_ONLY:
        return ""
    if source_column in {audit_schema.PRICE, audit_schema.QUANTITY}:
        return _holding_source_explanation(row, source_column)
    detail = _holding_detail_explanation(row, source_column)
    if source_column == audit_schema.BASE_MARKET_VALUE:
        return (
            f"{detail} This change is also reflected in the performance calculation "
            "through holdings.market_value."
        )
    return ""


def _holding_performance_input(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return the counted holding-value field related to a component row."""
    impact_policy = format_value(row.get(findings.IMPACT_POLICY))
    foreign_currency_policy = (
        f"{findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}"
        f"holdings.{source_column}_row_currency"
    )
    if impact_policy == foreign_currency_policy:
        return "holdings.base_market_value"
    return "holdings.market_value"


def _fx_rate_support_explanation(row: Mapping[str, object]) -> str:
    """Return an FX-rate explanation linked to the counted base value."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    target_field = format_value(row.get(source_allocation.FX_RATE_TARGET_FIELD))
    base_value_change = row.get(source_allocation.FX_RATE_BASE_VALUE_CHANGE)
    to_currency = format_value(row.get(findings.TO_CURRENCY))
    security_suffix = f" for {security_id}" if security_id else ""
    return (
        f"{_fx_rate_change_sentence(row)} This affects the performance calculation "
        f"through {target_field}; the counted {to_currency or 'base-currency'} "
        f"effect{security_suffix} is "
        f"{rows.change_amount_text(base_value_change)}."
    )


def _split_factor_explanation(row: Mapping[str, object]) -> str:
    """Return plain-language explanation for a split-factor support row."""
    return (
        f"{_split_factor_change_sentence(row)} This affects the performance "
        "calculation through holdings.market_value; holdings.quantity is "
        "supporting evidence."
    )


def _holding_timing_label(row: Mapping[str, object]) -> str:
    """Return beginning/ending label for inclusive-period holding dates."""
    input_date = row.get(findings.INPUT_DATE)
    from_date = row.get(findings.FROM_DATE)
    thru_date = row.get(findings.THRU_DATE)
    if (
        isinstance(input_date, _dt.date)
        and isinstance(from_date, _dt.date)
        and input_date == from_date - _dt.timedelta(days=1)
    ):
        return "beginning"
    if isinstance(input_date, _dt.date) and input_date == thru_date:
        return "ending"
    return ""


def _transaction_component_explanation(
    row: Mapping[str, object],
    source_column: str,
    *,
    exact_case: bool,
    estimated_impact: float | None = None,
) -> str:
    """Return plain-language explanation for a transaction component row."""
    if source_column in {
        audit_schema.COMMISSION,
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        explanation = _transaction_change_sentence(row, source_column)
        if (
            source_column == audit_schema.QUANTITY
            and _transaction_quantity_affects_holding_value(
                row,
                exact_case=exact_case,
            )
        ):
            return (
                f"{explanation} This affects the performance calculation through "
                "transactions.amount and holdings.market_value."
            )
        return (
            f"{explanation} This affects the performance calculation through "
            "transactions.amount."
        )
    explanation = _transaction_change_sentence(row, source_column)
    category = row.get(findings.TRANSACTION_CATEGORY)
    if estimated_impact is not None and category in {
        TRANSACTION_CATEGORY_BUY,
        TRANSACTION_CATEGORY_SELL,
    }:
        return (
            f"{explanation} This affects the performance calculation through "
            "weighted external flow."
        )
    if estimated_impact is not None and category in {
        TRANSACTION_CATEGORY_FEE_EXPENSE,
        TRANSACTION_CATEGORY_INCOME,
    }:
        return (
            f"{explanation} This affects the performance calculation through income."
        )
    if row.get(rows.TRANSACTION_FLOW_SUPPORTS_HOLDING):
        return (
            f"{explanation} This affects the performance calculation through "
            "holdings.market_value."
        )
    return explanation


def _field_change_text(value: object, source_column: str) -> str:
    """Return a field-aware change amount for reviewer explanations."""
    if source_column != audit_schema.QUANTITY:
        return rows.change_amount_text(value)
    number = rows.number_or_none(value)
    if number is None:
        return "the changed amount"
    whole, fraction = f"{abs(number):,.6f}".rsplit(".", maxsplit=1)
    fraction = fraction.rstrip("0").ljust(2, "0")
    return f"{whole}.{fraction}"


def _transaction_quantity_affects_holding_value(
    row: Mapping[str, object],
    *,
    exact_case: bool,
) -> bool:
    """Return whether transaction quantity affects a long-position holding value."""
    change_number = rows.number_or_none(rows.row_change_value(row))
    if change_number is None:
        return False
    transaction_code = transaction_code_matching_key(
        row.get(findings.TRANSACTION_CODE),
        exact_case=exact_case,
    )
    if transaction_code in transaction_boundary_codes("quantity_holding_neutral"):
        return False
    transaction_category = row.get(findings.TRANSACTION_CATEGORY)
    return transaction_category in {
        TRANSACTION_CATEGORY_BUY,
        TRANSACTION_CATEGORY_SELL,
    }


@cache
def _comparison_path_uses_exact_transaction_case(path: str) -> bool:
    """Return the cached exact-case setting for one comparison path."""
    specification = AuditSpecification(
        path,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    )
    return transaction_semantics_exact_case(
        specification.values,
        specification_path=specification.path,
    )


def _comparison_uses_exact_transaction_case(
    comparison_path: util.PathLike | None,
) -> bool:
    """Return whether reviewer guidance must honor exact transaction case."""
    if comparison_path is None:
        return False
    return _comparison_path_uses_exact_transaction_case(str(Path(comparison_path)))


def _transaction_code_prefix(row: Mapping[str, object]) -> str:
    """Return a short transaction-code prefix for review guidance."""
    transaction_code = format_value(row.get(findings.TRANSACTION_CODE))
    if transaction_code:
        return f"{transaction_code}: "
    fallback = _transaction_code_fallback(row)
    if not fallback:
        return ""
    return f"{fallback.replace('_', ' ')}: "


def _transaction_code_fallback(row: Mapping[str, object]) -> str:
    """Return a compact transaction label when the raw code is unavailable."""
    category = format_value(row.get(findings.TRANSACTION_CATEGORY))
    if category == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
        if row.get(findings.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_POSITIVE:
            return "deposit"
        return "withdrawal"
    if category == TRANSACTION_CATEGORY_FEE_EXPENSE:
        return "fee"
    if category == TRANSACTION_CATEGORY_INCOME:
        return "income"
    return category


def _yaml_path_label(comparison_path: util.PathLike | None) -> str:
    """Return a compact YAML path label for setup instructions."""
    if comparison_path is None:
        return "comparison YAML"
    return str(Path(comparison_path))


def _dataset_column_label(dataset: str, source_column: str) -> str:
    """Return ``dataset.column`` text for impact-method setup messages."""
    if dataset and source_column:
        return f"{dataset}.{source_column}"
    if dataset:
        return dataset
    if source_column:
        return source_column
    return "this input field"


def _missing_impact_input_setup(dataset: str, source_column: str) -> str:
    """Return setup text when a configured method lacks usable source inputs."""
    if dataset == audit_schema.TRANSACTIONS and source_column == audit_schema.AMOUNT:
        return (
            "Configured transaction impact method is present, but this row still "
            "cannot be estimated. Review return denominator, transaction sign/flow "
            "semantics, and transaction date inputs."
        )
    if dataset == audit_schema.HOLDINGS:
        return (
            "This holding changed. The beginning or ending portfolio value row "
            "shows the counted effect."
        )
    return (
        "Configured YAML impact method is present, but this row still cannot be "
        "estimated. Review the inputs required by that method."
    )


def _missing_impact_method_action(dataset: str, source_column: str) -> str:
    """Return action text for source rows without an additive impact method."""
    if dataset == audit_schema.TRANSACTIONS:
        return _add_method_action("transaction", source_column)
    if dataset == audit_schema.HOLDINGS:
        return _add_method_action("holding", source_column)
    return _add_method_action("input", source_column)


def _review_change_action(dataset_label: str, source_column: str) -> str:
    """Return standardized action text for a review-only changed value."""
    return f"Review {_source_change_label(dataset_label, source_column)} change."


def _add_method_action(dataset_label: str, source_column: str) -> str:
    """Return standardized action text for a missing impact method."""
    return (
        f"Review {_source_change_label(dataset_label, source_column)} change; "
        f"add {dataset_label} impact method before estimating."
    )


def _source_change_label(dataset_label: str, source_column: str) -> str:
    """Return compact dataset/field wording for action text."""
    if source_column:
        return f"{dataset_label} {source_column}"
    return dataset_label
