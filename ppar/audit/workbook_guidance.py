"""Build deterministic reviewer guidance for Audit workbook evidence rows."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import datetime as _dt
from pathlib import Path

# Project imports
import ppar.utilities as util
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_rows as rows
from ppar.audit import workbook_source_allocation as source_allocation
from ppar.audit.performance_comparison import findings
from ppar.audit.rendering import format_value
from ppar.audit.specification import SECURITY_COMPARISON_LEVEL
from ppar.audit.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
)
from ppar.audit.transaction_policy import transaction_boundary_codes

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
    "possible_cause_field_name",
    "possible_cause_row_comment",
    "possible_cause_summary",
    "review_guidance",
    "review_note",
]

_POSSIBLE_CAUSE_CONFIGURATION_NOTE = "Add YAML configuration to count it as explained."
_POSSIBLE_CAUSE_FIELDS = {
    (audit_schema.HOLDINGS, audit_schema.MARKET_VALUE),
    (audit_schema.TRANSACTIONS, audit_schema.AMOUNT),
}


def review_note(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
) -> str:
    """Return one reviewer-facing note for a changed workbook row.

    Args:
        row: Finding or intermediate workbook evidence row.
        estimated_impact: Additive estimated return impact, when available.
        row_use: Reviewer-facing use classification.
        impact_status: Reviewer-facing impact treatment status.

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
    source_explanation = _source_row_explanation(row, dataset, source_column)
    if source_explanation:
        return source_explanation
    if rows.has_evidence_only_policy(row):
        return (
            "Review-only evidence; this row is not counted in "
            '"Performance Differences" or "Explained Difference".'
        )
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
    """Return guidance for why a row does or does not explain performance.

    Args:
        row: Finding or intermediate workbook evidence row.
        estimated_impact: Additive estimated return impact, when available.
        comparison_path: Optional comparison YAML path used in setup guidance.
        impact_status: Reviewer-facing impact treatment status.
        row_kind: Internal evidence-row classification.

    Returns:
        Deterministic reviewer guidance for the cause table.
    """
    dataset = format_value(row.get(findings.DATASET))
    source_column = format_value(row.get(findings.SOURCE_COLUMN))
    if estimated_impact is not None:
        if row.get(rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
            return _transaction_reconstruction_flow_guidance(row)
        if dataset == audit_schema.HOLDINGS and source_column in {
            audit_schema.ACCRUED,
            audit_schema.BASE_ACCRUED,
            audit_schema.MARKET_VALUE,
            audit_schema.BASE_MARKET_VALUE,
            audit_schema.PRICE,
            audit_schema.QUANTITY,
        }:
            return _holding_detail_explanation(row, source_column)
        if dataset == audit_schema.TRANSACTIONS:
            return _transaction_component_explanation(row, source_column)
        return ""

    if dataset == audit_schema.TRANSACTIONS and source_column in {
        audit_schema.COMMISSION,
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _transaction_component_explanation(row, source_column)
    if row.get(rows.POSSIBLE_CAUSE_ROW):
        return _possible_cause_review_guidance(row, dataset, source_column)
    if rows.has_additive_policy(row) and impact_status == IMPACT_STATUS_MISSING_INPUT:
        return _missing_impact_input_setup(dataset, source_column)
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return _transaction_cash_balance_explanation(row)
    if row.get(rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return _transaction_reconstruction_flow_guidance(row)
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
        return _related_input_guidance(row, dataset, source_column)
    if rows.has_evidence_only_policy(row):
        return (
            "Review-only evidence; this row is not counted in "
            '"Performance Differences" or "Explained Difference".'
        )
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
        return _holding_detail_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        if source_column == audit_schema.AMOUNT:
            return _source_row_explanation(row, dataset, source_column)
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
) -> str:
    """Return source-data explanation text for a recognized source shape."""
    if dataset == audit_schema.HOLDINGS:
        return _holding_detail_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        if source_column == audit_schema.AMOUNT:
            if row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
                return _portfolio_external_flow_transaction_explanation(row)
            return _transaction_cash_balance_explanation(row)
        return _transaction_component_explanation(row, source_column)
    if dataset == audit_schema.SPLITS and source_column == audit_schema.SPLIT_FACTOR:
        return _split_factor_explanation(row)
    return ""


def _performance_dataset_review_note(source_column: str) -> str:
    """Return review guidance for reported performance-extract rows."""
    if source_column in {audit_schema.PORTFOLIO_RETURN, audit_schema.SECURITY_RETURN}:
        return "Reported return residual; no supported source-data row explains this difference."
    if source_column in {
        audit_schema.BEGIN_MARKET_VALUE,
        audit_schema.END_MARKET_VALUE,
        audit_schema.FLOW,
        audit_schema.INCOME,
    }:
        return "Performance-extract input; not a separate additive cause."
    return "Performance-extract diagnostic; not a separate additive cause."


def _related_input_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return guidance for an input component's related performance field."""
    if dataset == audit_schema.HOLDINGS and source_column in {
        audit_schema.ACCRUED,
        audit_schema.BASE_ACCRUED,
        audit_schema.MARKET_VALUE,
        audit_schema.BASE_MARKET_VALUE,
        audit_schema.PRICE,
        audit_schema.QUANTITY,
    }:
        return _holding_detail_explanation(row, source_column)
    if dataset == audit_schema.TRANSACTIONS:
        return _transaction_component_explanation(row, source_column)
    return "Review-only supporting evidence for the related counted row."


def _transaction_reconstruction_flow_guidance(row: Mapping[str, object]) -> str:
    """Return guidance for transaction rows absorbed by reconstruction formulas."""
    comparison_level = row.get("_workbook_reconstruction_comparison_level")
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _transaction_component_explanation(row, audit_schema.AMOUNT)
    if row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
        return _portfolio_external_flow_transaction_explanation(row)
    return _transaction_cash_balance_explanation(row)


def _possible_cause_review_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return concise guidance for evidence that may explain a residual."""
    if row.get(rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        explanation = _transaction_cash_balance_explanation(row)
    elif dataset == audit_schema.TRANSACTIONS and source_column == audit_schema.AMOUNT:
        explanation = _transaction_amount_possible_cause_explanation(row)
    else:
        explanation = _source_row_explanation(row, dataset, source_column)
    if not explanation:
        explanation = possible_cause_row_comment(row)
    return f"{explanation} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"


def _transaction_amount_possible_cause_explanation(
    row: Mapping[str, object],
) -> str:
    """Return compact possible-cause wording for transaction amount changes."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = rows.row_change_value(row)
    return (
        f"{_transaction_code_prefix(row)}{security_prefix}transactions.amount "
        f"{rows.increased_or_decreased(change_value)} by "
        f"{rows.change_amount_text(change_value)}."
    )


def _portfolio_external_flow_transaction_explanation(
    row: Mapping[str, object],
) -> str:
    """Return source-data wording for a portfolio external-flow transaction."""
    flow_delta = rows.row_change_value(row)
    weighted_flow_delta = (rows.number_or_none(flow_delta) or 0.0) * (
        source_allocation.source_flow_weight(row)
    )
    return (
        f"{_transaction_code_prefix(row)}External flow "
        f"{rows.increased_or_decreased(flow_delta)} by "
        f"{rows.change_amount_text(flow_delta)}; weighted external flow "
        f"{rows.increased_or_decreased(weighted_flow_delta)} by "
        f"{rows.change_amount_text(weighted_flow_delta)}."
    )


def _transaction_cash_balance_explanation(row: Mapping[str, object]) -> str:
    """Return source-data wording for a transaction's ending cash-balance effect."""
    return (
        f"{_transaction_code_prefix(row)}Caused cash-balance "
        "ending holdings.market_value "
        f"to {_cash_balance_direction(row)} by "
        f"{rows.change_amount_text(rows.row_change_value(row))}."
    )


def _cash_balance_direction(row: Mapping[str, object]) -> str:
    """Return increase/decrease wording for the cash effect of a transaction."""
    if row.get(findings.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_POSITIVE:
        return "increase"
    return "decrease"


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
    change_text = rows.change_amount_text(change_value)
    if timing_label == "beginning":
        return (
            "Inherited beginning-value difference from the preceding period: "
            f"{security_prefix}{holdings_label}.{source_column} "
            f"{rows.increased_or_decreased(change_value)} by {change_text}. "
            "This value is retained because it is an input to Modified Dietz."
        )
    return (
        f"{security_prefix}{holdings_label}.{source_column} "
        f"{rows.increased_or_decreased(change_value)} by {change_text}."
    )


def _fx_rate_support_explanation(row: Mapping[str, object]) -> str:
    """Return an FX-rate explanation linked to the counted base value."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    target_field = format_value(row.get(source_allocation.FX_RATE_TARGET_FIELD))
    base_value_change = row.get(source_allocation.FX_RATE_BASE_VALUE_CHANGE)
    snapshot_a = rows.number_or_none(row.get(findings.SNAPSHOT_A_VALUE))
    snapshot_b = rows.number_or_none(row.get(findings.SNAPSHOT_B_VALUE))
    from_currency = format_value(row.get(findings.FROM_CURRENCY))
    to_currency = format_value(row.get(findings.TO_CURRENCY))
    pair_prefix = (
        f"{from_currency}-to-{to_currency} FX rate"
        if from_currency and to_currency
        else "FX rate"
    )
    quote_suffix = (
        f" {to_currency} per {from_currency}"
        if from_currency and to_currency
        else ""
    )
    rate_change = (
        "changed"
        if snapshot_a is None or snapshot_b is None
        else f"changed from {snapshot_a:g} to {snapshot_b:g}"
    )
    return (
        f"{pair_prefix} {rate_change}{quote_suffix}; "
        f"{security_prefix}{target_field} shows the counted "
        f"{to_currency or 'base-currency'} effect of "
        f"{rows.change_amount_text(base_value_change)}."
    )


def _split_factor_explanation(row: Mapping[str, object]) -> str:
    """Return plain-language explanation for a split-factor support row."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    split_factor = rows.row_change_value(row)
    return (
        f"split: Caused {security_prefix}holdings.quantity and related "
        "holdings.market_value to increase using a "
        f"{_split_factor_text(split_factor)} split factor."
    )


def _split_factor_text(value: object) -> str:
    """Return compact split-factor text for workbook explanations."""
    number = rows.number_or_none(value)
    if number is None:
        return "changed"
    if float(number).is_integer():
        return f"{abs(number):.1f}"
    return f"{abs(number):g}"


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
) -> str:
    """Return plain-language explanation for a transaction component row."""
    security_id = format_value(row.get(findings.SECURITY_ID))
    security_text = f" for {security_id}" if security_id else ""
    field_text = f"transactions.{source_column}"
    if source_column == audit_schema.COMMISSION:
        change_value = rows.row_change_value(row)
        change_number = rows.number_or_none(change_value)
        change_verb = (
            "decrease" if change_number is not None and change_number < 0 else "increase"
        )
        transaction_amount = (
            f"{security_id} transactions.amount"
            if security_id
            else "transactions.amount"
        )
        return (
            f"{_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb} by {rows.change_amount_text(change_value)}."
        )
    if source_column in {audit_schema.PRICE, audit_schema.QUANTITY}:
        change_value = rows.row_change_value(row)
        change_number = rows.number_or_none(change_value)
        change_verb = (
            "decrease" if change_number is not None and change_number < 0 else "increase"
        )
        transaction_amount = (
            f"{security_id} transactions.amount"
            if security_id
            else "transactions.amount"
        )
        if source_column == audit_schema.QUANTITY:
            quantity_effect = _transaction_quantity_holding_effect(row)
            if quantity_effect:
                holdings_quantity = (
                    f"{security_id} holdings.quantity"
                    if security_id
                    else "holdings.quantity"
                )
                return (
                    f"{_transaction_code_prefix(row)}Caused {transaction_amount} "
                    f"to {change_verb} and {holdings_quantity} to {quantity_effect}."
                )
        return (
            f"{_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb}."
        )
    if (
        source_column == audit_schema.AMOUNT
        and row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW
    ):
        field_text = "external flow"
    elif (
        source_column == audit_schema.AMOUNT
        and row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_FEE_EXPENSE
    ):
        field_text = "fee/expense"
    elif (
        source_column == audit_schema.AMOUNT
        and row.get(findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_INCOME
    ):
        field_text = "income"
    return (
        f"{_transaction_code_prefix(row)}The {field_text}{security_text} changed by "
        f"{rows.change_amount_text(rows.row_change_value(row))}."
    )


def _transaction_quantity_holding_effect(row: Mapping[str, object]) -> str:
    """Return buy/sell holding direction for a transaction quantity row."""
    change_number = rows.number_or_none(rows.row_change_value(row))
    if change_number is None:
        return ""
    transaction_code = format_value(row.get(findings.TRANSACTION_CODE)).lower()
    if transaction_code in transaction_boundary_codes("quantity_holding_neutral"):
        return ""
    transaction_category = row.get(findings.TRANSACTION_CATEGORY)
    if transaction_category == TRANSACTION_CATEGORY_BUY:
        return "decrease" if change_number < 0 else "increase"
    if transaction_category == TRANSACTION_CATEGORY_SELL:
        return "increase" if change_number < 0 else "decrease"
    return ""


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
