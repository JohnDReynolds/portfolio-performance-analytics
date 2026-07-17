"""Classify and normalize finding rows for Audit workbook presentation."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping

# Project imports
from ppar.audit import field_roles
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_layout
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings
from ppar.audit.specification import SECURITY_COMPARISON_LEVEL

NO_UNDERLYING_CAUSE_DATASET = "no_underlying_causes_found"
TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW = (
    "_workbook_transaction_supports_reconstruction_flow"
)
UNSELECTED_RELATED_ESTIMATE = "_workbook_unselected_related_estimate"
NON_ADDITIVE_PORTFOLIO_TRANSACTION = "_workbook_non_additive_portfolio_transaction"
TRANSACTION_FLOW_SUPPORTS_HOLDING = "_workbook_transaction_flow_supports_holding"
SPLIT_FACTOR_SUPPORTS_HOLDING = "_workbook_split_factor_supports_holding"
POSSIBLE_CAUSE_ROW = "_workbook_possible_cause_row"

ROW_KIND_UNDERLYING_CAUSE = "underlying_cause"
ROW_KIND_REPORTED_DIAGNOSTIC = "reported_diagnostic"
ROW_KIND_CONTEXT = "context"
ROW_KIND_DIAGNOSTIC = "diagnostic"
ROW_KIND_OTHER = "other"

USE_EXPLAINS_CHANGE = "Explains Change"
USE_REVIEW_CONTEXT = "Review Context"
USE_DIAGNOSTIC = "Diagnostic"

__all__ = [
    "USE_DIAGNOSTIC",
    "USE_EXPLAINS_CHANGE",
    "USE_REVIEW_CONTEXT",
    "NON_ADDITIVE_PORTFOLIO_TRANSACTION",
    "POSSIBLE_CAUSE_ROW",
    "SPLIT_FACTOR_SUPPORTS_HOLDING",
    "TRANSACTION_FLOW_SUPPORTS_HOLDING",
    "UNSELECTED_RELATED_ESTIMATE",
    "change_amount_text",
    "evidence_as_of_date",
    "formula_support_row",
    "has_additive_policy",
    "has_evidence_only_policy",
    "has_text",
    "higher_or_lower",
    "increased_or_decreased",
    "is_context_row",
    "is_reported_diagnostic_row",
    "is_underlying_cause_row",
    "non_additive_row",
    "number_or_none",
    "primary_review_period_key",
    "review_period_key",
    "row_change_value",
    "security_review_period_key",
    "use_priority",
    "workbook_row_kind",
]


def number_or_none(value: object) -> float | None:
    """Return a float for numeric values, preserving missing values.

    Args:
        value: Candidate numeric value.

    Returns:
        A float for non-Boolean integers and floats; otherwise ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def change_amount_text(value: object) -> str:
    """Return a compact absolute amount for reviewer-facing explanations."""
    number = number_or_none(value)
    if number is None:
        return "the changed amount"
    return f"{abs(number):,.2f}"


def row_change_value(row: Mapping[str, object]) -> object:
    """Return the changed amount from either workbook or finding row shape."""
    change = row.get(workbook_layout.CHANGE)
    if change is not None:
        return change
    return row.get(findings.DELTA_B_MINUS_A)


def increased_or_decreased(value: object) -> str:
    """Return increased/decreased wording for a numeric B-minus-A value."""
    number = number_or_none(value)
    if number is not None and number < 0:
        return "decreased"
    return "increased"


def higher_or_lower(value: object) -> str:
    """Return higher/lower wording for a numeric B-minus-A value."""
    number = number_or_none(value)
    if number is not None and number < 0:
        return "lower"
    return "higher"


def review_period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the portfolio-period review key for a finding or workbook row."""
    return (
        row.get(findings.PORTFOLIO_ID),
        row.get(findings.FROM_DATE),
        row.get(findings.THRU_DATE),
    )


def primary_review_period_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the review-period key at the configured comparison grain."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return security_review_period_key(row)
    return review_period_key(row)


def security_review_period_key(
    row: Mapping[str, object],
) -> tuple[object, object, object, object]:
    """Return the security-period review key for a finding or workbook row."""
    return (*review_period_key(row), row.get(findings.SECURITY_ID))


def evidence_as_of_date(row: Mapping[str, object]) -> object | None:
    """Return the date represented by a finding or workbook evidence row."""
    input_date = row.get(findings.INPUT_DATE)
    if input_date is not None:
        return input_date
    return row.get(findings.THRU_DATE)


def use_priority(row_use: str) -> int:
    """Return sort priority for a reviewer-facing workbook row use."""
    return {
        USE_EXPLAINS_CHANGE: 0,
        USE_REVIEW_CONTEXT: 1,
        USE_DIAGNOSTIC: 2,
    }.get(row_use, 9)


def formula_support_row(
    row: Mapping[str, object],
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return a non-additive row marked as reconstruction support.

    Args:
        row: Finding or intermediate workbook row.
        comparison_level: Audit report level that owns the reconstruction row.

    Returns:
        A copied row with additive fields cleared and formula-support metadata.
    """
    row_dict = non_additive_row(row)
    if (
        row_dict.get(findings.DATASET) == audit_schema.TRANSACTIONS
        and row_dict.get(findings.SOURCE_COLUMN) == audit_schema.AMOUNT
    ):
        row_dict[TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW] = True
        row_dict["_workbook_reconstruction_comparison_level"] = comparison_level
    return row_dict


def non_additive_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return a workbook row with explained-difference fields cleared."""
    row_dict = dict(row)
    row_dict[explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[explain.IMPACT_BASIS] = explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[explain.IMPACT_METHOD] = None
    return row_dict


def is_underlying_cause_row(row: Mapping[str, object]) -> bool:
    """Return whether a row is an identifiable input-cause candidate."""
    return workbook_row_kind(row) == ROW_KIND_UNDERLYING_CAUSE


def is_reported_diagnostic_row(row: Mapping[str, object]) -> bool:
    """Return whether a row is a reported-performance diagnostic."""
    return workbook_row_kind(row) == ROW_KIND_REPORTED_DIAGNOSTIC


def is_context_row(row: Mapping[str, object]) -> bool:
    """Return whether a row is context-only evidence."""
    return workbook_row_kind(row) == ROW_KIND_CONTEXT


def workbook_row_kind(row: Mapping[str, object]) -> str:
    """Return the workbook presentation role for a finding row."""
    if row.get(findings.DATASET) == NO_UNDERLYING_CAUSE_DATASET:
        row_kind = ROW_KIND_DIAGNOSTIC
    elif field_roles.is_reported_performance_component(
        row.get(findings.DATASET),
        row.get(findings.SOURCE_COLUMN),
    ):
        row_kind = ROW_KIND_REPORTED_DIAGNOSTIC
    elif row.get(findings.EVIDENCE_ROLE) == findings.CONTEXT.value:
        row_kind = ROW_KIND_CONTEXT
    elif has_evidence_only_policy(row):
        row_kind = ROW_KIND_CONTEXT
    elif row.get(findings.DATASET) in {
        audit_schema.PORTFOLIO_PERFORMANCE,
        audit_schema.SECURITY_PERFORMANCE,
    }:
        row_kind = ROW_KIND_REPORTED_DIAGNOSTIC
    elif row.get(findings.EVIDENCE_ROLE) == findings.DIRECT_INPUT.value:
        row_kind = ROW_KIND_UNDERLYING_CAUSE
    else:
        row_kind = ROW_KIND_OTHER
    return row_kind


def has_evidence_only_policy(row: Mapping[str, object]) -> bool:
    """Return whether a row has explicit YAML evidence-only treatment."""
    policies = (
        row.get(findings.IMPACT_POLICY),
        row.get(findings.TRANSACTION_IMPACT_POLICY),
    )
    return any(
        isinstance(policy, str)
        and policy.startswith(findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)
        for policy in policies
    )


def has_additive_policy(row: Mapping[str, object]) -> bool:
    """Return whether a row has a configured additive impact policy."""
    policies = (
        row.get(findings.IMPACT_POLICY),
        row.get(findings.TRANSACTION_IMPACT_POLICY),
    )
    return any(
        has_text(policy)
        and not str(policy).startswith(findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)
        for policy in policies
    )


def has_text(value: object) -> bool:
    """Return whether a value has non-blank text."""
    return isinstance(value, str) and bool(value.strip())
