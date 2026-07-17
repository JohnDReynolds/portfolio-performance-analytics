"""Classify normalized performance-comparison fields by accounting role."""

from __future__ import annotations

# Python imports
from typing import Final

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols

PERFORMANCE_INPUT: Final[str] = "performance_input"
INPUT_COMPONENT: Final[str] = "input_component"
CONTEXT: Final[str] = "context"
REPORTED_PERFORMANCE_COMPONENT: Final[str] = "reported_performance_component"
UNCLASSIFIED: Final[str] = "unclassified"

_FIELD_ROLES: Final[dict[tuple[str, str], str]] = {
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.BASE_MARKET_VALUE): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.ACCRUED): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.BASE_ACCRUED): PERFORMANCE_INPUT,
    (pc_cols.TRANSACTIONS, pc_cols.AMOUNT): PERFORMANCE_INPUT,
    (pc_cols.TRANSACTIONS, pc_cols.BASE_AMOUNT): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.QUANTITY): INPUT_COMPONENT,
    (pc_cols.HOLDINGS, pc_cols.PRICE): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.QUANTITY): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.PRICE): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.COMMISSION): INPUT_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.PORTFOLIO_RETURN): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.INCOME): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.GAIN_LOSS): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.BEGIN_MARKET_VALUE): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.END_MARKET_VALUE): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.FLOW): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.SECURITY_RETURN): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.CONTRIBUTION): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.WEIGHT): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.BEGIN_MARKET_VALUE): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.END_MARKET_VALUE): (REPORTED_PERFORMANCE_COMPONENT),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.INCOME): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.GAIN_LOSS): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.FX_RATES, pc_cols.FX_RATE): INPUT_COMPONENT,
    (pc_cols.FX_RATES, pc_cols.LOCAL_EXPOSURE): INPUT_COMPONENT,
    (pc_cols.SPLITS, pc_cols.SPLIT_FACTOR): CONTEXT,
    (pc_cols.HOLDINGS, pc_cols.COST): CONTEXT,
}


def field_role(dataset: object, source_column: object) -> str:
    """Return the accounting role for a normalized dataset field.

    Args:
        dataset: Normalized dataset name.
        source_column: Normalized source column name.

    Returns:
        One of ``performance_input``, ``input_component``, ``context``,
        ``reported_performance_component``, or ``unclassified``. Unknown
        fields remain unclassified so comparison and policy validation can
        fail closed.
    """
    if not isinstance(dataset, str) or not isinstance(source_column, str):
        return UNCLASSIFIED
    return _FIELD_ROLES.get((dataset, source_column), UNCLASSIFIED)


def assert_comparison_fields_classified(
    comparison_fields: dict[str, tuple[str, ...]],
) -> None:
    """Raise unless every field on a comparison surface has an explicit role.

    Args:
        comparison_fields: Compared columns keyed by normalized dataset.

    Raises:
        PpaError: If a field could produce a finding without an explicit
            accounting role.
    """
    unclassified = sorted(
        (dataset, source_column)
        for dataset, source_columns in comparison_fields.items()
        for source_column in source_columns
        if field_role(dataset, source_column) == UNCLASSIFIED
    )
    if not unclassified:
        return
    labels = ", ".join(
        f"{dataset}.{source_column}" for dataset, source_column in unclassified
    )
    raise PpaError(
        "SN-12 fail-closed policy invariant failed: comparison fields require "
        f"explicit accounting roles: {labels}.",
        504,
    )


def requires_explicit_impact_policy(dataset: object, source_column: object) -> bool:
    """Return whether a classified field requires explicit impact treatment.

    Args:
        dataset: Normalized dataset name.
        source_column: Normalized source column name.

    Returns:
        ``True`` for direct performance inputs and their input components.

    Raises:
        PpaError: If the field has no explicit accounting role.
    """
    role = field_role(dataset, source_column)
    if role == UNCLASSIFIED:
        raise PpaError(
            "SN-12 fail-closed policy invariant failed: changed field "
            f"{dataset}.{source_column} has no explicit accounting role.",
            504,
        )
    return role in {PERFORMANCE_INPUT, INPUT_COMPONENT}


def is_performance_input(dataset: object, source_column: object) -> bool:
    """Return whether a field directly feeds performance calculation."""
    return field_role(dataset, source_column) == PERFORMANCE_INPUT


def is_input_component(dataset: object, source_column: object) -> bool:
    """Return whether a field explains or reconciles a performance input."""
    return field_role(dataset, source_column) == INPUT_COMPONENT


def is_context(dataset: object, source_column: object) -> bool:
    """Return whether a field is review context only."""
    return field_role(dataset, source_column) == CONTEXT


def is_reported_performance_component(dataset: object, source_column: object) -> bool:
    """Return whether a field is reported performance output or support."""
    return field_role(dataset, source_column) == REPORTED_PERFORMANCE_COMPONENT


def is_classified(dataset: object, source_column: object) -> bool:
    """Return whether a normalized field has an explicit accounting role."""
    return field_role(dataset, source_column) != UNCLASSIFIED
