"""Classify normalized performance-comparison fields by accounting role."""

from __future__ import annotations

# Python imports
from typing import Final

# Project imports
from ppar.performance_comparison import schema as pc_cols

PERFORMANCE_INPUT: Final[str] = "performance_input"
INPUT_COMPONENT: Final[str] = "input_component"
CONTEXT: Final[str] = "context"
REPORTED_PERFORMANCE_COMPONENT: Final[str] = "reported_performance_component"

_FIELD_ROLES: Final[dict[tuple[str, str], str]] = {
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.ACCRUED): PERFORMANCE_INPUT,
    (pc_cols.TRANSACTIONS, pc_cols.AMOUNT): PERFORMANCE_INPUT,
    (pc_cols.CASH, pc_cols.CASH_BALANCE): PERFORMANCE_INPUT,
    (pc_cols.CASH, pc_cols.MARKET_VALUE): PERFORMANCE_INPUT,
    (pc_cols.HOLDINGS, pc_cols.QUANTITY): INPUT_COMPONENT,
    (pc_cols.HOLDINGS, pc_cols.PRICE): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.QUANTITY): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.PRICE): INPUT_COMPONENT,
    (pc_cols.TRANSACTIONS, pc_cols.COMMISSION): INPUT_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.PORTFOLIO_RETURN): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.INCOME): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.GAIN_LOSS): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.BEGIN_MARKET_VALUE): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.END_MARKET_VALUE): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.FLOW): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.SECURITY_RETURN): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.CONTRIBUTION): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.WEIGHT): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.BEGIN_MARKET_VALUE): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.END_MARKET_VALUE): (
        REPORTED_PERFORMANCE_COMPONENT
    ),
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.INCOME): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.SECURITY_PERFORMANCE, pc_cols.GAIN_LOSS): REPORTED_PERFORMANCE_COMPONENT,
    (pc_cols.FX_RATES, pc_cols.FX_RATE): CONTEXT,
    (pc_cols.HOLDINGS, pc_cols.COST): CONTEXT,
}


def field_role(dataset: object, source_column: object) -> str:
    """Return the accounting role for a normalized dataset field.

    Args:
        dataset: Normalized dataset name.
        source_column: Normalized source column name.

    Returns:
        One of ``performance_input``, ``input_component``, ``context``, or
        ``reported_performance_component``. Unknown fields default to
        ``context`` so they cannot silently become additive.
    """
    if not isinstance(dataset, str) or not isinstance(source_column, str):
        return CONTEXT
    return _FIELD_ROLES.get((dataset, source_column), CONTEXT)


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
