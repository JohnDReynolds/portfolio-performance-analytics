"""Stable vocabulary for the Audit Performance Comparison sub-feature."""

from enum import StrEnum


class CauseArea(StrEnum):
    """Stable coarse cause areas used by performance explanation output."""

    SECURITY_RETURN_OR_CONTRIBUTION = "security_return_or_contribution"
    MARKET_VALUE_OR_HOLDING = "market_value_or_holding"
    TRANSACTION_ACTIVITY = "transaction_activity"
    PORTFOLIO_PERFORMANCE_INPUT = "portfolio_performance_input"
    CLASSIFICATION_OR_REFERENCE = "classification_or_reference"
    UNEXPLAINED = "unexplained"
