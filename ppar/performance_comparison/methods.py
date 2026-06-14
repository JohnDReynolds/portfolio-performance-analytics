"""Shared performance comparison YAML method names.

The comparison YAML uses string tokens because users edit the file directly.
These enums keep the implementation anchored to one source of truth while
preserving the plain string values written in YAML, reports, and audit outputs.
"""

from __future__ import annotations

# Python imports
from enum import StrEnum


class ContributionImpactMethod(StrEnum):
    """Supported `contribution_impact_methods` YAML method names."""

    SOURCE_FIELD_DELTA_OVER_BEGIN_MARKET_VALUE = (
        "source_field_delta_over_begin_market_value"
    )
    VENDOR_CONTRIBUTION_DELTA = "vendor_contribution_delta"
    SECURITY_RETURN_DELTA_TIMES_WEIGHT = "security_return_delta_times_weight"


class TransactionImpactMethod(StrEnum):
    """Supported `transaction_impact_methods` YAML method names."""

    EVIDENCE_ONLY = "evidence_only"
    MODIFIED_DIETZ = "modified_dietz"
    TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR = (
        "transaction_amount_delta_over_return_denominator"
    )
