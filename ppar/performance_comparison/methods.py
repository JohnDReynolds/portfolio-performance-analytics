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


class HoldingImpactMethod(StrEnum):
    """Supported `holding_impact_methods` YAML method names."""

    EVIDENCE_ONLY = "evidence_only"
    QUANTITY_DELTA_TIMES_SNAPSHOT_A_UNIT_MARKET_VALUE_OVER_RETURN_DENOMINATOR = (
        "quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator"
    )
    MARKET_VALUE_DELTA_OVER_RETURN_DENOMINATOR = (
        "market_value_delta_over_return_denominator"
    )
    ACCRUED_DELTA_OVER_RETURN_DENOMINATOR = (
        "accrued_delta_over_return_denominator"
    )


class PriceImpactMethod(StrEnum):
    """Supported `price_impact_methods` YAML method names."""

    PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT = (
        "price_delta_over_snapshot_a_price_times_weight"
    )


class CashImpactMethod(StrEnum):
    """Supported `cash_impact_methods` YAML method names."""

    CASH_DELTA_OVER_RETURN_DENOMINATOR = "cash_delta_over_return_denominator"


class FxRateImpactMethod(StrEnum):
    """Supported `fx_rate_impact_methods` YAML method names."""

    EVIDENCE_ONLY = "evidence_only"


class SecurityMasterImpactMethod(StrEnum):
    """Supported `security_master_impact_methods` YAML method names."""

    EVIDENCE_ONLY = "evidence_only"


class ModifiedDietzFlowTiming(StrEnum):
    """Supported Modified Dietz flow date source options."""

    TRADE_DATE = "trade_date"
    SETTLEMENT_DATE = "settlement_date"


class ModifiedDietzDayCount(StrEnum):
    """Supported Modified Dietz day-count conventions."""

    ACTUAL_DAYS = "actual_days"


class ModifiedDietzInclusionRule(StrEnum):
    """Supported Modified Dietz flow inclusion timing rules."""

    BEGINNING_OF_DAY = "beginning_of_day"
    END_OF_DAY = "end_of_day"


class ModifiedDietzDoubleCountPolicy(StrEnum):
    """Supported Modified Dietz double-counting guardrail policies."""

    CROSS_CHECK_ONLY = "cross_check_only"


class ReturnReconstructionMethod(StrEnum):
    """Supported return-reconstruction method names."""

    MODIFIED_DIETZ = "modified_dietz"
    MODIFIED_SIMPLE_DIETZ = "modified_simple_dietz"
    SIMPLE_DIETZ = "simple_dietz"


class ReturnReconstructionValueSource(StrEnum):
    """Supported return-reconstruction beginning/ending value sources."""

    HOLDINGS = "holdings"


class ReturnReconstructionFlowSource(StrEnum):
    """Supported return-reconstruction flow sources."""

    TRANSACTIONS = "transactions"


class ReturnBasis(StrEnum):
    """Supported reported-return basis labels."""

    GROSS = "gross"
    NET = "net"


class ReturnReconstructionSignConvention(StrEnum):
    """Supported return-reconstruction amount sign conventions."""

    SIGNED_AMOUNT = "signed_amount"
