"""Stable product vocabulary for Audit Data Issues output.

The enum values in this module are serialized product contracts. Registry
metadata describes the current Data Issues implementation without providing a
generic rules engine or changing finding behavior.
"""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class DataIssueType(StrEnum):
    """Stable machine-readable identifiers for implemented Data Issues checks."""

    DUPLICATE_TRANSACTIONS = "duplicate_transactions"
    DIVIDEND_RATE = "dividend_rate"
    HOLDINGS_ACCRUED_RATE = "holdings_accrued_rate"
    HOLDINGS_PRICE_RANGE = "holdings_price_range"
    MISSING_DIVIDEND = "missing_dividend"
    PA_SA_RATE = "pa_sa_rate"
    PORTFOLIO_MARKET_VALUE_CONTINUITY = "portfolio_market_value_continuity"
    SECURITY_MARKET_VALUE_CONTINUITY = "security_market_value_continuity"
    TRANSACTIONS_PRICE_RANGE = "transactions_price_range"


class DataIssueCategory(StrEnum):
    """Stable reviewer groupings for current and approved Data Issues families."""

    CONTINUITY = "continuity"
    DUPLICATE = "duplicate"
    PRICE = "price"
    INCOME = "income"
    ACCRUED_INTEREST = "accrued_interest"
    POSITION_VALUE = "position_value"
    CORPORATE_ACTION = "corporate_action"


@dataclass(frozen=True)
class DataIssueDefinition:
    """Product facts for one implemented Data Issues check.

    Attributes:
        category: Stable reviewer grouping for the issue.
        mandatory: Whether the check remains active regardless of optional-check
            enablement.
        default_enabled: Whether the check executes when its optional enablement
            is omitted. Mandatory checks are always enabled.
        required_datasets: Normalized datasets needed to evaluate the check.
        supports_absolute_tolerance: Whether the check accepts an absolute
            tolerance in YAML.
        supports_percent_tolerance: Whether the check accepts a percent
            tolerance in YAML.
        reviewer_meaning: Concise, non-conclusive meaning for a reviewer.
    """

    category: DataIssueCategory
    mandatory: bool
    default_enabled: bool
    required_datasets: tuple[str, ...]
    supports_absolute_tolerance: bool
    supports_percent_tolerance: bool
    reviewer_meaning: str


_NUMERIC_CHECK = {
    "supports_absolute_tolerance": True,
    "supports_percent_tolerance": True,
}

DATA_ISSUE_REGISTRY: Final[
    Mapping[DataIssueType, DataIssueDefinition]
] = MappingProxyType(
    {
        DataIssueType.DUPLICATE_TRANSACTIONS: DataIssueDefinition(
            category=DataIssueCategory.DUPLICATE,
            mandatory=False,
            default_enabled=True,
            required_datasets=("transactions",),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning="Exact transaction rows repeat within one snapshot.",
        ),
        DataIssueType.DIVIDEND_RATE: DataIssueDefinition(
            category=DataIssueCategory.INCOME,
            mandatory=False,
            default_enabled=True,
            required_datasets=("transactions", "holdings"),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day dividend rates differ across portfolios.",
        ),
        DataIssueType.HOLDINGS_ACCRUED_RATE: DataIssueDefinition(
            category=DataIssueCategory.ACCRUED_INTEREST,
            mandatory=False,
            default_enabled=True,
            required_datasets=("holdings",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day holdings accrued rates differ across portfolios.",
        ),
        DataIssueType.HOLDINGS_PRICE_RANGE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            mandatory=False,
            default_enabled=True,
            required_datasets=("holdings",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day holdings prices differ across portfolios.",
        ),
        DataIssueType.MISSING_DIVIDEND: DataIssueDefinition(
            category=DataIssueCategory.INCOME,
            mandatory=False,
            default_enabled=True,
            required_datasets=("holdings", "transactions"),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning="A conservatively eligible position may lack a dividend.",
        ),
        DataIssueType.PA_SA_RATE: DataIssueDefinition(
            category=DataIssueCategory.ACCRUED_INTEREST,
            mandatory=False,
            default_enabled=True,
            required_datasets=("transactions", "holdings"),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day purchase/sale accrued rates differ across portfolios.",
        ),
        DataIssueType.PORTFOLIO_MARKET_VALUE_CONTINUITY: (
            DataIssueDefinition(
                category=DataIssueCategory.CONTINUITY,
                mandatory=True,
                default_enabled=True,
                required_datasets=("portfolio_performance",),
                **_NUMERIC_CHECK,
                reviewer_meaning=(
                    "Prior ending and next beginning portfolio market values differ."
                ),
            )
        ),
        DataIssueType.SECURITY_MARKET_VALUE_CONTINUITY: DataIssueDefinition(
            category=DataIssueCategory.CONTINUITY,
            mandatory=True,
            default_enabled=True,
            required_datasets=("security_performance",),
            **_NUMERIC_CHECK,
            reviewer_meaning=(
                "Prior ending and next beginning security market values differ."
            ),
        ),
        DataIssueType.TRANSACTIONS_PRICE_RANGE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            mandatory=False,
            default_enabled=True,
            required_datasets=("transactions",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day transaction prices differ across portfolios.",
        ),
    }
)
