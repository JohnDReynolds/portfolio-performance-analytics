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
    DELIVER_IN_ORIGINAL_COST_INCOMPLETE = "deliver_in_original_cost_incomplete"
    DIVIDEND_RATE = "dividend_rate"
    HOLDINGS_ACCRUED_RATE = "holdings_accrued_rate"
    HOLDINGS_NONPOSITIVE_PRICE = "holdings_nonpositive_price"
    HOLDINGS_PRICE_RANGE = "holdings_price_range"
    HOLDINGS_STALE_PRICE = "holdings_stale_price"
    LARGE_PRICE_VARIATION = "large_price_variation"
    MISSING_DIVIDEND = "missing_dividend"
    PA_SA_RATE = "pa_sa_rate"
    TRANSACTION_SECURITY_TYPE_MISMATCH = "transaction_security_type_mismatch"
    TRANSACTIONS_NONPOSITIVE_PRICE = "transactions_nonpositive_price"
    TRANSACTIONS_PRICE_RANGE = "transactions_price_range"


class DataIssueCategory(StrEnum):
    """Stable reviewer groupings for current and approved Data Issues families."""

    DUPLICATE = "duplicate"
    PRICE = "price"
    INCOME = "income"
    ACCRUED_INTEREST = "accrued_interest"
    POSITION_VALUE = "position_value"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class DataIssueDefinition:
    """Product facts for one implemented Data Issues check.

    Attributes:
        category: Stable reviewer grouping for the issue.
        default_enabled: Whether the check executes when its enablement is
            omitted.
        required_datasets: Normalized datasets needed to evaluate the check.
        supports_absolute_tolerance: Whether the check accepts an absolute
            tolerance in YAML.
        supports_percent_tolerance: Whether the check accepts a percent
            tolerance in YAML.
        reviewer_meaning: Concise, non-conclusive meaning for a reviewer.
        requires_only_filter: Whether explicit enablement requires a nonempty
            ``only`` population filter.
        supports_minimum_calendar_days: Whether the check accepts a positive
            integer calendar-day threshold in YAML.
    """

    category: DataIssueCategory
    default_enabled: bool
    required_datasets: tuple[str, ...]
    supports_absolute_tolerance: bool
    supports_percent_tolerance: bool
    reviewer_meaning: str
    requires_only_filter: bool = False
    supports_minimum_calendar_days: bool = False


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
            default_enabled=True,
            required_datasets=("transactions",),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning="Exact transaction rows repeat within one snapshot.",
        ),
        DataIssueType.DELIVER_IN_ORIGINAL_COST_INCOMPLETE: DataIssueDefinition(
            category=DataIssueCategory.POSITION_VALUE,
            default_enabled=False,
            required_datasets=("transactions",),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "Original cost amount or date is absent for a transaction in "
                "the explicitly configured deliver-in population."
            ),
            requires_only_filter=True,
        ),
        DataIssueType.DIVIDEND_RATE: DataIssueDefinition(
            category=DataIssueCategory.INCOME,
            default_enabled=True,
            required_datasets=("transactions", "holdings"),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day dividend rates differ across portfolios.",
        ),
        DataIssueType.HOLDINGS_ACCRUED_RATE: DataIssueDefinition(
            category=DataIssueCategory.ACCRUED_INTEREST,
            default_enabled=True,
            required_datasets=("holdings",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day holdings accrued rates differ across portfolios.",
        ),
        DataIssueType.HOLDINGS_NONPOSITIVE_PRICE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=False,
            required_datasets=("holdings",),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "A nonzero holding has a zero or negative price in the configured "
                "population."
            ),
            requires_only_filter=True,
        ),
        DataIssueType.HOLDINGS_PRICE_RANGE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=True,
            required_datasets=("holdings",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day holdings prices differ across portfolios.",
        ),
        DataIssueType.HOLDINGS_STALE_PRICE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=False,
            required_datasets=("holdings", "security_master"),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "The same positive holding price recurs across supplied holding "
                "observations for at least the configured calendar-day threshold."
            ),
            requires_only_filter=True,
            supports_minimum_calendar_days=True,
        ),
        DataIssueType.LARGE_PRICE_VARIATION: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=False,
            required_datasets=("portfolio_performance",),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "Split-normalized holding and transaction prices vary beyond a "
                "named rule's minimum tolerance within one portfolio period."
            ),
        ),
        DataIssueType.MISSING_DIVIDEND: DataIssueDefinition(
            category=DataIssueCategory.INCOME,
            default_enabled=True,
            required_datasets=("holdings", "transactions"),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning="A conservatively eligible position may lack a dividend.",
        ),
        DataIssueType.PA_SA_RATE: DataIssueDefinition(
            category=DataIssueCategory.ACCRUED_INTEREST,
            default_enabled=True,
            required_datasets=("transactions", "holdings"),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day purchase/sale accrued rates differ across portfolios.",
        ),
        DataIssueType.TRANSACTIONS_NONPOSITIVE_PRICE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=False,
            required_datasets=("transactions", "security_master"),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "A nonzero-quantity transaction has a zero or negative price in "
                "the configured price-bearing population."
            ),
            requires_only_filter=True,
        ),
        DataIssueType.TRANSACTION_SECURITY_TYPE_MISMATCH: DataIssueDefinition(
            category=DataIssueCategory.CLASSIFICATION,
            default_enabled=False,
            required_datasets=("transactions", "security_master"),
            supports_absolute_tolerance=False,
            supports_percent_tolerance=False,
            reviewer_meaning=(
                "A transaction security type differs from the snapshot reference "
                "type for the same exact-case security ID."
            ),
            requires_only_filter=True,
        ),
        DataIssueType.TRANSACTIONS_PRICE_RANGE: DataIssueDefinition(
            category=DataIssueCategory.PRICE,
            default_enabled=True,
            required_datasets=("transactions",),
            **_NUMERIC_CHECK,
            reviewer_meaning="Same-day transaction prices differ across portfolios.",
        ),
    }
)
