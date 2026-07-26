"""Maintainer-only transaction coverage groupings.

These sets support fixture and research checks. Runtime product code must not
import this module or treat these groupings as economic transaction policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ppar.transaction_codes import transaction_code_matching_key

TRANSACTION_EVIDENCE_GROUPS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "packaged_formula": frozenset({"by", "dv", "in", "sl"}),
            "contextual_packaged": frozenset({"ai", "ti"}),
            "ambiguous_context_required": frozenset({"dp", "li", "lo", "ti", "wd"}),
            "context_only": frozenset({"epus", "exus"}),
            "fixed_income_safe": frozenset({"in"}),
            "fixed_income_accrued_interest": frozenset({"pa", "sa"}),
            "fixed_income_backlog": frozenset({"ai", "pd"}),
            "capital_return_backlog": frozenset({"pd", "rc"}),
            "short_side_backlog": frozenset({"cs", "ss"}),
            "standalone_backlog": frozenset({"epus"}),
        }
    )
)

FIXED_INCOME_SAFE_TRANSACTION_CODES: Final[frozenset[str]] = (
    TRANSACTION_EVIDENCE_GROUPS["fixed_income_safe"]
)
FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES: Final[frozenset[str]] = (
    TRANSACTION_EVIDENCE_GROUPS["fixed_income_accrued_interest"]
)
FIXED_INCOME_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = (
    TRANSACTION_EVIDENCE_GROUPS["fixed_income_backlog"]
)
CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = (
    TRANSACTION_EVIDENCE_GROUPS["capital_return_backlog"]
)
SHORT_SIDE_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = (
    TRANSACTION_EVIDENCE_GROUPS["short_side_backlog"]
)
FIXED_INCOME_FORMULA_INPUTS: Final[tuple[str, ...]] = (
    "ordinary interest transaction amounts",
    "configured holdings.accrued changes",
    "paired purchase/sale accrued-interest adjunct amounts",
)
FIXED_INCOME_OUT_OF_SCOPE: Final[tuple[str, ...]] = (
    "amortization/accretion engine",
    "bond principal schedule reconstruction",
    "yield calculation",
    "tax-lot accounting",
)
CAPITAL_RETURN_POSSIBLE_ROLES: Final[tuple[str, ...]] = (
    "performance income",
    "corporate-action evidence",
    "review-only evidence",
)
CAPITAL_RETURN_REQUIRED_EVIDENCE: Final[tuple[str, ...]] = (
    "security identity",
    "amount sign",
    "cost-basis or principal context",
    "local mapping or REP/report semantics",
)
SHORT_SIDE_REQUIRED_EVIDENCE: Final[tuple[str, ...]] = (
    "short security type",
    "cash, margin, or short-account symbols",
    "amount and quantity signs",
    "local mapping or REP/report semantics",
)


def transaction_evidence_groups(code: object) -> tuple[str, ...]:
    """Return maintainer evidence groups containing one exact-case code."""
    normalized = transaction_code_matching_key(code)
    if not normalized:
        return ()
    return tuple(
        group
        for group, codes in TRANSACTION_EVIDENCE_GROUPS.items()
        if normalized in codes
    )


def registered_transaction_evidence_codes() -> frozenset[str]:
    """Return all codes covered by maintainer evidence groups."""
    return frozenset(
        code
        for group_codes in TRANSACTION_EVIDENCE_GROUPS.values()
        for code in group_codes
    )


def fixed_income_transaction_boundary(code: object) -> str:
    """Return the maintainer evidence treatment for a fixed-income code."""
    normalized = transaction_code_matching_key(code)
    if normalized in FIXED_INCOME_SAFE_TRANSACTION_CODES:
        return "safe_income"
    if normalized in FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES:
        return "accrued_interest_adjunct"
    if normalized in FIXED_INCOME_BACKLOG_TRANSACTION_CODES:
        return "backlog"
    return "not_fixed_income_boundary"


def transaction_backlog_gate(code: object) -> str:
    """Return the maintainer backlog family for one exact-case code."""
    normalized = transaction_code_matching_key(code)
    if normalized in CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES:
        return "capital_return_policy"
    if normalized in SHORT_SIDE_BACKLOG_TRANSACTION_CODES:
        return "short_side_evidence"
    return "not_backlog_gate"
