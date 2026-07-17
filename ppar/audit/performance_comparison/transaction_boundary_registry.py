"""Reviewer-facing transaction boundary registry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from ppar.performance_comparison.backlog_gates import (
    CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES,
    SHORT_SIDE_BACKLOG_TRANSACTION_CODES,
)
from ppar.performance_comparison.fixed_income import (
    FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES,
    FIXED_INCOME_BACKLOG_TRANSACTION_CODES,
    FIXED_INCOME_SAFE_TRANSACTION_CODES,
)

PACKAGED_FORMULA_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"by", "dv", "in", "sl"}
)
AMBIGUOUS_CONTEXT_REQUIRED_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"dp", "li", "lo", "wd"}
)
REVIEW_ONLY_TEST_TRANSACTION_CODES: Final[frozenset[str]] = frozenset({";"})
CONTEXT_ONLY_TRANSACTION_CODES: Final[frozenset[str]] = frozenset({"exus"})
STANDALONE_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = frozenset({"epus"})

TRANSACTION_BOUNDARY_REGISTRY: Final[MappingProxyType[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "packaged_formula": PACKAGED_FORMULA_TRANSACTION_CODES,
            "fixed_income_safe": FIXED_INCOME_SAFE_TRANSACTION_CODES,
            "fixed_income_accrued_interest": (
                FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES
            ),
            "ambiguous_context_required": (
                AMBIGUOUS_CONTEXT_REQUIRED_TRANSACTION_CODES
            ),
            "review_only_test": REVIEW_ONLY_TEST_TRANSACTION_CODES,
            "context_only": CONTEXT_ONLY_TRANSACTION_CODES,
            "fixed_income_backlog": FIXED_INCOME_BACKLOG_TRANSACTION_CODES,
            "capital_return_backlog": CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES,
            "short_side_backlog": SHORT_SIDE_BACKLOG_TRANSACTION_CODES,
            "standalone_backlog": STANDALONE_BACKLOG_TRANSACTION_CODES,
        }
    )
)


def transaction_boundary_groups(code: object) -> tuple[str, ...]:
    """Return boundary groups containing one transaction code.

    Args:
        code: Source transaction code.

    Returns:
        Boundary group names containing the normalized code. Unknown, blank, and
        unregistered codes return an empty tuple.
    """
    normalized = "" if code is None else str(code).strip().lower()
    if not normalized:
        return ()
    return tuple(
        group
        for group, codes in TRANSACTION_BOUNDARY_REGISTRY.items()
        if normalized in codes
    )


def registered_transaction_codes() -> frozenset[str]:
    """Return all transaction codes covered by the boundary registry."""
    codes: set[str] = set()
    for group_codes in TRANSACTION_BOUNDARY_REGISTRY.values():
        codes.update(group_codes)
    return frozenset(codes)
