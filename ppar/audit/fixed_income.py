"""Fixed-income boundary helpers for performance comparison review."""

from __future__ import annotations

from typing import Final

FIXED_INCOME_SAFE_TRANSACTION_CODES: Final[frozenset[str]] = frozenset({"in"})
FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"pa", "sa"}
)
FIXED_INCOME_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"ai", "pd"}
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


def fixed_income_transaction_boundary(code: object) -> str:
    """Return fixed-income boundary treatment for one transaction code.

    Args:
        code: Source transaction code.

    Returns:
        ``"safe_income"`` for ordinary interest rows,
        ``"accrued_interest_adjunct"`` for promoted purchase/sale accrued
        interest adjuncts, ``"backlog"`` for fixed-income rows that still need
        more evidence, and ``"not_fixed_income_boundary"`` otherwise.
    """
    normalized = "" if code is None else str(code).strip().lower()
    if normalized in FIXED_INCOME_SAFE_TRANSACTION_CODES:
        return "safe_income"
    if normalized in FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES:
        return "accrued_interest_adjunct"
    if normalized in FIXED_INCOME_BACKLOG_TRANSACTION_CODES:
        return "backlog"
    return "not_fixed_income_boundary"
