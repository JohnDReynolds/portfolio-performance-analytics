"""Backlog boundary helpers for high-risk transaction families."""

from __future__ import annotations

from typing import Final

CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"pd", "rc"}
)
SHORT_SIDE_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = frozenset({"cs", "ss"})
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


def transaction_backlog_gate(code: object) -> str:
    """Return the explicit backlog gate for one transaction code.

    Args:
        code: Source transaction code.

    Returns:
        ``"capital_return_policy"`` for return-of-capital or principal-paydown
        rows, ``"short_side_evidence"`` for short-sale or cover-short rows, and
        ``"not_backlog_gate"`` otherwise.
    """
    normalized = "" if code is None else str(code).strip().lower()
    if normalized in CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES:
        return "capital_return_policy"
    if normalized in SHORT_SIDE_BACKLOG_TRANSACTION_CODES:
        return "short_side_evidence"
    return "not_backlog_gate"
