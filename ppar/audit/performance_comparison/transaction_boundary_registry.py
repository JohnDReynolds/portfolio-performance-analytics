"""Reviewer-facing transaction boundary registry."""

from __future__ import annotations

from typing import Final

from ppar.audit.transaction_policy import (
    transaction_boundary_codes,
    transaction_boundary_registry,
)

PACKAGED_FORMULA_TRANSACTION_CODES: Final[frozenset[str]] = (
    transaction_boundary_codes("packaged_formula")
)
CONTEXTUAL_PACKAGED_TRANSACTION_CODES: Final[frozenset[str]] = (
    transaction_boundary_codes("contextual_packaged")
)
AMBIGUOUS_CONTEXT_REQUIRED_TRANSACTION_CODES: Final[frozenset[str]] = (
    transaction_boundary_codes("ambiguous_context_required")
)
REVIEW_ONLY_TEST_TRANSACTION_CODES: Final[frozenset[str]] = (
    transaction_boundary_codes("review_only_test")
)
CONTEXT_ONLY_TRANSACTION_CODES: Final[frozenset[str]] = transaction_boundary_codes(
    "context_only"
)
STANDALONE_BACKLOG_TRANSACTION_CODES: Final[frozenset[str]] = (
    transaction_boundary_codes("standalone_backlog")
)

TRANSACTION_BOUNDARY_REGISTRY: Final = transaction_boundary_registry()


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
