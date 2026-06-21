"""Transaction diagnostic presentation helpers for performance comparison."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping

# Project imports
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.findings import (
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_ID_UNMATCHED,
    TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_SEMANTICS_SOURCE_MIXED,
    TRANSACTION_SEMANTICS_SOURCE_SOURCE,
    TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
)


def transaction_match_review_note(match_status: object) -> str:
    """Return a reviewer-facing explanation for one transaction match status.

    Args:
        match_status: Transaction match-status value from the findings table.

    Returns:
        Human-readable text explaining how to interpret the matching status.
    """
    if match_status == TRANSACTION_MATCH_STATUS_ID_MATCH:
        return "Changed fields were compared on rows matched by transaction_id."
    if match_status == TRANSACTION_MATCH_STATUS_ID_UNMATCHED:
        return "Rows were not paired by transaction_id; review as adds/drops."
    if match_status == TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED:
        return (
            "No stable transaction_id was available; strict fallback keys left "
            "rows unmatched rather than inferring an edit."
        )
    return "Review this transaction matching status before interpreting activity."


def transaction_matching_diagnostic_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    """Return stable ordering for transaction matching diagnostic rows.

    Args:
        row: Transaction matching diagnostic row.

    Returns:
        Sort key that puts known match statuses in review order before unknown
        statuses.
    """
    order = {
        TRANSACTION_MATCH_STATUS_ID_MATCH.value: 0,
        TRANSACTION_MATCH_STATUS_ID_UNMATCHED.value: 1,
        TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED.value: 2,
    }
    match_status = str(row[TRANSACTION_MATCH_STATUS])
    return order.get(match_status, len(order)), str(match_status)


def readable_transaction_semantics_source(value: object) -> str:
    """Return reviewer-facing text for a transaction semantics provenance value.

    Args:
        value: Raw transaction semantics provenance value.

    Returns:
        Human-readable provenance label.
    """
    if value == TRANSACTION_SEMANTICS_SOURCE_SOURCE:
        return "source"
    if value == TRANSACTION_SEMANTICS_SOURCE_YAML_RULE:
        return "YAML transaction_rules"
    if value == TRANSACTION_SEMANTICS_SOURCE_MIXED:
        return "mixed source and YAML transaction_rules"
    if value == TRANSACTION_SEMANTICS_SOURCE_UNKNOWN:
        return "unknown"
    if value is None or value == "":
        return "not provided"
    return str(value)


def format_label_counts(counts: Mapping[str, int]) -> str:
    """Return stable readable counts for free-form labels.

    Args:
        counts: Label counts keyed by display label.

    Returns:
        Comma-delimited ``label: count`` text with labels sorted alphabetically.
    """
    return ", ".join(
        f"{label}: {counts[label]}"
        for label in sorted(counts)
        if counts.get(label, 0) > 0
    )


def parse_transaction_semantics_sources(value: object) -> dict[str, int]:
    """Return provenance counts parsed from a transaction summary string.

    Args:
        value: Summary text such as ``"source: 2, yaml_rule: 1"``.

    Returns:
        Counts keyed by transaction semantics provenance source.
    """
    if not isinstance(value, str) or not value:
        return {}

    counts: dict[str, int] = {}
    for part in value.split(","):
        source, separator, count_text = part.strip().partition(":")
        if not source or not separator:
            continue
        try:
            count = int(count_text.strip())
        except ValueError:
            continue
        counts[source.strip()] = counts.get(source.strip(), 0) + count
    return counts


def format_transaction_semantics_source_counts(counts: Mapping[str, int]) -> str:
    """Return stable readable transaction semantics provenance counts.

    Args:
        counts: Counts keyed by transaction semantics provenance source.

    Returns:
        Comma-delimited counts with known provenance labels in business order.
    """
    ordered_sources = (
        TRANSACTION_SEMANTICS_SOURCE_SOURCE,
        TRANSACTION_SEMANTICS_SOURCE_MIXED,
        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
        TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    )
    parts = [
        f"{source}: {counts[source]}"
        for source in ordered_sources
        if counts.get(source, 0) > 0
    ]
    other_sources = sorted(source for source in counts if source not in ordered_sources)
    parts.extend(
        f"{source}: {counts[source]}"
        for source in other_sources
        if counts.get(source, 0) > 0
    )
    return ", ".join(parts)


def transaction_field_sort_key(field: str) -> tuple[int, str]:
    """Return stable business-oriented transaction field ordering.

    Args:
        field: Transaction source-column name.

    Returns:
        Sort key that places amount, quantity, and price first.
    """
    order = {
        pc_cols.AMOUNT: 0,
        pc_cols.QUANTITY: 1,
        pc_cols.PRICE: 2,
    }
    return (order.get(field, 99), field)
