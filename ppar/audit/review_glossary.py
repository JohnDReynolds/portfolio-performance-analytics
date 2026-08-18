"""Define concise reviewer-facing explanations for specialized Audit terms."""

from __future__ import annotations

# Python imports
from typing import Final

_TERM_TOOLTIPS: Final[dict[str, str]] = {
    "Total Quantity": (
        "Total portfolios or periods evaluated, including those with no reported "
        "performance difference."
    ),
    "No Performance Differences": (
        "Count with no reported performance difference beyond the configured "
        "comparison tolerance."
    ),
    "Fully Explained": (
        "The reported performance difference is accounted for by supported, "
        "quantified causes within the configured tolerance."
    ),
    "Fully Explained Differences": (
        "Count of reported performance differences accounted for by supported, "
        "quantified causes within the configured tolerance."
    ),
    "Partly Explained": (
        "Supported, quantified causes account for part, but not all, of the "
        "reported performance difference."
    ),
    "Partly Explained Differences": (
        "Count of reported performance differences for which supported, quantified "
        "causes account for part, but not all, of the difference."
    ),
    "Unexplained": (
        "PPAR did not quantify a supported cause for the reported performance "
        "difference."
    ),
    "Unexplained Differences": (
        "Count of reported performance differences for which PPAR did not quantify "
        "a supported cause."
    ),
    "Missing YAML Specifications": (
        "PPAR cannot finalize the explanation because required YAML treatment is "
        "missing."
    ),
    "Setup Incomplete": (
        "Count of reported performance differences that cannot be finalized because "
        "required YAML treatment is missing."
    ),
    "Explained Cause": (
        "A supported, quantified cause included in the Explained Difference."
    ),
    "Possible Cause": (
        "Relevant evidence that may help explain the difference but is not counted "
        "in the Explained Difference."
    ),
    "Supporting Evidence": (
        "Evidence supporting an explanation without being independently counted as "
        "another cause."
    ),
    "Review Context": (
        "Relevant changed data shown for review but not counted in the Explained "
        "Difference."
    ),
    "Issue Type": "Cross-reference consistency check reported by Audit.",
    "Quantity": "Number of Data Issues of this type.",
}
_TOOLTIP_VALUE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"input_role", "review_status", "row_type"}
)


def audit_term_tooltip(term: object) -> str:
    """Return the concise Audit definition for a controlled review term.

    Args:
        term: User-facing term or value from an Audit review table.

    Returns:
        Tooltip text when the term has a curated definition; otherwise an empty
        string.
    """
    normalized_term = " ".join(str(term).split())
    return _TERM_TOOLTIPS.get(normalized_term, "")


def explanation_status_tooltip() -> str:
    """Return the compact glossary for the four performance review statuses."""
    return " ".join(
        (
            f"Fully Explained: {audit_term_tooltip('Fully Explained')}",
            f"Partly Explained: {audit_term_tooltip('Partly Explained')}",
            f"Unexplained: {audit_term_tooltip('Unexplained')}",
            "Setup Incomplete: "
            f"{audit_term_tooltip('Missing YAML Specifications')}",
        )
    )


def audit_value_tooltip(column: str, value: object) -> str:
    """Return a curated tooltip for a controlled Audit table value.

    Args:
        column: Internal review-table column name.
        value: Value displayed in the review table cell.

    Returns:
        Tooltip text for selected controlled-value columns; otherwise an empty
        string.
    """
    if column not in _TOOLTIP_VALUE_COLUMNS:
        return ""
    return audit_term_tooltip(value)
