"""Resolve explicit or inferred Axys source column names."""

from __future__ import annotations

# Python imports
from collections.abc import Callable

# Project imports
from ppar.errors import PpaError

ErrorMessage = Callable[[str], str]


def resolve_column(
    field_name: str,
    aliases: tuple[str, ...],
    available_columns: set[str],
    error_message: ErrorMessage,
    *,
    explicit_column: object | None = None,
    ambiguous_message: str,
    error_code: int,
) -> str | None:
    """Return an explicit column, inferred alias, or ``None`` when missing.

    Args:
        field_name: Logical field being resolved.
        aliases: Candidate CSV column names in inference priority order.
        available_columns: CSV header columns available for matching.
        error_message: Callback adding Axys source context to validation
            details.
        explicit_column: Explicitly configured CSV column name, if present.
        ambiguous_message: Error message prefix used when multiple aliases
            match.
        error_code: PPA error code to use for ambiguity errors.

    Returns:
        The explicit or inferred CSV column name, or ``None`` if no alias
        matches and no explicit column was supplied.

    Raises:
        PpaError: If more than one alias matches the available CSV columns.
    """
    if explicit_column is not None:
        return str(explicit_column) if explicit_column in available_columns else None

    matches = [alias for alias in aliases if alias in available_columns]
    if len(matches) > 1:
        raise PpaError(
            error_message(f"{ambiguous_message} for {field_name!r}: {matches}."),
            error_code,
        )
    return matches[0] if matches else None
