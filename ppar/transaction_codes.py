"""Exact transaction-code matching helpers without economic policy."""

from __future__ import annotations


def transaction_code_matching_key(value: object) -> str:
    """Return a stripped native-case transaction-code comparison key.

    Args:
        value: Native source transaction code.

    Returns:
        Stripped native-case code. Missing values return an empty string.

    Notes:
        Case is intentionally preserved. Local economic meaning belongs in
        exact-case Audit ``transaction_rules``.
    """
    if value is None:
        return ""
    return str(value).strip()
