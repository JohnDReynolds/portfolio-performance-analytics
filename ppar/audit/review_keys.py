"""Build stable review keys for performance-comparison artifacts."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import rendering as _pc_rendering

__all__ = [
    "REVIEW_KEY",
    "period_key",
    "period_review_key",
    "row_review_key",
    "with_period_review_key",
    "with_security_review_key",
]

REVIEW_KEY = "review_key"


def period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the portfolio-period grouping key for a report row.

    Args:
        row: Row-like mapping containing portfolio and period columns.

    Returns:
        Tuple of portfolio id, from date, and through date.
    """
    return (
        row[_pc_findings.PORTFOLIO_ID],
        row[_pc_findings.FROM_DATE],
        row[_pc_findings.THRU_DATE],
    )


def period_review_key(row: Mapping[str, object]) -> str:
    """Return a stable text key for joining period-level review artifacts.

    Args:
        row: Row-like mapping containing portfolio and period columns.

    Returns:
        Stable ``portfolio::from_date::thru_date`` text key.
    """
    return "::".join(
        [
            _pc_rendering.format_value(row.get(_pc_findings.PORTFOLIO_ID)),
            _pc_rendering.format_value(row.get(_pc_findings.FROM_DATE)),
            _pc_rendering.format_value(row.get(_pc_findings.THRU_DATE)),
        ]
    )


def with_period_review_key(table: pl.DataFrame) -> pl.DataFrame:
    """Add ``review_key`` to tables that already carry portfolio-period columns.

    Args:
        table: Source table.

    Returns:
        Table with ``review_key`` as the first column when period columns are
        available. Tables that already have a review key, or cannot support
        one, are returned unchanged.
    """
    period_columns = {
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
    }
    if REVIEW_KEY in table.columns or not period_columns.issubset(table.columns):
        return table
    table_with_key = table.with_columns(
        pl.concat_str(
            [
                pl.col(_pc_findings.PORTFOLIO_ID).cast(pl.String),
                pl.col(_pc_findings.FROM_DATE).cast(pl.String),
                pl.col(_pc_findings.THRU_DATE).cast(pl.String),
            ],
            separator="::",
        ).alias(REVIEW_KEY)
    )
    return table_with_key.select(
        [REVIEW_KEY, *[column for column in table.columns if column != REVIEW_KEY]]
    )


def with_security_review_key(table: pl.DataFrame) -> pl.DataFrame:
    """Add ``review_key`` to tables that already carry security-period columns.

    Args:
        table: Source table.

    Returns:
        Table with ``review_key`` as the first column when security-period
        columns are available. Tables that already have a review key, or cannot
        support one, are returned unchanged.
    """
    security_columns = {
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.SECURITY_ID,
    }
    if REVIEW_KEY in table.columns or not security_columns.issubset(table.columns):
        return table
    table_with_key = table.with_columns(
        pl.concat_str(
            [
                pl.col(_pc_findings.PORTFOLIO_ID).cast(pl.String),
                pl.col(_pc_findings.FROM_DATE).cast(pl.String),
                pl.col(_pc_findings.THRU_DATE).cast(pl.String),
                pl.col(_pc_findings.SECURITY_ID).cast(pl.String),
            ],
            separator="::",
        ).alias(REVIEW_KEY)
    )
    return table_with_key.select(
        [REVIEW_KEY, *[column for column in table.columns if column != REVIEW_KEY]]
    )


def row_review_key(row: Mapping[str, object]) -> str:
    """Return a row's review key when enough period fields are available.

    Args:
        row: Row-like mapping from a report or workbook table.

    Returns:
        Existing ``review_key`` value, derived period key, or an empty string
        when the row does not have enough period fields.
    """
    if _has_text(row.get(REVIEW_KEY)):
        return _pc_rendering.format_value(row.get(REVIEW_KEY))
    period_columns = {
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
    }
    if not period_columns.issubset(row.keys()):
        return ""
    return period_review_key(row)


def _has_text(value: object) -> bool:
    """Return whether a value has non-blank text."""
    return isinstance(value, str) and bool(value.strip())
