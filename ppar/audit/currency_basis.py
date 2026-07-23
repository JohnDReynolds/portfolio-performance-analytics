"""Define normalized monetary-field currency-basis policy."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import schema as pc_cols

__all__ = [
    "CurrencyBasis",
    "PORTFOLIO_BASE_CURRENCY_BASIS",
    "ROW_CURRENCY_BASIS",
    "FROM_CURRENCY_BASIS",
    "CURRENCY_PAIR_BASIS",
    "base_currency_monetary_value",
    "monetary_field_currency_basis",
    "normalize_currency_columns",
    "row_uses_foreign_currency",
]


class CurrencyBasis(StrEnum):
    """Supported normalized monetary-field currency bases."""

    ROW_CURRENCY = "row_currency"
    PORTFOLIO_BASE_CURRENCY = "portfolio_base_currency"
    FROM_CURRENCY = "from_currency"
    CURRENCY_PAIR = "currency_pair"


ROW_CURRENCY_BASIS: Final[str] = CurrencyBasis.ROW_CURRENCY.value
PORTFOLIO_BASE_CURRENCY_BASIS: Final[str] = (
    CurrencyBasis.PORTFOLIO_BASE_CURRENCY.value
)
FROM_CURRENCY_BASIS: Final[str] = CurrencyBasis.FROM_CURRENCY.value
CURRENCY_PAIR_BASIS: Final[str] = CurrencyBasis.CURRENCY_PAIR.value

_ROW_CURRENCY_MONETARY_FIELDS: Final[set[tuple[str, str]]] = {
    (pc_cols.HOLDINGS, pc_cols.PRICE),
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE),
    (pc_cols.HOLDINGS, pc_cols.COST),
    (pc_cols.HOLDINGS, pc_cols.ACCRUED),
    (pc_cols.TRANSACTIONS, pc_cols.PRICE),
    (pc_cols.TRANSACTIONS, pc_cols.AMOUNT),
    (pc_cols.TRANSACTIONS, pc_cols.COMMISSION),
}
_PORTFOLIO_BASE_MONETARY_FIELDS: Final[set[tuple[str, str]]] = {
    (pc_cols.HOLDINGS, pc_cols.BASE_MARKET_VALUE),
    (pc_cols.HOLDINGS, pc_cols.BASE_ACCRUED),
    (pc_cols.TRANSACTIONS, pc_cols.BASE_AMOUNT),
}
_CURRENCY_COLUMNS: Final[tuple[str, ...]] = (
    pc_cols.CURRENCY,
    pc_cols.BASE_CURRENCY,
    pc_cols.FROM_CURRENCY,
    pc_cols.TO_CURRENCY,
)


def normalize_currency_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a frame with supplied currency codes stripped and uppercased.

    Args:
        frame: Normalized source frame.

    Returns:
        Frame with existing currency columns normalized. Blank values remain
        blank so required-value validation can report them explicitly.
    """
    columns = [column for column in _CURRENCY_COLUMNS if column in frame.columns]
    if not columns:
        return frame
    return frame.with_columns(
        pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase().alias(column)
        for column in columns
    )


def monetary_field_currency_basis(dataset: object, source_column: object) -> str | None:
    """Return the normalized currency basis for a monetary field.

    Detailed unqualified monetary fields use the row's ``currency``. Detailed
    ``base_`` fields use the portfolio's ``base_currency``. FX rates use their
    explicit currency pair.

    Args:
        dataset: Normalized dataset name.
        source_column: Normalized source column name.

    Returns:
        A stable currency-basis label, or ``None`` for nonmonetary fields.
    """
    key = dataset, source_column
    if key in _ROW_CURRENCY_MONETARY_FIELDS:
        return ROW_CURRENCY_BASIS
    if key in _PORTFOLIO_BASE_MONETARY_FIELDS:
        return PORTFOLIO_BASE_CURRENCY_BASIS
    if key == (pc_cols.FX_RATES, pc_cols.LOCAL_EXPOSURE):
        return FROM_CURRENCY_BASIS
    if key == (pc_cols.FX_RATES, pc_cols.FX_RATE):
        return CURRENCY_PAIR_BASIS
    return None


def row_uses_foreign_currency(row: Mapping[str, object]) -> bool:
    """Return whether a row explicitly states different row and base currencies."""
    row_currency = _normalized_currency(row.get(pc_cols.CURRENCY))
    base_currency = _normalized_currency(row.get(pc_cols.BASE_CURRENCY))
    return bool(
        row_currency
        and base_currency
        and row_currency != base_currency
    )


def base_currency_monetary_value(
    row: Mapping[str, object],
    *,
    local_field: str,
    base_field: str,
) -> float | None:
    """Return a base-currency value without treating foreign local data as base.

    An explicit base value always wins. The unqualified value is a safe fallback
    only when row and base currencies are equal or currency metadata is absent
    for a legacy single-currency extract.

    Args:
        row: Normalized detailed source row.
        local_field: Unqualified field whose unit is the row currency.
        base_field: Explicit portfolio-base-currency counterpart.

    Returns:
        The numeric base-currency value, or ``None`` when translation is needed.
    """
    base_value = _numeric_value(row.get(base_field))
    if base_value is not None:
        return base_value
    if row_uses_foreign_currency(row):
        return None
    return _numeric_value(row.get(local_field))


def _normalized_currency(value: object) -> str:
    """Return an uppercase currency code or an empty string."""
    return "" if value is None else str(value).strip().upper()


def _numeric_value(value: object) -> float | None:
    """Return a float for non-boolean numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
