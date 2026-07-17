"""Apply authoritative portfolio base currency to portfolio-scoped source rows."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
import ppar.utilities as util

_NORMALIZED_PORTFOLIO_ID = "_ppar_normalized_portfolio_id"
_NORMALIZED_ROW_CURRENCY = "_ppar_normalized_row_currency"
_AUTHORITATIVE_BASE_CURRENCY = "_ppar_authoritative_base_currency"


def with_authoritative_base_currency(
    frame: pl.DataFrame,
    portfolio_performance: pl.DataFrame,
    *,
    dataset_name: str,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> pl.DataFrame:
    """Return source rows validated against portfolio base currency.

    Portfolio performance is the authoritative source of portfolio reporting
    currency. When it supplies a base currency, this function fills a missing
    row-level value and rejects a contradictory value. Legacy comparisons that
    do not supply portfolio base currency remain supported.

    Args:
        frame: Normalized portfolio-scoped source rows.
        portfolio_performance: Normalized portfolio performance rows from the
            same snapshot.
        dataset_name: Normalized name of ``frame`` for error reporting.
        path: Source CSV path for ``frame``.
        specification_path: Comparison YAML path for error reporting.

    Returns:
        Source rows whose base currency agrees with the authoritative portfolio
        value wherever that value is available.

    Raises:
        PpaError: If one portfolio has conflicting performance-row currencies
            or a source row contradicts its portfolio currency.
    """
    if (
        pc_cols.BASE_CURRENCY not in portfolio_performance.columns
        or pc_cols.PORTFOLIO_ID not in frame.columns
    ):
        return frame

    base_currency_by_portfolio = _base_currency_by_portfolio(
        portfolio_performance,
        specification_path=specification_path,
    )
    if not base_currency_by_portfolio:
        return frame

    authoritative_currencies = pl.DataFrame(
        {
            _NORMALIZED_PORTFOLIO_ID: list(base_currency_by_portfolio),
            _AUTHORITATIVE_BASE_CURRENCY: list(
                base_currency_by_portfolio.values()
            ),
        }
    )
    row_currency = (
        _normalized_currency_expression(pc_cols.BASE_CURRENCY)
        if pc_cols.BASE_CURRENCY in frame.columns
        else pl.lit(None, dtype=pl.String)
    )
    joined = (
        frame.with_columns(
            _normalized_text_expression(pc_cols.PORTFOLIO_ID).alias(
                _NORMALIZED_PORTFOLIO_ID
            ),
            row_currency.alias(_NORMALIZED_ROW_CURRENCY),
        )
        .join(
            authoritative_currencies,
            on=_NORMALIZED_PORTFOLIO_ID,
            how="left",
            maintain_order="left",
        )
    )
    conflicts = joined.filter(
        pl.col(_AUTHORITATIVE_BASE_CURRENCY).is_not_null()
        & pl.col(_NORMALIZED_ROW_CURRENCY).is_not_null()
        & (
            pl.col(_NORMALIZED_ROW_CURRENCY)
            != pl.col(_AUTHORITATIVE_BASE_CURRENCY)
        )
    ).head(5)
    if not conflicts.is_empty():
        samples = "; ".join(
            (
                f"portfolio_id={portfolio_id}, row={row_currency_value}, "
                f"portfolio={authoritative_currency}"
            )
            for portfolio_id, row_currency_value, authoritative_currency in conflicts.select(
                _NORMALIZED_PORTFOLIO_ID,
                _NORMALIZED_ROW_CURRENCY,
                _AUTHORITATIVE_BASE_CURRENCY,
            ).iter_rows()
        )
        raise PpaError(
            (
                f"{specification_path}: {dataset_name} file {path} contains "
                "base_currency values that conflict with authoritative "
                f"portfolio_performance values. Sample rows: {samples}"
            ),
            504,
        )

    existing_currency = (
        pl.col(pc_cols.BASE_CURRENCY)
        if pc_cols.BASE_CURRENCY in frame.columns
        else pl.lit(None, dtype=pl.String)
    )
    updated = joined.with_columns(
        pl.when(pl.col(_AUTHORITATIVE_BASE_CURRENCY).is_not_null())
        .then(pl.col(_AUTHORITATIVE_BASE_CURRENCY))
        .otherwise(existing_currency)
        .alias(pc_cols.BASE_CURRENCY)
    ).drop(
        _NORMALIZED_PORTFOLIO_ID,
        _NORMALIZED_ROW_CURRENCY,
        _AUTHORITATIVE_BASE_CURRENCY,
    )
    return updated.select(
        *frame.columns,
        *(() if pc_cols.BASE_CURRENCY in frame.columns else (pc_cols.BASE_CURRENCY,)),
    )


def _base_currency_by_portfolio(
    portfolio_performance: pl.DataFrame,
    *,
    specification_path: util.PathLike,
) -> dict[str, str]:
    """Return one authoritative base currency per portfolio."""
    normalized = portfolio_performance.select(
        _normalized_text_expression(pc_cols.PORTFOLIO_ID).alias(
            pc_cols.PORTFOLIO_ID
        ),
        _normalized_currency_expression(pc_cols.BASE_CURRENCY).alias(
            pc_cols.BASE_CURRENCY
        ),
    ).filter(
        (pl.col(pc_cols.PORTFOLIO_ID) != "")
        & pl.col(pc_cols.BASE_CURRENCY).is_not_null()
    )
    if normalized.is_empty():
        return {}
    currencies = normalized.group_by(pc_cols.PORTFOLIO_ID).agg(
        pl.col(pc_cols.BASE_CURRENCY).unique().sort()
    )
    conflicts = currencies.filter(
        pl.col(pc_cols.BASE_CURRENCY).list.len() > 1
    ).sort(pc_cols.PORTFOLIO_ID)
    if not conflicts.is_empty():
        samples = "; ".join(
            f"portfolio_id={portfolio_id}, currencies={','.join(values)}"
            for portfolio_id, values in conflicts.head(5).iter_rows()
        )
        raise PpaError(
            (
                f"{specification_path}: portfolio_performance must provide one "
                f"base_currency per portfolio. Conflicts: {samples}"
            ),
            504,
        )
    return dict(
        currencies.select(
            pc_cols.PORTFOLIO_ID,
            pl.col(pc_cols.BASE_CURRENCY).list.first(),
        ).iter_rows()
    )


def _normalized_currency_expression(column: str) -> pl.Expr:
    """Return an expression for normalized optional currency text."""
    return (
        _normalized_text_expression(column)
        .str.to_uppercase()
        .replace("", None)
    )


def _normalized_text_expression(column: str) -> pl.Expr:
    """Return an expression matching normalized Python text conversion."""
    return pl.col(column).cast(pl.String, strict=False).fill_null("").str.strip_chars()


def _normalized_currency(value: object) -> str | None:
    """Return an uppercase currency value or ``None`` for a blank value."""
    text = _normalized_text(value)
    return text.upper() if text else None


def _normalized_text(value: object) -> str:
    """Return a stripped text representation for source identifiers."""
    return "" if value is None else str(value).strip()
