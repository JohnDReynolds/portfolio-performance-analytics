"""Apply authoritative portfolio base currency to portfolio-scoped source rows."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as pc_cols
import ppar.utilities as util


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

    rows: list[dict[str, object]] = []
    conflicts: list[str] = []
    for row in frame.iter_rows(named=True):
        updated_row = dict(row)
        portfolio_id = _normalized_text(row.get(pc_cols.PORTFOLIO_ID))
        authoritative_currency = base_currency_by_portfolio.get(portfolio_id)
        if authoritative_currency is None:
            rows.append(updated_row)
            continue

        row_currency = _normalized_currency(row.get(pc_cols.BASE_CURRENCY))
        if row_currency is not None and row_currency != authoritative_currency:
            conflicts.append(
                f"portfolio_id={portfolio_id}, row={row_currency}, "
                f"portfolio={authoritative_currency}"
            )
        updated_row[pc_cols.BASE_CURRENCY] = authoritative_currency
        rows.append(updated_row)

    if conflicts:
        samples = "; ".join(conflicts[:5])
        raise PpaError(
            (
                f"{specification_path}: {dataset_name} file {path} contains "
                "base_currency values that conflict with authoritative "
                f"portfolio_performance values. Sample rows: {samples}"
            ),
            504,
        )

    if pc_cols.BASE_CURRENCY in frame.columns:
        return pl.DataFrame(rows, schema=frame.schema)
    if frame.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.String).alias(pc_cols.BASE_CURRENCY)
        )
    return pl.DataFrame(rows).select(*frame.columns, pc_cols.BASE_CURRENCY)


def _base_currency_by_portfolio(
    portfolio_performance: pl.DataFrame,
    *,
    specification_path: util.PathLike,
) -> dict[str, str]:
    """Return one authoritative base currency per portfolio."""
    currencies: dict[str, set[str]] = {}
    for row in portfolio_performance.select(
        pc_cols.PORTFOLIO_ID,
        pc_cols.BASE_CURRENCY,
    ).iter_rows(named=True):
        portfolio_id = _normalized_text(row.get(pc_cols.PORTFOLIO_ID))
        base_currency = _normalized_currency(row.get(pc_cols.BASE_CURRENCY))
        if not portfolio_id or base_currency is None:
            continue
        currencies.setdefault(portfolio_id, set()).add(base_currency)

    conflicts = {
        portfolio_id: values
        for portfolio_id, values in currencies.items()
        if len(values) > 1
    }
    if conflicts:
        samples = "; ".join(
            f"portfolio_id={portfolio_id}, currencies={','.join(sorted(values))}"
            for portfolio_id, values in sorted(conflicts.items())[:5]
        )
        raise PpaError(
            (
                f"{specification_path}: portfolio_performance must provide one "
                f"base_currency per portfolio. Conflicts: {samples}"
            ),
            504,
        )
    return {
        portfolio_id: next(iter(values))
        for portfolio_id, values in currencies.items()
    }


def _normalized_currency(value: object) -> str | None:
    """Return an uppercase currency value or ``None`` for a blank value."""
    text = _normalized_text(value)
    return text.upper() if text else None


def _normalized_text(value: object) -> str:
    """Return a stripped text representation for source identifiers."""
    return "" if value is None else str(value).strip()
