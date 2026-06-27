"""Link dated comparison evidence to configured portfolio periods."""

from __future__ import annotations

# Python imports
import datetime as dt
from collections.abc import Mapping
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import schema as pc_cols

DATED_EVIDENCE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.TRANSACTIONS: pc_cols.TRANSACTION_DATE,
    pc_cols.HOLDINGS: pc_cols.HOLDING_DATE,
    pc_cols.CASH: pc_cols.CASH_DATE,
}


def portfolio_periods_from_snapshots(
    snapshot_a: pl.DataFrame,
    snapshot_b: pl.DataFrame,
) -> pl.DataFrame:
    """Return unique portfolio-period rows from two portfolio snapshots.

    Args:
        snapshot_a: Normalized portfolio performance rows for snapshot A.
        snapshot_b: Normalized portfolio performance rows for snapshot B.

    Returns:
        Stable table containing unique ``portfolio_id``, ``from_date``, and
        ``thru_date`` combinations.
    """
    columns = [pc_cols.PORTFOLIO_ID, pc_cols.FROM_DATE, pc_cols.THRU_DATE]
    return pl.concat([snapshot_a.select(columns), snapshot_b.select(columns)]).unique().sort(
        columns
    )


def security_periods_from_snapshots(
    snapshot_a: pl.DataFrame,
    snapshot_b: pl.DataFrame,
) -> pl.DataFrame:
    """Return unique portfolio-security-period rows from two snapshots.

    Args:
        snapshot_a: Normalized security performance rows for snapshot A.
        snapshot_b: Normalized security performance rows for snapshot B.

    Returns:
        Stable table containing unique ``portfolio_id``, ``security_id``,
        ``from_date``, and ``thru_date`` combinations.
    """
    columns = [
        pc_cols.PORTFOLIO_ID,
        pc_cols.SECURITY_ID,
        pc_cols.FROM_DATE,
        pc_cols.THRU_DATE,
    ]
    return pl.concat([snapshot_a.select(columns), snapshot_b.select(columns)]).unique().sort(
        columns
    )


def period_context_for_dated_evidence(
    row: Mapping[str, object],
    dataset: str,
    portfolio_periods: pl.DataFrame | None,
) -> tuple[object | None, object | None]:
    """Return period context for a dated evidence row.

    Args:
        row: Joined comparison row.
        dataset: Normalized dataset name for the row.
        portfolio_periods: Unique portfolio period rows, or ``None`` when no
            period linking should be attempted.

    Returns:
        ``(from_date, thru_date)`` when the row already has period context or
        when its source date falls inside a configured portfolio period.
        Otherwise returns ``(None, None)``.
    """
    from_date = row.get(pc_cols.FROM_DATE)
    thru_date = row.get(pc_cols.THRU_DATE)
    if from_date is not None or thru_date is not None:
        return from_date, thru_date
    if dataset not in DATED_EVIDENCE_COLUMNS or portfolio_periods is None:
        return None, None
    return _dated_evidence_period_context(row, dataset, portfolio_periods)


def _dated_evidence_period_context(
    row: Mapping[str, object],
    dataset: str,
    portfolio_periods: pl.DataFrame,
) -> tuple[object | None, object | None]:
    """Return the portfolio period containing a dated evidence row."""
    portfolio_id = row.get(pc_cols.PORTFOLIO_ID)
    evidence_date = row.get(DATED_EVIDENCE_COLUMNS[dataset])
    if portfolio_id is None or not isinstance(evidence_date, dt.date):
        return None, None

    candidates: list[tuple[dt.date, dt.date]] = []
    portfolio_rows = portfolio_periods.filter(pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
    for period in portfolio_rows.iter_rows(named=True):
        from_date = period[pc_cols.FROM_DATE]
        thru_date = period[pc_cols.THRU_DATE]
        if not isinstance(from_date, dt.date) or not isinstance(thru_date, dt.date):
            continue
        if from_date <= evidence_date <= thru_date:
            candidates.append((from_date, thru_date))
    if not candidates:
        return None, None

    candidates.sort(key=lambda period: ((period[1] - period[0]).days, period[0]))
    return candidates[0]
