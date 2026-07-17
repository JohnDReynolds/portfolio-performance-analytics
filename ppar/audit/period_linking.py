"""Link dated comparison evidence to configured portfolio periods."""

from __future__ import annotations

# Python imports
import datetime as dt
from collections.abc import Mapping
from typing import Final, TypeAlias

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
import ppar.utilities as util

DATED_EVIDENCE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.TRANSACTIONS: pc_cols.TRANSACTION_DATE,
    pc_cols.HOLDINGS: pc_cols.HOLDING_DATE,
    pc_cols.SPLITS: pc_cols.SPLIT_DATE,
    pc_cols.FX_RATES: pc_cols.RATE_DATE,
}
PortfolioPeriodLookup: TypeAlias = Mapping[
    object, tuple[tuple[dt.date, dt.date], ...]
]
_SOURCE_ORDER = "_ppar_source_order"
_SCOPE_ORDER = "_ppar_scope_order"
_PRIOR_FROM_DATE = "_ppar_prior_from_date"
_PRIOR_THRU_DATE = "_ppar_prior_thru_date"


def validate_portfolio_periods(
    periods: pl.DataFrame,
    *,
    dataset_name: str,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Reject reversed or overlapping performance periods.

    Args:
        periods: Normalized performance rows from one snapshot.
        dataset_name: Normalized performance dataset name.
        path: Source CSV path.
        specification_path: Comparison YAML path.

    Raises:
        PpaError: If a period is reversed or two periods overlap within the
            same portfolio. Overlap would make dated evidence multiply assigned.
    """
    scope_columns = [pc_cols.PORTFOLIO_ID]
    if dataset_name == pc_cols.SECURITY_PERFORMANCE:
        scope_columns.append(pc_cols.SECURITY_ID)
    relevant_columns = [*scope_columns, pc_cols.FROM_DATE, pc_cols.THRU_DATE]
    period_rows = periods.select(relevant_columns).with_row_index(_SOURCE_ORDER)

    reversed_periods = period_rows.filter(
        pl.col(pc_cols.FROM_DATE) > pl.col(pc_cols.THRU_DATE)
    )
    if not reversed_periods.is_empty():
        reversed_period = reversed_periods.row(0, named=True)
        portfolio_id = reversed_period[pc_cols.PORTFOLIO_ID]
        from_date = reversed_period[pc_cols.FROM_DATE]
        thru_date = reversed_period[pc_cols.THRU_DATE]
        raise PpaError(
            (
                f"{specification_path}: SN-07 period-boundary validation failed "
                f"for {dataset_name} file {path}: portfolio_id={portfolio_id}, "
                f"from_date={from_date} is after thru_date={thru_date}."
            ),
            504,
        )

    scope_order = period_rows.group_by(scope_columns, maintain_order=True).agg(
        pl.col(_SOURCE_ORDER).min().alias(_SCOPE_ORDER)
    )
    ordered_periods = (
        period_rows.filter(
            pl.col(pc_cols.FROM_DATE).is_not_null()
            & pl.col(pc_cols.THRU_DATE).is_not_null()
        )
        # Exact duplicates remain the comparison-key validator's Error 112.
        .unique(subset=relevant_columns, keep="first", maintain_order=True)
        .join(
            scope_order,
            on=scope_columns,
            how="left",
            maintain_order="left",
        )
        .sort([_SCOPE_ORDER, pc_cols.FROM_DATE, pc_cols.THRU_DATE])
        .with_columns(
            pl.col(pc_cols.FROM_DATE)
            .shift(1)
            .over(scope_columns)
            .alias(_PRIOR_FROM_DATE),
            pl.col(pc_cols.THRU_DATE)
            .shift(1)
            .over(scope_columns)
            .alias(_PRIOR_THRU_DATE),
        )
    )
    overlapping_periods = ordered_periods.filter(
        pl.col(_PRIOR_THRU_DATE).is_not_null()
        & (pl.col(pc_cols.FROM_DATE) <= pl.col(_PRIOR_THRU_DATE))
    )
    if overlapping_periods.is_empty():
        return

    overlap = overlapping_periods.row(0, named=True)
    scope = tuple(overlap[column] for column in scope_columns)
    raise PpaError(
        (
            f"{specification_path}: SN-07 period-boundary validation failed "
            f"for {dataset_name} file {path}: scope={scope} "
            f"has overlapping periods {overlap[_PRIOR_FROM_DATE]}.."
            f"{overlap[_PRIOR_THRU_DATE]} and {overlap[pc_cols.FROM_DATE]}.."
            f"{overlap[pc_cols.THRU_DATE]}."
        ),
        504,
    )


def validate_dated_evidence_assignments(
    frame: pl.DataFrame,
    *,
    dataset_name: str,
    periods: pl.DataFrame,
    path: util.PathLike,
    specification_path: util.PathLike,
    date_columns: tuple[str, ...] | None = None,
    allow_unassigned: bool = False,
) -> None:
    """Require every portfolio-scoped evidence date to map to one period.

    Rows for portfolios absent from the performance extract are audit-only and
    cannot be counted, so they remain outside this assignment contract.

    Args:
        frame: Normalized dated evidence rows from one snapshot.
        dataset_name: Normalized source dataset name.
        periods: Normalized portfolio performance periods from the same snapshot.
        path: Source CSV path.
        specification_path: Comparison YAML path.
        date_columns: Date fields to validate. Defaults to the dataset's primary
            evidence date.
        allow_unassigned: Whether zero-match rows may remain visible,
            noncounted review evidence. Multiple matches are always unsafe.

    Raises:
        PpaError: If evidence for a compared portfolio maps to zero or more than
            one performance period.
    """
    lookup = portfolio_period_lookup(periods)
    columns = date_columns or (DATED_EVIDENCE_COLUMNS[dataset_name],)
    for row in frame.iter_rows(named=True):
        portfolio_id = row.get(pc_cols.PORTFOLIO_ID)
        portfolio_periods = lookup.get(portfolio_id)
        if not portfolio_periods:
            continue
        for date_column in columns:
            evidence_date = row.get(date_column)
            if evidence_date is None:
                continue
            candidates = _period_candidates(evidence_date, portfolio_periods)
            if len(candidates) == 1:
                continue
            if not candidates and allow_unassigned:
                continue
            assignment = "no" if not candidates else "multiple"
            raise PpaError(
                (
                    f"{specification_path}: SN-07 period-boundary validation failed "
                    f"for {dataset_name} file {path}: portfolio_id={portfolio_id}, "
                    f"{date_column}={evidence_date} maps to {assignment} performance "
                    "periods; exactly one is required before the row can affect "
                    "performance."
                ),
                504,
            )


def portfolio_period_lookup(periods: pl.DataFrame) -> PortfolioPeriodLookup:
    """Index valid periods once by portfolio for repeated evidence linking."""
    periods_by_portfolio: dict[object, list[tuple[dt.date, dt.date]]] = {}
    for period in periods.iter_rows(named=True):
        portfolio_id = period[pc_cols.PORTFOLIO_ID]
        from_date = period[pc_cols.FROM_DATE]
        thru_date = period[pc_cols.THRU_DATE]
        if not isinstance(from_date, dt.date) or not isinstance(thru_date, dt.date):
            continue
        periods_by_portfolio.setdefault(portfolio_id, []).append(
            (from_date, thru_date)
        )
    return {
        portfolio_id: tuple(portfolio_periods)
        for portfolio_id, portfolio_periods in periods_by_portfolio.items()
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
    portfolio_periods: pl.DataFrame | PortfolioPeriodLookup | None,
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
    portfolio_periods: pl.DataFrame | PortfolioPeriodLookup,
) -> tuple[object | None, object | None]:
    """Return the portfolio period containing a dated evidence row."""
    portfolio_id = row.get(pc_cols.PORTFOLIO_ID)
    evidence_date = row.get(DATED_EVIDENCE_COLUMNS[dataset])
    if portfolio_id is None or not isinstance(evidence_date, dt.date):
        return None, None

    periods = (
        portfolio_period_lookup(portfolio_periods).get(portfolio_id, ())
        if isinstance(portfolio_periods, pl.DataFrame)
        else portfolio_periods.get(portfolio_id, ())
    )
    candidates = _period_candidates(evidence_date, periods)
    if not candidates:
        return None, None
    if len(candidates) > 1:
        raise PpaError(
            (
                "SN-07 period-boundary validation failed: dated evidence for "
                f"portfolio_id={portfolio_id}, date={evidence_date} maps to "
                f"multiple periods {candidates}."
            ),
            504,
        )
    return candidates[0]


def _period_candidates(
    evidence_date: object,
    periods: tuple[tuple[dt.date, dt.date], ...] | list[tuple[dt.date, dt.date]],
) -> list[tuple[dt.date, dt.date]]:
    """Return all periods that inclusively contain one evidence date."""
    if not isinstance(evidence_date, dt.date):
        return []
    return sorted(
        (
            (from_date, thru_date)
            for from_date, thru_date in periods
            if from_date <= evidence_date <= thru_date
        ),
        key=lambda period: (period[0], period[1]),
    )
