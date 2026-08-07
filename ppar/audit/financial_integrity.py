"""Validate financial-input currency, unit, and dated-evidence contracts."""

from __future__ import annotations

# Python imports
import math
import re
from typing import Final, TypeAlias, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.holdings import HoldingsLoader
from ppar.audit.period_linking import (
    validate_dated_evidence_assignments,
)
from ppar.audit.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.audit.security_performance import SecurityPerformanceLoader
from ppar.audit import source_loader
from ppar.audit.specification import AuditSpecification
from ppar.audit.splits import SplitsLoader
from ppar.audit.transactions import TransactionsLoader
import ppar.common as util

_CURRENCY_COLUMNS: Final[tuple[str, ...]] = (
    pc_cols.CURRENCY,
    pc_cols.BASE_CURRENCY,
)
_CURRENCY_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Z]{3}")
_BASE_COUNTERPARTS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    pc_cols.HOLDINGS: (
        (pc_cols.MARKET_VALUE, pc_cols.BASE_MARKET_VALUE),
        (pc_cols.ACCRUED, pc_cols.BASE_ACCRUED),
    ),
    pc_cols.TRANSACTIONS: ((pc_cols.AMOUNT, pc_cols.BASE_AMOUNT),),
}
_MONETARY_ABSOLUTE_TOLERANCE: Final[float] = 0.01
_SnapshotFrames: TypeAlias = tuple[
    pl.DataFrame,
    pl.DataFrame | None,
    pl.DataFrame | None,
    pl.DataFrame | None,
    pl.DataFrame | None,
]


def validate_financial_input_integrity(
    specification: AuditSpecification,
) -> None:
    """Enforce Phase 3 currency, unit, and period source contracts.

    Args:
        specification: Parsed comparison specification.

    Raises:
        PpaError: If currency metadata is unsafe, monetary units conflict, or
            portfolio-scoped dated evidence cannot be assigned exactly once.
    """
    if source_loader.financial_validation_is_cached(specification.path):
        return

    portfolio_loader = PortfolioPerformanceLoader(specification)
    security_loader = SecurityPerformanceLoader(specification)
    holdings_loader = HoldingsLoader(specification)
    transactions_loader = TransactionsLoader(specification)
    splits_loader = SplitsLoader(specification)
    snapshots: dict[SnapshotKey, _SnapshotFrames] = {}
    for snapshot_key in ("a", "b"):
        security_performance = security_loader.load(snapshot_key)
        if pc_cols.PORTFOLIO_PERFORMANCE in specification.files:
            periods = portfolio_loader.load(snapshot_key)
        elif security_performance is not None:
            periods = security_performance.select(
                pc_cols.PORTFOLIO_ID,
                pc_cols.FROM_DATE,
                pc_cols.THRU_DATE,
                *(
                    [pc_cols.BASE_CURRENCY]
                    if pc_cols.BASE_CURRENCY in security_performance.columns
                    else []
                ),
            ).unique()
        else:
            continue
        snapshot = (
            periods,
            security_performance,
            holdings_loader.load(snapshot_key),
            transactions_loader.load(snapshot_key),
            splits_loader.load(snapshot_key),
        )
        snapshots[snapshot_key] = snapshot
        _validate_snapshot(
            specification,
            snapshot_key,
            *snapshot,
        )
    _validate_changed_evidence_assignments(
        specification,
        snapshots["a"],
        snapshots["b"],
    )
    source_loader.cache_financial_validation(specification.path)


def _validate_snapshot(
    specification: AuditSpecification,
    snapshot_key: SnapshotKey,
    periods: pl.DataFrame,
    security_performance: pl.DataFrame | None,
    holdings: pl.DataFrame | None,
    transactions: pl.DataFrame | None,
    _splits: pl.DataFrame | None,
) -> None:
    """Validate one snapshot's detailed financial inputs."""
    period_dataset = (
        pc_cols.PORTFOLIO_PERFORMANCE
        if pc_cols.PORTFOLIO_PERFORMANCE in specification.files
        else pc_cols.SECURITY_PERFORMANCE
    )
    _validate_currency_codes(
        periods,
        dataset_name=period_dataset,
        path=_source_path(specification, period_dataset, snapshot_key),
        specification_path=specification.path,
    )
    if security_performance is not None:
        _validate_currency_codes(
            security_performance,
            dataset_name=pc_cols.SECURITY_PERFORMANCE,
            path=_source_path(
                specification,
                pc_cols.SECURITY_PERFORMANCE,
                snapshot_key,
            ),
            specification_path=specification.path,
        )
    for dataset_name, frame in (
        (pc_cols.HOLDINGS, holdings),
        (pc_cols.TRANSACTIONS, transactions),
    ):
        if frame is None:
            continue
        path = _source_path(specification, dataset_name, snapshot_key)
        _validate_currency_codes(
            frame,
            dataset_name=dataset_name,
            path=path,
            specification_path=specification.path,
        )
        _validate_monetary_units(
            frame,
            dataset_name=dataset_name,
            path=path,
            specification_path=specification.path,
        )


def _validate_changed_evidence_assignments(
    specification: AuditSpecification,
    snapshot_a: _SnapshotFrames,
    snapshot_b: _SnapshotFrames,
) -> None:
    """Validate period assignment for rows that differ between snapshots."""
    periods_a, _, holdings_a, transactions_a, splits_a = snapshot_a
    periods_b, _, holdings_b, transactions_b, splits_b = snapshot_b
    for raw_snapshot_key, periods, holdings, transactions, splits in (
        ("a", periods_a, holdings_a, transactions_a, splits_a),
        ("b", periods_b, holdings_b, transactions_b, splits_b),
    ):
        snapshot_key = cast(SnapshotKey, raw_snapshot_key)
        other_frames = (
            holdings_b if snapshot_key == "a" else holdings_a,
            transactions_b if snapshot_key == "a" else transactions_a,
            splits_b if snapshot_key == "a" else splits_a,
        )
        for dataset_name, frame, other_frame in zip(
            (
                pc_cols.HOLDINGS,
                pc_cols.TRANSACTIONS,
                pc_cols.SPLITS,
            ),
            (holdings, transactions, splits),
            other_frames,
            strict=True,
        ):
            changed = _rows_not_identical_in_other_snapshot(frame, other_frame)
            if changed is None or changed.is_empty():
                continue
            path = _source_path(specification, dataset_name, snapshot_key)
            if dataset_name == pc_cols.SPLITS:
                if holdings is not None:
                    _validate_split_assignments(
                        changed,
                        holdings,
                        periods,
                        path=path,
                        specification_path=specification.path,
                    )
                continue
            date_columns = None
            if dataset_name == pc_cols.TRANSACTIONS:
                date_columns = (pc_cols.TRANSACTION_DATE, pc_cols.SETTLEMENT_DATE)
            validate_dated_evidence_assignments(
                changed,
                dataset_name=dataset_name,
                periods=periods,
                path=path,
                specification_path=specification.path,
                date_columns=date_columns,
                allow_unassigned=True,
            )


def _rows_not_identical_in_other_snapshot(
    frame: pl.DataFrame | None,
    other_frame: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Return rows whose complete normalized representation is not conserved."""
    if frame is None:
        return None
    if other_frame is None:
        return frame
    columns = sorted(set(frame.columns) | set(other_frame.columns))
    row_index = "__ppar_integrity_row_index"
    occurrence = "__ppar_integrity_occurrence"
    normalized_frame = _integrity_comparison_columns(
        frame,
        other_frame,
        columns,
    ).with_row_index(row_index)
    normalized_other = _integrity_comparison_columns(
        other_frame,
        frame,
        columns,
    )
    grouping_columns = list(columns)
    normalized_frame = normalized_frame.with_columns(
        pl.int_range(pl.len()).over(grouping_columns).alias(occurrence)
    )
    normalized_other = normalized_other.with_columns(
        pl.int_range(pl.len()).over(grouping_columns).alias(occurrence)
    )
    unmatched_indexes = (
        normalized_frame.join(
            normalized_other.select(*grouping_columns, occurrence),
            on=[*grouping_columns, occurrence],
            how="anti",
            nulls_equal=True,
        )
        .get_column(row_index)
        .sort()
    )
    return frame[unmatched_indexes]


def _integrity_comparison_columns(
    frame: pl.DataFrame,
    counterpart: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """Return aligned columns for a multiset comparison between snapshots."""
    expressions: list[pl.Expr] = []
    for column in columns:
        if (
            column in frame.columns
            and column in counterpart.columns
            and frame.schema[column] == counterpart.schema[column]
        ):
            expressions.append(pl.col(column))
            continue
        if column in frame.columns:
            data_type = frame.schema[column]
            expressions.append(
                pl.when(pl.col(column).is_null())
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(
                    pl.concat_str(
                        pl.lit(f"{data_type}:"),
                        pl.col(column).cast(pl.String),
                    )
                )
                .alias(column)
            )
            continue
        expressions.append(
            pl.lit(None, dtype=pl.String).alias(column)
        )
    return frame.select(expressions)


def _validate_split_assignments(
    splits: pl.DataFrame,
    holdings: pl.DataFrame,
    periods: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Validate split dates for each compared portfolio holding the security."""
    portfolios_by_security: dict[object, set[object]] = {}
    for row in holdings.iter_rows(named=True):
        portfolios_by_security.setdefault(row.get(pc_cols.SECURITY_ID), set()).add(
            row.get(pc_cols.PORTFOLIO_ID)
        )
    contextual_rows: list[dict[str, object]] = []
    for row in splits.iter_rows(named=True):
        for portfolio_id in portfolios_by_security.get(
            row.get(pc_cols.SECURITY_ID),
            set(),
        ):
            contextual_row = dict(row)
            contextual_row[pc_cols.PORTFOLIO_ID] = portfolio_id
            contextual_rows.append(contextual_row)
    if not contextual_rows:
        return
    validate_dated_evidence_assignments(
        pl.DataFrame(contextual_rows),
        dataset_name=pc_cols.SPLITS,
        periods=periods,
        path=path,
        specification_path=specification_path,
    )


def _validate_currency_codes(
    frame: pl.DataFrame,
    *,
    dataset_name: str,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Reject supplied currency values outside the normalized three-letter contract."""
    for column in _CURRENCY_COLUMNS:
        if column not in frame.columns:
            continue
        for row_number, value in enumerate(frame.get_column(column), start=2):
            if value is None or str(value).strip() == "":
                continue
            code = str(value).strip().upper()
            if _CURRENCY_CODE_PATTERN.fullmatch(code):
                continue
            raise PpaError(
                (
                    f"{specification_path}: SN-06 currency validation failed for "
                    f"{dataset_name} file {path}: row {row_number} column {column!r} "
                    f"must be a three-letter currency code; found {value!r}."
                ),
                504,
            )


def _validate_monetary_units(
    frame: pl.DataFrame,
    *,
    dataset_name: str,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Require safe base counterparts for foreign countable monetary values."""
    counterparts = _BASE_COUNTERPARTS.get(dataset_name, ())
    if not counterparts:
        return
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        row_currency = _currency(row.get(pc_cols.CURRENCY))
        base_currency = _currency(row.get(pc_cols.BASE_CURRENCY))
        if not row_currency or not base_currency:
            continue
        for local_field, base_field in counterparts:
            local_value = _number(row.get(local_field))
            base_value = _number(row.get(base_field))
            if row_currency != base_currency:
                if local_value not in {None, 0.0} and base_value is None:
                    _raise_missing_base_value(
                        specification_path,
                        dataset_name,
                        path,
                        row_number,
                        row_currency,
                        base_currency,
                        local_field,
                        base_field,
                    )
                continue
            if local_value is None or base_value is None:
                continue
            if math.isclose(
                local_value,
                base_value,
                rel_tol=1e-12,
                abs_tol=_MONETARY_ABSOLUTE_TOLERANCE,
            ):
                continue
            raise PpaError(
                (
                    f"{specification_path}: SN-06 unit validation failed for "
                    f"{dataset_name} file {path}: row {row_number} uses "
                    f"currency=base_currency={row_currency}, so {local_field} "
                    f"({local_value}) and {base_field} ({base_value}) must agree."
                ),
                504,
            )


def _raise_missing_base_value(
    specification_path: util.PathLike,
    dataset_name: str,
    path: util.PathLike,
    row_number: int,
    row_currency: str,
    base_currency: str,
    local_field: str,
    base_field: str,
) -> None:
    """Raise the standard foreign-value source-contract error."""
    raise PpaError(
        (
            f"{specification_path}: SN-06 unit validation failed for {dataset_name} "
            f"file {path}: row {row_number} has {local_field} in {row_currency} "
            f"but no {base_field} in portfolio base currency {base_currency}."
        ),
        504,
    )


def _source_path(
    specification: AuditSpecification,
    dataset_name: str,
    snapshot_key: SnapshotKey,
) -> util.PathLike:
    """Return one configured snapshot source path."""
    comparison_file = specification.files[dataset_name]
    return (
        comparison_file.snapshot_a_path
        if snapshot_key == "a"
        else comparison_file.snapshot_b_path
    )


def _currency(value: object) -> str:
    """Return one normalized currency value."""
    return "" if value is None else str(value).strip().upper()


def _number(value: object) -> float | None:
    """Return a finite non-boolean float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
