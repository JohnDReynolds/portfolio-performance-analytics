"""Load normalized holding comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit.base_currency import with_authoritative_base_currency
from ppar.audit.currency_basis import normalize_currency_columns
from ppar.audit import source_loader
from ppar.errors import PpaError
from ppar.audit.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.audit.specification import AuditSpecification
import ppar.utilities as util


def _performance_calculation_configured(
    specification: AuditSpecification,
) -> bool:
    """Return whether holdings supply a configured performance calculation."""
    return (
        specification.portfolio_return_reconstruction is not None
        or specification.security_return_reconstruction is not None
    )


def _holding_column_aliases(
    specification: AuditSpecification,
) -> tuple[source_loader.ColumnAliases, source_loader.ColumnAliases]:
    """Return required and optional holding aliases for this configuration."""
    required_aliases = dict(aliases.HOLDINGS_REQUIRED_ALIASES)
    optional_aliases = dict(aliases.HOLDINGS_OPTIONAL_ALIASES)
    if _performance_calculation_configured(specification):
        for column in pc_cols.HOLDINGS_PERFORMANCE_CALCULATION_REQUIRED_COLUMNS:
            if column in required_aliases:
                continue
            required_aliases[column] = optional_aliases.pop(column)
    return required_aliases, optional_aliases


def _validate_performance_calculation_market_values(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Require a finite market value for every performance-calculation holding."""
    market_value = pl.col(pc_cols.MARKET_VALUE).cast(pl.Float64, strict=False)
    invalid_rows = frame.filter(
        market_value.is_null()
        | market_value.is_nan()
        | market_value.is_infinite()
    )
    if invalid_rows.is_empty():
        return
    raise PpaError(
        (
            f"{specification_path}: holdings column {pc_cols.MARKET_VALUE!r} "
            f"must contain a finite value on every row used for performance "
            f"calculation in {str(path)!r}."
        ),
        502,
    )


def _validate_foreign_currency_base_market_values(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Require a finite base value for each explicitly foreign holding."""
    if not {
        pc_cols.CURRENCY,
        pc_cols.BASE_CURRENCY,
        pc_cols.BASE_MARKET_VALUE,
    }.issubset(frame.columns):
        return
    currency = pl.col(pc_cols.CURRENCY).fill_null("").str.strip_chars()
    base_currency = pl.col(pc_cols.BASE_CURRENCY).fill_null("").str.strip_chars()
    base_market_value = pl.col(pc_cols.BASE_MARKET_VALUE).cast(
        pl.Float64,
        strict=False,
    )
    invalid_rows = frame.filter(
        (currency != "")
        & (base_currency != "")
        & (currency != base_currency)
        & (
            base_market_value.is_null()
            | base_market_value.is_nan()
            | base_market_value.is_infinite()
        )
    )
    if invalid_rows.is_empty():
        return
    raise PpaError(
        (
            f"{specification_path}: holdings column "
            f"{pc_cols.BASE_MARKET_VALUE!r} must contain a finite value for "
            "every foreign-currency holding used for performance calculation "
            f"in {str(path)!r}."
        ),
        502,
    )


class HoldingsLoader:
    """Load normalized holding rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: AuditSpecification) -> None:
        """Initialize the holdings loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized holding rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Holding rows with normalized comparison column names, or ``None``
            when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.HOLDINGS,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.HOLDINGS,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        required_aliases, optional_aliases = _holding_column_aliases(
            self._specification
        )
        frame = source_loader.read_schema_mapped_csv(
            path,
            pc_cols.HOLDINGS_COLUMNS,
            pc_cols.HOLDINGS,
            required_aliases,
            optional_aliases,
            self._specification,
            snapshot_key,
        ).with_columns(
            pl.col(pc_cols.HOLDING_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = source_loader.require_numeric_columns(
            frame,
            columns=(
                pc_cols.QUANTITY,
                pc_cols.PRICE,
                pc_cols.MARKET_VALUE,
                pc_cols.BASE_MARKET_VALUE,
                pc_cols.COST,
                pc_cols.ACCRUED,
                pc_cols.BASE_ACCRUED,
            ),
            dataset_name=pc_cols.HOLDINGS,
            path=path,
            specification_path=self._specification.path,
        )
        if _performance_calculation_configured(self._specification):
            _validate_performance_calculation_market_values(
                frame,
                path=path,
                specification_path=self._specification.path,
            )
        frame = normalize_currency_columns(
            with_authoritative_base_currency(
                frame,
                PortfolioPerformanceLoader(self._specification).load(snapshot_key),
                dataset_name=pc_cols.HOLDINGS,
                path=path,
                specification_path=self._specification.path,
            )
        )
        if _performance_calculation_configured(self._specification):
            _validate_foreign_currency_base_market_values(
                frame,
                path=path,
                specification_path=self._specification.path,
            )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.HOLDINGS,
            snapshot_key,
            path,
            frame,
        )
