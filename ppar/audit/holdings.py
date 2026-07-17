"""Load normalized holding comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import aliases
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.base_currency import with_authoritative_base_currency
from ppar.performance_comparison.currency_basis import normalize_currency_columns
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util


class HoldingsLoader:
    """Load normalized holding rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
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

        frame = source_loader.read_mapped_csv(
            path,
            pc_cols.HOLDINGS_COLUMNS,
            pc_cols.HOLDINGS,
            aliases.HOLDINGS_REQUIRED_ALIASES,
            aliases.HOLDINGS_OPTIONAL_ALIASES,
            self._specification.path,
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
        frame = normalize_currency_columns(
            with_authoritative_base_currency(
                frame,
                PortfolioPerformanceLoader(self._specification).load(snapshot_key),
                dataset_name=pc_cols.HOLDINGS,
                path=path,
                specification_path=self._specification.path,
            )
        )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.HOLDINGS,
            snapshot_key,
            path,
            frame,
        )
