"""Load normalized security performance comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import aliases
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.base_currency import with_authoritative_base_currency
from ppar.performance_comparison.currency_basis import normalize_currency_columns
from ppar.performance_comparison.period_linking import validate_portfolio_periods
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.performance_comparison.specification import (
    SECURITY_COMPARISON_LEVEL,
    PerformanceComparisonSpecification,
)
import ppar.utilities as util


class SecurityPerformanceLoader:
    """Load normalized security performance rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize the security performance loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized security performance rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Security performance rows with normalized comparison column names,
            or ``None`` when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.SECURITY_PERFORMANCE,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            if self._specification.comparison_level == SECURITY_COMPARISON_LEVEL:
                raise PpaError(self._error_message(util.file_path_error(path or "")), 802)
            return None
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.SECURITY_PERFORMANCE,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_schema_mapped_csv(
            path,
            pc_cols.SECURITY_PERFORMANCE_COLUMNS,
            pc_cols.SECURITY_PERFORMANCE,
            aliases.SECURITY_PERFORMANCE_REQUIRED_ALIASES,
            aliases.SECURITY_PERFORMANCE_OPTIONAL_ALIASES,
            self._specification,
            snapshot_key,
        ).with_columns(
            pl.col(pc_cols.FROM_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            pl.col(pc_cols.THRU_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = normalize_currency_columns(
            source_loader.require_numeric_columns(
                frame,
                columns=(
                    pc_cols.SECURITY_RETURN,
                    pc_cols.WEIGHT,
                    pc_cols.CONTRIBUTION,
                    pc_cols.BEGIN_MARKET_VALUE,
                    pc_cols.END_MARKET_VALUE,
                    pc_cols.INCOME,
                    pc_cols.GAIN_LOSS,
                ),
                dataset_name=pc_cols.SECURITY_PERFORMANCE,
                path=path,
                specification_path=self._specification.path,
            )
        )
        validate_portfolio_periods(
            frame,
            dataset_name=pc_cols.SECURITY_PERFORMANCE,
            path=path,
            specification_path=self._specification.path,
        )
        if pc_cols.PORTFOLIO_PERFORMANCE in self._specification.files:
            frame = normalize_currency_columns(
                with_authoritative_base_currency(
                    frame,
                    PortfolioPerformanceLoader(self._specification).load(snapshot_key),
                    dataset_name=pc_cols.SECURITY_PERFORMANCE,
                    path=path,
                    specification_path=self._specification.path,
                )
            )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.SECURITY_PERFORMANCE,
            snapshot_key,
            path,
            frame,
        )

    def _error_message(self, message: str) -> str:
        """Return an error message with comparison specification context."""
        return (
            f"{message}  |  "
            f"comparison_specification_path={self._specification.path}"
        )
