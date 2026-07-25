"""Load normalized portfolio performance comparison sources."""

from __future__ import annotations

# Python imports
from typing import Literal

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit.currency_basis import normalize_currency_columns
from ppar.audit.period_linking import validate_portfolio_periods
from ppar.audit import source_loader
from ppar.audit.specification import AuditSpecification
import ppar.common as util

SnapshotKey = Literal["a", "b"]


class PortfolioPerformanceLoader:
    """Load normalized portfolio performance rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: AuditSpecification) -> None:
        """Initialize the portfolio performance loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame:
        """Load one snapshot's normalized portfolio performance rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Portfolio performance rows with normalized comparison column names.

        Raises:
            PpaError: If the source file is missing or required columns cannot
                be resolved.
        """
        path = self._portfolio_performance_path(snapshot_key)
        if not util.file_path_exists(path):
            raise PpaError(self._error_message(util.file_path_error(path)), 802)
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.PORTFOLIO_PERFORMANCE,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_schema_mapped_csv(
            path,
            pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS,
            pc_cols.PORTFOLIO_PERFORMANCE,
            aliases.PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES,
            aliases.PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES,
            self._specification,
            snapshot_key,
        ).with_columns(
            pl.col(pc_cols.FROM_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            pl.col(pc_cols.THRU_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = normalize_currency_columns(
            source_loader.require_numeric_columns(
                frame,
                columns=(pc_cols.PORTFOLIO_RETURN,),
                dataset_name=pc_cols.PORTFOLIO_PERFORMANCE,
                path=path,
                specification_path=self._specification.path,
            )
        )
        validate_portfolio_periods(
            frame,
            dataset_name=pc_cols.PORTFOLIO_PERFORMANCE,
            path=path,
            specification_path=self._specification.path,
        )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.PORTFOLIO_PERFORMANCE,
            snapshot_key,
            path,
            frame,
        )

    def _portfolio_performance_path(self, snapshot_key: SnapshotKey) -> util.PathLike:
        """Return the resolved portfolio performance path for a snapshot."""
        comparison_file = self._specification.files[pc_cols.PORTFOLIO_PERFORMANCE]
        return (
            comparison_file.snapshot_a_path
            if snapshot_key == "a"
            else comparison_file.snapshot_b_path
        )

    def _error_message(self, message: str) -> str:
        """Return an error message with comparison specification context."""
        return (
            f"{message}  |  "
            f"comparison_specification_path={self._specification.path}"
        )
