"""Load normalized price comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import aliases
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import SnapshotKey
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util


class PricesLoader:
    """Load normalized price rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize the price loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized price rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Price rows with normalized comparison column names, or ``None``
            when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.PRICES,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None

        frame = source_loader.read_mapped_csv(
            path,
            pc_cols.PRICES_COLUMNS,
            pc_cols.PRICES,
            aliases.PRICES_REQUIRED_ALIASES,
            aliases.PRICES_OPTIONAL_ALIASES,
            self._specification.path,
        ).with_columns(
            pl.col(pc_cols.PRICE_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        return source_loader.require_numeric_columns(
            frame,
            columns=(pc_cols.PRICE,),
            dataset_name=pc_cols.PRICES,
            path=path,
            specification_path=self._specification.path,
        )
