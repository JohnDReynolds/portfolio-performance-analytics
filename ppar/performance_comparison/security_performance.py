"""Load normalized security performance comparison sources."""

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
        path = self._security_performance_path(snapshot_key)
        if path is None or not util.file_path_exists(path):
            return None

        return (
            source_loader.read_schema_mapped_csv(
                path,
                pc_cols.SECURITY_PERFORMANCE_COLUMNS,
                pc_cols.SECURITY_PERFORMANCE,
                aliases.SECURITY_PERFORMANCE_REQUIRED_ALIASES,
                aliases.SECURITY_PERFORMANCE_OPTIONAL_ALIASES,
                self._specification,
                snapshot_key,
            )
            .with_columns(
                pl.col(pc_cols.FROM_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col(pc_cols.THRU_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

    def _security_performance_path(self, snapshot_key: SnapshotKey) -> util.PathLike | None:
        """Return the resolved security performance path for a snapshot."""
        comparison_file = self._specification.files.get(pc_cols.SECURITY_PERFORMANCE)
        if comparison_file is None:
            return None
        return (
            comparison_file.snapshot_a_path
            if snapshot_key == "a"
            else comparison_file.snapshot_b_path
        )
