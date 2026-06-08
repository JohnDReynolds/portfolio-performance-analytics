"""Load normalized security master comparison sources."""

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


class SecurityMasterLoader:
    """Load normalized security master rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize the security master loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized security master rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Security master rows with normalized comparison column names, or
            ``None`` when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.SECURITY_MASTER,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None

        return source_loader.read_schema_mapped_csv(
            path,
            pc_cols.SECURITY_MASTER_COLUMNS,
            pc_cols.SECURITY_MASTER,
            aliases.SECURITY_MASTER_REQUIRED_ALIASES,
            aliases.SECURITY_MASTER_OPTIONAL_ALIASES,
            self._specification,
            snapshot_key,
        )
