"""Load normalized split-factor comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit import source_loader
from ppar.audit.portfolio_performance import SnapshotKey
from ppar.audit.specification import AuditSpecification
import ppar.common as util


class SplitsLoader:
    """Load normalized security-level split-factor rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: AuditSpecification) -> None:
        """Initialize the split-factor loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized split-factor rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Split rows with normalized comparison column names, or ``None``
            when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved or if split factors are nonnumeric.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.SPLITS,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.SPLITS,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_schema_mapped_csv(
            path,
            pc_cols.SPLITS_COLUMNS,
            pc_cols.SPLITS,
            aliases.SPLITS_REQUIRED_ALIASES,
            aliases.SPLITS_OPTIONAL_ALIASES,
            self._specification,
            snapshot_key,
        ).with_columns(
            pl.col(pc_cols.SPLIT_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = source_loader.require_numeric_columns(
            frame,
            columns=(pc_cols.SPLIT_FACTOR,),
            dataset_name=pc_cols.SPLITS,
            path=path,
            specification_path=self._specification.path,
        )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.SPLITS,
            snapshot_key,
            path,
            frame,
        )
