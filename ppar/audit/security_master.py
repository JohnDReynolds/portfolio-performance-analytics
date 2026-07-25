"""Load optional normalized security-master data for Audit enrichment."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit import source_loader
from ppar.audit.portfolio_performance import SnapshotKey
from ppar.audit.specification import AuditSpecification
from ppar.errors import PpaError
import ppar.utilities as util


class SecurityMasterLoader:
    """Load one exact-case security-master row per normalized security ID.

    Attributes:
        _specification: Parsed Audit specification containing snapshot paths and
            optional security-master configuration.
    """

    def __init__(self, specification: AuditSpecification) -> None:
        """Initialize the loader.

        Args:
            specification: Parsed Audit specification.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Return one snapshot's normalized security-master rows.

        Args:
            snapshot_key: Snapshot side, either ``"a"`` or ``"b"``.

        Returns:
            A normalized frame, or ``None`` when the optional dataset is not
            configured or its optional source file is absent.

        Raises:
            PpaError: If required columns are missing, security IDs are blank,
                or exact-case security IDs are duplicated.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.SECURITY_MASTER,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.SECURITY_MASTER,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_schema_mapped_csv(
            path,
            pc_cols.SECURITY_MASTER_COLUMNS,
            pc_cols.SECURITY_MASTER,
            aliases.SECURITY_MASTER_REQUIRED_ALIASES,
            aliases.SECURITY_MASTER_OPTIONAL_ALIASES,
            self._specification,
            snapshot_key,
        )
        self._validate_security_ids(frame, path)
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.SECURITY_MASTER,
            snapshot_key,
            path,
            frame,
        )

    def _validate_security_ids(
        self,
        frame: pl.DataFrame,
        path: util.PathLike,
    ) -> None:
        """Require nonblank, unique, exact-case security IDs."""
        security_ids = frame.get_column(pc_cols.SECURITY_ID).cast(pl.String)
        blank_rows = frame.filter(
            pl.col(pc_cols.SECURITY_ID).is_null()
            | (pl.col(pc_cols.SECURITY_ID).cast(pl.String).str.strip_chars() == "")
        )
        if not blank_rows.is_empty():
            raise PpaError(
                self._error_message(
                    f"security_master file {path} contains a blank security_id."
                ),
                504,
            )
        duplicate_ids = (
            pl.DataFrame({pc_cols.SECURITY_ID: security_ids})
            .group_by(pc_cols.SECURITY_ID)
            .len()
            .filter(pl.col("len") > 1)
            .get_column(pc_cols.SECURITY_ID)
            .to_list()
        )
        if duplicate_ids:
            sample = ", ".join(repr(value) for value in duplicate_ids[:5])
            raise PpaError(
                self._error_message(
                    "security_master must contain one exact-case row per "
                    f"security_id; duplicates: {sample}."
                ),
                504,
            )

    def _error_message(self, message: str) -> str:
        """Return an error with Audit specification context."""
        return f"{message}  |  audit_specification_path={self._specification.path}"
