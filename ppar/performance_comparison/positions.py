"""Load normalized position comparison sources."""

from __future__ import annotations

# Python imports
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.portfolio_performance import SnapshotKey
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util

_REQUIRED_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    pc_cols.PORTFOLIO_ID: (
        "PORTFOLIO_ID",
        "PORTFOLIO_CODE",
        "PORT",
        "PORTFOLIO",
        "ACCOUNT",
        "ACCT",
    ),
    pc_cols.SECURITY_ID: ("SECURITY_ID", "SEC", "SECURITY", "SEC_ID", "SECNO"),
    pc_cols.POSITION_DATE: ("POSITION_DATE",),
}
_OPTIONAL_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    pc_cols.QUANTITY: ("QUANTITY", "QTY", "UNITS", "SHARES"),
    pc_cols.PRICE: ("PRICE", "PX", "MARKET_PRICE"),
    pc_cols.MARKET_VALUE: ("MARKET_VALUE", "MV", "MKT_VAL"),
    pc_cols.COST: ("COST", "COST_BASIS", "BOOK_COST", "TAX_COST", "ORIG_COST"),
    pc_cols.ACCRUED: ("ACCRUED", "ACCRUED_INCOME", "ACCRUED_INT", "ACCRUAL"),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
}


class PositionsLoader:
    """Load normalized position rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize the position loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized position rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Position rows with normalized comparison column names, or ``None``
            when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = self._positions_path(snapshot_key)
        if path is None or not util.file_path_exists(path):
            return None

        mappings = self._csv_to_internal_mappings(path)
        selected_columns = [
            column_name
            for column_name in pc_cols.POSITIONS_COLUMNS
            if column_name in mappings.values()
        ]
        return (
            pl.read_csv(path)
            .rename(mappings)
            .select(selected_columns)
            .with_columns(
                pl.col(pc_cols.POSITION_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

    def _positions_path(self, snapshot_key: SnapshotKey) -> util.PathLike | None:
        """Return the resolved positions path for a snapshot."""
        comparison_file = self._specification.files.get(pc_cols.POSITIONS)
        if comparison_file is None:
            return None
        return (
            comparison_file.snapshot_a_path
            if snapshot_key == "a"
            else comparison_file.snapshot_b_path
        )

    def _csv_to_internal_mappings(self, path: util.PathLike) -> dict[str, str]:
        """Return source-to-normalized column mappings for a CSV header."""
        available_columns = set(pl.read_csv(path, n_rows=0).columns)
        mappings: dict[str, str] = {}
        missing_columns: list[str] = []

        for internal_column, aliases in _REQUIRED_COLUMN_ALIASES.items():
            source_column = self._resolve_required_column(
                internal_column,
                aliases,
                available_columns,
            )
            if source_column is None:
                missing_columns.append(
                    f"{internal_column!r}; tried aliases {list(aliases)}"
                )
                continue
            mappings[source_column] = internal_column

        if missing_columns:
            raise PpaError(
                self._error_message(
                    f"Missing {missing_columns} in {str(path)!r}.  |  "
                    f"CSV columns available are: {sorted(available_columns)}"
                ),
                502,
            )

        for internal_column, aliases in _OPTIONAL_COLUMN_ALIASES.items():
            source_column = self._resolve_optional_column(
                internal_column,
                aliases,
                available_columns,
            )
            if source_column is not None:
                mappings[source_column] = internal_column

        return mappings

    def _resolve_required_column(
        self,
        internal_column: str,
        aliases: tuple[str, ...],
        available_columns: set[str],
    ) -> str | None:
        """Resolve a required source column from known aliases."""
        matches = [alias for alias in aliases if alias in available_columns]
        if len(matches) > 1:
            raise PpaError(
                self._error_message(
                    f"Ambiguous positions source columns for "
                    f"{internal_column!r}: {matches}."
                ),
                502,
            )
        return matches[0] if matches else None

    def _resolve_optional_column(
        self,
        internal_column: str,
        aliases: tuple[str, ...],
        available_columns: set[str],
    ) -> str | None:
        """Resolve an optional source column using alias priority order."""
        matches = [alias for alias in aliases if alias in available_columns]
        if len(matches) > 1:
            raise PpaError(
                self._error_message(
                    f"Ambiguous positions source columns for "
                    f"{internal_column!r}: {matches}."
                ),
                502,
            )
        return matches[0] if matches else None

    def _error_message(self, message: str) -> str:
        """Return an error message with comparison specification context."""
        return (
            f"{message}  |  "
            f"comparison_specification_path={self._specification.path}"
        )
