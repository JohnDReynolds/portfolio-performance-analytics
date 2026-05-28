"""Load normalized portfolio performance comparison sources."""

from __future__ import annotations

# Python imports
from typing import Final, Literal

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util

SnapshotKey = Literal["a", "b"]

_REQUIRED_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    pc_cols.PORTFOLIO_ID: (
        "PORTFOLIO_ID",
        "PORTFOLIO_CODE",
        "PORT",
        "PORTFOLIO",
        "ACCOUNT",
        "ACCT",
    ),
    pc_cols.FROM_DATE: ("FROM_DATE",),
    pc_cols.THRU_DATE: ("THRU_DATE",),
    pc_cols.PORTFOLIO_RETURN: (
        "PORT_RETURN",
        "RETURN",
        "RET",
    ),
}
_OPTIONAL_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    pc_cols.PORTFOLIO_NAME: ("PORTFOLIO_NAME",),
    pc_cols.BEGIN_MARKET_VALUE: ("BEGIN_MV", "BEG_MV", "BMV", "BEGIN_VALUE"),
    pc_cols.END_MARKET_VALUE: ("END_MV", "EMV", "ENDING_VALUE"),
    pc_cols.FLOW: ("FLOW", "NET_FLOW", "CONTRIB_WITHDRAW", "CASH_FLOW"),
    pc_cols.INCOME: ("INCOME", "INC", "DIV_INT", "INV_INCOME"),
    pc_cols.GAIN_LOSS: ("GAIN_LOSS", "GL", "GAIN", "REAL_UNREAL_GL"),
    pc_cols.PERIOD_ID: ("PERIOD_ID",),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
}


class PortfolioPerformanceLoader:
    """Load normalized portfolio performance rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
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

        mappings = self._csv_to_internal_mappings(path)
        selected_columns = [
            column_name
            for column_name in pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS
            if column_name in mappings.values()
        ]
        return (
            pl.read_csv(path)
            .rename(mappings)
            .select(selected_columns)
            .with_columns(
                pl.col(pc_cols.FROM_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col(pc_cols.THRU_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

    def _portfolio_performance_path(self, snapshot_key: SnapshotKey) -> util.PathLike:
        """Return the resolved portfolio performance path for a snapshot."""
        comparison_file = self._specification.files[pc_cols.PORTFOLIO_PERFORMANCE]
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
                    f"Ambiguous portfolio performance source columns for "
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
                    f"Ambiguous portfolio performance source columns for "
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
