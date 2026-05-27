"""Load normalized Axys portfolio and security performance sources."""

from __future__ import annotations

# Python imports
import datetime as dt
from typing import Final, Literal

# Third-party imports
import polars as pl

# Project imports
from ppar.axys.specification import AxysSpecification, ErrorMessage
import ppar.columns as cols
from ppar.errors import PpaError
import ppar.utilities as util

PerformanceSourceType = Literal[
    "portfolio_performance_columns",
    "security_performance_columns",
]
_PORTFOLIO_CODE_ALIASES: Final[tuple[str, ...]] = (
    "ACCT",
    "ACCOUNT",
    "PORT",
    "PORTFOLIO_CODE",
    "PORTFOLIO_ID",
)
_PERFORMANCE_COLUMN_KEYS: Final[dict[PerformanceSourceType, dict[str, str]]] = {
    "portfolio_performance_columns": {
        cols.FROM_DATE: "from_date",
        cols.THRU_DATE: "thru_date",
        cols.PORTFOLIO_CODE: "portfolio_code",
        cols.PORTFOLIO_NAME: "portfolio_name",
        cols.PORTFOLIO_RETURN: "portfolio_return",
    },
    "security_performance_columns": {
        cols.FROM_DATE: "from_date",
        cols.CONTRIBUTION: "contribution",
        cols.THRU_DATE: "thru_date",
        cols.IDENTIFIER: "identifier",
        cols.PORTFOLIO_CODE: "portfolio_code",
        cols.RETURN: "return",
        cols.WEIGHT: "weight",
    },
}
_PERFORMANCE_COLUMN_ALIASES: Final[dict[PerformanceSourceType, dict[str, tuple[str, ...]]]] = {
    "portfolio_performance_columns": {
        cols.FROM_DATE: ("FROM_DATE",),
        cols.THRU_DATE: ("THRU_DATE",),
        cols.PORTFOLIO_CODE: _PORTFOLIO_CODE_ALIASES,
        cols.PORTFOLIO_NAME: ("PORTFOLIO_NAME",),
        cols.PORTFOLIO_RETURN: ("PERF", "PERFORMANCE", "PORT_RETURN", "RET", "RETURN"),
    },
    "security_performance_columns": {
        cols.FROM_DATE: ("FROM_DATE",),
        cols.CONTRIBUTION: ("CONTRIBUTION_W_X_R", "CONTRIBUTION_WXR", "CONTRIBUTION"),
        cols.THRU_DATE: ("THRU_DATE",),
        cols.IDENTIFIER: ("SECURITY_ID", "SEC", "SECURITY", "SEC_ID"),
        cols.PORTFOLIO_CODE: _PORTFOLIO_CODE_ALIASES,
        cols.RETURN: ("SEC_RETURN", "RET", "RETURN", "PERF", "PERFORMANCE"),
        cols.WEIGHT: ("BEGIN_WEIGHT", "WEIGHT", "WGT", "PCT_ASSETS", "PERCENT_ASSETS"),
    },
}
_LEGACY_PERFORMANCE_COLUMN_KEYS: Final[dict[str, str]] = {
    cols.PORTFOLIO_RETURN: cols.PORTFOLIO_RETURN,
}

_PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.FROM_DATE,
    cols.THRU_DATE,
    cols.PORTFOLIO_CODE,
    cols.PORTFOLIO_NAME,
    cols.PORTFOLIO_RETURN,
}
_SECURITY_PERFORMANCE_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.FROM_DATE,
    cols.CONTRIBUTION,
    cols.THRU_DATE,
    cols.IDENTIFIER,
    cols.PORTFOLIO_CODE,
    cols.RETURN,
    cols.WEIGHT,
}


class AxysPerformanceSourceLoader:
    """Normalize Axys portfolio- and security-performance CSV sources.

    Attributes:
        _specification: Parsed Axys source configuration.
        _error_message: Callback used to add facade-level validation context.
        _from_date: Optional inclusive earliest reporting date to retain.
        _thru_date: Optional inclusive latest thru date to retain.
    """

    def __init__(
        self,
        specification: AxysSpecification,
        error_message: ErrorMessage,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
    ) -> None:
        """Initialize a performance source loader.

        Args:
            specification: Parsed Axys configuration.
            error_message: Callback that adds facade-level source context to
                validation messages.
            from_date: Optional inclusive earliest reporting date to retain.
            thru_date: Optional inclusive latest thru date to retain.
        """
        self._specification = specification
        self._error_message = error_message
        self._from_date = from_date
        self._thru_date = thru_date

    def load(
        self,
        file_path: util.PathLike,
        column_name_mappings_name: PerformanceSourceType,
        portfolio_code: str | None = None,
    ) -> pl.DataFrame:
        """Load a performance CSV with normalized columns and date filters.

        Args:
            file_path: Path to the portfolio- or security-performance CSV.
            column_name_mappings_name: Specification section defining the
                source-to-package column mapping.
            portfolio_code: Optional portfolio code used to filter source rows.

        Returns:
            Normalized performance rows containing the columns required for the
            selected source kind.

        Raises:
            PpaError: If the source path does not exist or required mapped
                columns are missing from the specification or CSV file.
        """
        path = self._specification.resolve_path(file_path)
        if not util.file_path_exists(path):
            raise PpaError(self._error_message(util.file_path_error(path)), None)

        required_columns = (
            _PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS
            if column_name_mappings_name == "portfolio_performance_columns"
            else _SECURITY_PERFORMANCE_REQUIRED_COLUMNS
        )
        csv_to_internal_mappings = self._csv_to_internal_mappings(
            path,
            column_name_mappings_name,
            required_columns,
        )

        lazy_frame = (
            pl.scan_csv(path)
            .rename(csv_to_internal_mappings)
            .select(required_columns)
            .with_columns(
                pl.col(cols.FROM_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col(cols.THRU_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )
        if portfolio_code is not None:
            lazy_frame = lazy_frame.filter(pl.col(cols.PORTFOLIO_CODE) == portfolio_code)
        if self._from_date is not None:
            lazy_frame = lazy_frame.filter(
                pl.lit(self._from_date) <= pl.col(cols.THRU_DATE)
            )
        if self._thru_date is not None:
            lazy_frame = lazy_frame.filter(pl.col(cols.THRU_DATE) <= pl.lit(self._thru_date))
        return lazy_frame.collect()

    def _csv_to_internal_mappings(
        self,
        path: util.PathLike,
        column_name_mappings_name: PerformanceSourceType,
        required_columns: set[str],
    ) -> dict[str, str]:
        """Return CSV-to-internal column mappings for a performance source.

        Args:
            path: Source CSV path used for validation context and header
                inspection.
            column_name_mappings_name: Specification section defining the
                source-to-package column mapping.
            required_columns: Internal columns required for the source kind.

        Returns:
            Mapping from source CSV column names to internal package column
            names.

        Raises:
            PpaError: If required mapped columns are missing from either the
                specification or the CSV header.
        """
        configured_column_mappings: dict[str, str] = self._specification.values.get(
            column_name_mappings_name, {}
        )
        column_mappings = self._normalize_column_mapping_keys(
            configured_column_mappings,
            column_name_mappings_name,
        )
        header = pl.read_csv(path, n_rows=0)
        available_columns = set(header.columns)
        missing_columns: list[str] = []
        csv_to_internal_mappings: dict[str, str] = {}

        for internal_column in required_columns:
            source_column = column_mappings.get(internal_column)
            if source_column is not None:
                if source_column not in available_columns:
                    missing_columns.append(
                        f"{internal_column!r} configured as {source_column!r}"
                    )
                    continue
                csv_to_internal_mappings[source_column] = internal_column
                continue

            inferred_column = self._infer_source_column(
                path,
                column_name_mappings_name,
                internal_column,
                available_columns,
            )
            if inferred_column is None:
                missing_columns.append(self._missing_column_message(
                    column_name_mappings_name,
                    internal_column,
                ))
                continue
            csv_to_internal_mappings[inferred_column] = internal_column

        if missing_columns:
            raise PpaError(
                self._error_message(
                    f"Missing {missing_columns} in {str(path)!r}.  |  "
                    f"CSV columns available are: {sorted(available_columns)}"
                ),
                502,
            )

        return csv_to_internal_mappings

    def _normalize_column_mapping_keys(
        self,
        column_mappings: dict[str, str],
        column_name_mappings_name: PerformanceSourceType,
    ) -> dict[str, str]:
        """Return configured source columns keyed by internal package column.

        Args:
            column_mappings: Raw YAML column mapping section.
            column_name_mappings_name: Specification section being normalized.

        Returns:
            Mapping from internal package column names to configured CSV
            column names.
        """
        canonical_keys = _PERFORMANCE_COLUMN_KEYS[column_name_mappings_name]
        key_to_internal_column = {
            yaml_key: internal_column
            for internal_column, yaml_key in canonical_keys.items()
        }
        key_to_internal_column.update(_LEGACY_PERFORMANCE_COLUMN_KEYS)
        return {
            key_to_internal_column.get(key, key): value
            for key, value in column_mappings.items()
        }

    def _infer_source_column(
        self,
        path: util.PathLike,
        column_name_mappings_name: PerformanceSourceType,
        internal_column: str,
        available_columns: set[str],
    ) -> str | None:
        """Infer a source CSV column from known Axys/IMEX aliases.

        Args:
            path: Source CSV path used for error context.
            column_name_mappings_name: Specification section being loaded.
            internal_column: Internal package column to resolve.
            available_columns: CSV header columns.

        Returns:
            The inferred CSV column, or ``None`` when no known alias exists.

        Raises:
            PpaError: If more than one alias exists for the same internal
                column.
        """
        aliases = _PERFORMANCE_COLUMN_ALIASES[column_name_mappings_name][internal_column]
        matches = [alias for alias in aliases if alias in available_columns]
        if len(matches) > 1:
            raise PpaError(
                self._error_message(
                    f"Ambiguous inferred source columns for {internal_column!r} "
                    f"in {str(path)!r}: {matches}. Configure "
                    f"{_PERFORMANCE_COLUMN_KEYS[column_name_mappings_name][internal_column]!r} "
                    "explicitly."
                ),
                502,
            )
        return matches[0] if matches else None

    @staticmethod
    def _missing_column_message(
        column_name_mappings_name: PerformanceSourceType,
        internal_column: str,
    ) -> str:
        """Return an error fragment for a missing performance source column."""
        yaml_key = _PERFORMANCE_COLUMN_KEYS[column_name_mappings_name][internal_column]
        aliases = _PERFORMANCE_COLUMN_ALIASES[column_name_mappings_name][internal_column]
        return f"{yaml_key!r} for {internal_column!r}; tried aliases {list(aliases)}"
