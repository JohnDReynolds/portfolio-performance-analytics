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

PerformanceSourceType = Literal["portperf_columns", "secperf_columns"]

_PORTPERF_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.FROM_DATE,
    cols.THRU_DATE,
    cols.PORTFOLIO_CODE,
    cols.PORTFOLIO_NAME,
    cols.PORTFOLIO_RETURN,
}
_SECPERF_REQUIRED_COLUMNS: Final[set[str]] = {
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
            _PORTPERF_REQUIRED_COLUMNS
            if column_name_mappings_name == "portperf_columns"
            else _SECPERF_REQUIRED_COLUMNS
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
        column_mappings: dict[str, str] = self._specification.values.get(
            column_name_mappings_name, {}
        )
        available_columns = set(column_mappings)
        missing_columns = required_columns - available_columns
        csv_to_internal_mappings: dict[str, str] = {}

        if not missing_columns:
            csv_to_internal_mappings = {value: key for key, value in column_mappings.items()}
            header = pl.read_csv(path, n_rows=0)
            available_columns = {
                csv_to_internal_mappings[column]
                for column in csv_to_internal_mappings
                if column in header.columns
            }
            missing_columns = required_columns - available_columns

        if missing_columns:
            raise PpaError(
                self._error_message(
                    f"Missing {sorted(missing_columns)} in {str(path)!r}.  |  "
                    f"Columns available are: {sorted(available_columns)}"
                ),
                502,
            )

        return csv_to_internal_mappings
