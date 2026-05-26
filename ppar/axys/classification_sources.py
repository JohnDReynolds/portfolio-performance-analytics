"""Load normalized Axys classification and mapping sources."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from typing import Any, Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.axys.specification import AxysSpecification, ErrorMessage, SourceType
from ppar.errors import PpaError
import ppar.utilities as util

_CLASSIFICATION_FIELDS_ALLOWED: Final[set[str]] = {
    "default_file_path",
    "identifier_column",
    "name_column",
    "is_security_master",
    "filter_column",
    "filter_value",
    "mapping",
}
_MAPPING_FIELDS_ALLOWED: Final[set[str]] = {
    "default_file_path",
    "identifier_column",
    "name_column",
    "is_security_master",
    "filter_column",
    "filter_value",
}
_CLASSIFICATION_MAPPING_FIELDS_REQUIRED: Final[set[str]] = {
    "default_file_path",
    "identifier_column",
    "name_column",
}
_CLASSIFICATION_MAPPING_COLUMN_NAMES: Final[set[str]] = {
    "identifier_column",
    "name_column",
    "filter_column",
}


class AxysClassificationSourceLoader:
    """Normalize Axys classification and mapping CSV sources.

    Attributes:
        _specification: Parsed Axys source configuration.
        _error_message: Callback used to add facade-level validation context.
    """

    def __init__(
        self,
        specification: AxysSpecification,
        error_message: ErrorMessage,
        source_path_overrides: Mapping[str, util.PathLike] | None = None,
    ) -> None:
        """Initialize a classification/mapping source loader.

        Args:
            specification: Parsed Axys configuration.
            error_message: Callback that adds facade-level source context to
                validation messages.
            source_path_overrides: Optional source file paths keyed by
                configured classification or mapping source name.

        Raises:
            PpaError: If a source path override references an unknown
                classification or mapping source.
        """
        self._specification = specification
        self._error_message = error_message
        self._source_path_overrides = dict(source_path_overrides or {})
        self._validate_source_path_overrides()

    def load(
        self,
        source_type: SourceType,
        source_name: str | None,
        unique_security_ids: list[str],
    ) -> pl.DataFrame:
        """Load a normalized classification or mapping source.

        Args:
            source_type: Kind of supporting source to load.
            source_name: Configured source name, or ``None`` to return an empty
                DataFrame.
            unique_security_ids: Security identifiers retained in loaded
                portfolio output.

        Returns:
            Two-column DataFrame containing normalized identifier/name pairs.

        Raises:
            PpaError: If the source is unknown, its specification is invalid,
                its CSV does not exist, or its declared columns do not exist.
        """
        if not source_name:
            return pl.DataFrame()

        data_sources: dict[str, dict[str, Any]] = self._specification.values.get(
            f"{source_type}s", {}
        )
        if source_name not in data_sources:
            raise PpaError(
                self._error_message(f"Unknown {source_type} {source_name!r}"),
                504,
            )

        data_source = data_sources[source_name]
        self._validate_source_definition(source_type, source_name, data_source)

        file_path = self._source_file_path(source_name, data_source)
        if not util.file_path_exists(file_path):
            raise PpaError(self._error_message(util.file_path_error(file_path)), None)

        lazy_frame = pl.scan_csv(file_path)
        self._validate_csv_columns(source_type, source_name, data_source, lazy_frame)

        if data_source.get("is_security_master", False):
            lazy_frame = lazy_frame.filter(
                pl.col(data_source["identifier_column"]).is_in(unique_security_ids)
            )
        if {"filter_column", "filter_value"}.issubset(data_source):
            lazy_frame = lazy_frame.filter(
                pl.col(data_source["filter_column"]) == data_source["filter_value"]
            )

        rename_mappings = {
            data_source["identifier_column"]: "identifier_column",
            data_source["name_column"]: "name_column",
        }
        return (
            lazy_frame.collect()
            .rename(rename_mappings)
            .select(("identifier_column", "name_column"))
            .unique(subset=["identifier_column"], keep="any")
        )

    def _validate_csv_columns(
        self,
        source_type: SourceType,
        source_name: str,
        data_source: dict[str, Any],
        lazy_frame: pl.LazyFrame,
    ) -> None:
        """Validate that configured source columns exist in the CSV.

        Args:
            source_type: Kind of supporting source being loaded.
            source_name: Configured source name being loaded.
            data_source: Source definition from the Axys specification.
            lazy_frame: Lazy CSV scan used to inspect available columns.

        Raises:
            PpaError: If any configured CSV column name does not exist.
        """
        specified_column_names = {
            data_source[field]
            for field in _CLASSIFICATION_MAPPING_COLUMN_NAMES
            if field in data_source
        }
        nonexistent_column_names = specified_column_names - set(
            lazy_frame.collect_schema().names()
        )
        if nonexistent_column_names:
            raise PpaError(
                self._error_message(
                    f"Nonexistent column names for {source_type} {source_name!r}: "
                    f"{nonexistent_column_names}"
                ),
                504,
            )

    def _validate_source_definition(
        self,
        source_type: SourceType,
        source_name: str,
        data_source: dict[str, Any],
    ) -> None:
        """Validate a classification or mapping source specification.

        Args:
            source_type: Kind of supporting source being loaded.
            source_name: Configured source name being loaded.
            data_source: Source definition from the Axys specification.

        Raises:
            PpaError: If required fields are missing, unknown fields are
                present, ``is_security_master`` is not boolean, or a
                non-security-master classification does not identify a
                configured mapping.
        """
        allowed_fields = (
            _CLASSIFICATION_FIELDS_ALLOWED
            if source_type == "classification"
            else _MAPPING_FIELDS_ALLOWED
        )
        unknown_fields = set(data_source) - allowed_fields
        if unknown_fields:
            raise PpaError(
                self._error_message(
                    f"Unknown fields for {source_type} {source_name!r}: {unknown_fields}"
                ),
                504,
            )

        missing_fields = _CLASSIFICATION_MAPPING_FIELDS_REQUIRED - set(data_source)
        if missing_fields:
            raise PpaError(
                self._error_message(
                    f"Missing fields for {source_type} {source_name!r}: {missing_fields}"
                ),
                504,
            )

        is_security_master = data_source.get("is_security_master", False)
        if not isinstance(is_security_master, bool):
            raise PpaError(
                self._error_message(
                    f"Invalid is_security_master value for {source_type} {source_name!r}: "
                    f"{is_security_master!r} must be a boolean."
                ),
                504,
            )

        mapping_name = data_source.get("mapping")
        if source_type == "classification" and not is_security_master:
            self._validate_classification_mapping(source_name, mapping_name)

    def _validate_classification_mapping(
        self,
        source_name: str,
        mapping_name: object,
    ) -> None:
        """Validate a non-security-master classification mapping reference.

        Args:
            source_name: Configured classification source name.
            mapping_name: Mapping reference from the classification source
                definition.

        Raises:
            PpaError: If the classification does not reference a configured
                mapping.
        """
        if not isinstance(mapping_name, str) or not mapping_name:
            raise PpaError(
                self._error_message(
                    f"Missing mapping for classification {source_name!r}. "
                    "Non-security-master classifications must specify a mapping."
                ),
                504,
            )
        if mapping_name not in self._specification.values.get("mappings", {}):
            raise PpaError(
                self._error_message(
                    f"Unknown mapping {mapping_name!r} for classification {source_name!r}"
                ),
                504,
            )

    def _source_file_path(self, source_name: str, data_source: dict[str, Any]) -> util.PathLike:
        """Return the override or configured default path for a source.

        Args:
            source_name: Configured classification or mapping source name.
            data_source: Source definition from the Axys specification.

        Returns:
            Resolved source file path.
        """
        override_path = self._source_path_overrides.get(source_name)
        file_path = (
            override_path
            if override_path is not None
            else cast(util.PathLike, data_source["default_file_path"])
        )
        return self._specification.resolve_path(file_path)

    def _validate_source_path_overrides(self) -> None:
        """Validate that file path overrides reference configured sources.

        Raises:
            PpaError: If any override key is not a configured classification or
                mapping source name.
        """
        configured_source_names = set(
            self._specification.values.get("classifications", {})
        ) | set(self._specification.values.get("mappings", {}))
        unknown_source_names = set(self._source_path_overrides) - configured_source_names
        if unknown_source_names:
            raise PpaError(
                self._error_message(
                    f"Unknown source path override names: {unknown_source_names}"
                ),
                504,
            )
