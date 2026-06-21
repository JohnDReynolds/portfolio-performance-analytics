"""Load normalized Axys classification and mapping sources."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from typing import Any, Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.axys.specification import AxysSpecification, ErrorMessage, SourceType
from ppar.axys.column_aliases import resolve_column
import ppar.analytics.schema as cols
from ppar.errors import PpaError
import ppar.utilities as util

_CLASSIFICATION_FIELDS_ALLOWED: Final[set[str]] = {
    "display_name",
    "file_path",
    "identifier_column",
    "name_column",
    "is_security_master",
    "filter_column",
    "filter_value",
    "mapping",
}
_MAPPING_FIELDS_ALLOWED: Final[set[str]] = {
    "classification_column",
    "display_name_column",
}
_FILE_BACKED_CLASSIFICATION_FIELDS_REQUIRED: Final[set[str]] = {
    "file_path",
    "identifier_column",
    "name_column",
}
_CLASSIFICATION_MAPPING_COLUMN_NAMES: Final[set[str]] = {
    "identifier_column",
    "name_column",
    "filter_column",
}
_MAPPING_FIELDS_REQUIRED: Final[set[str]] = {"classification_column"}
_SECURITY_MASTER_FIELDS_REQUIRED: Final[set[str]] = {
    "identifier_column",
    "name_column",
}
_SECURITY_MASTER_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "identifier_column": ("SECURITY_ID", "SEC", "SECURITY", "SEC_ID"),
    "name_column": ("SECURITY_NAME", "DESC", "DESCRIPTION", "NAME", "SEC_DESC"),
}
_SECURITY_MASTER_COLUMNS_KEY: Final[str] = "security_master_columns"
_SECURITY_MASTER_PATH_KEY: Final[str] = "security_master_path"
_SECURITY_CLASSIFICATION_NAME: Final[str] = "Security"
_FILTER_TO_SECURITY_IDS: Final[str] = "_filter_to_security_ids"
_SOURCE_FILE_PATH: Final[str] = "_source_file_path"
_NORMALIZED_SOURCE_COLUMNS: Final[tuple[str, str]] = (cols.IDENTIFIER, cols.NAME)


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
                configured classification source name.

        Raises:
            PpaError: If a source path override references an unknown
                classification source.
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

        data_source = self._source_definition(source_type, source_name)
        if data_source is None:
            raise PpaError(
                self._error_message(f"Unknown {source_type} {source_name!r}"),
                504,
            )

        self._validate_source_definition(source_type, source_name, data_source)
        effective_source = self._effective_source_definition(
            source_type,
            source_name,
            data_source,
        )

        file_path = self._source_file_path(source_type, source_name, effective_source)
        if not util.file_path_exists(file_path):
            raise PpaError(self._error_message(util.file_path_error(file_path)), None)

        lazy_frame = pl.scan_csv(file_path)
        self._validate_csv_columns(source_type, source_name, effective_source, lazy_frame)

        if effective_source.get(_FILTER_TO_SECURITY_IDS, False):
            lazy_frame = lazy_frame.filter(
                pl.col(effective_source["identifier_column"]).is_in(unique_security_ids)
            )
        if {"filter_column", "filter_value"}.issubset(effective_source):
            lazy_frame = lazy_frame.filter(
                pl.col(effective_source["filter_column"])
                == effective_source["filter_value"]
            )

        rename_mappings = {
            effective_source["identifier_column"]: cols.IDENTIFIER,
            effective_source["name_column"]: cols.NAME,
        }
        return (
            lazy_frame.collect()
            .rename(rename_mappings)
            .select(_NORMALIZED_SOURCE_COLUMNS)
            .unique(subset=[cols.IDENTIFIER], keep="any")
        )

    def _source_definition(
        self,
        source_type: SourceType,
        source_name: str,
    ) -> dict[str, Any] | None:
        """Return an explicit source definition or synthesized classification.

        Args:
            source_type: Kind of supporting source being loaded.
            source_name: Configured source name being loaded.

        Returns:
            Source definition from the Axys specification, a synthesized
            mapping-backed classification definition, or ``None`` when the
            source is unknown.
        """
        data_sources = (
            self._specification.classifications
            if source_type == "classification"
            else self._specification.mappings
        )
        data_source = data_sources.get(source_name)
        if data_source is not None:
            return data_source

        if source_type == "classification":
            if source_name == _SECURITY_CLASSIFICATION_NAME:
                return {"is_security_master": True}
            mapping_source = self._specification.mappings.get(source_name)
            if (
                isinstance(mapping_source, dict)
                and "display_name_column" in mapping_source
            ):
                return {
                    "mapping": source_name,
                    "name_column": mapping_source["display_name_column"],
                }
            if isinstance(mapping_source, dict):
                raise PpaError(
                    self._error_message(
                        f"Mapping {source_name!r} cannot be used as a classification "
                        "without display_name_column."
                    ),
                    504,
                )
        return None

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
                present, ``is_security_master`` is not boolean, required
                source fields are missing, or a non-security-master
                classification does not identify a configured mapping.
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

        is_security_master = data_source.get("is_security_master", False)
        if not isinstance(is_security_master, bool):
            raise PpaError(
                self._error_message(
                    f"Invalid is_security_master value for {source_type} {source_name!r}: "
                    f"{is_security_master!r} must be a boolean."
                ),
                504,
            )

        if source_type == "mapping":
            self._validate_mapping_definition(source_name, data_source)
            return

        if is_security_master:
            return

        mapping_name = data_source.get("mapping")
        self._validate_classification_mapping(source_name, mapping_name)
        self._validate_classification_source_fields(source_name, data_source)

    def _validate_mapping_definition(
        self,
        source_name: str,
        data_source: dict[str, Any],
    ) -> None:
        """Validate a mapping definition that points into the security master.

        Args:
            source_name: Configured mapping source name.
            data_source: Mapping definition from the Axys specification.

        Raises:
            PpaError: If the required ``classification_column`` field is
                missing.
        """
        missing_fields = _MAPPING_FIELDS_REQUIRED - set(data_source)
        if missing_fields:
            raise PpaError(
                self._error_message(
                    f"Missing fields for mapping {source_name!r}: {missing_fields}"
                ),
                504,
            )

    def _validate_classification_source_fields(
        self,
        source_name: str,
        data_source: dict[str, Any],
    ) -> None:
        """Validate explicit or security-master-backed classification fields.

        Args:
            source_name: Configured classification source name.
            data_source: Classification definition from the Axys specification.

        Raises:
            PpaError: If the classification has neither a complete explicit
                source definition nor a security-master-backed definition.
        """
        has_file_path = "file_path" in data_source
        if has_file_path:
            missing_fields = _FILE_BACKED_CLASSIFICATION_FIELDS_REQUIRED - set(
                data_source
            )
            if missing_fields:
                raise PpaError(
                    self._error_message(
                        f"Missing fields for classification {source_name!r}: "
                        f"{missing_fields}"
                    ),
                    504,
                )
            return

        if "identifier_column" in data_source:
            raise PpaError(
                self._error_message(
                    f"Missing fields for classification {source_name!r}: "
                    "{'file_path'}"
                ),
                504,
            )

        missing_fields = {"name_column"} - set(data_source)
        if missing_fields:
            raise PpaError(
                self._error_message(
                    f"Missing fields for classification {source_name!r}: "
                    f"{missing_fields}"
                ),
                504,
            )

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
        if mapping_name not in self._specification.mappings:
            raise PpaError(
                self._error_message(
                    f"Unknown mapping {mapping_name!r} for classification {source_name!r}"
                ),
                504,
            )

    def _source_file_path(
        self,
        source_type: SourceType,
        source_name: str,
        data_source: dict[str, Any],
    ) -> util.PathLike:
        """Return the override or configured default path for a source.

        Args:
            source_type: Kind of supporting source being loaded.
            source_name: Configured classification or mapping source name.
            data_source: Source definition from the Axys specification.

        Returns:
            Resolved source file path.
        """
        override_path = (
            self._source_path_overrides.get(source_name)
            if source_type == "classification"
            else None
        )
        file_path = override_path if override_path is not None else data_source[_SOURCE_FILE_PATH]
        return self._specification.resolve_path(file_path)

    def _effective_source_definition(
        self,
        source_type: SourceType,
        source_name: str,
        data_source: dict[str, Any],
    ) -> dict[str, Any]:
        """Return inherited path and column settings for a loadable source.

        Args:
            source_type: Kind of supporting source being loaded.
            source_name: Configured source name being loaded.
            data_source: Raw source definition from the Axys specification.

        Returns:
            Source definition with explicit file path, identifier column, name
            column, and security-ID filtering behavior.

        Raises:
            PpaError: If required security master settings are missing.
        """
        if source_type == "mapping":
            security_master = self._security_master_definition(source_name)
            return {
                _SOURCE_FILE_PATH: security_master[_SOURCE_FILE_PATH],
                "identifier_column": security_master["identifier_column"],
                "name_column": data_source["classification_column"],
                _FILTER_TO_SECURITY_IDS: True,
            }

        if data_source.get("is_security_master", False):
            security_master = self._security_master_definition(source_name)
            return {
                _SOURCE_FILE_PATH: self._explicit_or_security_master_path(
                    data_source,
                    security_master,
                ),
                "identifier_column": data_source.get(
                    "identifier_column", security_master["identifier_column"]
                ),
                "name_column": data_source.get(
                    "name_column", security_master["name_column"]
                ),
                _FILTER_TO_SECURITY_IDS: True,
            }

        if "file_path" in data_source:
            return {
                **data_source,
                _SOURCE_FILE_PATH: data_source["file_path"],
                _FILTER_TO_SECURITY_IDS: False,
            }

        mapping_name = cast(str, data_source["mapping"])
        mapping = self._specification.mappings[mapping_name]
        security_master = self._security_master_definition(source_name)
        return {
            **data_source,
            _SOURCE_FILE_PATH: security_master[_SOURCE_FILE_PATH],
            "identifier_column": mapping["classification_column"],
            _FILTER_TO_SECURITY_IDS: False,
        }

    def _security_master_definition(self, source_name: str) -> dict[str, Any]:
        """Return validated top-level security master path and columns.

        Args:
            source_name: Source requiring security master settings, used for
                validation context.

        Returns:
            Source definition containing the security master path and columns.

        Raises:
            PpaError: If the security master path or required columns are not
                configured.
        """
        security_master_path = self._specification.values.get(_SECURITY_MASTER_PATH_KEY)
        configured_columns_value = self._specification.values.get(
            _SECURITY_MASTER_COLUMNS_KEY,
            {},
        )
        if not security_master_path:
            raise PpaError(
                self._error_message(
                    f"{_SECURITY_MASTER_PATH_KEY} is required for source {source_name!r}."
                ),
                504,
            )
        if not isinstance(configured_columns_value, dict):
            raise PpaError(
                self._error_message(f"{_SECURITY_MASTER_COLUMNS_KEY} must be a mapping."),
                504,
            )
        configured_columns = cast(dict[str, Any], configured_columns_value)

        security_master_columns = self._resolve_security_master_columns(
            source_name,
            security_master_path,
            configured_columns,
        )
        return {
            _SOURCE_FILE_PATH: security_master_path,
            "identifier_column": security_master_columns["identifier_column"],
            "name_column": security_master_columns["name_column"],
        }

    def _resolve_security_master_columns(
        self,
        source_name: str,
        security_master_path: util.PathLike,
        configured_columns: dict[str, Any],
    ) -> dict[str, str]:
        """Return explicit or inferred security master column names.

        Args:
            source_name: Source requiring security master settings, used for
                validation context.
            security_master_path: Configured security master source path.
            configured_columns: Explicit YAML security master column mappings.

        Returns:
            Mapping for ``identifier_column`` and ``name_column``.

        Raises:
            PpaError: If a column cannot be inferred or if multiple aliases
                match the security master header.
        """
        path = self._specification.resolve_path(security_master_path)
        if not util.file_path_exists(path):
            raise PpaError(self._error_message(util.file_path_error(path)), None)

        available_columns = set(pl.read_csv(path, n_rows=0).columns)
        resolved_columns: dict[str, str] = {}
        missing_fields: list[str] = []
        for field_name in _SECURITY_MASTER_FIELDS_REQUIRED:
            source_column = resolve_column(
                field_name,
                _SECURITY_MASTER_COLUMN_ALIASES[field_name],
                available_columns,
                self._error_message,
                explicit_column=configured_columns.get(field_name),
                ambiguous_message=(
                    "Ambiguous inferred security master column. "
                    f"Configure {field_name!r} explicitly. "
                    f"Source requiring security master: {source_name!r}"
                ),
                error_code=504,
            )
            if source_column is None:
                missing_fields.append(
                    f"{field_name!r}; tried aliases "
                    f"{list(_SECURITY_MASTER_COLUMN_ALIASES[field_name])}"
                )
                continue
            if source_column not in available_columns:
                missing_fields.append(
                    f"{field_name!r} configured as {source_column!r}"
                )
                continue
            resolved_columns[field_name] = source_column

        if missing_fields:
            raise PpaError(
                self._error_message(
                    f"Missing {missing_fields} for {_SECURITY_MASTER_COLUMNS_KEY}. "
                    f"CSV columns available are: {sorted(available_columns)}. "
                    f"Source requiring security master: {source_name!r}"
                ),
                504,
            )
        return resolved_columns

    @staticmethod
    def _explicit_or_security_master_path(
        data_source: dict[str, Any],
        security_master: dict[str, Any],
    ) -> util.PathLike:
        """Return an explicit source path or inherited security master path."""
        return cast(
            util.PathLike,
            data_source.get("file_path", security_master[_SOURCE_FILE_PATH]),
        )

    def _validate_source_path_overrides(self) -> None:
        """Validate that file path overrides reference configured sources.

        Raises:
            PpaError: If any override key is not a configured classification
                source name.
        """
        configured_source_names = set(self._specification.classifications)
        unknown_source_names = set(self._source_path_overrides) - configured_source_names
        if unknown_source_names:
            raise PpaError(
                self._error_message(
                    f"Unknown source path override names: {unknown_source_names}"
                ),
                504,
            )
