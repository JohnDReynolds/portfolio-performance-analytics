"""Shared CSV loading helpers for performance comparison sources."""

from __future__ import annotations

# Python imports
from pathlib import Path
from typing import TypeAlias

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.specification import (
    ComparisonSnapshot,
    PerformanceComparisonSpecification,
)
from ppar.errors import PpaError
import ppar.utilities as util

ColumnAliases: TypeAlias = dict[str, tuple[str, ...]]

_SCHEMA_COLUMN_SECTIONS = {
    pc_cols.PORTFOLIO_PERFORMANCE: "portfolio_performance_columns",
    pc_cols.SECURITY_PERFORMANCE: "security_performance_columns",
    pc_cols.SECURITY_MASTER: "security_master_columns",
}
_SCHEMA_COLUMN_KEYS: dict[str, dict[str, str]] = {
    pc_cols.PORTFOLIO_PERFORMANCE: {
        "portfolio_code": pc_cols.PORTFOLIO_ID,
        "portfolio_id": pc_cols.PORTFOLIO_ID,
        "portfolio_name": pc_cols.PORTFOLIO_NAME,
        "from_date": pc_cols.FROM_DATE,
        "thru_date": pc_cols.THRU_DATE,
        "portfolio_return": pc_cols.PORTFOLIO_RETURN,
    },
    pc_cols.SECURITY_PERFORMANCE: {
        "portfolio_code": pc_cols.PORTFOLIO_ID,
        "portfolio_id": pc_cols.PORTFOLIO_ID,
        "identifier": pc_cols.SECURITY_ID,
        "security_id": pc_cols.SECURITY_ID,
        "security_name": pc_cols.SECURITY_NAME,
        "from_date": pc_cols.FROM_DATE,
        "thru_date": pc_cols.THRU_DATE,
        "return": pc_cols.SECURITY_RETURN,
        "security_return": pc_cols.SECURITY_RETURN,
        "weight": pc_cols.WEIGHT,
        "contribution": pc_cols.CONTRIBUTION,
    },
    pc_cols.SECURITY_MASTER: {
        "identifier_column": pc_cols.SECURITY_ID,
        "identifier": pc_cols.SECURITY_ID,
        "security_id": pc_cols.SECURITY_ID,
        "name_column": pc_cols.SECURITY_NAME,
        "security_name": pc_cols.SECURITY_NAME,
    },
}


def csv_to_internal_mappings(
    path: util.PathLike,
    dataset_name: str,
    required_aliases: ColumnAliases,
    optional_aliases: ColumnAliases,
    specification_path: util.PathLike,
) -> dict[str, str]:
    """Return source-to-normalized column mappings for a CSV header.

    Args:
        path: CSV source path.
        dataset_name: Human-readable normalized dataset name for error messages.
        required_aliases: Required normalized columns and allowed source aliases.
        optional_aliases: Optional normalized columns and allowed source aliases.
        specification_path: Comparison specification path for error context.

    Returns:
        Mapping from source CSV column name to normalized internal column name.

    Raises:
        PpaError: If a required normalized column is missing or if any
            normalized column resolves to multiple source columns.
    """
    available_columns = set(pl.read_csv(path, n_rows=0).columns)
    mappings: dict[str, str] = {}
    missing_columns: list[str] = []

    for internal_column, aliases in required_aliases.items():
        source_column = _resolve_column(
            path,
            dataset_name,
            internal_column,
            aliases,
            available_columns,
            specification_path,
        )
        if source_column is None:
            missing_columns.append(f"{internal_column!r}; tried aliases {list(aliases)}")
            continue
        mappings[source_column] = internal_column

    if missing_columns:
        raise PpaError(
            _error_message(
                f"Missing {missing_columns} in {str(path)!r}.  |  "
                f"CSV columns available are: {sorted(available_columns)}",
                specification_path,
            ),
            502,
        )

    for internal_column, aliases in optional_aliases.items():
        source_column = _resolve_column(
            path,
            dataset_name,
            internal_column,
            aliases,
            available_columns,
            specification_path,
        )
        if source_column is not None:
            mappings[source_column] = internal_column

    return mappings


def read_mapped_csv(
    path: util.PathLike,
    columns: tuple[str, ...],
    dataset_name: str,
    required_aliases: ColumnAliases,
    optional_aliases: ColumnAliases,
    specification_path: util.PathLike,
) -> pl.DataFrame:
    """Read a CSV and return only normalized columns resolved from aliases.

    Args:
        path: CSV source path.
        columns: Normalized output column order.
        dataset_name: Human-readable normalized dataset name for error messages.
        required_aliases: Required normalized columns and allowed source aliases.
        optional_aliases: Optional normalized columns and allowed source aliases.
        specification_path: Comparison specification path for error context.

    Returns:
        DataFrame with source columns renamed to normalized internal names.

    Raises:
        PpaError: If column mappings cannot be resolved unambiguously.
    """
    mappings = csv_to_internal_mappings(
        path,
        dataset_name,
        required_aliases,
        optional_aliases,
        specification_path,
    )
    selected_columns = [
        column_name
        for column_name in columns
        if column_name in mappings.values()
    ]
    return pl.read_csv(path).rename(mappings).select(selected_columns)


def read_schema_mapped_csv(
    path: util.PathLike,
    columns: tuple[str, ...],
    dataset_name: str,
    default_required_aliases: ColumnAliases,
    default_optional_aliases: ColumnAliases,
    specification: PerformanceComparisonSpecification,
    snapshot_key: str,
) -> pl.DataFrame:
    """Read a CSV using built-in aliases plus snapshot schema overrides.

    Args:
        path: CSV source path.
        columns: Normalized output column order.
        dataset_name: Normalized comparison dataset name.
        default_required_aliases: Built-in required aliases.
        default_optional_aliases: Built-in optional aliases.
        specification: Parsed comparison specification.
        snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

    Returns:
        DataFrame with source columns renamed to normalized internal names.
    """
    snapshot = snapshot_by_key(specification, snapshot_key)
    return read_mapped_csv(
        path,
        columns,
        dataset_name,
        aliases_with_schema_overrides(
            dataset_name,
            default_required_aliases,
            snapshot,
            specification.path,
        ),
        aliases_with_schema_overrides(
            dataset_name,
            default_optional_aliases,
            snapshot,
            specification.path,
        ),
        specification.path,
    )


def require_numeric_columns(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    dataset_name: str,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> pl.DataFrame:
    """Raise if present numeric columns contain nonblank nonnumeric values.

    Args:
        frame: Normalized source frame.
        columns: Normalized numeric columns to validate when present.
        dataset_name: Normalized dataset name for error messages.
        path: Source CSV path.
        specification_path: Comparison YAML path for error context.

    Returns:
        The original DataFrame, unchanged.

    Raises:
        PpaError: If a present numeric column contains a nonblank value that
            cannot be converted to a number.
    """
    for column in columns:
        if column not in frame.columns:
            continue
        invalid_rows = frame.filter(
            pl.col(column).is_not_null()
            & pl.col(column).cast(pl.Float64, strict=False).is_null()
        )
        if invalid_rows.is_empty():
            continue
        invalid_value = invalid_rows.get_column(column)[0]
        raise PpaError(
            _error_message(
                (
                    f"{dataset_name} column {column!r} contains a nonnumeric "
                    f"value {invalid_value!r} in {str(path)!r}."
                ),
                specification_path,
            ),
            502,
        )
    return frame


def snapshot_by_key(
    specification: PerformanceComparisonSpecification,
    snapshot_key: str,
) -> ComparisonSnapshot:
    """Return the configured snapshot for a neutral snapshot key."""
    if snapshot_key == "a":
        return specification.snapshot_a
    if snapshot_key == "b":
        return specification.snapshot_b
    raise PpaError(f"Unknown snapshot key: {snapshot_key}", 999)


def optional_file_path(
    specification: PerformanceComparisonSpecification,
    dataset_name: str,
    snapshot_key: str,
) -> util.PathLike | None:
    """Return a resolved optional file path for a snapshot.

    Args:
        specification: Parsed comparison specification.
        dataset_name: Normalized comparison dataset name.
        snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

    Returns:
        The resolved snapshot-specific file path, or ``None`` when the
        optional dataset is not configured.

    Raises:
        PpaError: If ``snapshot_key`` is not a known neutral snapshot key.
    """
    comparison_file = specification.files.get(dataset_name)
    if comparison_file is None:
        return None
    if snapshot_key == "a":
        return comparison_file.snapshot_a_path
    if snapshot_key == "b":
        return comparison_file.snapshot_b_path
    raise PpaError(f"Unknown snapshot key: {snapshot_key}", 999)


def aliases_with_schema_overrides(
    dataset_name: str,
    default_aliases: ColumnAliases,
    snapshot: ComparisonSnapshot,
    specification_path: util.PathLike,
) -> ColumnAliases:
    """Return aliases with explicit schema mappings placed first.

    Args:
        dataset_name: Normalized comparison dataset name.
        default_aliases: Built-in aliases for the dataset.
        snapshot: Snapshot configuration that may reference a schema YAML.
        specification_path: Comparison specification path for error context.

    Returns:
        Alias mapping with configured source columns tried before built-in
        defaults for the same normalized column.

    Raises:
        PpaError: If a referenced schema YAML cannot be parsed or has invalid
            column mapping shapes.
    """
    schema_aliases = _schema_aliases(dataset_name, snapshot, specification_path)
    if not schema_aliases:
        return default_aliases

    merged_aliases = dict(default_aliases)
    for internal_column, schema_column_aliases in schema_aliases.items():
        merged_aliases[internal_column] = schema_column_aliases
    return merged_aliases


def _resolve_column(
    path: util.PathLike,
    dataset_name: str,
    internal_column: str,
    aliases: tuple[str, ...],
    available_columns: set[str],
    specification_path: util.PathLike,
) -> str | None:
    """Resolve a source column from aliases, rejecting duplicate matches."""
    matches = [alias for alias in aliases if alias in available_columns]
    if len(matches) > 1:
        display_dataset_name = dataset_name.replace("_", " ")
        raise PpaError(
            _error_message(
                f"Ambiguous {display_dataset_name} source columns for "
                f"{internal_column!r}: {matches}.  |  "
                f"CSV path is {str(path)!r}.",
                specification_path,
            ),
            502,
        )
    return matches[0] if matches else None


def _schema_aliases(
    dataset_name: str,
    snapshot: ComparisonSnapshot,
    specification_path: util.PathLike,
) -> ColumnAliases:
    """Return aliases from a referenced schema YAML for one dataset."""
    section_name = _SCHEMA_COLUMN_SECTIONS.get(dataset_name)
    if section_name is None or snapshot.schema_path is None:
        return {}

    schema_values = _read_schema_yaml(snapshot.schema_path, specification_path)
    section = schema_values.get(section_name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise PpaError(
            _error_message(f"{section_name} must be a mapping.", specification_path),
            504,
        )

    key_mappings = _SCHEMA_COLUMN_KEYS[dataset_name]
    aliases: ColumnAliases = {}
    for schema_key, source_column in section.items():
        if schema_key not in key_mappings:
            continue
        if not isinstance(source_column, str) or not source_column:
            raise PpaError(
                _error_message(
                    f"{section_name}.{schema_key} must be a string.",
                    specification_path,
                ),
                504,
            )
        aliases[key_mappings[schema_key]] = (source_column,)
    return aliases


def _read_schema_yaml(
    schema_path: Path,
    specification_path: util.PathLike,
) -> dict[object, object]:
    """Read a referenced schema YAML as a mapping."""
    with open(schema_path, "r", encoding=util.ENCODING) as file:
        try:
            values = yaml.safe_load(file)
        except Exception as error:
            raise PpaError(
                _error_message(f"Invalid schema YAML: {error}", specification_path),
                504,
            ) from error
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise PpaError(
            _error_message("Referenced schema YAML must be a mapping.", specification_path),
            504,
        )
    return values


def _error_message(message: str, specification_path: util.PathLike) -> str:
    """Return an error message with comparison specification context."""
    return f"{message}  |  comparison_specification_path={specification_path}"
