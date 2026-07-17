"""Shared CSV loading helpers for performance comparison sources."""

from __future__ import annotations

# Python imports
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TypeAlias

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.audit import schema as pc_cols
from ppar.audit.specification import (
    ComparisonSnapshot,
    AuditSpecification,
)
from ppar.errors import PpaError
import ppar.utilities as util

ColumnAliases: TypeAlias = dict[str, tuple[str, ...]]
_SourceFrameCache: TypeAlias = dict[Path, pl.DataFrame]
_NormalizedFrameKey: TypeAlias = tuple[Path, str, str, Path]
_NormalizedFrameCache: TypeAlias = dict[_NormalizedFrameKey, pl.DataFrame]

_SOURCE_FRAME_CACHE: ContextVar[_SourceFrameCache | None] = ContextVar(
    "performance_comparison_source_frame_cache",
    default=None,
)
_NORMALIZED_FRAME_CACHE: ContextVar[_NormalizedFrameCache | None] = ContextVar(
    "performance_comparison_normalized_frame_cache",
    default=None,
)
_FINANCIAL_VALIDATION_CACHE: ContextVar[set[Path] | None] = ContextVar(
    "performance_comparison_financial_validation_cache",
    default=None,
)

_SCHEMA_COLUMN_SECTIONS = {
    pc_cols.PORTFOLIO_PERFORMANCE: "portfolio_performance_columns",
    pc_cols.SECURITY_PERFORMANCE: "security_performance_columns",
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
}


@contextmanager
def source_frame_cache() -> Iterator[None]:
    """Reuse raw and normalized frames within one performance-comparison run.

    Nested scopes share their parent's cache. The cache owns only untouched raw
    CSV frames and fully validated normalized frames. Polars operations return
    new frames, so callers can safely derive later tables without mutating the
    cached source objects.

    Yields:
        Control to the report or script run using the cache.
    """
    existing_cache = _SOURCE_FRAME_CACHE.get()
    if existing_cache is not None:
        yield
        return

    source_token = _SOURCE_FRAME_CACHE.set({})
    normalized_token = _NORMALIZED_FRAME_CACHE.set({})
    validation_token = _FINANCIAL_VALIDATION_CACHE.set(set())
    try:
        yield
    finally:
        _FINANCIAL_VALIDATION_CACHE.reset(validation_token)
        _NORMALIZED_FRAME_CACHE.reset(normalized_token)
        _SOURCE_FRAME_CACHE.reset(source_token)


def cached_normalized_frame(
    specification_path: util.PathLike,
    dataset_name: str,
    snapshot_key: str,
    source_path: util.PathLike,
) -> pl.DataFrame | None:
    """Return a validated normalized frame cached for this Audit run.

    Args:
        specification_path: Comparison YAML controlling normalization.
        dataset_name: Normalized dataset name.
        snapshot_key: Snapshot side, normally ``"a"`` or ``"b"``.
        source_path: Physical source file used by the loader.

    Returns:
        The cached frame, or ``None`` outside a cache scope or before the first
        successful load.
    """
    cache = _NORMALIZED_FRAME_CACHE.get()
    if cache is None:
        return None
    return cache.get(
        _normalized_frame_key(
            specification_path,
            dataset_name,
            snapshot_key,
            source_path,
        )
    )


def cache_normalized_frame(
    specification_path: util.PathLike,
    dataset_name: str,
    snapshot_key: str,
    source_path: util.PathLike,
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Cache and return one fully validated normalized source frame.

    Args:
        specification_path: Comparison YAML controlling normalization.
        dataset_name: Normalized dataset name.
        snapshot_key: Snapshot side, normally ``"a"`` or ``"b"``.
        source_path: Physical source file used by the loader.
        frame: Fully normalized and validated frame.

    Returns:
        ``frame`` unchanged.
    """
    cache = _NORMALIZED_FRAME_CACHE.get()
    if cache is not None:
        cache[
            _normalized_frame_key(
                specification_path,
                dataset_name,
                snapshot_key,
                source_path,
            )
        ] = frame
    return frame


def _normalized_frame_key(
    specification_path: util.PathLike,
    dataset_name: str,
    snapshot_key: str,
    source_path: util.PathLike,
) -> _NormalizedFrameKey:
    """Return the run-scoped normalized-frame cache key."""
    return (
        Path(specification_path).expanduser().resolve(),
        dataset_name,
        snapshot_key,
        Path(source_path).expanduser().resolve(),
    )


def financial_validation_is_cached(specification_path: util.PathLike) -> bool:
    """Return whether financial inputs passed validation in this audit run."""
    cache = _FINANCIAL_VALIDATION_CACHE.get()
    if cache is None:
        return False
    return Path(specification_path).expanduser().resolve() in cache


def cache_financial_validation(specification_path: util.PathLike) -> None:
    """Record successful financial-input validation in this audit run."""
    cache = _FINANCIAL_VALIDATION_CACHE.get()
    if cache is not None:
        cache.add(Path(specification_path).expanduser().resolve())


def _read_source_csv(path: util.PathLike) -> pl.DataFrame:
    """Read a raw CSV once in the active source-frame cache scope."""
    resolved_path = Path(path).expanduser().resolve()
    cache = _SOURCE_FRAME_CACHE.get()
    if cache is None:
        return pl.read_csv(resolved_path)
    if resolved_path not in cache:
        cache[resolved_path] = pl.read_csv(resolved_path)
    return cache[resolved_path]


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
    available_columns = set(_read_source_csv(path).columns)
    return _mappings_from_available_columns(
        path,
        dataset_name,
        required_aliases,
        optional_aliases,
        specification_path,
        available_columns,
    )


def _mappings_from_available_columns(
    path: util.PathLike,
    dataset_name: str,
    required_aliases: ColumnAliases,
    optional_aliases: ColumnAliases,
    specification_path: util.PathLike,
    available_columns: set[str],
) -> dict[str, str]:
    """Resolve normalized mappings from columns already read from a CSV."""
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
    source_frame = _read_source_csv(path)
    mappings = _mappings_from_available_columns(
        path,
        dataset_name,
        required_aliases,
        optional_aliases,
        specification_path,
        set(source_frame.columns),
    )
    selected_columns = [
        column_name
        for column_name in columns
        if column_name in mappings.values()
    ]
    return source_frame.rename(mappings).select(selected_columns)


def read_schema_mapped_csv(
    path: util.PathLike,
    columns: tuple[str, ...],
    dataset_name: str,
    default_required_aliases: ColumnAliases,
    default_optional_aliases: ColumnAliases,
    specification: AuditSpecification,
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
    specification: AuditSpecification,
    snapshot_key: str,
) -> ComparisonSnapshot:
    """Return the configured snapshot for a neutral snapshot key."""
    if snapshot_key == "a":
        return specification.snapshot_a
    if snapshot_key == "b":
        return specification.snapshot_b
    raise PpaError(f"Unknown snapshot key: {snapshot_key}", 999)


def optional_file_path(
    specification: AuditSpecification,
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
