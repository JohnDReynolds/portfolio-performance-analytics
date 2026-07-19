"""Construct stable PPAR security identifiers from Axys/APX source fields."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
import ppar.utilities as util

_SECURITY_ID_KEY: Final = "security_id"
_COMPONENTS_KEY: Final = "components"
_SEPARATOR_KEY: Final = "separator"
_DATASETS_KEY: Final = "datasets"
_DEFAULT_SEPARATOR: Final = ""
_CONFIGURATION_FIELDS: Final = {
    _COMPONENTS_KEY,
    _SEPARATOR_KEY,
    _DATASETS_KEY,
}
_DATASET_FIELDS: Final = {
    _COMPONENTS_KEY,
    _SEPARATOR_KEY,
}
_SECURITY_DATASETS: Final = {
    "holdings",
    "security_performance",
    "security_reference",
    "splits",
    "transactions",
}


@dataclass(frozen=True)
class SecurityIdConstruction:
    """Describe ordered Axys/APX fields used to construct a security key.

    Attributes:
        components: Exact-case source CSV columns, in concatenation order.
        separator: Optional text inserted between adjacent component values.
    """

    components: tuple[str, ...]
    separator: str

    @property
    def schema_overrides(self) -> dict[str, type[pl.DataType]]:
        """Return CSV schema overrides that preserve component text exactly."""
        return {component: pl.String for component in self.components}


def security_id_construction(
    values: Mapping[str, object],
    dataset_name: str,
    error_message: Callable[[str], str],
    *,
    error_code: int = 504,
) -> SecurityIdConstruction | None:
    """Return validated security-ID construction for one source dataset.

    Args:
        values: Parsed YAML root mapping.
        dataset_name: Normalized source dataset name.
        error_message: Callback that adds product-specific error context.
        error_code: PPAR error code used for invalid configuration.

    Returns:
        Construction settings for a security-bearing dataset, or ``None``
        when the YAML does not configure composite security identity.

    Raises:
        PpaError: If the configuration shape, component names, or separator is
            invalid.
    """
    if dataset_name not in _SECURITY_DATASETS:
        return None
    raw_configuration = values.get(_SECURITY_ID_KEY)
    if raw_configuration is None:
        return None
    configuration = _require_mapping(
        raw_configuration,
        _SECURITY_ID_KEY,
        error_message,
        error_code,
    )
    _reject_unknown_fields(
        configuration,
        _CONFIGURATION_FIELDS,
        _SECURITY_ID_KEY,
        error_message,
        error_code,
    )

    dataset_configuration: Mapping[str, object] = {}
    raw_datasets = configuration.get(_DATASETS_KEY, {})
    datasets = _require_mapping(
        raw_datasets,
        f"{_SECURITY_ID_KEY}.{_DATASETS_KEY}",
        error_message,
        error_code,
    )
    _reject_unknown_fields(
        datasets,
        _SECURITY_DATASETS,
        f"{_SECURITY_ID_KEY}.{_DATASETS_KEY}",
        error_message,
        error_code,
    )
    raw_dataset_configuration = datasets.get(dataset_name)
    if raw_dataset_configuration is not None:
        dataset_configuration = _require_mapping(
            raw_dataset_configuration,
            f"{_SECURITY_ID_KEY}.{_DATASETS_KEY}.{dataset_name}",
            error_message,
            error_code,
        )
        _reject_unknown_fields(
            dataset_configuration,
            _DATASET_FIELDS,
            f"{_SECURITY_ID_KEY}.{_DATASETS_KEY}.{dataset_name}",
            error_message,
            error_code,
        )

    components_value = dataset_configuration.get(
        _COMPONENTS_KEY,
        configuration.get(_COMPONENTS_KEY),
    )
    components = _validate_components(
        components_value,
        dataset_name,
        error_message,
        error_code,
    )
    separator_value = dataset_configuration.get(
        _SEPARATOR_KEY,
        configuration.get(_SEPARATOR_KEY, _DEFAULT_SEPARATOR),
    )
    separator = _validate_separator(
        separator_value,
        dataset_name,
        error_message,
        error_code,
    )
    return SecurityIdConstruction(components=components, separator=separator)


def with_constructed_security_id(
    frame: pl.DataFrame,
    construction: SecurityIdConstruction,
    *,
    output_column: str,
    dataset_name: str,
    source_path: util.PathLike,
    error_message: Callable[[str], str],
    error_code: int = 502,
) -> pl.DataFrame:
    """Add a validated composite security identifier to a source frame.

    Args:
        frame: Raw source CSV frame.
        construction: Ordered source columns and separator.
        output_column: Temporary or normalized constructed-ID column name.
        dataset_name: Normalized source dataset name for errors.
        source_path: Source CSV path for errors.
        error_message: Callback that adds product-specific error context.
        error_code: PPAR error code used for invalid source data.

    Returns:
        A new frame containing ``output_column``.

    Raises:
        PpaError: If a component column is missing, blank, padded with
            whitespace, or produces an ambiguous composite identifier.

    Notes:
        Symbols may contain the configured separator. PPAR therefore checks
        the observed component tuples for ambiguous concatenation instead of
        rejecting legitimate Axys/APX symbols such as ``MARGIN_USD``.
    """
    missing_columns = set(construction.components) - set(frame.columns)
    if missing_columns:
        raise PpaError(
            error_message(
                f"Missing security_id component columns {sorted(missing_columns)} "
                f"in {str(source_path)!r} for {dataset_name}. CSV columns "
                f"available are: {sorted(frame.columns)}"
            ),
            error_code,
        )

    string_expressions = {
        component: pl.col(component).cast(pl.String, strict=False)
        for component in construction.components
    }
    for component, string_expression in string_expressions.items():
        stripped_expression = string_expression.str.strip_chars()
        invalid_rows = frame.filter(
            string_expression.is_null()
            | stripped_expression.eq("")
            | string_expression.ne(stripped_expression)
        )
        if not invalid_rows.is_empty():
            value = invalid_rows.get_column(component)[0]
            raise PpaError(
                error_message(
                    f"security_id component {component!r} contains a blank, null, "
                    f"or whitespace-padded value {value!r} in {str(source_path)!r} "
                    f"for {dataset_name}."
                ),
                error_code,
            )
    result = frame.with_columns(
        pl.concat_str(
            [string_expressions[component] for component in construction.components],
            separator=construction.separator,
        ).alias(output_column)
    )
    collisions = (
        result.select((*construction.components, output_column))
        .unique()
        .group_by(output_column)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not collisions.is_empty():
        identifier = collisions.get_column(output_column)[0]
        raise PpaError(
            error_message(
                f"Distinct security_id component tuples produce ambiguous "
                f"identifier {identifier!r} in {str(source_path)!r} for "
                f"{dataset_name}. Choose a different separator."
            ),
            error_code,
        )
    return result


def _require_mapping(
    value: object,
    field_path: str,
    error_message: Callable[[str], str],
    error_code: int,
) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise a configuration error."""
    if not isinstance(value, Mapping):
        raise PpaError(error_message(f"{field_path} must be a mapping."), error_code)
    return cast(Mapping[str, object], value)


def _reject_unknown_fields(
    values: Mapping[str, object],
    allowed_fields: set[str],
    field_path: str,
    error_message: Callable[[str], str],
    error_code: int,
) -> None:
    """Raise when a security-ID configuration mapping has unknown fields."""
    unknown_fields = set(values) - allowed_fields
    if unknown_fields:
        raise PpaError(
            error_message(
                f"Unknown fields for {field_path}: "
                f"{sorted(map(str, unknown_fields))}"
            ),
            error_code,
        )


def _validate_components(
    value: object,
    dataset_name: str,
    error_message: Callable[[str], str],
    error_code: int,
) -> tuple[str, ...]:
    """Return validated ordered security-ID component column names."""
    if not isinstance(value, list) or len(value) < 2:
        raise PpaError(
            error_message(
                f"security_id components for {dataset_name} must be a list of "
                "at least two source column names."
            ),
            error_code,
        )
    if any(not isinstance(component, str) or not component for component in value):
        raise PpaError(
            error_message(
                f"security_id components for {dataset_name} must be nonempty strings."
            ),
            error_code,
        )
    components = tuple(value)
    if len(set(components)) != len(components):
        raise PpaError(
            error_message(
                f"security_id components for {dataset_name} must be distinct: "
                f"{list(components)}"
            ),
            error_code,
        )
    return components


def _validate_separator(
    value: object,
    dataset_name: str,
    error_message: Callable[[str], str],
    error_code: int,
) -> str:
    """Return a validated composite security-ID separator."""
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise PpaError(
            error_message(
                f"security_id separator for {dataset_name} must be a single-line "
                "string."
            ),
            error_code,
        )
    return value
