"""Shared helpers for nested source-file YAML configuration."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Mapping
from typing import cast

# Project imports
from ppar.errors import PpaError


def source_file_definition(
    values: Mapping[str, object],
    file_name: str,
    error_message: Callable[[str], str],
    *,
    error_code: int = 504,
) -> Mapping[str, object]:
    """Return one nested ``files`` definition or an empty mapping.

    Args:
        values: Parsed YAML root mapping.
        file_name: Dataset key inside ``files``.
        error_message: Callback adding product-specific error context.
        error_code: PPAR error code for malformed configuration.

    Returns:
        The configured file definition, or an empty mapping when omitted.

    Raises:
        PpaError: If ``files`` or the requested definition is not a mapping.
    """
    raw_files = values.get("files", {})
    if not isinstance(raw_files, Mapping):
        raise PpaError(error_message("files must be a mapping."), error_code)
    raw_definition = raw_files.get(file_name, {})
    if isinstance(raw_definition, str):
        return {"path": raw_definition}
    if not isinstance(raw_definition, Mapping):
        raise PpaError(
            error_message(f"files.{file_name} must be a string or mapping."),
            error_code,
        )
    return cast(Mapping[str, object], raw_definition)


def source_file_columns(
    values: Mapping[str, object],
    file_name: str,
    error_message: Callable[[str], str],
    *,
    error_code: int = 504,
) -> Mapping[str, object]:
    """Return one dataset's nested source-column mappings.

    Args:
        values: Parsed YAML root mapping.
        file_name: Dataset key inside ``files``.
        error_message: Callback adding product-specific error context.
        error_code: PPAR error code for malformed configuration.

    Returns:
        The configured ``columns`` mapping, or an empty mapping when omitted.

    Raises:
        PpaError: If the file definition or ``columns`` value is malformed.
    """
    definition = source_file_definition(
        values,
        file_name,
        error_message,
        error_code=error_code,
    )
    raw_columns = definition.get("columns", {})
    if not isinstance(raw_columns, Mapping):
        raise PpaError(
            error_message(f"files.{file_name}.columns must be a mapping."),
            error_code,
        )
    return cast(Mapping[str, object], raw_columns)
