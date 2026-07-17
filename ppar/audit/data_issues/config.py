"""Strict validation for Data Issues configuration."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
import math
from typing import Final

# Project imports
from ppar.audit.data_issues.vocabulary import (
    DATA_ISSUE_REGISTRY,
    DataIssueDefinition,
    DataIssueType,
)

DATA_ISSUES_CONFIG_KEY: Final[str] = "data_issues"
_RETIRED_DATA_AUDIT_CONFIG_KEY: Final[str] = "data_audit_checks"

_ENABLED_KEY: Final[str] = "enabled"
_ONLY_KEY: Final[str] = "only"
_EXCLUDE_KEY: Final[str] = "exclude"
_ABSOLUTE_TOLERANCE_KEY: Final[str] = "absolute_tolerance"
_PERCENT_TOLERANCE_KEY: Final[str] = "percent_tolerance"
_OPTIONAL_CHECK_KEYS: Final[frozenset[str]] = frozenset(
    {_ENABLED_KEY, _ONLY_KEY, _EXCLUDE_KEY}
)
_FILTER_FIELD_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "snapshot",
        "portfolio",
        "portfolio_id",
        "security",
        "security_id",
        "security_type",
        "asset_class",
        "transaction_code",
    }
)


def validate_data_issues_config(values: Mapping[str, object]) -> None:
    """Validate the complete strict ``data_issues`` YAML contract.

    Args:
        values: Parsed comparison YAML root mapping.

    Raises:
        ValueError: If the section, a check, or a setting has an invalid shape,
            unsupported key, or unsafe value. Messages include the actionable
            YAML path.
    """
    if _RETIRED_DATA_AUDIT_CONFIG_KEY in values:
        raise ValueError(
            f"{_RETIRED_DATA_AUDIT_CONFIG_KEY} is no longer supported; use "
            f"{DATA_ISSUES_CONFIG_KEY} instead."
        )

    config_key, raw_config = _configured_section(values)
    if config_key is None:
        return
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{config_key} must be a mapping.")

    supported_issue_names = frozenset(issue_type.value for issue_type in DataIssueType)
    unsupported_keys = sorted(
        str(key)
        for key in raw_config
        if key != _ENABLED_KEY and key not in supported_issue_names
    )
    if unsupported_keys:
        raise ValueError(
            f"{config_key} has unknown issue types or unsupported keys: "
            f"{', '.join(unsupported_keys)}."
        )

    if _ENABLED_KEY in raw_config:
        _validate_boolean(raw_config[_ENABLED_KEY], f"{config_key}.enabled")

    for issue_type in DataIssueType:
        if issue_type.value not in raw_config:
            continue
        _validate_check(
            issue_type,
            DATA_ISSUE_REGISTRY[issue_type],
            raw_config[issue_type.value],
            config_key=config_key,
        )


def data_issues_config_summary(values: Mapping[str, object]) -> dict[str, object]:
    """Return effective Data Issues enablement and mandatory-check policy.

    Args:
        values: Parsed and validated comparison YAML root mapping.

    Returns:
        Stable summary fields for ``validate_config`` output.

    Raises:
        ValueError: If the Data Issues configuration is invalid.
    """
    validate_data_issues_config(values)
    _, raw_config = _configured_section(values)
    raw_config = {} if raw_config is None else raw_config
    config = raw_config if isinstance(raw_config, Mapping) else {}
    optional_master_enabled = config.get(_ENABLED_KEY, True)

    optional_checks = [
        issue_type.value
        for issue_type, definition in DATA_ISSUE_REGISTRY.items()
        if not definition.mandatory
        and optional_master_enabled is True
        and _effective_check_enabled(config, issue_type, definition)
    ]
    mandatory_checks = [
        issue_type.value
        for issue_type, definition in DATA_ISSUE_REGISTRY.items()
        if definition.mandatory
    ]
    return {
        "optional_master_enabled": optional_master_enabled,
        "optional_checks_enabled": ", ".join(optional_checks) or "none",
        "mandatory_checks": ", ".join(mandatory_checks),
        "policy": (
            "mandatory continuity checks remain active; optional checks are "
            + ("enabled by default" if optional_master_enabled else "disabled")
        ),
    }


def _validate_check(
    issue_type: DataIssueType,
    definition: DataIssueDefinition,
    raw_check: object,
    *,
    config_key: str,
) -> None:
    """Validate one issue-type configuration mapping."""
    path = f"{config_key}.{issue_type.value}"
    if not isinstance(raw_check, Mapping):
        raise ValueError(f"{path} must be a mapping.")

    supported_keys = _supported_check_keys(definition)
    unsupported_keys = sorted(str(key) for key in raw_check if key not in supported_keys)
    if unsupported_keys:
        raise ValueError(f"{path} has unsupported keys: {', '.join(unsupported_keys)}.")

    if _ENABLED_KEY in raw_check:
        _validate_boolean(raw_check[_ENABLED_KEY], f"{path}.enabled")
    for filter_key in (_ONLY_KEY, _EXCLUDE_KEY):
        if filter_key in raw_check:
            _validate_filter(raw_check[filter_key], f"{path}.{filter_key}")
    for tolerance_key in (_ABSOLUTE_TOLERANCE_KEY, _PERCENT_TOLERANCE_KEY):
        if tolerance_key in raw_check:
            _validate_tolerance(raw_check[tolerance_key], f"{path}.{tolerance_key}")


def _supported_check_keys(definition: DataIssueDefinition) -> frozenset[str]:
    """Return YAML keys supported by one registry definition."""
    keys: set[str] = set() if definition.mandatory else set(_OPTIONAL_CHECK_KEYS)
    if definition.supports_absolute_tolerance:
        keys.add(_ABSOLUTE_TOLERANCE_KEY)
    if definition.supports_percent_tolerance:
        keys.add(_PERCENT_TOLERANCE_KEY)
    return frozenset(keys)


def _validate_boolean(value: object, path: str) -> None:
    """Require an actual YAML Boolean value."""
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a Boolean.")


def _validate_tolerance(value: object, path: str) -> None:
    """Require a finite nonnegative, non-Boolean numeric tolerance."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{path} must be a finite nonnegative number.")


def _validate_filter(value: object, path: str) -> None:
    """Validate a supported exact-match filter mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    for field_name, raw_values in value.items():
        field_path = f"{path}.{field_name}"
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"{path} field names must be nonempty strings.")
        normalized_name = field_name.strip().lower().rsplit(".", maxsplit=1)[-1]
        if normalized_name not in _FILTER_FIELD_ALIASES:
            raise ValueError(f"{field_path} is not a supported filter field.")
        if isinstance(raw_values, list):
            if not raw_values:
                raise ValueError(f"{field_path} must contain at least one scalar value.")
            for index, item in enumerate(raw_values):
                _validate_filter_scalar(item, f"{field_path}[{index}]")
        else:
            _validate_filter_scalar(raw_values, field_path)


def _validate_filter_scalar(value: object, path: str) -> None:
    """Require one nonempty scalar suitable for exact string matching."""
    if isinstance(value, (Mapping, list, tuple, set)) or value is None:
        raise ValueError(f"{path} must be a scalar value.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be a finite scalar value.")
    if not str(value).strip():
        raise ValueError(f"{path} must be a nonempty scalar value.")


def _effective_check_enabled(
    config: Mapping[object, object],
    issue_type: DataIssueType,
    definition: DataIssueDefinition,
) -> bool:
    """Return one optional check's effective validated enablement."""
    raw_check = config.get(issue_type.value, {})
    check = raw_check if isinstance(raw_check, Mapping) else {}
    enabled = check.get(_ENABLED_KEY, definition.default_enabled)
    return enabled is True


def _configured_section(
    values: Mapping[str, object],
) -> tuple[str | None, object | None]:
    """Return the configured Data Issues key and value."""
    if DATA_ISSUES_CONFIG_KEY in values:
        return DATA_ISSUES_CONFIG_KEY, values[DATA_ISSUES_CONFIG_KEY]
    return None, None
