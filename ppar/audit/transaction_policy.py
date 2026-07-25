"""Load executable transaction policy from the packaged YAML contract."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Final

import yaml

from ppar.errors import PpaError

_POLICY_PACKAGE: Final[str] = "ppar.setup_templates.axys_apx_audit"
_POLICY_FILE_NAME: Final[str] = "transaction_semantics_policy.yaml"
_BOUNDARY_GROUPS_KEY: Final[str] = "boundary_groups"


def transaction_boundary_registry() -> Mapping[str, frozenset[str]]:
    """Return transaction boundary groups from the packaged YAML policy."""
    return _loaded_policy()


def transaction_boundary_codes(group: str) -> frozenset[str]:
    """Return configured codes for one boundary group.

    Args:
        group: Boundary-group name from the packaged policy.

    Returns:
        Lowercase transaction codes assigned to the group.

    Raises:
        PpaError: If the requested boundary group is absent.
    """
    registry = transaction_boundary_registry()
    if group not in registry:
        raise PpaError(
            f"{_POLICY_FILE_NAME}: missing transaction boundary group {group!r}.",
            999,
        )
    return registry[group]


def transaction_code_matching_key(
    value: object,
) -> str:
    """Return a stripped native-case transaction-code comparison key.

    Args:
        value: Native source transaction code.

    Returns:
        Stripped native-case code. Missing values return an empty string.
    """
    if value is None:
        return ""
    return str(value).strip()


@cache
def _loaded_policy() -> Mapping[str, frozenset[str]]:
    """Load and validate the immutable packaged transaction policy."""
    resource = files(_POLICY_PACKAGE).joinpath(_POLICY_FILE_NAME)
    try:
        values = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PpaError(
            f"Unable to load packaged transaction policy {_POLICY_FILE_NAME}: {error}",
            999,
        ) from error
    if not isinstance(values, dict):
        raise PpaError(f"{_POLICY_FILE_NAME}: root must be a mapping.", 999)
    if values.get("schema_version") != 1:
        raise PpaError(f"{_POLICY_FILE_NAME}: schema_version must be 1.", 999)

    groups = _validated_boundary_groups(values.get(_BOUNDARY_GROUPS_KEY))
    return MappingProxyType(groups)


def _validated_boundary_groups(value: object) -> dict[str, frozenset[str]]:
    """Return validated boundary groups."""
    if not isinstance(value, dict) or not value:
        raise PpaError(
            f"{_POLICY_FILE_NAME}: {_BOUNDARY_GROUPS_KEY} must be a nonempty mapping.",
            999,
        )
    groups: dict[str, frozenset[str]] = {}
    for raw_group, raw_codes in value.items():
        if not isinstance(raw_group, str) or not raw_group.strip():
            raise PpaError(
                f"{_POLICY_FILE_NAME}: boundary group names must be nonblank strings.",
                999,
            )
        if not isinstance(raw_codes, list) or not raw_codes:
            raise PpaError(
                f"{_POLICY_FILE_NAME}: boundary group {raw_group!r} must be a nonempty list.",
                999,
            )
        if any(not isinstance(code, str) or not code.strip() for code in raw_codes):
            raise PpaError(
                f"{_POLICY_FILE_NAME}: boundary codes must be nonblank strings.",
                999,
            )
        groups[raw_group.strip()] = frozenset(
            code.strip().lower() for code in raw_codes
        )
    return groups
