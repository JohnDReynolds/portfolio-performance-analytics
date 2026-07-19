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
_DEFAULT_RULES_KEY: Final[str] = "default_transaction_rules"
_BOUNDARY_GROUPS_KEY: Final[str] = "boundary_groups"
_DEFAULT_RULE_VALUES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "transaction_category": frozenset(
            {
                "buy",
                "corporate_action",
                "external_flow",
                "fee_expense",
                "income",
                "sell",
                "transfer",
            }
        ),
        "cash_flow_sign": frozenset({"negative", "none", "positive"}),
        "performance_flow_sign": frozenset(
            {"external", "neutral", "performance"}
        ),
    }
)


def default_transaction_rules() -> Mapping[str, Mapping[str, str]]:
    """Return packaged compatibility rules keyed by normalized source code."""
    return _loaded_policy()[0]


def transaction_boundary_registry() -> Mapping[str, frozenset[str]]:
    """Return transaction boundary groups from the packaged YAML policy."""
    return _loaded_policy()[1]


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
    *,
    exact_case: bool = False,
) -> str:
    """Return a transaction-code comparison key without inventing semantics.

    Args:
        value: Native source transaction code.
        exact_case: Preserve source case when ``True``; otherwise retain the
            legacy lowercase comparison behavior.

    Returns:
        Stripped exact-case or legacy lowercase code. Missing values return an
        empty string.
    """
    if value is None:
        return ""
    code = str(value).strip()
    return code if exact_case else code.lower()


@cache
def _loaded_policy() -> tuple[
    Mapping[str, Mapping[str, str]],
    Mapping[str, frozenset[str]],
]:
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

    rules = _validated_default_rules(values.get(_DEFAULT_RULES_KEY))
    groups = _validated_boundary_groups(values.get(_BOUNDARY_GROUPS_KEY))
    return MappingProxyType(rules), MappingProxyType(groups)


def _validated_default_rules(value: object) -> dict[str, Mapping[str, str]]:
    """Return validated default rule mappings."""
    if not isinstance(value, dict) or not value:
        raise PpaError(
            f"{_POLICY_FILE_NAME}: {_DEFAULT_RULES_KEY} must be a nonempty mapping.",
            999,
        )
    rules: dict[str, Mapping[str, str]] = {}
    for raw_code, raw_rule in value.items():
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise PpaError(
                f"{_POLICY_FILE_NAME}: default rule keys must be nonblank strings.",
                999,
            )
        if not isinstance(raw_rule, dict):
            raise PpaError(
                f"{_POLICY_FILE_NAME}: default rule {raw_code!r} must be a mapping.",
                999,
            )
        normalized_rule: dict[str, str] = {}
        for key, raw_value in raw_rule.items():
            if not isinstance(key, str) or not isinstance(raw_value, str):
                raise PpaError(
                    f"{_POLICY_FILE_NAME}: default rule values must be strings.",
                    999,
                )
            if key not in _DEFAULT_RULE_VALUES:
                raise PpaError(
                    f"{_POLICY_FILE_NAME}: default rule {raw_code!r} has "
                    f"unsupported key {key!r}.",
                    999,
                )
            if raw_value not in _DEFAULT_RULE_VALUES[key]:
                raise PpaError(
                    f"{_POLICY_FILE_NAME}: default rule {raw_code!r}.{key} "
                    f"has unsupported value {raw_value!r}.",
                    999,
                )
            normalized_rule[key] = raw_value
        rules[raw_code.strip().upper()] = MappingProxyType(normalized_rule)
    return rules


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
