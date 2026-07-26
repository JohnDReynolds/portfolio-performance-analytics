"""Summarize transaction semantics for validation and report handoff artifacts."""

from __future__ import annotations

# Python imports
from collections.abc import Iterable, Mapping

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import schema as _pc_cols
from ppar.transaction_codes import transaction_code_matching_key


def transaction_semantics_summary(
    frames: Iterable[pl.DataFrame],
    *,
    rule_codes: set[str] | None = None,
) -> dict[str, object]:
    """Return structured transaction semantics summary metadata.

    Args:
        frames: Transaction-like DataFrames with normalized transaction columns.
        rule_codes: Optional normalized transaction-code rules configured in
            YAML.
    Returns:
        JSON-serializable transaction summary fields.
    """
    observed_codes: set[str] = set()
    semantics_source_counts: dict[str, int] = {}
    unknown_category_count = 0
    for frame in frames:
        observed_codes.update(native_transaction_codes(frame))
        unknown_category_count += _column_value_count(
            frame,
            _pc_cols.TRANSACTION_CATEGORY,
            "unknown",
        )
        semantics_source_counts.update(
            _combined_counts(
                semantics_source_counts,
                _column_counts(frame, _pc_cols.TRANSACTION_SEMANTICS_SOURCE),
            )
        )
    codes_without_rules = (
        {
            code
            for code in observed_codes
            if transaction_code_matching_key(code) not in rule_codes
        }
        if rule_codes is not None
        else set()
    )
    return {
        "observed_codes": sorted(observed_codes),
        "codes_without_yaml_rules": sorted(codes_without_rules),
        "unknown_category_count": unknown_category_count,
        "semantics_source_counts": dict(sorted(semantics_source_counts.items())),
        "ambiguous_context_blocked_count": 0,
    }


def transaction_codes(frame: pl.DataFrame) -> set[str]:
    """Return native-case transaction codes in a transaction-like frame."""
    if _pc_cols.TRANSACTION_CODE not in frame.columns:
        return set()
    codes = set()
    for value in frame.get_column(_pc_cols.TRANSACTION_CODE):
        code = native_transaction_code(value)
        if code:
            codes.add(code)
    return codes


def native_transaction_codes(frame: pl.DataFrame) -> set[str]:
    """Return native transaction-code strings observed in a transaction-like frame."""
    if _pc_cols.TRANSACTION_CODE not in frame.columns:
        return set()
    codes = set()
    for value in frame.get_column(_pc_cols.TRANSACTION_CODE):
        code = native_transaction_code(value)
        if code:
            codes.add(code)
    return codes


def transaction_rule_codes(
    values: Mapping[str, object],
) -> set[str]:
    """Return native-case transaction code keys configured in YAML rules."""
    rules_value = values.get("transaction_rules", {})
    if not isinstance(rules_value, dict):
        return set()
    return {
        code
        for raw_code in rules_value
        if (code := transaction_code_matching_key(raw_code))
    }


def native_transaction_code(value: object) -> str:
    """Return a stripped native transaction code, or blank for missing values."""
    if value is None:
        return ""
    code = str(value).strip()
    return code if code else ""


def format_codes(codes: object) -> str:
    """Return a stable, readable transaction code list."""
    if not isinstance(codes, list):
        return "none"
    text_codes = [str(code) for code in codes if str(code).strip()]
    return ", ".join(sorted(text_codes)) if text_codes else "none"


def format_semantics_source_counts(counts: object) -> str:
    """Return readable transaction semantics-source counts."""
    if not isinstance(counts, dict) or not counts:
        return "none"
    labels = {
        "source": "source",
        "yaml_rule": "YAML rule",
        "mixed": "mixed",
        "unknown": "unknown",
    }
    parts = [
        f"{labels.get(str(key), str(key))}: {value}"
        for key, value in sorted(counts.items())
    ]
    return ", ".join(parts)


def _column_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    """Return nonblank string counts for one DataFrame column."""
    if column not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for value in frame.get_column(column):
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _column_value_count(frame: pl.DataFrame, column: str, expected: str) -> int:
    """Return how many rows have a normalized string value in one column."""
    if column not in frame.columns:
        return 0
    count = 0
    for value in frame.get_column(column):
        if str(value).strip().lower() == expected:
            count += 1
    return count


def _combined_counts(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> dict[str, int]:
    """Return combined integer counts from two mappings."""
    combined = dict(left)
    for key, value in right.items():
        combined[key] = combined.get(key, 0) + value
    return combined
