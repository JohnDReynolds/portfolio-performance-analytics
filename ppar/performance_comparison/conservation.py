"""Enforce lossless evidence and single-owner explanation contracts."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
import datetime as dt
import hashlib
import json
import math
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import findings as pc_findings
from ppar.performance_comparison.safety_invariants import DifferenceDisposition

FINDING_SEQUENCE: Final[str] = "finding_sequence"
FINDING_FINGERPRINT: Final[str] = "finding_fingerprint"
SAFETY_DISPOSITION: Final[str] = "safety_disposition"
ECONOMIC_EFFECT_ID: Final[str] = "economic_effect_id"
COUNTED_CAUSE_OWNER: Final[str] = "counted_cause_owner"

_AS_OF_DATE: Final[str] = "as_of_date"
_ESTIMATED_IMPACT: Final[str] = "estimated_impact"
_PORTFOLIO_LEVEL: Final[str] = "portfolio"
_SECURITY_LEVEL: Final[str] = "security"
_RECONSTRUCTION_FORMULA_CODE: Final[str] = "reconstruction_formula_input"


def finding_audit_trail(findings: pl.DataFrame) -> pl.DataFrame:
    """Return the lossless finding table with an explicit visible disposition.

    Source findings are review evidence in the complete audit trail. Counted
    economic effects are represented separately by cause rows because one
    source difference can affect more than one performance period.

    Args:
        findings: Complete comparison findings, including suppressed rows.

    Returns:
        The original rows and columns plus deterministic sequence, fingerprint,
        and disposition columns.
    """
    if findings.is_empty():
        return findings.with_columns(
            pl.Series(FINDING_SEQUENCE, [], dtype=pl.UInt32),
            pl.Series(FINDING_FINGERPRINT, [], dtype=pl.String),
            pl.Series(SAFETY_DISPOSITION, [], dtype=pl.String),
        )

    fingerprint_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for sequence, finding in enumerate(findings.iter_rows(named=True), start=1):
        base_fingerprint = _row_fingerprint(finding, findings.columns)
        occurrence = fingerprint_counts.get(base_fingerprint, 0) + 1
        fingerprint_counts[base_fingerprint] = occurrence
        rows.append(
            {
                **finding,
                FINDING_SEQUENCE: sequence,
                FINDING_FINGERPRINT: f"{base_fingerprint}:{occurrence}",
                SAFETY_DISPOSITION: DifferenceDisposition.REVIEW_EVIDENCE.value,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def assert_complete_finding_audit_trail(
    findings: pl.DataFrame,
    audit_trail: pl.DataFrame,
) -> None:
    """Raise unless every finding remains present with a visible disposition.

    Args:
        findings: Complete comparison findings before disposition metadata.
        audit_trail: Result from :func:`finding_audit_trail`.

    Raises:
        PpaError: If any source finding is missing, changed, duplicated by the
            disposition step, or lacks a permitted visible disposition.
    """
    required_columns = {
        *findings.columns,
        FINDING_SEQUENCE,
        FINDING_FINGERPRINT,
        SAFETY_DISPOSITION,
    }
    if not required_columns.issubset(audit_trail.columns):
        missing = sorted(required_columns - set(audit_trail.columns))
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: complete audit trail "
            f"is missing columns {missing}.",
            999,
        )
    if audit_trail.height != findings.height:
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: complete audit trail "
            f"contains {audit_trail.height} rows for {findings.height} findings.",
            999,
        )
    if not audit_trail.select(findings.columns).equals(findings):
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: finding values or order "
            "changed while assigning dispositions.",
            999,
        )
    expected_sequences = list(range(1, findings.height + 1))
    if audit_trail[FINDING_SEQUENCE].to_list() != expected_sequences:
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: finding sequence is not "
            "complete and contiguous.",
            999,
        )
    fingerprints = audit_trail[FINDING_FINGERPRINT]
    if fingerprints.null_count() or fingerprints.n_unique() != audit_trail.height:
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: finding fingerprints "
            "are missing or duplicated.",
            999,
        )
    permitted = {disposition.value for disposition in DifferenceDisposition}
    actual = set(audit_trail[SAFETY_DISPOSITION].drop_nulls().to_list())
    if audit_trail[SAFETY_DISPOSITION].null_count() or not actual.issubset(permitted):
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: a finding lacks a "
            "permitted visible disposition.",
            999,
        )


def persisted_finding_audit_trail_issues(table: pl.DataFrame) -> list[str]:
    """Return validation issues for a persisted complete finding audit trail.

    Args:
        table: Findings CSV table read from a report bundle.

    Returns:
        Human-readable SN-01 and SN-05 validation issues.
    """
    required_columns = {
        FINDING_SEQUENCE,
        FINDING_FINGERPRINT,
        SAFETY_DISPOSITION,
        pc_findings.SOURCE_RECORD_LOCATOR,
    }
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        return [f"findings audit trail is missing safety columns {missing_columns}"]
    issues: list[str] = []
    locators = table[pc_findings.SOURCE_RECORD_LOCATOR]
    if locators.null_count() or any(
        not str(value).strip() for value in locators.drop_nulls().to_list()
    ):
        issues.append("findings audit trail source-record locators are missing")
    expected_sequences = list(range(1, table.height + 1))
    if table[FINDING_SEQUENCE].to_list() != expected_sequences:
        issues.append("findings audit trail sequence is not complete and contiguous")
    fingerprints = table[FINDING_FINGERPRINT]
    if fingerprints.null_count() or fingerprints.n_unique() != table.height:
        issues.append("findings audit trail fingerprints are missing or duplicated")
    permitted_dispositions = {
        disposition.value for disposition in DifferenceDisposition
    }
    dispositions = table[SAFETY_DISPOSITION]
    if dispositions.null_count() or not set(dispositions.drop_nulls()).issubset(
        permitted_dispositions
    ):
        issues.append("findings audit trail contains an invalid safety disposition")
    return issues


def cause_conservation_table(
    causes: pl.DataFrame,
    *,
    comparison_level: str,
) -> pl.DataFrame:
    """Return cause rows with disposition and economic-effect ownership.

    Args:
        causes: Performance Difference Causes rows.
        comparison_level: ``portfolio`` or ``security`` report grain.

    Returns:
        Cause rows with safety disposition, economic effect, and counted-owner
        metadata appended.

    Raises:
        ValueError: If ``comparison_level`` is unsupported.
    """
    if comparison_level not in {_PORTFOLIO_LEVEL, _SECURITY_LEVEL}:
        raise ValueError(f"Unsupported comparison level: {comparison_level!r}")
    if causes.is_empty():
        return causes.with_columns(
            pl.Series(SAFETY_DISPOSITION, [], dtype=pl.String),
            pl.Series(ECONOMIC_EFFECT_ID, [], dtype=pl.String),
            pl.Series(COUNTED_CAUSE_OWNER, [], dtype=pl.String),
        )

    rows: list[dict[str, object]] = []
    for row in causes.iter_rows(named=True):
        impact = _number_or_none(row.get(_ESTIMATED_IMPACT))
        disposition = (
            DifferenceDisposition.COUNTED_CAUSE
            if impact is not None
            else DifferenceDisposition.REVIEW_EVIDENCE
        )
        effect_id = _economic_effect_id(row, comparison_level=comparison_level)
        rows.append(
            {
                **row,
                SAFETY_DISPOSITION: disposition.value,
                ECONOMIC_EFFECT_ID: effect_id,
                COUNTED_CAUSE_OWNER: (
                    _counted_cause_owner(row) if impact is not None else None
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def assert_cause_conservation(
    original_causes: pl.DataFrame,
    causes: pl.DataFrame,
    *,
    comparison_level: str,
) -> None:
    """Raise unless cause rows are lossless and counted effects have one owner.

    Args:
        original_causes: Cause rows before conservation metadata.
        causes: Cause rows returned by :func:`cause_conservation_table`.
        comparison_level: ``portfolio`` or ``security`` report grain.

    Raises:
        PpaError: If a cause disappears, has an inconsistent disposition, uses
            an ineligible counted representation, or shares ownership.
    """
    if causes.height != original_causes.height or not causes.select(
        original_causes.columns
    ).equals(original_causes):
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: cause rows changed while "
            "assigning dispositions.",
            999,
        )
    _assert_cause_dispositions(causes)
    _assert_counted_representations(causes)
    _assert_counted_period_boundaries(causes)
    _assert_single_effect_owners(causes, comparison_level=comparison_level)


def _assert_cause_dispositions(causes: pl.DataFrame) -> None:
    """Raise unless cause dispositions agree with their explained amounts."""
    for row in causes.iter_rows(named=True):
        impact = _number_or_none(row.get(_ESTIMATED_IMPACT))
        disposition = row.get(SAFETY_DISPOSITION)
        owner = row.get(COUNTED_CAUSE_OWNER)
        effect_id = row.get(ECONOMIC_EFFECT_ID)
        if not isinstance(effect_id, str) or not effect_id:
            raise PpaError(
                "SN-01 no-lost-differences invariant failed: a cause row has no "
                "economic effect identifier.",
                999,
            )
        if impact is None:
            valid = (
                disposition == DifferenceDisposition.REVIEW_EVIDENCE.value
                and owner is None
            )
        else:
            valid = (
                disposition == DifferenceDisposition.COUNTED_CAUSE.value
                and isinstance(owner, str)
                and bool(owner)
            )
        if not valid:
            raise PpaError(
                "SN-01 no-lost-differences invariant failed: cause disposition "
                "does not agree with Performance Difference Explained.",
                999,
            )


def _assert_counted_representations(causes: pl.DataFrame) -> None:
    """Raise when a support-only representation owns an explained amount."""
    counted = causes.filter(
        pl.col(SAFETY_DISPOSITION) == DifferenceDisposition.COUNTED_CAUSE.value
    )
    for row in counted.iter_rows(named=True):
        dataset = row.get(pc_findings.DATASET)
        source_column = row.get(pc_findings.SOURCE_COLUMN)
        code = row.get(pc_findings.FINDING_CODE)
        if code == _RECONSTRUCTION_FORMULA_CODE:
            continue
        if dataset == pc_cols.TRANSACTIONS and source_column not in {
            pc_cols.AMOUNT,
            pc_cols.BASE_AMOUNT,
        }:
            _raise_ineligible_owner(row)
        if dataset in {pc_cols.FX_RATES, pc_cols.SPLITS}:
            _raise_ineligible_owner(row)


def _assert_counted_period_boundaries(causes: pl.DataFrame) -> None:
    """Raise when dated evidence owns impact outside its permitted boundary."""
    counted = causes.filter(
        pl.col(SAFETY_DISPOSITION) == DifferenceDisposition.COUNTED_CAUSE.value
    )
    for row in counted.iter_rows(named=True):
        dataset = row.get(pc_findings.DATASET)
        input_date = _date_or_none(row.get(_AS_OF_DATE))
        from_date = _date_or_none(row.get(pc_findings.FROM_DATE))
        thru_date = _date_or_none(row.get(pc_findings.THRU_DATE))
        if input_date is None or from_date is None or thru_date is None:
            continue
        permitted = from_date <= input_date <= thru_date
        if dataset in {pc_cols.HOLDINGS, pc_cols.FX_RATES}:
            permitted = permitted or input_date == from_date - dt.timedelta(days=1)
        if permitted:
            continue
        raise PpaError(
            "SN-07 period-boundary invariant failed: counted cause "
            f"{dataset}.{row.get(pc_findings.SOURCE_COLUMN)} dated {input_date} "
            f"is outside performance period {from_date}..{thru_date}.",
            999,
        )


def _assert_single_effect_owners(
    causes: pl.DataFrame,
    *,
    comparison_level: str,
) -> None:
    """Raise when one economic effect has more than one counted owner."""
    counted = causes.filter(
        pl.col(SAFETY_DISPOSITION) == DifferenceDisposition.COUNTED_CAUSE.value
    )
    if counted.is_empty():
        return
    duplicates = (
        counted.group_by(ECONOMIC_EFFECT_ID)
        .agg(
            pl.len().alias("owner_count"),
            pl.col(COUNTED_CAUSE_OWNER).alias("owners"),
        )
        .filter(pl.col("owner_count") > 1)
    )
    if duplicates.is_empty():
        return
    duplicate = duplicates.row(0, named=True)
    raise PpaError(
        "SN-02 no-double-counting invariant failed for "
        f"{comparison_level} economic effect {duplicate[ECONOMIC_EFFECT_ID]!r}: "
        f"counted owners={duplicate['owners']!r}.",
        999,
    )


def _raise_ineligible_owner(row: Mapping[str, object]) -> None:
    """Raise for a support-only field carrying a counted explanation."""
    dataset_field = (
        f"{row.get(pc_findings.DATASET)}.{row.get(pc_findings.SOURCE_COLUMN)}"
    )
    raise PpaError(
        "SN-02 no-double-counting invariant failed: support-only field "
        f"{dataset_field} owns Performance Difference Explained.",
        999,
    )


def _economic_effect_id(
    row: Mapping[str, object],
    *,
    comparison_level: str,
) -> str:
    """Return a stable identifier for one economic effect at the report grain."""
    dataset = row.get(pc_findings.DATASET)
    source_column = row.get(pc_findings.SOURCE_COLUMN)
    payload: list[object] = [
        comparison_level,
        row.get(pc_findings.PORTFOLIO_ID),
        row.get(pc_findings.FROM_DATE),
        row.get(pc_findings.THRU_DATE),
        row.get(pc_findings.SECURITY_ID),
        row.get(_AS_OF_DATE),
        dataset,
        _economic_effect_family(dataset, source_column),
    ]
    if dataset == pc_cols.TRANSACTIONS:
        payload.extend(
            (
                row.get(pc_findings.TRANSACTION_CODE),
                row.get(pc_findings.TRANSACTION_CATEGORY),
                row.get(pc_findings.SNAPSHOT_A_VALUE),
                row.get(pc_findings.SNAPSHOT_B_VALUE),
            )
        )
    encoded = json.dumps(
        [_json_value(value) for value in payload],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "effect:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _economic_effect_family(dataset: object, source_column: object) -> str:
    """Return the accounting family that may have only one counted owner."""
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.MARKET_VALUE,
        pc_cols.BASE_MARKET_VALUE,
        pc_cols.QUANTITY,
        pc_cols.PRICE,
    }:
        return "holding_value"
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.BASE_ACCRUED,
    }:
        return "holding_accrued"
    if dataset == pc_cols.TRANSACTIONS:
        return "transaction"
    return f"{dataset}.{source_column}"


def _counted_cause_owner(row: Mapping[str, object]) -> str:
    """Return a concise human-readable owner label for a counted cause."""
    dataset_field = (
        f"{row.get(pc_findings.DATASET)}.{row.get(pc_findings.SOURCE_COLUMN)}"
    )
    security = row.get(pc_findings.SECURITY_ID)
    as_of_date = row.get(_AS_OF_DATE)
    transaction_code = row.get(pc_findings.TRANSACTION_CODE)
    suffix = ":".join(
        str(value)
        for value in (security, as_of_date, transaction_code)
        if value not in {None, ""}
    )
    return dataset_field if not suffix else f"{dataset_field}@{suffix}"


def _row_fingerprint(row: Mapping[str, object], columns: list[str]) -> str:
    """Return a deterministic fingerprint for one complete finding row."""
    payload = {column: _json_value(row.get(column)) for column in columns}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value: object) -> object:
    """Return a deterministic JSON-compatible scalar value."""
    if value is None or isinstance(value, str | bool | int):
        normalized = value
    elif isinstance(value, float):
        if math.isnan(value):
            normalized = "NaN"
        elif math.isinf(value):
            normalized = "Infinity" if value > 0 else "-Infinity"
        else:
            normalized = value
    elif isinstance(value, dt.datetime | dt.date):
        normalized = value.isoformat()
    else:
        normalized = str(value)
    return normalized


def _number_or_none(value: object) -> float | None:
    """Return a finite numeric value or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _date_or_none(value: object) -> dt.date | None:
    """Return a date for Python date/datetime values."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None
