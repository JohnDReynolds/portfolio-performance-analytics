"""Provide stable source-record identity and bidirectional report lineage."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import datetime as dt
from enum import Enum
import hashlib
import json
import math
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import conservation as pc_conservation
from ppar.performance_comparison import findings as pc_findings

SOURCE_LINEAGE_TYPE: Final[str] = "source_lineage_type"
SOURCE_FINDING_FINGERPRINTS: Final[str] = "source_finding_fingerprints"

SOURCE_FINDING_LINEAGE: Final[str] = "source_finding"
DERIVED_FORMULA_LINEAGE: Final[str] = "derived_formula"
REPORT_DISPOSITION_LINEAGE: Final[str] = "report_disposition"

_RECONSTRUCTION_FORMULA_CODE: Final[str] = "reconstruction_formula_input"
_NO_UNDERLYING_CAUSE_DATASET: Final[str] = "no_underlying_causes_found"
_RECONSTRUCTION_COMPONENTS: Final[str] = "_workbook_reconstruction_components"
_LINEAGE_ROW_INDEX: Final[str] = "__ppar_lineage_row_index"
_DERIVED_LOCATOR: Final[str] = "__ppar_derived_locator"
_FINGERPRINT_LOCATOR: Final[str] = "__ppar_fingerprint_locator"


def source_record_locator(
    dataset: str,
    source_file: str | None,
    row: Mapping[str, object],
    key_columns: Sequence[str],
    *,
    qualifier: str | None = None,
) -> str:
    """Return a stable locator for one normalized logical source record.

    Args:
        dataset: Normalized dataset name.
        source_file: Configured source file, when available.
        row: Normalized row or key-value mapping.
        key_columns: Columns defining the logical comparison key.
        qualifier: Optional discriminator for a derived grouping such as an
            ambiguous duplicate-key diagnostic.

    Returns:
        A compact locator stable across physical source-file row reordering.

    Notes:
        The locator deliberately hashes normalized logical keys, not values
        being compared. Snapshot A and B versions of the same source record
        therefore share a locator.
    """
    payload = {
        "dataset": dataset,
        "source_file": source_file,
        "key": [[column, _canonical_value(row.get(column))] for column in key_columns],
        "qualifier": qualifier,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"source:{dataset}:{digest}"


def assert_finding_source_lineage(findings: pl.DataFrame) -> None:
    """Raise unless every source finding has a stable record locator.

    Args:
        findings: Complete comparison findings, including suppressed rows.

    Raises:
        PpaError: If the locator column is absent, blank, or inconsistent for
            the same dataset/file/logical record representation.
    """
    locator_column = pc_findings.SOURCE_RECORD_LOCATOR
    if locator_column not in findings.columns:
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: findings do not "
            "include source-record locators.",
            999,
        )
    if findings.is_empty():
        return
    invalid = findings.filter(
        pl.col(locator_column).is_null()
        | (pl.col(locator_column).cast(pl.String).str.strip_chars() == "")
    )
    if not invalid.is_empty():
        row = invalid.row(0, named=True)
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: finding "
            f"{row.get(pc_findings.FINDING_CODE)!r} for "
            f"{row.get(pc_findings.DATASET)!r} has no source-record locator.",
            999,
        )


def cause_lineage_table(
    causes: pl.DataFrame,
    findings: pl.DataFrame,
    *,
    finding_audit_trail: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return cause rows with explicit backward lineage to source findings.

    Args:
        causes: Internal Performance Difference Causes table.
        findings: Active source findings used to build the cause table.
        finding_audit_trail: Optional precomputed complete finding audit trail.

    Returns:
        Cause rows with lineage type, stable locator, and source-finding
        fingerprints. Formula and no-cause disposition rows receive explicit
        derived lineage instead of pretending to be source records.

    Raises:
        PpaError: If a source-backed cause cannot be traced to a finding.
    """
    audit_trail = (
        pc_conservation.finding_audit_trail(findings)
        if finding_audit_trail is None
        else finding_audit_trail
    )
    if causes.is_empty():
        result = causes.with_columns(
            pl.Series(SOURCE_LINEAGE_TYPE, [], dtype=pl.String),
            pl.Series(SOURCE_FINDING_FINGERPRINTS, [], dtype=pl.String),
        )
    else:
        result = _cause_lineage_rows(causes, audit_trail)
    assert_bidirectional_report_lineage(
        findings,
        result,
        finding_audit_trail=audit_trail,
    )
    return result


def _cause_lineage_rows(
    causes: pl.DataFrame,
    audit_trail: pl.DataFrame,
) -> pl.DataFrame:
    """Add cause lineage in bulk while preserving exact cryptographic locators."""
    code = _optional_column(causes, pc_findings.FINDING_CODE)
    dataset = _optional_column(causes, pc_findings.DATASET)
    lineage_type = (
        pl.when(code == _RECONSTRUCTION_FORMULA_CODE)
        .then(pl.lit(DERIVED_FORMULA_LINEAGE))
        .when(dataset == _NO_UNDERLYING_CAUSE_DATASET)
        .then(pl.lit(REPORT_DISPOSITION_LINEAGE))
        .otherwise(pl.lit(SOURCE_FINDING_LINEAGE))
    )
    working = causes.with_row_index(_LINEAGE_ROW_INDEX)
    if pc_findings.SOURCE_RECORD_LOCATOR not in working.columns:
        working = working.with_columns(
            pl.lit(None, dtype=pl.String).alias(
                pc_findings.SOURCE_RECORD_LOCATOR
            )
        )
    working = working.with_columns(lineage_type.alias(SOURCE_LINEAGE_TYPE))

    invalid_formula = working.filter(
        (pl.col(SOURCE_LINEAGE_TYPE) == DERIVED_FORMULA_LINEAGE)
        & _missing_reconstruction_components_expr(working)
    )
    if not invalid_formula.is_empty():
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: a derived "
            "formula cause has no reconstruction component provenance.",
            999,
        )

    source_causes = working.filter(
        pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE
    )
    invalid_locator = _invalid_source_locator_rows(source_causes)
    if not invalid_locator.is_empty():
        row = invalid_locator.row(0, named=True)
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: a report "
            f"cause for {row.get(pc_findings.DATASET)}."
            f"{row.get(pc_findings.SOURCE_COLUMN)} has no source-record locator.",
            999,
        )

    fingerprint_map = _fingerprints_by_locator(audit_trail)
    working = working.join(
        fingerprint_map,
        left_on=pc_findings.SOURCE_RECORD_LOCATOR,
        right_on=_FINGERPRINT_LOCATOR,
        how="left",
        maintain_order="left",
    )
    untraceable = working.filter(
        (pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE)
        & pl.col(SOURCE_FINDING_FINGERPRINTS).is_null()
    )
    if not untraceable.is_empty():
        locator = untraceable[pc_findings.SOURCE_RECORD_LOCATOR][0]
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: report cause "
            f"locator {locator!r} does not trace back to a source finding.",
            999,
        )

    derived_rows = working.filter(
        pl.col(SOURCE_LINEAGE_TYPE) != SOURCE_FINDING_LINEAGE
    )
    derived_locator_map = pl.DataFrame(
        {
            _LINEAGE_ROW_INDEX: derived_rows[_LINEAGE_ROW_INDEX],
            _DERIVED_LOCATOR: [
                _derived_record_locator(row, str(row[SOURCE_LINEAGE_TYPE]))
                for row in derived_rows.iter_rows(named=True)
            ],
        },
        schema={_LINEAGE_ROW_INDEX: pl.UInt32, _DERIVED_LOCATOR: pl.String},
    )
    working = working.join(
        derived_locator_map,
        on=_LINEAGE_ROW_INDEX,
        how="left",
        maintain_order="left",
    ).with_columns(
        pl.when(pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE)
        .then(pl.col(pc_findings.SOURCE_RECORD_LOCATOR))
        .otherwise(pl.col(_DERIVED_LOCATOR))
        .alias(pc_findings.SOURCE_RECORD_LOCATOR),
        pl.when(pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE)
        .then(pl.col(SOURCE_FINDING_FINGERPRINTS))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(SOURCE_FINDING_FINGERPRINTS),
    )
    output_columns = list(causes.columns)
    if pc_findings.SOURCE_RECORD_LOCATOR not in output_columns:
        output_columns.append(pc_findings.SOURCE_RECORD_LOCATOR)
    for column_name in (SOURCE_LINEAGE_TYPE, SOURCE_FINDING_FINGERPRINTS):
        if column_name not in output_columns:
            output_columns.append(column_name)
    return working.select(output_columns)


def _fingerprints_by_locator(audit_trail: pl.DataFrame) -> pl.DataFrame:
    """Return sorted unique finding fingerprints for each valid locator."""
    locator = pc_findings.SOURCE_RECORD_LOCATOR
    fingerprint = pc_conservation.FINDING_FINGERPRINT
    if audit_trail.is_empty() or not {locator, fingerprint}.issubset(
        audit_trail.columns
    ):
        return pl.DataFrame(
            schema={
                _FINGERPRINT_LOCATOR: pl.String,
                SOURCE_FINDING_FINGERPRINTS: pl.String,
            }
        )
    return (
        audit_trail.select(locator, fingerprint)
        .filter(
            pl.col(locator).is_not_null()
            & pl.col(fingerprint).is_not_null()
        )
        .group_by(locator, maintain_order=True)
        .agg(
            pl.col(fingerprint)
            .unique()
            .sort()
            .str.join("|")
            .alias(SOURCE_FINDING_FINGERPRINTS)
        )
        .rename({locator: _FINGERPRINT_LOCATOR})
    )


def _invalid_source_locator_rows(source_causes: pl.DataFrame) -> pl.DataFrame:
    """Return source-backed rows whose locator is not a nonempty string."""
    locator = pc_findings.SOURCE_RECORD_LOCATOR
    if source_causes.is_empty():
        return source_causes
    if source_causes.schema.get(locator) != pl.String:
        return source_causes
    return source_causes.filter(pl.col(locator).is_null() | (pl.col(locator) == ""))


def _missing_reconstruction_components_expr(causes: pl.DataFrame) -> pl.Expr:
    """Return the existing falsey-component test as a Polars expression."""
    dtype = causes.schema.get(_RECONSTRUCTION_COMPONENTS)
    if dtype is None:
        return pl.lit(True)
    if dtype == pl.String:
        return pl.col(_RECONSTRUCTION_COMPONENTS).is_null() | (
            pl.col(_RECONSTRUCTION_COMPONENTS) == ""
        )
    if dtype == pl.Boolean:
        return pl.col(_RECONSTRUCTION_COMPONENTS).is_null() | ~pl.col(
            _RECONSTRUCTION_COMPONENTS
        )
    if dtype.is_numeric():
        return pl.col(_RECONSTRUCTION_COMPONENTS).is_null() | (
            pl.col(_RECONSTRUCTION_COMPONENTS) == 0
        )
    return pl.col(_RECONSTRUCTION_COMPONENTS).is_null()


def _optional_column(table: pl.DataFrame, column_name: str) -> pl.Expr:
    """Return an optional column expression with absent values as null."""
    return (
        pl.col(column_name)
        if column_name in table.columns
        else pl.lit(None)
    )


def assert_bidirectional_report_lineage(
    findings: pl.DataFrame,
    causes: pl.DataFrame,
    *,
    finding_audit_trail: pl.DataFrame | None = None,
) -> None:
    """Raise unless source findings and report causes retain valid lineage.

    Args:
        findings: Active source findings used by the report.
        causes: Cause table returned by :func:`cause_lineage_table`.
        finding_audit_trail: Optional precomputed complete finding audit trail.

    Raises:
        PpaError: If source-record identity is missing or a source-backed cause
            lacks backward finding fingerprints.
    """
    assert_finding_source_lineage(findings)
    finding_audit = (
        pc_conservation.finding_audit_trail(findings)
        if finding_audit_trail is None
        else finding_audit_trail
    )
    pc_conservation.assert_complete_finding_audit_trail(findings, finding_audit)
    available_fingerprints = set(
        finding_audit[pc_conservation.FINDING_FINGERPRINT].drop_nulls().to_list()
    )
    required = {
        pc_findings.SOURCE_RECORD_LOCATOR,
        SOURCE_LINEAGE_TYPE,
        SOURCE_FINDING_FINGERPRINTS,
    }
    if not required.issubset(causes.columns):
        raise PpaError(
            "SN-05 bidirectional-lineage invariant failed: cause table lacks "
            f"lineage columns {sorted(required - set(causes.columns))}.",
            999,
        )
    for cause in causes.iter_rows(named=True):
        lineage_type = cause.get(SOURCE_LINEAGE_TYPE)
        locator = cause.get(pc_findings.SOURCE_RECORD_LOCATOR)
        fingerprints = cause.get(SOURCE_FINDING_FINGERPRINTS)
        if not isinstance(locator, str) or not locator:
            raise PpaError(
                "SN-05 bidirectional-lineage invariant failed: a report row "
                "has no stable lineage locator.",
                999,
            )
        if lineage_type == SOURCE_FINDING_LINEAGE:
            if not isinstance(fingerprints, str) or not fingerprints:
                raise PpaError(
                    "SN-05 bidirectional-lineage invariant failed: a "
                    "source-backed cause lacks finding fingerprints.",
                    999,
                )
            linked_fingerprints = set(fingerprints.split("|"))
            if not linked_fingerprints.issubset(available_fingerprints):
                raise PpaError(
                    "SN-05 bidirectional-lineage invariant failed: a cause "
                    "references a finding fingerprint outside the audit trail.",
                    999,
                )
        elif lineage_type not in {
            DERIVED_FORMULA_LINEAGE,
            REPORT_DISPOSITION_LINEAGE,
        }:
            raise PpaError(
                "SN-05 bidirectional-lineage invariant failed: report row has "
                f"unknown lineage type {lineage_type!r}.",
                999,
            )


def persisted_cause_lineage_issues(table: pl.DataFrame) -> list[str]:
    """Return SN-05 validation issues for persisted cause lineage.

    Args:
        table: Cause-lineage CSV table read from a report bundle.

    Returns:
        Human-readable validation issues. An empty list means the persisted
        lineage metadata satisfies the structural contract.
    """
    required_columns = {
        pc_findings.SOURCE_RECORD_LOCATOR,
        SOURCE_LINEAGE_TYPE,
        SOURCE_FINDING_FINGERPRINTS,
        pc_conservation.SAFETY_DISPOSITION,
        pc_conservation.ECONOMIC_EFFECT_ID,
    }
    missing = sorted(required_columns - set(table.columns))
    if missing:
        return [f"cause lineage is missing safety columns {missing}"]
    issues: list[str] = []
    locators = table[pc_findings.SOURCE_RECORD_LOCATOR]
    if _contains_blank(locators):
        issues.append("cause lineage contains a missing source-record locator")
    permitted_types = {
        SOURCE_FINDING_LINEAGE,
        DERIVED_FORMULA_LINEAGE,
        REPORT_DISPOSITION_LINEAGE,
    }
    actual_types = table[SOURCE_LINEAGE_TYPE]
    if actual_types.null_count() or not set(actual_types.drop_nulls()).issubset(
        permitted_types
    ):
        issues.append("cause lineage contains an invalid lineage type")
    source_fingerprints = table.filter(
        pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE
    )[SOURCE_FINDING_FINGERPRINTS]
    if _contains_blank(source_fingerprints):
        issues.append("cause lineage contains an untraceable source-backed row")
    return issues


def persisted_cross_artifact_lineage_issues(
    finding_audit: pl.DataFrame,
    cause_lineage: pl.DataFrame,
) -> list[str]:
    """Return issues linking persisted causes to persisted findings.

    Args:
        finding_audit: Complete persisted findings table.
        cause_lineage: Persisted internal cause-lineage table.

    Returns:
        Human-readable issues for unknown source locators or fingerprints.
    """
    finding_columns = {
        pc_findings.SOURCE_RECORD_LOCATOR,
        pc_conservation.FINDING_FINGERPRINT,
    }
    cause_columns = {
        pc_findings.SOURCE_RECORD_LOCATOR,
        SOURCE_LINEAGE_TYPE,
        SOURCE_FINDING_FINGERPRINTS,
    }
    if not finding_columns.issubset(finding_audit.columns) or not cause_columns.issubset(
        cause_lineage.columns
    ):
        return []
    available_locators = set(
        finding_audit[pc_findings.SOURCE_RECORD_LOCATOR].drop_nulls().to_list()
    )
    available_fingerprints = set(
        finding_audit[pc_conservation.FINDING_FINGERPRINT].drop_nulls().to_list()
    )
    source_causes = cause_lineage.filter(
        pl.col(SOURCE_LINEAGE_TYPE) == SOURCE_FINDING_LINEAGE
    )
    issues: list[str] = []
    linked_locators = set(
        source_causes[pc_findings.SOURCE_RECORD_LOCATOR].drop_nulls().to_list()
    )
    if not linked_locators.issubset(available_locators):
        issues.append("cause lineage references a locator outside findings.csv")
    linked_fingerprints = {
        fingerprint
        for value in source_causes[SOURCE_FINDING_FINGERPRINTS].drop_nulls().to_list()
        for fingerprint in str(value).split("|")
    }
    if not linked_fingerprints.issubset(available_fingerprints):
        issues.append("cause lineage references a fingerprint outside findings.csv")
    return issues


def _derived_record_locator(row: Mapping[str, object], lineage_type: str) -> str:
    """Return a stable locator for a derived report row."""
    fields = (
        pc_findings.PORTFOLIO_ID,
        pc_findings.SECURITY_ID,
        pc_findings.FROM_DATE,
        pc_findings.THRU_DATE,
        pc_findings.DATASET,
        pc_findings.SOURCE_COLUMN,
        pc_findings.FINDING_CODE,
        _RECONSTRUCTION_COMPONENTS,
    )
    payload = {
        "lineage_type": lineage_type,
        "key": [[field, _canonical_value(row.get(field))] for field in fields],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"derived:{lineage_type}:{digest}"


def _canonical_value(value: object) -> object:
    """Return a deterministic JSON-compatible representation of a key value."""
    if value is None or isinstance(value, (str, bool, int)):
        result = value
    elif isinstance(value, float):
        if math.isnan(value):
            result = "NaN"
        elif math.isinf(value):
            result = "Infinity" if value > 0 else "-Infinity"
        else:
            result = format(value, ".17g")
    elif isinstance(value, (dt.date, dt.datetime, dt.time)):
        result = value.isoformat()
    elif isinstance(value, Enum):
        result = str(value.value)
    else:
        result = str(value)
    return result


def _contains_blank(values: pl.Series) -> bool:
    """Return whether a series contains null or blank values."""
    return bool(
        values.null_count()
        or any(not str(value).strip() for value in values.drop_nulls().to_list())
    )
