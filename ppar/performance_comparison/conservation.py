"""Enforce lossless evidence and single-owner explanation contracts."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
from typing import Final, cast

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
_MAX_FINGERPRINT_VALUE_CACHE: Final[int] = 4_096
_COMPACT_JSON_ENCODER = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"))
_EFFECT_KEY_PREFIX: Final[str] = "__ppar_effect_key_"


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

    fingerprint_columns = sorted(findings.columns)
    fingerprint_counts: dict[str, int] = {}
    fingerprints: list[str] = []
    for base_fingerprint in _finding_row_fingerprints(
        findings,
        fingerprint_columns,
    ):
        occurrence = fingerprint_counts.get(base_fingerprint, 0) + 1
        fingerprint_counts[base_fingerprint] = occurrence
        fingerprints.append(f"{base_fingerprint}:{occurrence}")
    return findings.with_columns(
        pl.Series(
            FINDING_SEQUENCE,
            range(1, findings.height + 1),
            dtype=pl.Int64,
        ),
        pl.Series(FINDING_FINGERPRINT, fingerprints, dtype=pl.String),
        pl.lit(DifferenceDisposition.REVIEW_EVIDENCE.value).alias(
            SAFETY_DISPOSITION
        ),
    )


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

    has_impact = _finite_impact_expr(causes)
    result = _with_economic_effect_ids(
        causes,
        comparison_level=comparison_level,
    ).with_columns(
        pl.when(has_impact)
        .then(pl.lit(DifferenceDisposition.COUNTED_CAUSE.value))
        .otherwise(pl.lit(DifferenceDisposition.REVIEW_EVIDENCE.value))
        .alias(SAFETY_DISPOSITION),
        pl.when(has_impact)
        .then(_counted_cause_owner_expr(causes))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(COUNTED_CAUSE_OWNER),
    )
    output_columns = list(causes.columns)
    for column_name in (
        SAFETY_DISPOSITION,
        ECONOMIC_EFFECT_ID,
        COUNTED_CAUSE_OWNER,
    ):
        if column_name not in output_columns:
            output_columns.append(column_name)
    return result.select(output_columns)


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
    if _has_invalid_string(causes, ECONOMIC_EFFECT_ID):
        raise PpaError(
            "SN-01 no-lost-differences invariant failed: a cause row has no "
            "economic effect identifier.",
            999,
        )
    has_impact = _finite_impact_expr(causes)
    owner_is_null = pl.col(COUNTED_CAUSE_OWNER).is_null()
    owner_is_present = _present_string_expr(causes, COUNTED_CAUSE_OWNER)
    valid = (
        (~has_impact)
        & (
            pl.col(SAFETY_DISPOSITION)
            == DifferenceDisposition.REVIEW_EVIDENCE.value
        )
        & owner_is_null
    ) | (
        has_impact
        & (
            pl.col(SAFETY_DISPOSITION)
            == DifferenceDisposition.COUNTED_CAUSE.value
        )
        & owner_is_present
    )
    if not causes.filter(~valid.fill_null(False)).is_empty():
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
    if counted.is_empty():
        return
    is_formula = pl.col(pc_findings.FINDING_CODE) == _RECONSTRUCTION_FORMULA_CODE
    ineligible = (~is_formula.fill_null(False)) & (
        (
            (pl.col(pc_findings.DATASET) == pc_cols.TRANSACTIONS)
            & ~pl.col(pc_findings.SOURCE_COLUMN).is_in(
                [pc_cols.AMOUNT, pc_cols.BASE_AMOUNT]
            )
        )
        | pl.col(pc_findings.DATASET).is_in([pc_cols.FX_RATES, pc_cols.SPLITS])
    )
    invalid = counted.filter(ineligible.fill_null(False)).head(1)
    if not invalid.is_empty():
        _raise_ineligible_owner(invalid.row(0, named=True))


def _assert_counted_period_boundaries(causes: pl.DataFrame) -> None:
    """Raise when dated evidence owns impact outside its permitted boundary."""
    counted = causes.filter(
        pl.col(SAFETY_DISPOSITION) == DifferenceDisposition.COUNTED_CAUSE.value
    )
    if counted.is_empty() or not _date_columns_are_temporal(counted):
        return
    input_date = pl.col(_AS_OF_DATE).cast(pl.Date)
    from_date = pl.col(pc_findings.FROM_DATE).cast(pl.Date)
    thru_date = pl.col(pc_findings.THRU_DATE).cast(pl.Date)
    dates_present = input_date.is_not_null() & from_date.is_not_null() & thru_date.is_not_null()
    permitted = input_date.is_between(from_date, thru_date, closed="both") | (
        pl.col(pc_findings.DATASET).is_in([pc_cols.HOLDINGS, pc_cols.FX_RATES])
        & (input_date == from_date.dt.offset_by("-1d"))
    )
    invalid = counted.filter(dates_present & ~permitted.fill_null(False)).head(1)
    if invalid.is_empty():
        return
    row = invalid.row(0, named=True)
    dataset = row.get(pc_findings.DATASET)
    input_value = _date_or_none(row.get(_AS_OF_DATE))
    from_value = _date_or_none(row.get(pc_findings.FROM_DATE))
    thru_value = _date_or_none(row.get(pc_findings.THRU_DATE))
    raise PpaError(
        "SN-07 period-boundary invariant failed: counted cause "
        f"{dataset}.{row.get(pc_findings.SOURCE_COLUMN)} dated {input_value} "
        f"is outside performance period {from_value}..{thru_value}.",
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


def _with_economic_effect_ids(
    causes: pl.DataFrame,
    *,
    comparison_level: str,
) -> pl.DataFrame:
    """Append exact effect identifiers after hashing distinct payloads once."""
    payload_columns = (
        pc_findings.PORTFOLIO_ID,
        pc_findings.FROM_DATE,
        pc_findings.THRU_DATE,
        pc_findings.SECURITY_ID,
        _AS_OF_DATE,
        pc_findings.DATASET,
        pc_findings.SOURCE_COLUMN,
        pc_findings.TRANSACTION_CODE,
        pc_findings.TRANSACTION_CATEGORY,
        pc_findings.SNAPSHOT_A_VALUE,
        pc_findings.SNAPSHOT_B_VALUE,
    )
    key_columns = tuple(
        f"{_EFFECT_KEY_PREFIX}{index}" for index in range(len(payload_columns))
    )
    keyed_causes = causes.with_columns(
        *(
            (
                pl.col(column_name)
                if column_name in causes.columns
                else pl.lit(None)
            ).alias(key_name)
            for column_name, key_name in zip(
                payload_columns,
                key_columns,
                strict=True,
            )
        )
    )
    distinct_keys = keyed_causes.select(key_columns).unique(maintain_order=True)
    effect_ids = [
        _economic_effect_id(
            dict(zip(payload_columns, values, strict=True)),
            comparison_level=comparison_level,
        )
        for values in distinct_keys.iter_rows()
    ]
    effect_map = distinct_keys.with_columns(
        pl.Series(ECONOMIC_EFFECT_ID, effect_ids, dtype=pl.String)
    )
    return keyed_causes.join(
        effect_map,
        on=list(key_columns),
        how="left",
        nulls_equal=True,
        maintain_order="left",
    ).drop(key_columns)


def _finite_impact_expr(causes: pl.DataFrame) -> pl.Expr:
    """Return a Boolean expression matching finite Python numeric impacts."""
    dtype = causes.schema.get(_ESTIMATED_IMPACT)
    if dtype is None or dtype == pl.Boolean or not dtype.is_numeric():
        return pl.lit(False)
    return pl.col(_ESTIMATED_IMPACT).cast(pl.Float64).is_finite().fill_null(False)


def _counted_cause_owner_expr(causes: pl.DataFrame) -> pl.Expr:
    """Return the existing human-readable owner label as a Polars expression."""
    dataset_field = pl.concat_str(
        _column_text_expr(causes, pc_findings.DATASET, null_text="None"),
        pl.lit("."),
        _column_text_expr(causes, pc_findings.SOURCE_COLUMN, null_text="None"),
    )
    suffix = pl.concat_str(
        *(
            pl.when(
                _column_text_expr(causes, column_name).is_not_null()
                & (_column_text_expr(causes, column_name) != "")
            )
            .then(_column_text_expr(causes, column_name))
            .otherwise(pl.lit(None, dtype=pl.String))
            for column_name in (
                pc_findings.SECURITY_ID,
                _AS_OF_DATE,
                pc_findings.TRANSACTION_CODE,
            )
        ),
        separator=":",
        ignore_nulls=True,
    )
    return pl.when(suffix == "").then(dataset_field).otherwise(
        pl.concat_str(dataset_field, pl.lit("@"), suffix)
    )


def _column_text_expr(
    table: pl.DataFrame,
    column_name: str,
    *,
    null_text: str | None = None,
) -> pl.Expr:
    """Return one optional column cast to its Python-string representation."""
    if column_name not in table.columns:
        return pl.lit(null_text, dtype=pl.String)
    result = pl.col(column_name).cast(pl.String)
    return result.fill_null(null_text) if null_text is not None else result


def _present_string_expr(table: pl.DataFrame, column_name: str) -> pl.Expr:
    """Return whether a column contains the nonempty strings required by SN-01."""
    if table.schema.get(column_name) != pl.String:
        return pl.lit(False)
    return pl.col(column_name).is_not_null() & (pl.col(column_name) != "")


def _has_invalid_string(table: pl.DataFrame, column_name: str) -> bool:
    """Return whether any row lacks a required nonempty Python string."""
    if table.schema.get(column_name) != pl.String:
        return not table.is_empty()
    return not table.filter(
        pl.col(column_name).is_null() | (pl.col(column_name) == "")
    ).is_empty()


def _date_columns_are_temporal(table: pl.DataFrame) -> bool:
    """Return whether all SN-07 date columns retain Python date semantics."""
    for column_name in (
        _AS_OF_DATE,
        pc_findings.FROM_DATE,
        pc_findings.THRU_DATE,
    ):
        dtype = table.schema.get(column_name)
        if dtype is None or not (
            dtype == pl.Date or dtype.base_type() == pl.Datetime
        ):
            return False
    return True


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


def _finding_row_fingerprints(
    findings: pl.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """Return exact persisted fingerprints without per-row JSON dictionaries.

    Notes:
        Fingerprint bytes are a persisted compatibility contract. The compiled
        encoders below reproduce ``json.dumps(..., ensure_ascii=True)`` exactly,
        including Unicode escaping, negative zero, non-finite floats, and ISO
        dates. Bounded per-column caches accelerate repeated categorical values
        without allowing memory use to grow with high-cardinality site data.
    """
    prefixes = tuple(
        ("{" if index == 0 else ",")
        + json.encoder.encode_basestring_ascii(column)
        + ":"
        for index, column in enumerate(columns)
    )
    value_encoders = tuple(
        _fingerprint_value_encoder(findings.schema[column], prefix)
        for column, prefix in zip(columns, prefixes, strict=True)
    )
    fingerprints: list[str] = []
    for values in findings.select(columns).iter_rows():
        encoded = (
            "".join(
                [
                    encoder(value)
                    for encoder, value in zip(value_encoders, values, strict=True)
                ]
            )
            + "}"
        )
        fingerprints.append(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return fingerprints


def _fingerprint_value_encoder(
    dtype: pl.DataType,
    prefix: str,
) -> Callable[[object], str]:
    """Return an exact JSON scalar encoder with its object-key prefix."""
    encoded_null = prefix + "null"
    if dtype == pl.String:
        encoder = _cached_fingerprint_value_encoder(
            prefix,
            lambda value: json.encoder.encode_basestring_ascii(
                cast(str, value)
            ),
        )
    elif dtype == pl.Boolean:
        def encode_boolean(value: object) -> str:
            return (
                encoded_null
                if value is None
                else prefix + ("true" if bool(value) else "false")
            )

        encoder = encode_boolean
    elif dtype == pl.Date or dtype.base_type() == pl.Datetime:
        encoder = _cached_fingerprint_value_encoder(
            prefix,
            lambda value: json.encoder.encode_basestring_ascii(
                cast(dt.date | dt.datetime, value).isoformat()
            ),
        )
    elif dtype.is_float():
        encoder = _fingerprint_float_encoder(prefix)
    elif dtype.is_integer():
        def encode_integer(value: object) -> str:
            return encoded_null if value is None else prefix + str(value)

        encoder = encode_integer
    elif dtype == pl.Null:
        def encode_null(_value: object) -> str:
            return encoded_null

        encoder = encode_null
    else:
        encoder = _cached_fingerprint_value_encoder(
            prefix,
            lambda value: _COMPACT_JSON_ENCODER.encode(_json_value(value)),
        )
    return encoder


def _fingerprint_float_encoder(prefix: str) -> Callable[[object], str]:
    """Return an exact JSON encoder for nullable floating-point values."""
    encoded_null = prefix + "null"

    def encode(value: object) -> str:
        if value is None:
            return encoded_null
        numeric_value = cast(float, value)
        if math.isnan(numeric_value):
            return prefix + '"NaN"'
        if math.isinf(numeric_value):
            return prefix + ('"Infinity"' if numeric_value > 0 else '"-Infinity"')
        return prefix + repr(numeric_value)

    return encode


def _cached_fingerprint_value_encoder(
    prefix: str,
    encode_value: Callable[[object], str],
) -> Callable[[object], str]:
    """Return a bounded exact encoder for repeated nullable scalar values."""
    cache: dict[object, str] = {None: prefix + "null"}

    def encode(value: object) -> str:
        try:
            return cache[value]
        except (KeyError, TypeError):
            encoded = prefix + encode_value(value)
        if len(cache) < _MAX_FINGERPRINT_VALUE_CACHE:
            try:
                cache[value] = encoded
            except TypeError:
                pass
        return encoded

    return encode


def _json_value(value: object) -> object:
    """Return a deterministic JSON-compatible scalar value."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return str(value)


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
