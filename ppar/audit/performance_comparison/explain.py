"""Build explanation-oriented tables from performance comparison findings."""

from __future__ import annotations

# Python imports
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import datetime as dt
import math

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import field_roles as _field_roles
from ppar.audit import schema as pc_cols
from ppar.audit.performance_comparison import _transaction_diagnostics as tx_diagnostics
from ppar.audit.performance_comparison.findings import (
    CASH_FLOW_SIGN,
    CONTEXT,
    DATASET,
    DELTA_B_MINUS_A,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_CURRENCY,
    FROM_DATE,
    IMPACT_POLICY,
    IMPACT_POLICY_EVIDENCE_ONLY_PREFIX,
    IMPACT_POLICY_FX_RATE_EXPOSURE,
    IMPACT_POLICY_HOLDING_ACCRUED,
    IMPACT_POLICY_HOLDING_MARKET_VALUE,
    IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
    IMPACT_POLICY_PRICE_WEIGHTED,
    INPUT_DATE,
    MESSAGE,
    PC_PORT_RET,
    PC_SEC_RET,
    PERFORMANCE_FLOW_SIGN,
    PORTFOLIO_ID,
    RELATED_OUTPUT,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
    IMPACT_INPUT_VALUE,
    SECURITY_ID,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SOURCE_RECORD_LOCATOR,
    SUPPRESSED,
    TARGET_OUTPUT,
    THRU_DATE,
    TO_CURRENCY,
    TRANSACTION_CODE,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
    TRANSACTION_IMPACT_POLICY_SECURITY_FLOW_MODIFIED_DIETZ,
    TRANSACTION_CATEGORY,
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_SEMANTICS_SOURCE,
)
from ppar.audit.performance_comparison.methods import (
    FxRateImpactMethod,
    ModifiedDietzDoubleCountPolicy,
    HoldingImpactMethod,
    PriceImpactMethod,
    TransactionImpactMethod,
)
from ppar.audit.performance_comparison.modified_dietz import (
    modified_dietz_flow_weight as _modified_dietz_flow_weight,
)
from ppar.audit.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_CASH_FLOW_SIGN_NONE,
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
    TRANSACTION_CATEGORY_UNKNOWN,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    transaction_impact_semantics_available,
)
from ppar.audit.performance_comparison.vocabulary import CauseArea

PORTFOLIO_RETURN_DELTA = "portfolio_return_delta"
SECURITY_RETURN_DELTA = "security_return_delta"
FINDING_COUNT = "finding_count"
PORTFOLIO_FINDING_COUNT = "portfolio_finding_count"
SECURITY_FINDING_COUNT = "security_finding_count"
DIRECT_INPUT_FINDING_COUNT = "direct_input_finding_count"
RELATED_OUTPUT_FINDING_COUNT = "related_output_finding_count"
CONTEXT_FINDING_COUNT = "context_finding_count"
EVIDENCE_GROUP = "evidence_group"
FX_RATE_FINDING_COUNT = "fx_rate_finding_count"
TRANSACTION_FINDING_COUNT = "transaction_finding_count"
HOLDING_FINDING_COUNT = "holding_finding_count"
HAS_SUPPRESSED_FINDINGS = "has_suppressed_findings"
PORTFOLIO_PERIOD_SUMMARY_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    PORTFOLIO_RETURN_DELTA,
    FINDING_COUNT,
    PORTFOLIO_FINDING_COUNT,
    DIRECT_INPUT_FINDING_COUNT,
    RELATED_OUTPUT_FINDING_COUNT,
    CONTEXT_FINDING_COUNT,
    FX_RATE_FINDING_COUNT,
    TRANSACTION_FINDING_COUNT,
    HOLDING_FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
)
PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    EVIDENCE_GROUP,
    DATASET,
    FINDING_COUNT,
)
SECURITY_PERIOD_SUMMARY_COLUMNS = (
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    SECURITY_RETURN_DELTA,
    FINDING_COUNT,
    SECURITY_FINDING_COUNT,
    DIRECT_INPUT_FINDING_COUNT,
    RELATED_OUTPUT_FINDING_COUNT,
    CONTEXT_FINDING_COUNT,
    TRANSACTION_FINDING_COUNT,
    HOLDING_FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
)
SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS = (
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    EVIDENCE_GROUP,
    DATASET,
    FINDING_COUNT,
)
REVIEW_RANK = "review_rank"
PRIORITY_SCORE = "priority_score"
ABSOLUTE_DELTA = "absolute_delta"
PRIORITY_REASON = "priority_reason"
ESTIMATED_RETURN_IMPACT = "estimated_return_impact"
IMPACT_BASIS = "impact_basis"
IMPACT_CONFIDENCE = "impact_confidence"
IMPACT_METHOD = "impact_method"
IMPACT_MESSAGE = "impact_message"
IMPACT_BASIS_NO_ESTIMATE = "no_estimate"
IMPACT_BASIS_HOLDING_ACCRUED = "holding_accrued"
IMPACT_BASIS_HOLDING_MARKET_VALUE = "holding_market_value"
IMPACT_BASIS_HOLDING_QUANTITY_UNIT_MARKET_VALUE = "holding_quantity_unit_market_value"
IMPACT_BASIS_PRICE_WEIGHTED = "price_weighted"
IMPACT_BASIS_SECURITY_HOLDING_ACCRUED = "security_holding_accrued"
IMPACT_BASIS_SECURITY_HOLDING_MARKET_VALUE = "security_holding_market_value"
IMPACT_BASIS_SECURITY_HOLDING_QUANTITY_UNIT_MARKET_VALUE = (
    "security_holding_quantity_unit_market_value"
)
IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT = "transaction_performance_amount"
IMPACT_BASIS_FX_RATE_LOCAL_EXPOSURE = "fx_rate_local_exposure"
IMPACT_BASIS_SECURITY_TRANSACTION_FLOW = "security_transaction_flow"
IMPACT_CONFIDENCE_LOW = "low"
IMPACT_CONFIDENCE_MEDIUM = "medium"
IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR = (
    TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value
)
IMPACT_METHOD_FX_RATE_DELTA_TIMES_LOCAL_EXPOSURE_OVER_DENOMINATOR = (
    FxRateImpactMethod.RATE_DELTA_TIMES_LOCAL_EXPOSURE_OVER_RETURN_DENOMINATOR.value
)
IMPACT_METHOD_SECURITY_TRANSACTION_FLOW_MODIFIED_DIETZ = (
    TransactionImpactMethod.MODIFIED_DIETZ.value
)
IMPACT_METHOD_HOLDING_MARKET_VALUE_DELTA_OVER_DENOMINATOR = (
    HoldingImpactMethod.MARKET_VALUE_DELTA_OVER_RETURN_DENOMINATOR.value
)
IMPACT_METHOD_HOLDING_ACCRUED_DELTA_OVER_DENOMINATOR = (
    HoldingImpactMethod.ACCRUED_DELTA_OVER_RETURN_DENOMINATOR.value
)
IMPACT_METHOD_HOLDING_QUANTITY_UNIT_MARKET_VALUE_OVER_DENOMINATOR = HoldingImpactMethod[
    "QUANTITY_DELTA_TIMES_SNAPSHOT_A_UNIT_MARKET_VALUE_OVER_RETURN_DENOMINATOR"
].value
IMPACT_METHOD_PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT = (
    PriceImpactMethod.PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT.value
)
ROOT_CAUSE_AREA = "root_cause_area"
ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION = (
    CauseArea.SECURITY_RETURN_OR_CONTRIBUTION.value
)
ROOT_CAUSE_MARKET_VALUE_OR_HOLDING = CauseArea.MARKET_VALUE_OR_HOLDING.value
ROOT_CAUSE_TRANSACTION_ACTIVITY = CauseArea.TRANSACTION_ACTIVITY.value
ROOT_CAUSE_FX_RATE = CauseArea.FX_RATE.value
ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT = CauseArea.PORTFOLIO_PERFORMANCE_INPUT.value
ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE = CauseArea.CLASSIFICATION_OR_REFERENCE.value
ROOT_CAUSE_UNEXPLAINED = CauseArea.UNEXPLAINED.value
TOP_CODES = "top_codes"
CHANGED_FIELDS = "changed_fields"
AMOUNT_DELTA = "amount_delta"
QUANTITY_DELTA = "quantity_delta"
PRICE_DELTA = "price_delta"
MISSING_IMPACT_INPUTS = "missing_impact_inputs"
TRANSACTION_SEMANTICS_SOURCES = "transaction_semantics_sources"
TRANSACTION_SIGN_AND_FLOW_SEMANTICS = "transaction sign and flow semantics"
EXTERNAL_FLOW_IMPACT_METHOD = "external-flow impact method"
EXTERNAL_FLOW_EVIDENCE_ONLY_POLICY = "external-flow evidence-only policy"
NEUTRAL_FLOW_IMPACT_METHOD = "neutral-flow impact method"
NO_CASH_TRANSACTION_IMPACT_METHOD = "no-cash transaction impact method"
TRANSACTION_IMPACT_METHOD = "transaction impact method"
CROSS_CHECK_TREATMENT = "cross_check_treatment"
CROSS_CHECK_ONLY = ModifiedDietzDoubleCountPolicy.CROSS_CHECK_ONLY.value
CROSS_CHECK_COUNT = "cross_check_count"
CROSS_CHECK_ESTIMATE_TOTAL = "cross_check_estimate_total"
CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL = "cross_check_absolute_estimate_total"
TRANSACTION_IMPACT_POLICIES = "transaction_impact_policies"
TRANSACTION_IMPACT_DIAGNOSTICS = "transaction_impact_diagnostics"
TRANSACTION_MATCH_STATUSES = "transaction_match_statuses"
TRANSACTION_MATCH_CONFIDENCE = "transaction_match_confidence"
TRANSACTION_MATCH_INTERPRETATION = "transaction_match_interpretation"
TRANSACTION_MATCH_REVIEW_NOTE = "transaction_match_review_note"
PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    REVIEW_RANK,
    PRIORITY_SCORE,
    PRIORITY_REASON,
    FINDING_CODE,
    DATASET,
    EVIDENCE_ROLE,
    SECURITY_ID,
    SOURCE_FILE,
    SOURCE_RECORD_LOCATOR,
    INPUT_DATE,
    SOURCE_COLUMN,
    FROM_CURRENCY,
    TO_CURRENCY,
    TRANSACTION_CODE,
    TRANSACTION_CATEGORY,
    CASH_FLOW_SIGN,
    PERFORMANCE_FLOW_SIGN,
    TRANSACTION_SEMANTICS_SOURCE,
    IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    DELTA_B_MINUS_A,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
    IMPACT_INPUT_VALUE,
    ABSOLUTE_DELTA,
    MESSAGE,
)
PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS = (
    *PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS,
    ESTIMATED_RETURN_IMPACT,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
    IMPACT_METHOD,
    IMPACT_MESSAGE,
)
PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    ROOT_CAUSE_AREA,
    FINDING_COUNT,
    ESTIMATED_RETURN_IMPACT,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
    TOP_CODES,
    IMPACT_MESSAGE,
)
SECURITY_PERIOD_CAUSE_SUMMARY_COLUMNS = (
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    ROOT_CAUSE_AREA,
    FINDING_COUNT,
    ESTIMATED_RETURN_IMPACT,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
    TOP_CODES,
    IMPACT_MESSAGE,
)
ROOT_CAUSE_AREA_COUNT = "root_cause_area_count"
ESTIMATED_CAUSE_AREA_COUNT = "estimated_cause_area_count"
EVIDENCE_ONLY_CAUSE_AREA_COUNT = "evidence_only_cause_area_count"
LOW_CONFIDENCE_ESTIMATE_COUNT = "low_confidence_estimate_count"
MEDIUM_CONFIDENCE_ESTIMATE_COUNT = "medium_confidence_estimate_count"
ESTIMATED_RETURN_IMPACT_TOTAL = "estimated_return_impact_total"
EVIDENCE_ONLY_AREAS = "evidence_only_areas"
IMPACT_COVERAGE_STATUS = "impact_coverage_status"
IMPACT_COVERAGE_REVIEW_NOTE = "impact_coverage_review_note"
IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES = "complete_estimates"
IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES = "partial_estimates"
IMPACT_COVERAGE_STATUS_EVIDENCE_ONLY = "evidence_only"
IMPACT_COVERAGE_STATUS_MISSING_INPUTS = "missing_inputs"
PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    PORTFOLIO_RETURN_DELTA,
    ROOT_CAUSE_AREA_COUNT,
    ESTIMATED_CAUSE_AREA_COUNT,
    EVIDENCE_ONLY_CAUSE_AREA_COUNT,
    LOW_CONFIDENCE_ESTIMATE_COUNT,
    MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
    ESTIMATED_RETURN_IMPACT_TOTAL,
    EVIDENCE_ONLY_AREAS,
    TRANSACTION_SEMANTICS_SOURCES,
    MISSING_IMPACT_INPUTS,
    IMPACT_COVERAGE_STATUS,
    IMPACT_COVERAGE_REVIEW_NOTE,
    IMPACT_MESSAGE,
)
TRANSACTION_ACTIVITY_SUMMARY_COLUMNS = (
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    TRANSACTION_CATEGORY,
    FINDING_COUNT,
    CHANGED_FIELDS,
    AMOUNT_DELTA,
    QUANTITY_DELTA,
    PRICE_DELTA,
    TRANSACTION_SEMANTICS_SOURCES,
    TRANSACTION_MATCH_STATUSES,
    MISSING_IMPACT_INPUTS,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
    IMPACT_MESSAGE,
)
TRANSACTION_MATCHING_DIAGNOSTIC_COLUMNS = (
    TRANSACTION_MATCH_STATUS,
    FINDING_COUNT,
    TRANSACTION_MATCH_CONFIDENCE,
    TRANSACTION_MATCH_INTERPRETATION,
    TRANSACTION_MATCH_REVIEW_NOTE,
)
PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS = (
    PORTFOLIO_ID,
    FROM_DATE,
    THRU_DATE,
    TRANSACTION_IMPACT_POLICIES,
    CROSS_CHECK_TREATMENT,
    CROSS_CHECK_COUNT,
    CROSS_CHECK_ESTIMATE_TOTAL,
    CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL,
    CHANGED_FIELDS,
    TRANSACTION_IMPACT_DIAGNOSTICS,
    IMPACT_MESSAGE,
)


def portfolio_period_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return a lightweight summary around portfolio-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be counted as
            active related evidence.

    Returns:
        One row per portfolio-period return delta with related evidence counts.
        This is an explanation bridge, not a causal attribution model.
    """
    if findings.is_empty():
        return _empty_portfolio_period_summary()

    active_findings = _active_findings(findings, include_suppressed)
    target_findings = active_findings.filter(
        (pl.col(FINDING_CODE) == PC_PORT_RET)
        & (pl.col(DATASET) == pc_cols.PORTFOLIO_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.PORTFOLIO_RETURN)
    )
    if target_findings.is_empty():
        return _empty_portfolio_period_summary()

    rows: list[dict[str, object]] = []
    active_index = _related_finding_count_index(
        active_findings,
        (PORTFOLIO_ID, FROM_DATE, THRU_DATE),
    )
    suppressed_keys = _suppressed_finding_keys(
        findings,
        (PORTFOLIO_ID, FROM_DATE, THRU_DATE),
    )
    for target in target_findings.iter_rows(named=True):
        related_keys = _portfolio_related_keys(target)
        related_counts = _combined_related_finding_counts(
            active_index,
            related_keys,
        )
        role_counts = _role_summary_counts_from_counter(related_counts.roles)
        rows.append(
            {
                PORTFOLIO_ID: target[PORTFOLIO_ID],
                FROM_DATE: target[FROM_DATE],
                THRU_DATE: target[THRU_DATE],
                PORTFOLIO_RETURN_DELTA: target[DELTA_B_MINUS_A],
                FINDING_COUNT: related_counts.total,
                PORTFOLIO_FINDING_COUNT: related_counts.datasets.get(
                    pc_cols.PORTFOLIO_PERFORMANCE,
                    0,
                ),
                **role_counts,
                FX_RATE_FINDING_COUNT: related_counts.datasets.get(pc_cols.FX_RATES, 0),
                TRANSACTION_FINDING_COUNT: related_counts.datasets.get(
                    pc_cols.TRANSACTIONS,
                    0,
                ),
                HOLDING_FINDING_COUNT: related_counts.datasets.get(pc_cols.HOLDINGS, 0),
                HAS_SUPPRESSED_FINDINGS: any(
                    key in suppressed_keys for key in related_keys
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(PORTFOLIO_PERIOD_SUMMARY_COLUMNS)


def portfolio_period_evidence_breakdown(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return role and dataset counts for portfolio-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be counted as
            active related evidence.

    Returns:
        Long-form evidence count table grouped by portfolio, period, evidence
        group, and dataset. The output is a reporting helper, not a causal
        attribution model.
    """
    if findings.is_empty():
        return _empty_portfolio_period_evidence_breakdown()

    active_findings = _active_findings(findings, include_suppressed)
    target_findings = active_findings.filter(
        (pl.col(FINDING_CODE) == PC_PORT_RET)
        & (pl.col(DATASET) == pc_cols.PORTFOLIO_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.PORTFOLIO_RETURN)
    )
    if target_findings.is_empty():
        return _empty_portfolio_period_evidence_breakdown()

    rows: list[dict[str, object]] = []
    active_index = _portfolio_period_index(active_findings)
    for target in target_findings.iter_rows(named=True):
        related_active = _indexed_portfolio_period_findings(active_findings, active_index, target)
        rows.extend(_evidence_breakdown_rows(target, related_active))
    return pl.DataFrame(rows, infer_schema_length=None).select(
        PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS
    )


def security_period_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return a lightweight summary around security-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be counted as
            active related evidence.

    Returns:
        One row per security-period return delta with related evidence counts.
        Empty or portfolio-only findings return an empty table with stable
        columns.
    """
    if findings.is_empty():
        return _empty_security_period_summary()

    active_findings = _active_findings(findings, include_suppressed)
    target_findings = findings.filter(
        (pl.col(FINDING_CODE) == PC_SEC_RET)
        & (pl.col(DATASET) == pc_cols.SECURITY_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
    )
    if target_findings.is_empty():
        return _empty_security_period_summary()

    rows: list[dict[str, object]] = []
    active_index = _related_finding_count_index(
        active_findings,
        (SECURITY_ID, PORTFOLIO_ID, FROM_DATE, THRU_DATE),
    )
    suppressed_keys = _suppressed_finding_keys(
        findings,
        (SECURITY_ID, PORTFOLIO_ID, FROM_DATE, THRU_DATE),
    )
    for target in target_findings.iter_rows(named=True):
        related_keys = _security_related_keys(target)
        related_counts = _combined_related_finding_counts(
            active_index,
            related_keys,
        )
        role_counts = _role_summary_counts_from_counter(related_counts.roles)
        rows.append(
            {
                PORTFOLIO_ID: target[PORTFOLIO_ID],
                SECURITY_ID: target[SECURITY_ID],
                FROM_DATE: target[FROM_DATE],
                THRU_DATE: target[THRU_DATE],
                SECURITY_RETURN_DELTA: target[DELTA_B_MINUS_A],
                FINDING_COUNT: related_counts.total,
                SECURITY_FINDING_COUNT: related_counts.datasets.get(
                    pc_cols.SECURITY_PERFORMANCE,
                    0,
                ),
                **role_counts,
                TRANSACTION_FINDING_COUNT: related_counts.datasets.get(
                    pc_cols.TRANSACTIONS,
                    0,
                ),
                HOLDING_FINDING_COUNT: related_counts.datasets.get(pc_cols.HOLDINGS, 0),
                HAS_SUPPRESSED_FINDINGS: any(
                    key in suppressed_keys for key in related_keys
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(SECURITY_PERIOD_SUMMARY_COLUMNS)


def security_period_evidence_breakdown(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return role and dataset counts for security-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be counted as
            active related evidence.

    Returns:
        Long-form evidence count table grouped by portfolio, security, period,
        evidence group, and dataset. Empty or portfolio-only findings return an
        empty table with stable columns.
    """
    if findings.is_empty():
        return _empty_security_period_evidence_breakdown()

    active_findings = _active_findings(findings, include_suppressed)
    target_findings = findings.filter(
        (pl.col(FINDING_CODE) == PC_SEC_RET)
        & (pl.col(DATASET) == pc_cols.SECURITY_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
    )
    if target_findings.is_empty():
        return _empty_security_period_evidence_breakdown()

    rows: list[dict[str, object]] = []
    active_index = _security_period_index(active_findings)
    for target in target_findings.iter_rows(named=True):
        related_active = _indexed_security_period_findings(active_findings, active_index, target)
        rows.extend(_security_evidence_breakdown_rows(target, related_active))
    return pl.DataFrame(rows, infer_schema_length=None).select(
        SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS
    )


_RANK_TARGET_ORDER = "__ppar_rank_target_order"
_RANK_EVIDENCE_ORDER = "__ppar_rank_evidence_order"
_RANK_TARGET_PORTFOLIO = "__ppar_rank_target_portfolio"
_RANK_TARGET_SECURITY = "__ppar_rank_target_security"
_RANK_TARGET_FROM_DATE = "__ppar_rank_target_from_date"
_RANK_TARGET_THRU_DATE = "__ppar_rank_target_thru_date"
_RANK_ROLE_SCORE = "__ppar_rank_role_score"
_RANK_DATASET_SCORE = "__ppar_rank_dataset_score"
_RANK_DELTA_SCORE = "__ppar_rank_delta_score"
_RANK_ABSOLUTE_SORT = "__ppar_rank_absolute_sort"
_RANK_STRING_SORT_COLUMNS = (
    DATASET,
    FINDING_CODE,
    SECURITY_ID,
    SOURCE_COLUMN,
    MESSAGE,
)


def _rank_period_evidence(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool,
    security_level: bool,
) -> pl.DataFrame:
    """Return portfolio- or security-period evidence ranked in Polars."""
    if findings.is_empty():
        return _empty_portfolio_period_evidence_ranking()

    active_findings = _active_findings(findings, include_suppressed)
    target_code = PC_SEC_RET if security_level else PC_PORT_RET
    target_dataset = (
        pc_cols.SECURITY_PERFORMANCE
        if security_level
        else pc_cols.PORTFOLIO_PERFORMANCE
    )
    target_source_column = (
        pc_cols.SECURITY_RETURN if security_level else pc_cols.PORTFOLIO_RETURN
    )
    targets = active_findings.filter(
        (pl.col(FINDING_CODE) == target_code)
        & (pl.col(DATASET) == target_dataset)
        & (pl.col(SOURCE_COLUMN) == target_source_column)
    )
    if targets.is_empty():
        return _empty_portfolio_period_evidence_ranking()

    target_rows = targets.with_row_index(_RANK_TARGET_ORDER).select(
        pl.col(PORTFOLIO_ID).alias(_RANK_TARGET_PORTFOLIO),
        pl.col(SECURITY_ID).alias(_RANK_TARGET_SECURITY),
        pl.col(FROM_DATE).alias(_RANK_TARGET_FROM_DATE),
        pl.col(THRU_DATE).alias(_RANK_TARGET_THRU_DATE),
        _RANK_TARGET_ORDER,
    )
    evidence_rows = active_findings.filter(
        pl.col(EVIDENCE_ROLE) != TARGET_OUTPUT
    ).with_row_index(_RANK_EVIDENCE_ORDER)
    relation = (
        _security_evidence_relation(target_rows, evidence_rows)
        if security_level
        else _portfolio_evidence_relation(target_rows, evidence_rows)
    )
    delta_type = active_findings.schema.get(DELTA_B_MINUS_A, pl.Null)
    absolute_delta = (
        pl.col(DELTA_B_MINUS_A).cast(pl.Float64).abs()
        if delta_type.is_numeric() and delta_type != pl.Boolean
        else pl.lit(None, dtype=pl.Float64)
    )
    ranked = _rank_evidence_relation(relation, absolute_delta).collect()
    if ranked.is_empty():
        return _empty_portfolio_period_evidence_ranking()
    return ranked


def _portfolio_evidence_relation(
    targets: pl.DataFrame,
    evidence: pl.DataFrame,
) -> pl.LazyFrame:
    """Return exact-period and undated portfolio evidence relationships."""
    target_rows = targets.lazy()
    evidence_rows = evidence.lazy()
    dated = target_rows.join(
        evidence_rows,
        left_on=[
            _RANK_TARGET_PORTFOLIO,
            _RANK_TARGET_FROM_DATE,
            _RANK_TARGET_THRU_DATE,
        ],
        right_on=[PORTFOLIO_ID, FROM_DATE, THRU_DATE],
        how="inner",
    )
    undated = target_rows.join(
        evidence_rows.filter(
            pl.col(FROM_DATE).is_null() & pl.col(THRU_DATE).is_null()
        ),
        left_on=_RANK_TARGET_PORTFOLIO,
        right_on=PORTFOLIO_ID,
        how="inner",
    )
    relations = [
        _rank_evidence_projection(frame, security_level=False)
        for frame in (dated, undated)
    ]
    return pl.concat(relations)


def _security_evidence_relation(
    targets: pl.DataFrame,
    evidence: pl.DataFrame,
) -> pl.LazyFrame:
    """Return exact and portfolio-optional security evidence relationships."""
    target_rows = targets.lazy()
    evidence_rows = evidence.lazy()
    exact_period = target_rows.join(
        evidence_rows,
        left_on=[
            _RANK_TARGET_SECURITY,
            _RANK_TARGET_PORTFOLIO,
            _RANK_TARGET_FROM_DATE,
            _RANK_TARGET_THRU_DATE,
        ],
        right_on=[SECURITY_ID, PORTFOLIO_ID, FROM_DATE, THRU_DATE],
        how="inner",
    )
    portfolio_optional_period = target_rows.join(
        evidence_rows.filter(pl.col(PORTFOLIO_ID).is_null()),
        left_on=[
            _RANK_TARGET_SECURITY,
            _RANK_TARGET_FROM_DATE,
            _RANK_TARGET_THRU_DATE,
        ],
        right_on=[SECURITY_ID, FROM_DATE, THRU_DATE],
        how="inner",
    )
    exact_undated = target_rows.join(
        evidence_rows.filter(
            pl.col(FROM_DATE).is_null() & pl.col(THRU_DATE).is_null()
        ),
        left_on=[_RANK_TARGET_SECURITY, _RANK_TARGET_PORTFOLIO],
        right_on=[SECURITY_ID, PORTFOLIO_ID],
        how="inner",
    )
    portfolio_optional_undated = target_rows.join(
        evidence_rows.filter(
            pl.col(PORTFOLIO_ID).is_null()
            & pl.col(FROM_DATE).is_null()
            & pl.col(THRU_DATE).is_null()
        ),
        left_on=_RANK_TARGET_SECURITY,
        right_on=SECURITY_ID,
        how="inner",
    )
    relations = [
        _rank_evidence_projection(frame, security_level=True)
        for frame in (
            exact_period,
            portfolio_optional_period,
            exact_undated,
            portfolio_optional_undated,
        )
    ]
    return pl.concat(relations)


def _rank_evidence_projection(
    relation: pl.LazyFrame,
    *,
    security_level: bool,
) -> pl.LazyFrame:
    """Project one relationship branch to a common evidence schema."""
    evidence_columns = [
        column
        for column in PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS[6:]
        if column != ABSOLUTE_DELTA
    ]
    projected_evidence = [
        (
            pl.col(_RANK_TARGET_SECURITY).alias(SECURITY_ID)
            if security_level and column == SECURITY_ID
            else pl.col(column)
        )
        for column in evidence_columns
    ]
    return relation.select(
        _RANK_TARGET_PORTFOLIO,
        _RANK_TARGET_SECURITY,
        _RANK_TARGET_FROM_DATE,
        _RANK_TARGET_THRU_DATE,
        _RANK_TARGET_ORDER,
        _RANK_EVIDENCE_ORDER,
        *projected_evidence,
    )


def _rank_evidence_relation(
    relation: pl.LazyFrame,
    absolute_delta: pl.Expr,
) -> pl.LazyFrame:
    """Score, sort, and rank an evidence relationship in one lazy plan."""
    role_text = pl.col(EVIDENCE_ROLE).cast(pl.String).fill_null("None")
    dataset_text = pl.col(DATASET).cast(pl.String).fill_null("None")
    role_score = role_text.replace_strict(
        {
            DIRECT_INPUT.value: 300,
            RELATED_OUTPUT.value: 200,
            CONTEXT.value: 100,
            TARGET_OUTPUT.value: 0,
        },
        default=0,
        return_dtype=pl.Int64,
    )
    dataset_score = dataset_text.replace_strict(
        {
            pc_cols.PORTFOLIO_PERFORMANCE: 40,
            pc_cols.TRANSACTIONS: 35,
            pc_cols.HOLDINGS: 35,
            pc_cols.FX_RATES: 25,
            pc_cols.SECURITY_PERFORMANCE: 10,
        },
        default=0,
        return_dtype=pl.Int64,
    )
    scored = relation.with_columns(
        absolute_delta.alias(ABSOLUTE_DELTA),
        role_text.alias(EVIDENCE_ROLE),
        dataset_text.alias(DATASET),
        role_score.alias(_RANK_ROLE_SCORE),
        dataset_score.alias(_RANK_DATASET_SCORE),
    ).with_columns(
        pl.when(pl.col(ABSOLUTE_DELTA).is_not_null())
        .then(10)
        .otherwise(0)
        .cast(pl.Int64)
        .alias(_RANK_DELTA_SCORE),
        pl.col(ABSOLUTE_DELTA).fill_null(-1.0).alias(_RANK_ABSOLUTE_SORT),
        *[
            pl.col(column)
            .cast(pl.String)
            .fill_null("None")
            .alias(f"__ppar_rank_sort_{column}")
            for column in _RANK_STRING_SORT_COLUMNS
        ],
    )
    scored = scored.with_columns(
        (
            pl.col(_RANK_ROLE_SCORE)
            + pl.col(_RANK_DATASET_SCORE)
            + pl.col(_RANK_DELTA_SCORE)
        ).alias(PRIORITY_SCORE),
        pl.format(
            "role={}:{}; dataset={}:{}; numeric_delta={}",
            EVIDENCE_ROLE,
            _RANK_ROLE_SCORE,
            DATASET,
            _RANK_DATASET_SCORE,
            _RANK_DELTA_SCORE,
        ).alias(PRIORITY_REASON),
    )
    sort_columns = [
        _RANK_TARGET_ORDER,
        PRIORITY_SCORE,
        _RANK_ABSOLUTE_SORT,
        *[f"__ppar_rank_sort_{column}" for column in _RANK_STRING_SORT_COLUMNS],
        _RANK_EVIDENCE_ORDER,
    ]
    ranked = scored.sort(
        sort_columns,
        descending=[False, True, True, False, False, False, False, False, False],
    ).with_columns(
        pl.col(_RANK_EVIDENCE_ORDER)
        .cum_count()
        .over(_RANK_TARGET_ORDER)
        .cast(pl.Int64)
        .alias(REVIEW_RANK)
    )
    return ranked.select(
        pl.col(_RANK_TARGET_PORTFOLIO).alias(PORTFOLIO_ID),
        pl.col(_RANK_TARGET_FROM_DATE).alias(FROM_DATE),
        pl.col(_RANK_TARGET_THRU_DATE).alias(THRU_DATE),
        *PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS[3:],
    )


def rank_portfolio_period_evidence(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return review-priority evidence rows for portfolio-period deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            the ranked evidence.

    Returns:
        One row per related non-target finding, ranked within each portfolio
        period. The score is a review-priority heuristic, not a causal
        contribution amount or explained return.
    """
    return _rank_period_evidence(
        findings,
        include_suppressed=include_suppressed,
        security_level=False,
    )


def rank_security_period_evidence(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return review-priority evidence rows for security-period deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            the ranked evidence.

    Returns:
        One row per related non-target finding, ranked within each security
        period. The score is a review-priority heuristic.
    """
    return _rank_period_evidence(
        findings,
        include_suppressed=include_suppressed,
        security_level=True,
    )


def portfolio_period_contribution_candidates(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return conservative contribution candidates for portfolio-period deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            contribution candidates.

    Returns:
        Ranked portfolio-period evidence rows with stable contribution-impact
        columns. Most rows may intentionally return ``no_estimate`` until a
        defensible linkage, denominator, and method are available.
    """
    ranking = rank_portfolio_period_evidence(
        findings,
        include_suppressed=include_suppressed,
    )
    if ranking.is_empty():
        return _empty_portfolio_period_contribution_candidates()

    rows = [_contribution_candidate_row(row) for row in ranking.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None).select(
        PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS
    )


def security_period_contribution_candidates(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return contribution candidates for security-period deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            contribution candidates.

    Returns:
        Ranked security-period evidence rows with stable contribution-impact
        columns.
    """
    ranking = rank_security_period_evidence(
        findings,
        include_suppressed=include_suppressed,
    )
    if ranking.is_empty():
        return _empty_portfolio_period_contribution_candidates()

    ranked_rows = list(ranking.iter_rows(named=True))
    denominators = _security_return_denominators(ranked_rows)
    rows = [
        _security_return_candidate_row(
            row,
            denominators.get(_security_return_candidate_key(row)),
        )
        for row in ranked_rows
    ]
    return pl.DataFrame(rows, infer_schema_length=None).select(
        PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS
    )


def top_evidence_table(
    findings: pl.DataFrame,
    top_evidence_limit: int,
    *,
    _candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return top contribution-candidate rows per portfolio period.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        top_evidence_limit: Maximum number of ranked evidence rows to return per
            portfolio period.

    Returns:
        Ranked contribution-candidate rows limited within each portfolio period.
    """
    candidates = (
        portfolio_period_contribution_candidates(findings) if _candidates is None else _candidates
    )
    if candidates.is_empty():
        return candidates

    # REVIEW_RANK is assigned within each portfolio period when candidates are
    # constructed, so this single filter is equivalent to sorting and taking
    # the head of every group separately.
    return candidates.filter(pl.col(REVIEW_RANK) <= top_evidence_limit).select(
        PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS
    )


def security_top_evidence_table(
    findings: pl.DataFrame,
    top_evidence_limit: int,
    *,
    _candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return top contribution-candidate rows per security period.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        top_evidence_limit: Maximum number of ranked evidence rows to return per
            security period.

    Returns:
        Ranked contribution-candidate rows limited within each security period.
    """
    candidates = (
        security_period_contribution_candidates(findings) if _candidates is None else _candidates
    )
    if candidates.is_empty():
        return candidates

    # REVIEW_RANK is assigned within each security period when candidates are
    # constructed, so this single filter replaces one eager sort per group.
    return candidates.filter(pl.col(REVIEW_RANK) <= top_evidence_limit).select(
        PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS
    )


def portfolio_period_cause_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
    _candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return cause-area summaries for portfolio-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            contribution candidates and cause-area summaries.

    Returns:
        One row per portfolio period and coarse root-cause area. The summary
        rolls up contribution candidates and intentionally leaves most impact
        estimates blank until defensible calculation rules exist.
    """
    candidates = (
        portfolio_period_contribution_candidates(
            findings,
            include_suppressed=include_suppressed,
        )
        if _candidates is None
        else _candidates
    )
    if candidates.is_empty():
        return _empty_portfolio_period_cause_summary()

    buckets: dict[tuple[object, object, object, str], list[dict[str, object]]] = {}
    for row in candidates.iter_rows(named=True):
        if _field_roles.is_reported_performance_component(
            row.get(DATASET),
            row.get(SOURCE_COLUMN),
        ):
            continue
        cause_area = _root_cause_area(row)
        key = (row[PORTFOLIO_ID], row[FROM_DATE], row[THRU_DATE], cause_area)
        buckets.setdefault(key, []).append(row)
    if not buckets:
        return _empty_portfolio_period_cause_summary()

    rows = [
        _portfolio_period_cause_summary_row(key, bucket_rows)
        for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_portfolio_period_cause_summary_sort_key)
    return pl.DataFrame(sorted_rows).select(PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS)


def security_period_cause_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
    _candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return cause-area summaries for security-period return deltas.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            contribution candidates and cause-area summaries.

    Returns:
        One row per portfolio/security/period and coarse root-cause area.
    """
    candidates = (
        security_period_contribution_candidates(
            findings,
            include_suppressed=include_suppressed,
        )
        if _candidates is None
        else _candidates
    )
    if candidates.is_empty():
        return _empty_security_period_cause_summary()

    buckets: dict[tuple[object, object, object, object, str], list[dict[str, object]]] = {}
    for row in candidates.iter_rows(named=True):
        if _field_roles.is_reported_performance_component(
            row.get(DATASET),
            row.get(SOURCE_COLUMN),
        ):
            continue
        cause_area = _root_cause_area(row)
        key = (
            row[PORTFOLIO_ID],
            row[SECURITY_ID],
            row[FROM_DATE],
            row[THRU_DATE],
            cause_area,
        )
        buckets.setdefault(key, []).append(row)
    if not buckets:
        return _empty_security_period_cause_summary()

    rows = [
        _security_period_cause_summary_row(key, bucket_rows)
        for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_security_period_cause_summary_sort_key)
    return pl.DataFrame(sorted_rows).select(SECURITY_PERIOD_CAUSE_SUMMARY_COLUMNS)


def portfolio_period_impact_coverage_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
    _candidates: pl.DataFrame | None = None,
    _periods: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return estimate-coverage status for each changed portfolio period.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            the underlying portfolio-period, cause-area, and transaction
            summaries.
        _candidates: Optional precomputed contribution candidates used by
            internal report-table caches.
        _periods: Optional precomputed portfolio-period summary used by
            internal report-table caches.

    Returns:
        One row per changed portfolio period. Counts are cause-area based, not
        finding-row based, because impact estimates are currently aggregated at
        the cause-area level.
    """
    periods = (
        portfolio_period_summary(
            findings,
            include_suppressed=include_suppressed,
        )
        if _periods is None
        else _periods
    )
    if periods.is_empty():
        return _empty_portfolio_period_impact_coverage_summary()

    causes = portfolio_period_cause_summary(
        findings,
        include_suppressed=include_suppressed,
        _candidates=_candidates,
    )
    transactions = transaction_activity_summary(
        findings,
        include_suppressed=include_suppressed,
    )
    causes_by_period = _summary_rows_by_period(causes)
    transactions_by_period = _summary_rows_by_period(transactions)
    rows = [
        _impact_coverage_summary_row(
            period,
            causes_by_period.get(_summary_period_key(period), []),
            transactions_by_period.get(_summary_period_key(period), []),
        )
        for period in periods.iter_rows(named=True)
    ]
    return pl.DataFrame(rows, infer_schema_length=None).select(
        PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS
    )


def portfolio_period_transaction_cross_checks(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return portfolio-period transaction impact cross-check diagnostics.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed transaction findings should be
            included in the summary.

    Returns:
        One row per portfolio period and transaction impact policy group. The
        estimates are review-only cross-checks and are intentionally separate
        from contribution totals.
    """
    if findings.is_empty() or TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE not in findings.columns:
        return _empty_portfolio_period_transaction_cross_checks()

    active_findings = _active_findings(findings, include_suppressed)
    cross_check_findings = active_findings.filter(
        (pl.col(DATASET) == pc_cols.TRANSACTIONS)
        & pl.col(TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE).is_not_null()
    )
    if cross_check_findings.is_empty():
        return _empty_portfolio_period_transaction_cross_checks()

    buckets: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in cross_check_findings.iter_rows(named=True):
        key = (
            row[PORTFOLIO_ID],
            row[FROM_DATE],
            row[THRU_DATE],
            row[TRANSACTION_IMPACT_POLICY],
        )
        buckets.setdefault(key, []).append(row)

    rows = [
        _portfolio_period_transaction_cross_check_row(key, bucket_rows)
        for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_portfolio_period_transaction_cross_check_sort_key)
    return pl.DataFrame(sorted_rows).select(PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS)


def transaction_activity_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return evidence-only summaries for changed transaction activity.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed transaction findings should be
            included in the summary.

    Returns:
        One row per portfolio, security, period, and normalized transaction
        category. Numeric source-field deltas are summed by field, but no
        return-impact estimate is produced until all transaction impact inputs
        are available and modeled.
    """
    if findings.is_empty() or TRANSACTION_CATEGORY not in findings.columns:
        return _empty_transaction_activity_summary()

    active_findings = _active_findings(findings, include_suppressed)
    transaction_findings = active_findings.filter(
        (pl.col(DATASET) == pc_cols.TRANSACTIONS)
        & pl.col(SOURCE_COLUMN).is_in(
            [
                pc_cols.AMOUNT,
                pc_cols.QUANTITY,
                pc_cols.PRICE,
            ]
        )
    )
    if transaction_findings.is_empty():
        return _empty_transaction_activity_summary()

    fallback_periods = _single_target_period_by_portfolio(active_findings)
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in transaction_findings.iter_rows(named=True):
        from_date, thru_date = _transaction_activity_period(row, fallback_periods)
        key = (
            row[PORTFOLIO_ID],
            row[SECURITY_ID],
            from_date,
            thru_date,
            row[TRANSACTION_CATEGORY] or TRANSACTION_CATEGORY_UNKNOWN,
        )
        buckets.setdefault(key, []).append(row)

    rows = [
        _transaction_activity_summary_row(key, bucket_rows) for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_transaction_activity_summary_sort_key)
    return pl.DataFrame(sorted_rows).select(TRANSACTION_ACTIVITY_SUMMARY_COLUMNS)


def transaction_matching_diagnostics(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return transaction matching status counts with reviewer notes.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed transaction findings should be
            included in the diagnostic counts.

    Returns:
        One row per transaction match-status label. The helper only reports how
        transaction rows were paired or left unpaired; it does not change
        matching behavior or infer edits from strict fallback keys.
    """
    if findings.is_empty() or TRANSACTION_MATCH_STATUS not in findings.columns:
        return _empty_transaction_matching_diagnostics()

    transaction_findings = _active_findings(findings, include_suppressed).filter(
        (pl.col(DATASET) == pc_cols.TRANSACTIONS)
        & pl.col(TRANSACTION_MATCH_STATUS).is_not_null()
        & (pl.col(TRANSACTION_MATCH_STATUS).cast(pl.String).str.len_chars() > 0)
    )
    if transaction_findings.is_empty():
        return _empty_transaction_matching_diagnostics()

    rows = []
    for row in (
        transaction_findings.group_by(TRANSACTION_MATCH_STATUS)
        .len(name=FINDING_COUNT)
        .iter_rows(named=True)
    ):
        match_status = row[TRANSACTION_MATCH_STATUS]
        rows.append(
            {
                TRANSACTION_MATCH_STATUS: match_status,
                FINDING_COUNT: row[FINDING_COUNT],
                TRANSACTION_MATCH_CONFIDENCE: (
                    tx_diagnostics.transaction_match_confidence(match_status)
                ),
                TRANSACTION_MATCH_INTERPRETATION: (
                    tx_diagnostics.transaction_match_interpretation(match_status)
                ),
                TRANSACTION_MATCH_REVIEW_NOTE: (
                    tx_diagnostics.transaction_match_review_note(match_status)
                ),
            }
        )
    sorted_rows = sorted(
        rows,
        key=tx_diagnostics.transaction_matching_diagnostic_sort_key,
    )
    return pl.DataFrame(sorted_rows).select(TRANSACTION_MATCHING_DIAGNOSTIC_COLUMNS)


def _transaction_activity_period(
    row: dict[str, object],
    fallback_periods: dict[object, tuple[object, object]],
) -> tuple[object | None, object | None]:
    """Return the period context to use for transaction activity grouping."""
    if row[FROM_DATE] is not None or row[THRU_DATE] is not None:
        return row[FROM_DATE], row[THRU_DATE]
    return fallback_periods.get(row[PORTFOLIO_ID], (None, None))


def _single_target_period_by_portfolio(
    findings: pl.DataFrame,
) -> dict[object, tuple[object, object]]:
    """Return unambiguous changed portfolio-return periods keyed by portfolio."""
    target_findings = findings.filter(
        (pl.col(FINDING_CODE) == PC_PORT_RET)
        & (pl.col(DATASET) == pc_cols.PORTFOLIO_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.PORTFOLIO_RETURN)
    )
    if target_findings.is_empty():
        return {}

    periods_by_portfolio: dict[object, set[tuple[object, object]]] = {}
    for row in target_findings.iter_rows(named=True):
        periods_by_portfolio.setdefault(row[PORTFOLIO_ID], set()).add(
            (row[FROM_DATE], row[THRU_DATE])
        )

    return {
        portfolio_id: next(iter(periods))
        for portfolio_id, periods in periods_by_portfolio.items()
        if len(periods) == 1
    }


def _active_findings(findings: pl.DataFrame, include_suppressed: bool) -> pl.DataFrame:
    """Return findings that should count as active evidence."""
    if include_suppressed:
        return findings
    return findings.filter(~pl.col(SUPPRESSED))


@dataclass
class _RelatedFindingCounts:
    """Mutable count accumulator for one indexed evidence relationship key."""

    total: int = 0
    roles: Counter[object] = field(default_factory=Counter)
    datasets: Counter[object] = field(default_factory=Counter)

    def add(self, evidence_role: object, dataset: object) -> None:
        """Add one related finding without materializing a DataFrame slice."""
        self.total += 1
        self.roles[evidence_role] += 1
        self.datasets[dataset] += 1

    def merge(self, other: _RelatedFindingCounts) -> None:
        """Merge one indexed key's counts into this accumulator."""
        self.total += other.total
        self.roles.update(other.roles)
        self.datasets.update(other.datasets)


def _related_finding_count_index(
    findings: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> dict[tuple[object, ...], _RelatedFindingCounts]:
    """Index summary counts once without repeatedly gathering DataFrame rows."""
    if findings.is_empty():
        return {}
    index: dict[tuple[object, ...], _RelatedFindingCounts] = {}
    columns = findings.select(*key_columns, EVIDENCE_ROLE, DATASET)
    for row in columns.iter_rows(named=True):
        key = tuple(row[column] for column in key_columns)
        counts = index.setdefault(key, _RelatedFindingCounts())
        counts.add(row[EVIDENCE_ROLE], row[DATASET])
    return index


def _suppressed_finding_keys(
    findings: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> set[tuple[object, ...]]:
    """Return relationship keys containing at least one suppressed finding."""
    if findings.is_empty() or SUPPRESSED not in findings.columns:
        return set()
    suppressed = findings.filter(pl.col(SUPPRESSED)).select(*key_columns)
    return set(suppressed.iter_rows())


def _combined_related_finding_counts(
    index: Mapping[tuple[object, ...], _RelatedFindingCounts],
    related_keys: tuple[tuple[object, ...], ...],
) -> _RelatedFindingCounts:
    """Return combined counts for every relationship key of one target row."""
    combined = _RelatedFindingCounts()
    for key in related_keys:
        counts = index.get(key)
        if counts is not None:
            combined.merge(counts)
    return combined


def _portfolio_related_keys(
    target: Mapping[str, object],
) -> tuple[tuple[object, ...], ...]:
    """Return exact-period and undated portfolio evidence keys."""
    portfolio_id = target[PORTFOLIO_ID]
    return (
        (portfolio_id, target[FROM_DATE], target[THRU_DATE]),
        (portfolio_id, None, None),
    )


def _security_related_keys(
    target: Mapping[str, object],
) -> tuple[tuple[object, ...], ...]:
    """Return exact and portfolio-optional security evidence keys."""
    security_id = target[SECURITY_ID]
    portfolio_id = target[PORTFOLIO_ID]
    from_date = target[FROM_DATE]
    thru_date = target[THRU_DATE]
    return (
        (security_id, portfolio_id, from_date, thru_date),
        (security_id, None, from_date, thru_date),
        (security_id, portfolio_id, None, None),
        (security_id, None, None, None),
    )


def _has_suppressed_findings(findings: pl.DataFrame) -> bool:
    """Return whether a related finding set includes suppressed rows."""
    if findings.is_empty() or SUPPRESSED not in findings.columns:
        return False
    return bool(findings.get_column(SUPPRESSED).any())


def _role_summary_counts(findings: pl.DataFrame) -> dict[str, int]:
    """Return standard role count fields for summary tables."""
    counts = _column_counts(findings, EVIDENCE_ROLE)
    return {
        DIRECT_INPUT_FINDING_COUNT: counts.get(DIRECT_INPUT, 0),
        RELATED_OUTPUT_FINDING_COUNT: counts.get(RELATED_OUTPUT, 0),
        CONTEXT_FINDING_COUNT: counts.get(CONTEXT, 0),
    }


def _role_summary_counts_from_counter(
    counts: Mapping[object, int],
) -> dict[str, int]:
    """Return standard role fields from pre-indexed evidence counts."""
    return {
        DIRECT_INPUT_FINDING_COUNT: counts.get(DIRECT_INPUT, 0),
        RELATED_OUTPUT_FINDING_COUNT: counts.get(RELATED_OUTPUT, 0),
        CONTEXT_FINDING_COUNT: counts.get(CONTEXT, 0),
    }


def _column_counts(findings: pl.DataFrame, column: str) -> dict[object, int]:
    """Return value counts without launching another dataframe query."""
    if findings.is_empty():
        return {}
    return dict(Counter(findings.get_column(column).to_list()))


def _portfolio_period_index(
    findings: pl.DataFrame,
) -> dict[tuple[object, ...], list[int]]:
    """Index finding row numbers once for portfolio-period lookups."""
    if findings.is_empty():
        return {}
    index: dict[tuple[object, ...], list[int]] = {}
    columns = findings.select(PORTFOLIO_ID, FROM_DATE, THRU_DATE)
    for row_number, key in enumerate(columns.iter_rows()):
        index.setdefault(key, []).append(row_number)
    return index


def _finding_row_index(
    findings: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> dict[tuple[object, ...], list[tuple[int, dict[str, object]]]]:
    """Index complete finding rows once for repeated evidence ranking lookups."""
    index: dict[
        tuple[object, ...],
        list[tuple[int, dict[str, object]]],
    ] = {}
    for row_number, row in enumerate(findings.iter_rows(named=True)):
        key = tuple(row[column] for column in key_columns)
        index.setdefault(key, []).append((row_number, row))
    return index


def _indexed_finding_rows(
    index: dict[tuple[object, ...], list[tuple[int, dict[str, object]]]],
    keys: tuple[tuple[object, ...], ...],
) -> list[dict[str, object]]:
    """Return indexed rows in their original finding-table order."""
    indexed_rows = [indexed_row for key in keys for indexed_row in index.get(key, ())]
    indexed_rows.sort(key=lambda indexed_row: indexed_row[0])
    return [row for _, row in indexed_rows]


def _indexed_portfolio_period_findings(
    findings: pl.DataFrame,
    index: dict[tuple[object, ...], list[int]],
    target: dict[str, object],
) -> pl.DataFrame:
    """Return exact-period and undated findings from a partitioned index."""
    portfolio_id = target[PORTFOLIO_ID]
    keys = (
        (portfolio_id, target[FROM_DATE], target[THRU_DATE]),
        (portfolio_id, None, None),
    )
    row_numbers = sorted(row_number for key in keys for row_number in index.get(key, ()))
    if not row_numbers:
        return findings.clear()
    return findings[row_numbers]


def _security_period_index(
    findings: pl.DataFrame,
) -> dict[tuple[object, ...], list[int]]:
    """Index finding row numbers once for security-period lookups."""
    if findings.is_empty():
        return {}
    index: dict[tuple[object, ...], list[int]] = {}
    columns = findings.select(SECURITY_ID, PORTFOLIO_ID, FROM_DATE, THRU_DATE)
    for row_number, key in enumerate(columns.iter_rows()):
        index.setdefault(key, []).append(row_number)
    return index


def _indexed_security_period_findings(
    findings: pl.DataFrame,
    index: dict[tuple[object, ...], list[int]],
    target: dict[str, object],
) -> pl.DataFrame:
    """Return related security findings from a partitioned index."""
    security_id = target[SECURITY_ID]
    portfolio_id = target[PORTFOLIO_ID]
    from_date = target[FROM_DATE]
    thru_date = target[THRU_DATE]
    keys = (
        (security_id, portfolio_id, from_date, thru_date),
        (security_id, None, from_date, thru_date),
        (security_id, portfolio_id, None, None),
        (security_id, None, None, None),
    )
    row_numbers = sorted(row_number for key in keys for row_number in index.get(key, ()))
    if not row_numbers:
        return findings.clear()
    return findings[row_numbers]


def _related_portfolio_period_findings(
    findings: pl.DataFrame,
    target: dict[str, object],
) -> pl.DataFrame:
    """Return findings related to a portfolio-period target."""
    portfolio_id = target[PORTFOLIO_ID]
    from_date = target[FROM_DATE]
    thru_date = target[THRU_DATE]
    return findings.filter(
        (pl.col(PORTFOLIO_ID) == portfolio_id)
        & (
            ((pl.col(FROM_DATE) == from_date) & (pl.col(THRU_DATE) == thru_date))
            | (pl.col(FROM_DATE).is_null() & pl.col(THRU_DATE).is_null())
        )
    )


def _related_security_period_findings(
    findings: pl.DataFrame,
    target: dict[str, object],
) -> pl.DataFrame:
    """Return findings related to a security-period target."""
    portfolio_id = target[PORTFOLIO_ID]
    security_id = target[SECURITY_ID]
    from_date = target[FROM_DATE]
    thru_date = target[THRU_DATE]
    return findings.filter(
        (pl.col(SECURITY_ID) == security_id)
        & ((pl.col(PORTFOLIO_ID) == portfolio_id) | pl.col(PORTFOLIO_ID).is_null())
        & (
            ((pl.col(FROM_DATE) == from_date) & (pl.col(THRU_DATE) == thru_date))
            | (pl.col(FROM_DATE).is_null() & pl.col(THRU_DATE).is_null())
        )
    )


def _dataset_count(findings: pl.DataFrame, dataset: str) -> int:
    """Return the number of findings for a normalized dataset."""
    if findings.is_empty():
        return 0
    return int(findings.filter(pl.col(DATASET) == dataset).height)


def _direct_input_count(findings: pl.DataFrame) -> int:
    """Return findings that are plausible direct performance inputs."""
    return _role_count(findings, DIRECT_INPUT)


def _context_count(findings: pl.DataFrame) -> int:
    """Return findings that provide investigation context."""
    return _role_count(findings, CONTEXT)


def _role_count(findings: pl.DataFrame, evidence_role: str) -> int:
    """Return the number of findings for an evidence role."""
    if findings.is_empty():
        return 0
    return int(findings.filter(pl.col(EVIDENCE_ROLE) == evidence_role).height)


def _role_dataset_count(
    findings: pl.DataFrame,
    evidence_role: str,
    dataset: str,
) -> int:
    """Return the number of findings for an evidence role and dataset."""
    if findings.is_empty():
        return 0
    return int(
        findings.filter(
            (pl.col(EVIDENCE_ROLE) == evidence_role) & (pl.col(DATASET) == dataset)
        ).height
    )


def _evidence_breakdown_rows(
    target: dict[str, object],
    related_findings: pl.DataFrame,
) -> list[dict[str, object]]:
    """Return role total and nonzero dataset rows for one portfolio period."""
    evidence_counts = [
        (TARGET_OUTPUT, None, _target_output_count(related_findings)),
        (DIRECT_INPUT, None, _direct_input_count(related_findings)),
        (RELATED_OUTPUT, None, _role_count(related_findings, RELATED_OUTPUT)),
        (CONTEXT, None, _context_count(related_findings)),
        (
            TARGET_OUTPUT,
            pc_cols.PORTFOLIO_PERFORMANCE,
            _role_dataset_count(
                related_findings,
                TARGET_OUTPUT,
                pc_cols.PORTFOLIO_PERFORMANCE,
            ),
        ),
        (
            DIRECT_INPUT,
            pc_cols.PORTFOLIO_PERFORMANCE,
            _role_dataset_count(
                related_findings,
                DIRECT_INPUT,
                pc_cols.PORTFOLIO_PERFORMANCE,
            ),
        ),
        (
            DIRECT_INPUT,
            pc_cols.FX_RATES,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.FX_RATES),
        ),
        (
            DIRECT_INPUT,
            pc_cols.TRANSACTIONS,
            _role_dataset_count(
                related_findings,
                DIRECT_INPUT,
                pc_cols.TRANSACTIONS,
            ),
        ),
        (
            DIRECT_INPUT,
            pc_cols.HOLDINGS,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.HOLDINGS),
        ),
        (
            RELATED_OUTPUT,
            pc_cols.SECURITY_PERFORMANCE,
            _role_dataset_count(
                related_findings,
                RELATED_OUTPUT,
                pc_cols.SECURITY_PERFORMANCE,
            ),
        ),
    ]

    rows: list[dict[str, object]] = []
    for evidence_group, dataset, finding_count in evidence_counts:
        if dataset is None or finding_count > 0:
            rows.append(
                _evidence_breakdown_row(
                    target,
                    evidence_group,
                    dataset,
                    finding_count,
                )
            )
    return rows


def _security_evidence_breakdown_rows(
    target: dict[str, object],
    related_findings: pl.DataFrame,
) -> list[dict[str, object]]:
    """Return role total and nonzero dataset rows for one security period."""
    target_findings = _security_target_findings(related_findings)
    related_output_findings = related_findings.filter(
        (pl.col(EVIDENCE_ROLE) == RELATED_OUTPUT)
        & ~(
            (pl.col(FINDING_CODE) == PC_SEC_RET)
            & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
        )
    )
    evidence_counts = [
        (TARGET_OUTPUT, None, target_findings.height),
        (DIRECT_INPUT, None, _direct_input_count(related_findings)),
        (RELATED_OUTPUT, None, related_output_findings.height),
        (CONTEXT, None, _context_count(related_findings)),
        (
            TARGET_OUTPUT,
            pc_cols.SECURITY_PERFORMANCE,
            _dataset_count(target_findings, pc_cols.SECURITY_PERFORMANCE),
        ),
        (
            DIRECT_INPUT,
            pc_cols.TRANSACTIONS,
            _role_dataset_count(
                related_findings,
                DIRECT_INPUT,
                pc_cols.TRANSACTIONS,
            ),
        ),
        (
            DIRECT_INPUT,
            pc_cols.HOLDINGS,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.HOLDINGS),
        ),
        (
            RELATED_OUTPUT,
            pc_cols.SECURITY_PERFORMANCE,
            _dataset_count(related_output_findings, pc_cols.SECURITY_PERFORMANCE),
        ),
    ]

    rows: list[dict[str, object]] = []
    for evidence_group, dataset, finding_count in evidence_counts:
        if dataset is None or finding_count > 0:
            rows.append(
                _security_evidence_breakdown_row(
                    target,
                    evidence_group,
                    dataset,
                    finding_count,
                )
            )
    return rows


def _security_target_findings(findings: pl.DataFrame) -> pl.DataFrame:
    """Return security return target findings within a related finding set."""
    if findings.is_empty():
        return findings
    return findings.filter(
        (pl.col(FINDING_CODE) == PC_SEC_RET)
        & (pl.col(DATASET) == pc_cols.SECURITY_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
    )


def _target_output_count(findings: pl.DataFrame, dataset: str | None = None) -> int:
    """Return portfolio return target findings."""
    del dataset
    return _role_count(findings, TARGET_OUTPUT)


def _evidence_breakdown_row(
    target: dict[str, object],
    evidence_group: str,
    dataset: str | None,
    finding_count: int,
) -> dict[str, object]:
    """Return one portfolio-period evidence breakdown row."""
    return {
        PORTFOLIO_ID: target[PORTFOLIO_ID],
        FROM_DATE: target[FROM_DATE],
        THRU_DATE: target[THRU_DATE],
        EVIDENCE_GROUP: evidence_group,
        DATASET: dataset,
        FINDING_COUNT: finding_count,
    }


def _security_evidence_breakdown_row(
    target: dict[str, object],
    evidence_group: str,
    dataset: str | None,
    finding_count: int,
) -> dict[str, object]:
    """Return one security-period evidence breakdown row."""
    return {
        PORTFOLIO_ID: target[PORTFOLIO_ID],
        SECURITY_ID: target[SECURITY_ID],
        FROM_DATE: target[FROM_DATE],
        THRU_DATE: target[THRU_DATE],
        EVIDENCE_GROUP: evidence_group,
        DATASET: dataset,
        FINDING_COUNT: finding_count,
    }


def _ranked_evidence_row(
    target: dict[str, object],
    finding: dict[str, object],
) -> dict[str, object]:
    """Return one portfolio-period evidence ranking row before rank numbering."""
    delta = finding[DELTA_B_MINUS_A]
    absolute_delta = _absolute_numeric_delta(delta)
    score, reason = _priority_score_and_reason(finding, absolute_delta)
    return {
        PORTFOLIO_ID: target[PORTFOLIO_ID],
        FROM_DATE: target[FROM_DATE],
        THRU_DATE: target[THRU_DATE],
        REVIEW_RANK: 0,
        PRIORITY_SCORE: score,
        PRIORITY_REASON: reason,
        FINDING_CODE: finding[FINDING_CODE],
        DATASET: finding[DATASET],
        EVIDENCE_ROLE: finding[EVIDENCE_ROLE],
        SECURITY_ID: finding[SECURITY_ID],
        SOURCE_FILE: finding[SOURCE_FILE],
        SOURCE_RECORD_LOCATOR: finding[SOURCE_RECORD_LOCATOR],
        INPUT_DATE: finding[INPUT_DATE],
        SOURCE_COLUMN: finding[SOURCE_COLUMN],
        FROM_CURRENCY: finding[FROM_CURRENCY],
        TO_CURRENCY: finding[TO_CURRENCY],
        TRANSACTION_CODE: finding[TRANSACTION_CODE],
        TRANSACTION_CATEGORY: finding[TRANSACTION_CATEGORY],
        CASH_FLOW_SIGN: finding[CASH_FLOW_SIGN],
        PERFORMANCE_FLOW_SIGN: finding[PERFORMANCE_FLOW_SIGN],
        TRANSACTION_SEMANTICS_SOURCE: finding[TRANSACTION_SEMANTICS_SOURCE],
        IMPACT_POLICY: finding[IMPACT_POLICY],
        TRANSACTION_IMPACT_POLICY: finding[TRANSACTION_IMPACT_POLICY],
        TRANSACTION_IMPACT_DIAGNOSTIC: finding[TRANSACTION_IMPACT_DIAGNOSTIC],
        TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE: (finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
        DELTA_B_MINUS_A: delta,
        SNAPSHOT_A_VALUE: finding[SNAPSHOT_A_VALUE],
        SNAPSHOT_B_VALUE: finding[SNAPSHOT_B_VALUE],
        RETURN_DENOMINATOR: finding[RETURN_DENOMINATOR],
        RETURN_WEIGHT: finding[RETURN_WEIGHT],
        IMPACT_INPUT_VALUE: finding[IMPACT_INPUT_VALUE],
        ABSOLUTE_DELTA: absolute_delta,
        MESSAGE: finding[MESSAGE],
    }


def _contribution_candidate_row(row: dict[str, object]) -> dict[str, object]:
    """Return one contribution candidate row from a ranked evidence row."""
    impact = _estimated_impact(row)
    return {
        **row,
        ESTIMATED_RETURN_IMPACT: impact[ESTIMATED_RETURN_IMPACT],
        IMPACT_BASIS: impact[IMPACT_BASIS],
        IMPACT_CONFIDENCE: impact[IMPACT_CONFIDENCE],
        IMPACT_METHOD: impact[IMPACT_METHOD],
        IMPACT_MESSAGE: impact[IMPACT_MESSAGE],
    }


def _security_return_candidate_row(
    row: dict[str, object],
    security_denominator: float | None,
) -> dict[str, object]:
    """Return one security-return candidate row from ranked security evidence."""
    impact = _estimated_security_return_impact(row, security_denominator)
    return {
        **row,
        ESTIMATED_RETURN_IMPACT: impact[ESTIMATED_RETURN_IMPACT],
        IMPACT_BASIS: impact[IMPACT_BASIS],
        IMPACT_CONFIDENCE: impact[IMPACT_CONFIDENCE],
        IMPACT_METHOD: impact[IMPACT_METHOD],
        IMPACT_MESSAGE: impact[IMPACT_MESSAGE],
    }


def _security_return_candidate_key(row: dict[str, object]) -> tuple[object, ...]:
    """Return the security-period key used for security-return estimates."""
    return (
        row.get(PORTFOLIO_ID),
        row.get(SECURITY_ID),
        row.get(FROM_DATE),
        row.get(THRU_DATE),
    )


def _security_return_denominators(
    rows: Iterable[dict[str, object]],
) -> dict[tuple[object, ...], float]:
    """Return reconstruction-derived denominators keyed by security period."""
    denominators: dict[tuple[object, ...], float] = {}
    for row in rows:
        denominator = _number_value(row.get(RETURN_DENOMINATOR))
        if denominator is None or denominator == 0.0:
            continue
        denominators[_security_return_candidate_key(row)] = denominator
    return denominators


def _estimated_security_return_impact(
    row: dict[str, object],
    security_denominator: float | None,
) -> dict[str, object]:
    """Return security-return impact fields for one ranked evidence row."""
    delta = _number_value(row[DELTA_B_MINUS_A])
    if delta is None:
        return _no_security_return_estimate()

    if security_denominator is None:
        return _no_security_return_estimate()

    if row[DATASET] == pc_cols.HOLDINGS and row[SOURCE_COLUMN] == pc_cols.MARKET_VALUE:
        return {
            ESTIMATED_RETURN_IMPACT: delta / security_denominator,
            IMPACT_BASIS: IMPACT_BASIS_SECURITY_HOLDING_MARKET_VALUE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_HOLDING_MARKET_VALUE_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: (
                "Approximate security-return impact uses the holding market value "
                "delta divided by snapshot A security market value."
            ),
        }

    if (
        row[DATASET] == pc_cols.HOLDINGS
        and row[SOURCE_COLUMN] in {pc_cols.ACCRUED, pc_cols.BASE_ACCRUED}
    ):
        return {
            ESTIMATED_RETURN_IMPACT: delta / security_denominator,
            IMPACT_BASIS: IMPACT_BASIS_SECURITY_HOLDING_ACCRUED,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_HOLDING_ACCRUED_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: (
                "Approximate security-return impact uses the holding accrued delta "
                "divided by snapshot A security market value."
            ),
        }

    if row[DATASET] == pc_cols.HOLDINGS and row[SOURCE_COLUMN] == pc_cols.QUANTITY:
        unit_market_value = _number_value(row[IMPACT_INPUT_VALUE])
        if unit_market_value is None:
            return _no_security_return_estimate()
        return {
            ESTIMATED_RETURN_IMPACT: (delta * unit_market_value) / security_denominator,
            IMPACT_BASIS: IMPACT_BASIS_SECURITY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: (IMPACT_METHOD_HOLDING_QUANTITY_UNIT_MARKET_VALUE_OVER_DENOMINATOR),
            IMPACT_MESSAGE: (
                "Approximate security-return impact uses the holding quantity delta "
                "multiplied by snapshot A unit market value, then divided by "
                "snapshot A security market value."
            ),
        }

    if _is_transaction_performance_amount_impact_candidate(row):
        if row.get(TRANSACTION_IMPACT_POLICY) == (
            TRANSACTION_IMPACT_POLICY_SECURITY_FLOW_MODIFIED_DIETZ
        ):
            return _estimated_security_transaction_flow_impact(
                row,
                delta,
                security_denominator,
            )
        if row.get(TRANSACTION_CATEGORY) != TRANSACTION_CATEGORY_INCOME:
            return _no_security_return_estimate()
        return {
            ESTIMATED_RETURN_IMPACT: delta / security_denominator,
            IMPACT_BASIS: IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: (
                "Approximate security-return impact uses the source-signed "
                "transaction amount delta divided by snapshot A security market "
                "value."
            ),
        }

    return _estimated_impact(row)


def _estimated_security_transaction_flow_impact(
    row: dict[str, object],
    delta: float,
    security_denominator: float,
) -> dict[str, object]:
    """Return a security-return estimate for changed buy/sell cash flow."""
    category = row.get(TRANSACTION_CATEGORY)
    if category not in {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}:
        return _no_security_return_estimate()

    from_date = _date_value(row.get(FROM_DATE))
    thru_date = _date_value(row.get(THRU_DATE))
    flow_date = _date_value(row.get(INPUT_DATE))
    if from_date is None or thru_date is None or flow_date is None:
        return _no_security_return_estimate()

    try:
        flow_weight = _modified_dietz_flow_weight(
            from_date=from_date,
            thru_date=thru_date,
            flow_date=flow_date,
            inclusion_rule="beginning_of_day",
        )
    except ValueError:
        return _no_security_return_estimate()

    security_flow_delta = -delta
    return {
        ESTIMATED_RETURN_IMPACT: (security_flow_delta * flow_weight) / security_denominator,
        IMPACT_BASIS: IMPACT_BASIS_SECURITY_TRANSACTION_FLOW,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_METHOD: IMPACT_METHOD_SECURITY_TRANSACTION_FLOW_MODIFIED_DIETZ,
        IMPACT_MESSAGE: (
            "Approximate security-return impact treats the changed buy/sell "
            "transaction amount as a security-level Modified Dietz flow."
        ),
    }


def _date_value(value: object) -> dt.date | None:
    """Return a date value from a date or datetime object."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _no_security_return_estimate() -> dict[str, object]:
    """Return an empty estimate for unsupported security-return evidence."""
    return {
        ESTIMATED_RETURN_IMPACT: None,
        IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_METHOD: None,
        IMPACT_MESSAGE: (
            "No defensible security-return impact estimate is available for this " "finding yet."
        ),
    }


def _estimated_impact(row: dict[str, object]) -> dict[str, object]:
    """Return the first-pass contribution-impact fields for one evidence row."""
    delta = row[DELTA_B_MINUS_A]
    if _is_fx_rate_local_exposure_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        exposure = _required_impact_number(row[IMPACT_INPUT_VALUE], IMPACT_INPUT_VALUE)
        denominator = _required_impact_number(row[RETURN_DENOMINATOR], RETURN_DENOMINATOR)
        return {
            ESTIMATED_RETURN_IMPACT: delta_float * exposure / denominator,
            IMPACT_BASIS: IMPACT_BASIS_FX_RATE_LOCAL_EXPOSURE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: (IMPACT_METHOD_FX_RATE_DELTA_TIMES_LOCAL_EXPOSURE_OVER_DENOMINATOR),
            IMPACT_MESSAGE: (
                "Approximate impact uses the FX-rate delta multiplied by the "
                "unchanged snapshot A local-currency exposure, divided by the "
                "portfolio return denominator. This is a normalized ppar "
                "screening estimate, not an assertion about Axys calculation "
                "mechanics."
            ),
        }
    if _is_transaction_performance_amount_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        denominator = _required_impact_number(row[RETURN_DENOMINATOR], RETURN_DENOMINATOR)
        return {
            ESTIMATED_RETURN_IMPACT: delta_float / denominator,
            IMPACT_BASIS: IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: _transaction_performance_amount_impact_message(row),
        }
    if _is_holding_market_value_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        denominator = _required_impact_number(row[RETURN_DENOMINATOR], RETURN_DENOMINATOR)
        return {
            ESTIMATED_RETURN_IMPACT: delta_float / denominator,
            IMPACT_BASIS: IMPACT_BASIS_HOLDING_MARKET_VALUE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_HOLDING_MARKET_VALUE_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: (
                "Approximate impact uses the holding market value delta "
                "divided by the return denominator. Treat as a low-confidence "
                "screening estimate because market value can reflect price, "
                "quantity, FX, accrued-interest, or booking changes."
            ),
        }
    if _is_holding_accrued_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        denominator = _required_impact_number(row[RETURN_DENOMINATOR], RETURN_DENOMINATOR)
        return {
            ESTIMATED_RETURN_IMPACT: delta_float / denominator,
            IMPACT_BASIS: IMPACT_BASIS_HOLDING_ACCRUED,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_HOLDING_ACCRUED_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: (
                "Approximate impact uses the holding accrued delta divided "
                "by the return denominator. Treat as a low-confidence "
                "screening estimate because accrued balances depend on source "
                "income accrual and pricing conventions."
            ),
        }
    if _is_holding_quantity_unit_market_value_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        denominator = _required_impact_number(row[RETURN_DENOMINATOR], RETURN_DENOMINATOR)
        unit_market_value = _required_impact_number(
            row[IMPACT_INPUT_VALUE], IMPACT_INPUT_VALUE
        )
        return {
            ESTIMATED_RETURN_IMPACT: (delta_float * unit_market_value) / denominator,
            IMPACT_BASIS: IMPACT_BASIS_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: (IMPACT_METHOD_HOLDING_QUANTITY_UNIT_MARKET_VALUE_OVER_DENOMINATOR),
            IMPACT_MESSAGE: (
                "Approximate impact uses the holding quantity delta multiplied "
                "by snapshot A unit market value, then divided by the return "
                "denominator. Treat as a low-confidence screening estimate "
                "because the unit value may embed price, FX, accrual, or "
                "classification effects."
            ),
        }
    if _is_price_weighted_impact_candidate(row):
        delta_float = _required_impact_number(delta, DELTA_B_MINUS_A)
        snapshot_a_price = _required_impact_number(
            row[SNAPSHOT_A_VALUE], SNAPSHOT_A_VALUE
        )
        weight = _required_impact_number(row[RETURN_WEIGHT], RETURN_WEIGHT)
        return {
            ESTIMATED_RETURN_IMPACT: (delta_float / snapshot_a_price) * weight,
            IMPACT_BASIS: IMPACT_BASIS_PRICE_WEIGHTED,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT,
            IMPACT_MESSAGE: (
                "Approximate impact uses the price delta divided by snapshot A "
                "price, multiplied by snapshot A security weight. Treat as a "
                "low-confidence screening estimate because holdings, FX, and "
                "accrual treatment may also affect market value."
            ),
        }
    if _has_evidence_only_impact_policy(row):
        return {
            ESTIMATED_RETURN_IMPACT: None,
            IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: None,
            IMPACT_MESSAGE: (
                "Configured as evidence-only in comparison YAML; this row is "
                "review evidence and does not receive an additive impact estimate."
            ),
        }
    return {
        ESTIMATED_RETURN_IMPACT: None,
        IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_METHOD: None,
        IMPACT_MESSAGE: (
            "No defensible return-impact estimate is available for this " "finding yet."
        ),
    }


def _is_transaction_performance_amount_impact_candidate(
    row: dict[str, object],
) -> bool:
    """Return whether a transaction amount row supports a performance estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    cash_flow_sign = row.get(CASH_FLOW_SIGN)
    return (
        row[DATASET] == pc_cols.TRANSACTIONS
        and row[SOURCE_COLUMN] in {pc_cols.AMOUNT, pc_cols.BASE_AMOUNT}
        and row.get(TRANSACTION_IMPACT_POLICY)
        in {
            TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            TRANSACTION_IMPACT_POLICY_SECURITY_FLOW_MODIFIED_DIETZ,
        }
        and row.get(PERFORMANCE_FLOW_SIGN) == TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
        and cash_flow_sign
        in {
            TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
        }
        and _number_value(delta) is not None
        and _number_value(denominator) not in (None, 0.0)
    )


def _is_fx_rate_local_exposure_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether an FX-rate row supports an exposure-linked estimate."""
    delta = _number_value(row.get(DELTA_B_MINUS_A))
    exposure = _number_value(row.get(IMPACT_INPUT_VALUE))
    denominator = _number_value(row.get(RETURN_DENOMINATOR))
    return (
        row.get(DATASET) == pc_cols.FX_RATES
        and row.get(SOURCE_COLUMN) == pc_cols.FX_RATE
        and row.get(IMPACT_POLICY) == IMPACT_POLICY_FX_RATE_EXPOSURE
        and delta is not None
        and exposure is not None
        and denominator is not None
        and denominator != 0.0
    )


def _is_holding_market_value_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether a holding market value row supports a rough estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    return (
        row[DATASET] == pc_cols.HOLDINGS
        and row[SOURCE_COLUMN] in {pc_cols.MARKET_VALUE, pc_cols.BASE_MARKET_VALUE}
        and row.get(IMPACT_POLICY) == IMPACT_POLICY_HOLDING_MARKET_VALUE
        and _number_value(delta) is not None
        and _number_value(denominator) not in (None, 0.0)
    )


def _is_holding_accrued_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether a holding accrued row supports a rough estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    return (
        row[DATASET] == pc_cols.HOLDINGS
        and row[SOURCE_COLUMN] in {pc_cols.ACCRUED, pc_cols.BASE_ACCRUED}
        and row.get(IMPACT_POLICY) == IMPACT_POLICY_HOLDING_ACCRUED
        and _number_value(delta) is not None
        and _number_value(denominator) not in (None, 0.0)
    )


def _is_holding_quantity_unit_market_value_impact_candidate(
    row: dict[str, object],
) -> bool:
    """Return whether a holding quantity row supports a rough estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    unit_market_value = row[IMPACT_INPUT_VALUE]
    return (
        row[DATASET] == pc_cols.HOLDINGS
        and row[SOURCE_COLUMN] == pc_cols.QUANTITY
        and row.get(IMPACT_POLICY) == IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE
        and _number_value(delta) is not None
        and _number_value(denominator) not in (None, 0.0)
        and _number_value(unit_market_value) is not None
    )


def _is_price_weighted_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether a price row supports a weighted price-return estimate."""
    delta = row[DELTA_B_MINUS_A]
    snapshot_a_price = row[SNAPSHOT_A_VALUE]
    weight = row[RETURN_WEIGHT]
    return (
        row[DATASET] == pc_cols.HOLDINGS
        and row[SOURCE_COLUMN] == pc_cols.PRICE
        and row.get(IMPACT_POLICY) == IMPACT_POLICY_PRICE_WEIGHTED
        and _number_value(delta) is not None
        and _number_value(snapshot_a_price) not in (None, 0.0)
        and _number_value(weight) not in (None, 0.0)
    )


def _has_evidence_only_impact_policy(row: dict[str, object]) -> bool:
    """Return whether a finding row is explicitly configured as evidence-only."""
    policies = (
        row.get(IMPACT_POLICY),
        row.get(TRANSACTION_IMPACT_POLICY),
    )
    return any(
        isinstance(policy, str) and policy.startswith(IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)
        for policy in policies
    )


def _transaction_performance_amount_impact_message(row: dict[str, object]) -> str:
    """Return a provenance-aware transaction amount impact message."""
    semantics_source = row.get(TRANSACTION_SEMANTICS_SOURCE)
    source_text = tx_diagnostics.readable_transaction_semantics_source(semantics_source)
    amount_basis = (
        "explicit base-currency transaction amount delta"
        if row.get(SOURCE_COLUMN) == pc_cols.BASE_AMOUNT
        else "source-signed transaction amount delta"
    )
    return (
        f"Approximate impact uses the {amount_basis} "
        "divided by the return denominator. Applies only when normalized "
        "sign/flow semantics mark the transaction as performance-affecting. "
        f"Transaction semantics source: {source_text}."
    )


def _has_transaction_impact_method_candidate(rows: list[dict[str, object]]) -> bool:
    """Return whether any transaction row has a currently supported method."""
    return any(_is_transaction_performance_amount_impact_candidate(row) for row in rows)


def _is_usable_number(value: object) -> bool:
    """Return whether a value can be used in impact arithmetic."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0


def _root_cause_area(row: dict[str, object]) -> str:
    """Return the coarse explanation bucket for a contribution candidate."""
    root_cause_area_by_dataset: dict[str, CauseArea] = {
        pc_cols.SECURITY_PERFORMANCE: CauseArea.SECURITY_RETURN_OR_CONTRIBUTION,
        pc_cols.HOLDINGS: CauseArea.MARKET_VALUE_OR_HOLDING,
        pc_cols.TRANSACTIONS: CauseArea.TRANSACTION_ACTIVITY,
        pc_cols.FX_RATES: CauseArea.FX_RATE,
        pc_cols.PORTFOLIO_PERFORMANCE: CauseArea.PORTFOLIO_PERFORMANCE_INPUT,
    }
    dataset = row[DATASET]
    if not isinstance(dataset, str):
        return CauseArea.UNEXPLAINED.value
    return root_cause_area_by_dataset.get(dataset, CauseArea.UNEXPLAINED).value


def _portfolio_period_cause_summary_row(
    key: tuple[object, object, object, str],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return one portfolio-period cause-area summary row."""
    portfolio_id, from_date, thru_date, root_cause_area = key
    estimated_impact = _summed_estimated_return_impact(rows)
    impact_basis = _summary_impact_basis(rows)
    impact_confidence = _summary_impact_confidence(rows)
    top_codes = _top_codes(rows)
    return {
        PORTFOLIO_ID: portfolio_id,
        FROM_DATE: from_date,
        THRU_DATE: thru_date,
        ROOT_CAUSE_AREA: root_cause_area,
        FINDING_COUNT: len(rows),
        ESTIMATED_RETURN_IMPACT: estimated_impact,
        IMPACT_BASIS: impact_basis,
        IMPACT_CONFIDENCE: impact_confidence,
        TOP_CODES: top_codes,
        IMPACT_MESSAGE: _summary_impact_message(
            root_cause_area,
            estimated_impact,
            top_codes,
            rows,
        ),
    }


def _security_period_cause_summary_row(
    key: tuple[object, object, object, object, str],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return one security-period cause-area summary row."""
    portfolio_id, security_id, from_date, thru_date, root_cause_area = key
    estimated_impact = _summed_estimated_return_impact(rows)
    impact_basis = _summary_impact_basis(rows)
    impact_confidence = _summary_impact_confidence(rows)
    top_codes = _top_codes(rows)
    return {
        PORTFOLIO_ID: portfolio_id,
        SECURITY_ID: security_id,
        FROM_DATE: from_date,
        THRU_DATE: thru_date,
        ROOT_CAUSE_AREA: root_cause_area,
        FINDING_COUNT: len(rows),
        ESTIMATED_RETURN_IMPACT: estimated_impact,
        IMPACT_BASIS: impact_basis,
        IMPACT_CONFIDENCE: impact_confidence,
        TOP_CODES: top_codes,
        IMPACT_MESSAGE: _summary_impact_message(
            root_cause_area,
            estimated_impact,
            top_codes,
            rows,
        ),
    }


def _impact_coverage_summary_row(
    period: dict[str, object],
    causes: list[dict[str, object]],
    transactions: list[dict[str, object]],
) -> dict[str, object]:
    """Return one cause-area estimate-coverage row for a portfolio period."""
    estimate_rows = [cause for cause in causes if cause.get(ESTIMATED_RETURN_IMPACT) is not None]
    evidence_only_rows = [
        cause for cause in causes if cause.get(IMPACT_BASIS) == IMPACT_BASIS_NO_ESTIMATE
    ]
    missing_inputs = _coverage_missing_impact_inputs(
        evidence_only_rows,
        transactions,
    )
    coverage_status = _impact_coverage_status(
        estimated_count=len(estimate_rows),
        evidence_only_count=len(evidence_only_rows),
        missing_inputs=missing_inputs,
    )

    return {
        PORTFOLIO_ID: period[PORTFOLIO_ID],
        FROM_DATE: period[FROM_DATE],
        THRU_DATE: period[THRU_DATE],
        PORTFOLIO_RETURN_DELTA: period[PORTFOLIO_RETURN_DELTA],
        ROOT_CAUSE_AREA_COUNT: len(causes),
        ESTIMATED_CAUSE_AREA_COUNT: len(estimate_rows),
        EVIDENCE_ONLY_CAUSE_AREA_COUNT: len(evidence_only_rows),
        LOW_CONFIDENCE_ESTIMATE_COUNT: _impact_confidence_count(
            estimate_rows,
            IMPACT_CONFIDENCE_LOW,
        ),
        MEDIUM_CONFIDENCE_ESTIMATE_COUNT: _impact_confidence_count(
            estimate_rows,
            IMPACT_CONFIDENCE_MEDIUM,
        ),
        ESTIMATED_RETURN_IMPACT_TOTAL: _sum_available_return_impacts(estimate_rows),
        EVIDENCE_ONLY_AREAS: _join_unique(
            str(cause[ROOT_CAUSE_AREA]) for cause in evidence_only_rows
        ),
        TRANSACTION_SEMANTICS_SOURCES: _period_transaction_semantics_sources(transactions),
        MISSING_IMPACT_INPUTS: missing_inputs,
        IMPACT_COVERAGE_STATUS: coverage_status,
        IMPACT_COVERAGE_REVIEW_NOTE: _impact_coverage_review_note(coverage_status),
        IMPACT_MESSAGE: _impact_coverage_message(
            estimated_count=len(estimate_rows),
            evidence_only_count=len(evidence_only_rows),
        ),
    }


def _impact_coverage_status(
    *,
    estimated_count: int,
    evidence_only_count: int,
    missing_inputs: str,
) -> str:
    """Return the reviewer-facing impact coverage status."""
    if missing_inputs:
        return IMPACT_COVERAGE_STATUS_MISSING_INPUTS
    if estimated_count > 0 and evidence_only_count == 0:
        return IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES
    if estimated_count > 0:
        return IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES
    return IMPACT_COVERAGE_STATUS_EVIDENCE_ONLY


def _impact_coverage_review_note(status: str) -> str:
    """Return reviewer guidance for an impact coverage status."""
    if status == IMPACT_COVERAGE_STATUS_MISSING_INPUTS:
        return "Resolve missing inputs before relying on impact totals."
    if status == IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        return "All current cause areas have selected impact estimates."
    if status == IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return "Review evidence-only areas before relying on impact totals."
    return "No cause areas have selected impact estimates yet."


def _impact_confidence_count(
    rows: list[dict[str, object]],
    confidence: str,
) -> int:
    """Return the number of estimated rows with the requested confidence."""
    return sum(1 for row in rows if row.get(IMPACT_CONFIDENCE) == confidence)


def _sum_available_return_impacts(rows: list[dict[str, object]]) -> float | None:
    """Return the sum of already-selected impact estimates."""
    estimates: list[float] = []
    for row in rows:
        estimate = row.get(ESTIMATED_RETURN_IMPACT)
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            continue
        estimates.append(float(estimate))
    if not estimates:
        return None
    return sum(estimates)


def _impact_coverage_message(
    *,
    estimated_count: int,
    evidence_only_count: int,
) -> str:
    """Return a concise explanation of estimate coverage for a period."""
    if estimated_count == 0:
        return "No cause areas have defensible return-impact estimates yet."
    if evidence_only_count == 0:
        return "All current cause areas have return-impact estimates."
    return (
        f"{estimated_count} cause area(s) have estimates; "
        f"{evidence_only_count} remain evidence-only."
    )


def _coverage_missing_impact_inputs(
    evidence_only_causes: list[dict[str, object]],
    transactions: list[dict[str, object]],
) -> str:
    """Return compact missing-input themes for evidence-only cause areas."""
    missing_inputs: list[str] = []
    for cause in evidence_only_causes:
        if _cause_is_configured_evidence_only(cause):
            continue
        cause_area = cause.get(ROOT_CAUSE_AREA)
        if cause_area == ROOT_CAUSE_TRANSACTION_ACTIVITY:
            transaction_inputs_found = False
            for transaction in transactions:
                transaction_inputs_found = True
                _extend_unique(
                    missing_inputs,
                    _split_missing_impact_inputs(transaction.get(MISSING_IMPACT_INPUTS)),
                )
            if not transaction_inputs_found:
                _extend_unique(
                    missing_inputs,
                    _split_transaction_cause_missing_inputs(cause.get(IMPACT_MESSAGE)),
                )
        elif cause_area in {
            ROOT_CAUSE_MARKET_VALUE_OR_HOLDING,
            ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE,
        }:
            _extend_unique(missing_inputs, ["return-impact method"])
        elif cause_area == ROOT_CAUSE_FX_RATE:
            _extend_unique(missing_inputs, ["currency exposure linkage"])
        else:
            _extend_unique(missing_inputs, ["defensible impact method"])
    return ", ".join(missing_inputs)


def _cause_is_configured_evidence_only(cause: dict[str, object]) -> bool:
    """Return whether a cause-area row is intentionally evidence-only."""
    message = cause.get(IMPACT_MESSAGE)
    return isinstance(message, str) and "Configured as evidence-only" in message


def _split_missing_impact_inputs(value: object) -> list[str]:
    """Return missing impact inputs parsed from a readable checklist string."""
    if not isinstance(value, str) or not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_transaction_cause_missing_inputs(value: object) -> list[str]:
    """Return transaction missing inputs parsed from a cause-summary message."""
    if not isinstance(value, str) or "Missing impact inputs:" not in value:
        return []
    _, missing_inputs = value.split("Missing impact inputs:", maxsplit=1)
    return _split_missing_impact_inputs(missing_inputs.rstrip("."))


def _extend_unique(target: list[str], values: list[str]) -> None:
    """Append values to target while preserving first-seen order."""
    for value in values:
        if value not in target:
            target.append(value)


def _join_unique(values: Iterable[str]) -> str:
    """Return a comma-separated string with first-seen duplicates removed."""
    unique_values: list[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return ", ".join(unique_values)


def _summary_period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the portfolio-period key shared by summary tables."""
    return row[PORTFOLIO_ID], row[FROM_DATE], row[THRU_DATE]


def _summary_rows_by_period(
    table: pl.DataFrame,
) -> dict[tuple[object, object, object], list[dict[str, object]]]:
    """Index summary rows once for repeated portfolio-period lookups."""
    rows_by_period: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in table.iter_rows(named=True):
        rows_by_period.setdefault(_summary_period_key(row), []).append(row)
    return rows_by_period


def _summed_estimated_return_impact(rows: list[dict[str, object]]) -> float | None:
    """Return the sum of available impact estimates, or ``None``."""
    estimates: list[float] = []
    estimated_rows = _preferred_estimate_rows(rows)
    for row in estimated_rows:
        estimate = row[ESTIMATED_RETURN_IMPACT]
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            continue
        estimates.append(float(estimate))
    if not estimates:
        return None
    return sum(estimates)


def _preferred_estimate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return estimate rows selected for additive summary totals.

    Notes:
        Performance inputs are the accounting fields that directly feed the
        return calculation. Input components can help explain those inputs, but
        should not also be summed when a related performance input is already
        available for the same cause bucket.
    """
    performance_input_rows = [
        row
        for row in rows
        if _field_roles.is_performance_input(row.get(DATASET), row.get(SOURCE_COLUMN))
    ]
    if performance_input_rows:
        return performance_input_rows
    holdings_price_rows = [
        row
        for row in rows
        if row.get(DATASET) == pc_cols.HOLDINGS and row.get(SOURCE_COLUMN) == pc_cols.PRICE
    ]
    if holdings_price_rows:
        return holdings_price_rows
    return rows


def _summary_impact_basis(rows: list[dict[str, object]]) -> str:
    """Return the aggregate impact basis for a cause-area bucket."""
    bases = {
        str(row[IMPACT_BASIS]) for row in rows if row[IMPACT_BASIS] != IMPACT_BASIS_NO_ESTIMATE
    }
    if IMPACT_BASIS_FX_RATE_LOCAL_EXPOSURE in bases:
        return IMPACT_BASIS_FX_RATE_LOCAL_EXPOSURE
    if IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT in bases:
        return IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT
    if IMPACT_BASIS_HOLDING_MARKET_VALUE in bases:
        return IMPACT_BASIS_HOLDING_MARKET_VALUE
    if IMPACT_BASIS_HOLDING_ACCRUED in bases:
        return IMPACT_BASIS_HOLDING_ACCRUED
    if IMPACT_BASIS_HOLDING_QUANTITY_UNIT_MARKET_VALUE in bases:
        return IMPACT_BASIS_HOLDING_QUANTITY_UNIT_MARKET_VALUE
    if IMPACT_BASIS_PRICE_WEIGHTED in bases:
        return IMPACT_BASIS_PRICE_WEIGHTED
    return IMPACT_BASIS_NO_ESTIMATE


def _summary_impact_confidence(rows: list[dict[str, object]]) -> str:
    """Return the aggregate impact confidence for a cause-area bucket."""
    confidences = {row[IMPACT_CONFIDENCE] for row in rows}
    if IMPACT_CONFIDENCE_MEDIUM in confidences:
        return IMPACT_CONFIDENCE_MEDIUM
    return IMPACT_CONFIDENCE_LOW


def _top_codes(rows: list[dict[str, object]], limit: int = 3) -> str:
    """Return a compact comma-separated list of representative finding codes."""
    ordered_codes: list[str] = []
    for row in sorted(rows, key=_portfolio_period_evidence_sort_key):
        code = str(row[FINDING_CODE])
        if code not in ordered_codes:
            ordered_codes.append(code)
    return ", ".join(ordered_codes[:limit])


def _summary_impact_message(
    root_cause_area: str,
    estimated_impact: float | None,
    top_codes: str,
    rows: list[dict[str, object]],
) -> str:
    """Return a short explanation for a cause-area summary row."""
    if estimated_impact is not None:
        return (
            "Estimated impact is based on currently supported contribution "
            f"candidate methods. Representative codes: {top_codes}."
        )
    if any(_has_evidence_only_impact_policy(row) for row in rows):
        return (
            "Configured as evidence-only in comparison YAML; this cause area "
            f"is review evidence. Representative codes: {top_codes}."
        )
    if root_cause_area == ROOT_CAUSE_TRANSACTION_ACTIVITY:
        return (
            "Transaction differences are grouped as evidence only. Missing "
            f"impact inputs: {_transaction_missing_impact_inputs_message(rows)}."
        )
    return (
        "Grouped evidence only; no defensible return-impact estimate is "
        f"available yet. Representative codes: {top_codes}."
    )


def _portfolio_period_cause_summary_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    """Return deterministic ordering for cause-area summaries."""
    estimated_impact = row[ESTIMATED_RETURN_IMPACT]
    absolute_impact = (
        abs(float(estimated_impact))
        if isinstance(estimated_impact, (int, float)) and not isinstance(estimated_impact, bool)
        else -1.0
    )
    finding_count = row[FINDING_COUNT]
    finding_count_sort = finding_count if isinstance(finding_count, int) else 0
    return (
        str(row[PORTFOLIO_ID]),
        str(row[FROM_DATE]),
        str(row[THRU_DATE]),
        -absolute_impact,
        -finding_count_sort,
        str(row[ROOT_CAUSE_AREA]),
    )


def _security_period_cause_summary_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    """Return deterministic ordering for security cause-area summaries."""
    portfolio_key = _portfolio_period_cause_summary_sort_key(row)
    return (
        portfolio_key[0],
        str(row[SECURITY_ID]),
        *portfolio_key[1:],
    )


def _transaction_activity_summary_row(
    key: tuple[object, ...],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return one transaction activity summary row."""
    portfolio_id, security_id, from_date, thru_date, transaction_category = key
    changed_fields = _changed_transaction_fields(rows)
    missing_impact_inputs = _transaction_missing_impact_inputs_message(
        rows,
        from_date=from_date,
        thru_date=thru_date,
    )
    impact_message = _transaction_activity_impact_message(missing_impact_inputs)
    return {
        PORTFOLIO_ID: portfolio_id,
        SECURITY_ID: security_id,
        FROM_DATE: from_date,
        THRU_DATE: thru_date,
        TRANSACTION_CATEGORY: transaction_category,
        FINDING_COUNT: len(rows),
        CHANGED_FIELDS: ", ".join(changed_fields),
        AMOUNT_DELTA: _field_delta(rows, pc_cols.AMOUNT),
        QUANTITY_DELTA: _field_delta(rows, pc_cols.QUANTITY),
        PRICE_DELTA: _field_delta(rows, pc_cols.PRICE),
        TRANSACTION_SEMANTICS_SOURCES: _transaction_semantics_source_counts(rows),
        TRANSACTION_MATCH_STATUSES: _transaction_match_status_counts(rows),
        MISSING_IMPACT_INPUTS: missing_impact_inputs,
        IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_MESSAGE: impact_message,
    }


def _portfolio_period_transaction_cross_check_row(
    key: tuple[object, object, object, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return one portfolio-period transaction cross-check summary row."""
    portfolio_id, from_date, thru_date, _transaction_impact_policy = key
    estimates = [
        value
        for row in rows
        if (value := _number_value(row.get(TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE))) is not None
    ]
    diagnostics = sorted(
        {
            str(row[TRANSACTION_IMPACT_DIAGNOSTIC])
            for row in rows
            if row.get(TRANSACTION_IMPACT_DIAGNOSTIC)
        }
    )
    policies = sorted(
        {str(row[TRANSACTION_IMPACT_POLICY]) for row in rows if row.get(TRANSACTION_IMPACT_POLICY)}
    )
    impact_message = (
        "Transaction impact cross-checks are review-only and are not "
        "included in estimated impact totals."
    )
    return {
        PORTFOLIO_ID: portfolio_id,
        FROM_DATE: from_date,
        THRU_DATE: thru_date,
        TRANSACTION_IMPACT_POLICIES: ", ".join(policies),
        CROSS_CHECK_TREATMENT: CROSS_CHECK_ONLY,
        CROSS_CHECK_COUNT: len(estimates),
        CROSS_CHECK_ESTIMATE_TOTAL: sum(estimates),
        CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL: sum(abs(estimate) for estimate in estimates),
        CHANGED_FIELDS: ", ".join(_changed_transaction_fields(rows)),
        TRANSACTION_IMPACT_DIAGNOSTICS: ", ".join(diagnostics),
        IMPACT_MESSAGE: impact_message,
    }


def _is_number(value: object) -> bool:
    """Return whether a value is a finite non-boolean number."""
    return _number_value(value) is not None


def _number_value(value: object) -> float | None:
    """Return a finite float for non-boolean numeric values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _required_impact_number(value: object, field_name: str) -> float:
    """Return a finite number admitted by an impact-candidate predicate.

    Args:
        value: Candidate value already screened by an impact predicate.
        field_name: Field included in an internal-invariant diagnostic.

    Returns:
        Finite numeric value.

    Raises:
        PpaError: If an impact predicate admitted a nonnumeric or nonfinite
            calculation input.
    """
    number = _number_value(value)
    if number is None:
        raise PpaError(
            f"Impact candidate admitted invalid {field_name}={value!r}.",
            999,
            context={"field": field_name, "value": repr(value)},
        )
    return number


def _portfolio_period_transaction_cross_check_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    """Return deterministic ordering for transaction cross-check summaries."""
    absolute_estimate = row[CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL]
    absolute_estimate_sort = _number_value(absolute_estimate) or 0.0
    return (
        str(row[PORTFOLIO_ID]),
        str(row[FROM_DATE]),
        str(row[THRU_DATE]),
        -absolute_estimate_sort,
        str(row[TRANSACTION_IMPACT_POLICIES]),
    )


def _transaction_activity_impact_message(missing_impact_inputs: str) -> str:
    """Return the transaction activity summary impact message."""
    if missing_impact_inputs:
        return (
            "Transaction activity summary is evidence-only. Missing impact "
            f"inputs: {missing_impact_inputs}."
        )
    return (
        "Transaction activity has modeled impact inputs; supported estimates "
        "are available in contribution candidates and cause summaries."
    )


def _period_transaction_semantics_sources(
    transactions: list[dict[str, object]],
) -> str:
    """Return aggregate transaction semantics provenance counts for a period."""
    counts: dict[str, int] = {}
    for transaction in transactions:
        for source, count in tx_diagnostics.parse_transaction_semantics_sources(
            transaction.get(TRANSACTION_SEMANTICS_SOURCES)
        ).items():
            counts[source] = counts.get(source, 0) + count
    return tx_diagnostics.format_transaction_semantics_source_counts(counts)


def _transaction_semantics_source_counts(rows: list[dict[str, object]]) -> str:
    """Return compact transaction semantics provenance counts for evidence rows."""
    counts: dict[str, int] = {}
    for row in rows:
        source = row.get(TRANSACTION_SEMANTICS_SOURCE)
        if not isinstance(source, str) or not source:
            continue
        counts[source] = counts.get(source, 0) + 1
    return tx_diagnostics.format_transaction_semantics_source_counts(counts)


def _transaction_match_status_counts(rows: list[dict[str, object]]) -> str:
    """Return compact transaction match-status counts for evidence rows."""
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get(TRANSACTION_MATCH_STATUS)
        if not isinstance(status, str) or not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return tx_diagnostics.format_label_counts(counts)


def _transaction_missing_impact_inputs_message(
    rows: list[dict[str, object]],
    *,
    from_date: object | None = None,
    thru_date: object | None = None,
) -> str:
    """Return a readable checklist of missing transaction impact inputs."""
    return ", ".join(
        _transaction_missing_impact_inputs(
            rows,
            from_date=from_date,
            thru_date=thru_date,
        )
    )


def _transaction_missing_impact_inputs(
    rows: list[dict[str, object]],
    *,
    from_date: object | None = None,
    thru_date: object | None = None,
) -> list[str]:
    """Return transaction impact eligibility inputs not present or not modeled."""
    if not rows:
        return [
            "portfolio",
            "security",
            "portfolio period",
            "normalized transaction category",
            "return denominator",
            TRANSACTION_SIGN_AND_FLOW_SEMANTICS,
        ]

    missing_inputs: list[str] = []
    if not any(row.get(PORTFOLIO_ID) is not None for row in rows):
        missing_inputs.append("portfolio")
    if not any(row.get(SECURITY_ID) is not None for row in rows):
        missing_inputs.append("security")
    has_period = from_date is not None and thru_date is not None
    has_period = has_period or any(
        row.get(FROM_DATE) is not None and row.get(THRU_DATE) is not None for row in rows
    )
    if not has_period:
        missing_inputs.append("portfolio period")
    if not any(
        row.get(TRANSACTION_CATEGORY) not in {None, "", TRANSACTION_CATEGORY_UNKNOWN}
        for row in rows
    ):
        missing_inputs.append("normalized transaction category")
    if not any(_is_usable_number(row.get(RETURN_DENOMINATOR)) for row in rows):
        missing_inputs.append("return denominator")

    # Sign/flow semantics must be source-supplied and normalized. They are an
    # eligibility gate; only currently modeled transaction treatments estimate.
    if not any(transaction_impact_semantics_available(row) for row in rows):
        missing_inputs.append(TRANSACTION_SIGN_AND_FLOW_SEMANTICS)
    if not missing_inputs and not _has_transaction_impact_method_candidate(rows):
        _extend_unique(missing_inputs, _transaction_unmodeled_method_inputs(rows))
    return missing_inputs


def _transaction_unmodeled_method_inputs(rows: list[dict[str, object]]) -> list[str]:
    """Return missing transaction method themes for modeled-but-unsupported rows."""
    missing_inputs: list[str] = []
    for row in rows:
        if _is_transaction_performance_amount_impact_candidate(row):
            continue
        performance_flow_sign = row.get(PERFORMANCE_FLOW_SIGN)
        cash_flow_sign = row.get(CASH_FLOW_SIGN)
        if performance_flow_sign == TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
            diagnostic_inputs = _transaction_impact_diagnostic_inputs(
                row.get(TRANSACTION_IMPACT_DIAGNOSTIC)
            )
            if diagnostic_inputs:
                _extend_unique(missing_inputs, diagnostic_inputs)
            elif (
                row.get(TRANSACTION_IMPACT_POLICY)
                == TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY
            ):
                _extend_unique(missing_inputs, [EXTERNAL_FLOW_EVIDENCE_ONLY_POLICY])
            else:
                _extend_unique(missing_inputs, [EXTERNAL_FLOW_IMPACT_METHOD])
        elif performance_flow_sign == TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL:
            _extend_unique(missing_inputs, [NEUTRAL_FLOW_IMPACT_METHOD])
        elif (
            performance_flow_sign == TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
            and cash_flow_sign == TRANSACTION_CASH_FLOW_SIGN_NONE
        ):
            _extend_unique(missing_inputs, [NO_CASH_TRANSACTION_IMPACT_METHOD])
        elif transaction_impact_semantics_available(row):
            _extend_unique(missing_inputs, [TRANSACTION_IMPACT_METHOD])
    if missing_inputs:
        return missing_inputs
    return ["return-impact method"]


def _transaction_impact_diagnostic_inputs(value: object) -> list[str]:
    """Return missing-impact themes from a transaction diagnostic string."""
    if not isinstance(value, str) or not value:
        return []
    if value == "external-flow evidence-only policy":
        return [EXTERNAL_FLOW_EVIDENCE_ONLY_POLICY]
    if value == "external-flow impact method missing":
        return [EXTERNAL_FLOW_IMPACT_METHOD]
    if value == "modified_dietz cross-check estimate":
        return ["modified_dietz cross-check only"]
    prefix = "modified_dietz missing inputs: "
    if value.startswith(prefix):
        missing = value.removeprefix(prefix)
        return [f"modified_dietz {part.strip()}" for part in missing.split(",") if part.strip()]
    return []


def _changed_transaction_fields(rows: list[dict[str, object]]) -> list[str]:
    """Return changed transaction source fields in stable order."""
    fields = {str(row[SOURCE_COLUMN]) for row in rows if row[SOURCE_COLUMN] is not None}
    return sorted(fields, key=tx_diagnostics.transaction_field_sort_key)


def _field_delta(rows: list[dict[str, object]], source_column: str) -> float | None:
    """Return summed numeric delta for one source column, or ``None``."""
    deltas: list[float] = []
    for row in rows:
        if row[SOURCE_COLUMN] != source_column:
            continue
        delta = row[DELTA_B_MINUS_A]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            continue
        deltas.append(float(delta))
    if not deltas:
        return None
    return sum(deltas)


def _transaction_activity_summary_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    """Return deterministic ordering for transaction activity summaries."""
    return (
        str(row[PORTFOLIO_ID]),
        str(row[FROM_DATE]),
        str(row[THRU_DATE]),
        str(row[SECURITY_ID]),
        str(row[TRANSACTION_CATEGORY]),
    )


def _priority_score_and_reason(
    finding: dict[str, object],
    absolute_delta: float | None,
) -> tuple[int, str]:
    """Return a transparent review-priority score and short reason."""
    role = str(finding[EVIDENCE_ROLE])
    dataset = str(finding[DATASET])
    role_score = _role_priority_score(role)
    dataset_score = _dataset_priority_score(dataset)
    delta_score = 10 if absolute_delta is not None else 0
    score = role_score + dataset_score + delta_score
    reason = (
        f"role={role}:{role_score}; "
        f"dataset={dataset}:{dataset_score}; "
        f"numeric_delta={delta_score}"
    )
    return score, reason


def _role_priority_score(evidence_role: str) -> int:
    """Return the review-priority weight for an evidence role."""
    return {
        DIRECT_INPUT.value: 300,
        RELATED_OUTPUT.value: 200,
        CONTEXT.value: 100,
        TARGET_OUTPUT.value: 0,
    }.get(evidence_role, 0)


def _dataset_priority_score(dataset: str) -> int:
    """Return the review-priority weight for a normalized dataset."""
    return {
        pc_cols.PORTFOLIO_PERFORMANCE: 40,
        pc_cols.TRANSACTIONS: 35,
        pc_cols.HOLDINGS: 35,
        pc_cols.FX_RATES: 25,
        pc_cols.SECURITY_PERFORMANCE: 10,
    }.get(dataset, 0)


def _absolute_numeric_delta(value: object) -> float | None:
    """Return the absolute delta when a finding has a numeric delta."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return abs(float(value))


def _portfolio_period_evidence_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    """Return descending score and deterministic tie-breakers for evidence rows."""
    absolute_delta = row[ABSOLUTE_DELTA]
    priority_score = row[PRIORITY_SCORE]
    delta_sort = absolute_delta if isinstance(absolute_delta, float) else -1.0
    score_sort = priority_score if isinstance(priority_score, int) else 0
    return (
        -score_sort,
        -delta_sort,
        str(row[DATASET]),
        str(row[FINDING_CODE]),
        str(row[SECURITY_ID]),
        str(row[SOURCE_COLUMN]),
        str(row[MESSAGE]),
    )


def _empty_portfolio_period_summary() -> pl.DataFrame:
    """Return an empty portfolio-period summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            PORTFOLIO_RETURN_DELTA: pl.Float64,
            FINDING_COUNT: pl.UInt32,
            PORTFOLIO_FINDING_COUNT: pl.UInt32,
            DIRECT_INPUT_FINDING_COUNT: pl.UInt32,
            RELATED_OUTPUT_FINDING_COUNT: pl.UInt32,
            CONTEXT_FINDING_COUNT: pl.UInt32,
            FX_RATE_FINDING_COUNT: pl.UInt32,
            TRANSACTION_FINDING_COUNT: pl.UInt32,
            HOLDING_FINDING_COUNT: pl.UInt32,
            HAS_SUPPRESSED_FINDINGS: pl.Boolean,
        }
    )


def _empty_portfolio_period_evidence_breakdown() -> pl.DataFrame:
    """Return an empty evidence breakdown with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            EVIDENCE_GROUP: pl.String,
            DATASET: pl.String,
            FINDING_COUNT: pl.UInt32,
        }
    )


def _empty_security_period_summary() -> pl.DataFrame:
    """Return an empty security-period summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            SECURITY_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            SECURITY_RETURN_DELTA: pl.Float64,
            FINDING_COUNT: pl.UInt32,
            SECURITY_FINDING_COUNT: pl.UInt32,
            DIRECT_INPUT_FINDING_COUNT: pl.UInt32,
            RELATED_OUTPUT_FINDING_COUNT: pl.UInt32,
            CONTEXT_FINDING_COUNT: pl.UInt32,
            TRANSACTION_FINDING_COUNT: pl.UInt32,
            HOLDING_FINDING_COUNT: pl.UInt32,
            HAS_SUPPRESSED_FINDINGS: pl.Boolean,
        }
    )


def _empty_security_period_evidence_breakdown() -> pl.DataFrame:
    """Return an empty security-period evidence breakdown with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            SECURITY_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            EVIDENCE_GROUP: pl.String,
            DATASET: pl.String,
            FINDING_COUNT: pl.UInt32,
        }
    )


def _empty_portfolio_period_evidence_ranking() -> pl.DataFrame:
    """Return an empty portfolio-period evidence ranking with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            REVIEW_RANK: pl.UInt32,
            PRIORITY_SCORE: pl.Int64,
            PRIORITY_REASON: pl.String,
            FINDING_CODE: pl.String,
            DATASET: pl.String,
            EVIDENCE_ROLE: pl.String,
            SECURITY_ID: pl.String,
            SOURCE_FILE: pl.String,
            SOURCE_RECORD_LOCATOR: pl.String,
            INPUT_DATE: pl.Date,
            SOURCE_COLUMN: pl.String,
            FROM_CURRENCY: pl.String,
            TO_CURRENCY: pl.String,
            TRANSACTION_CODE: pl.String,
            TRANSACTION_CATEGORY: pl.String,
            CASH_FLOW_SIGN: pl.String,
            PERFORMANCE_FLOW_SIGN: pl.String,
            TRANSACTION_SEMANTICS_SOURCE: pl.String,
            IMPACT_POLICY: pl.String,
            TRANSACTION_IMPACT_POLICY: pl.String,
            TRANSACTION_IMPACT_DIAGNOSTIC: pl.String,
            TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE: pl.Float64,
            DELTA_B_MINUS_A: pl.Float64,
            SNAPSHOT_A_VALUE: pl.String,
            SNAPSHOT_B_VALUE: pl.String,
            RETURN_DENOMINATOR: pl.Float64,
            RETURN_WEIGHT: pl.Float64,
            IMPACT_INPUT_VALUE: pl.Float64,
            ABSOLUTE_DELTA: pl.Float64,
            MESSAGE: pl.String,
        }
    )


def _empty_portfolio_period_contribution_candidates() -> pl.DataFrame:
    """Return empty contribution candidates with stable columns."""
    schema = _empty_portfolio_period_evidence_ranking().schema
    return pl.DataFrame(
        schema={
            **schema,
            ESTIMATED_RETURN_IMPACT: pl.Float64,
            IMPACT_BASIS: pl.String,
            IMPACT_CONFIDENCE: pl.String,
            IMPACT_METHOD: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_portfolio_period_cause_summary() -> pl.DataFrame:
    """Return empty cause-area summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            ROOT_CAUSE_AREA: pl.String,
            FINDING_COUNT: pl.UInt32,
            ESTIMATED_RETURN_IMPACT: pl.Float64,
            IMPACT_BASIS: pl.String,
            IMPACT_CONFIDENCE: pl.String,
            TOP_CODES: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_security_period_cause_summary() -> pl.DataFrame:
    """Return empty security cause-area summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            SECURITY_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            ROOT_CAUSE_AREA: pl.String,
            FINDING_COUNT: pl.UInt32,
            ESTIMATED_RETURN_IMPACT: pl.Float64,
            IMPACT_BASIS: pl.String,
            IMPACT_CONFIDENCE: pl.String,
            TOP_CODES: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_portfolio_period_impact_coverage_summary() -> pl.DataFrame:
    """Return empty impact coverage summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            PORTFOLIO_RETURN_DELTA: pl.Float64,
            ROOT_CAUSE_AREA_COUNT: pl.UInt32,
            ESTIMATED_CAUSE_AREA_COUNT: pl.UInt32,
            EVIDENCE_ONLY_CAUSE_AREA_COUNT: pl.UInt32,
            LOW_CONFIDENCE_ESTIMATE_COUNT: pl.UInt32,
            MEDIUM_CONFIDENCE_ESTIMATE_COUNT: pl.UInt32,
            ESTIMATED_RETURN_IMPACT_TOTAL: pl.Float64,
            EVIDENCE_ONLY_AREAS: pl.String,
            TRANSACTION_SEMANTICS_SOURCES: pl.String,
            MISSING_IMPACT_INPUTS: pl.String,
            IMPACT_COVERAGE_STATUS: pl.String,
            IMPACT_COVERAGE_REVIEW_NOTE: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_portfolio_period_transaction_cross_checks() -> pl.DataFrame:
    """Return empty transaction cross-check summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            TRANSACTION_IMPACT_POLICIES: pl.String,
            CROSS_CHECK_TREATMENT: pl.String,
            CROSS_CHECK_COUNT: pl.UInt32,
            CROSS_CHECK_ESTIMATE_TOTAL: pl.Float64,
            CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL: pl.Float64,
            CHANGED_FIELDS: pl.String,
            TRANSACTION_IMPACT_DIAGNOSTICS: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_transaction_activity_summary() -> pl.DataFrame:
    """Return empty transaction activity summary with stable columns."""
    return pl.DataFrame(
        schema={
            PORTFOLIO_ID: pl.String,
            SECURITY_ID: pl.String,
            FROM_DATE: pl.Date,
            THRU_DATE: pl.Date,
            TRANSACTION_CATEGORY: pl.String,
            FINDING_COUNT: pl.UInt32,
            CHANGED_FIELDS: pl.String,
            AMOUNT_DELTA: pl.Float64,
            QUANTITY_DELTA: pl.Float64,
            PRICE_DELTA: pl.Float64,
            TRANSACTION_SEMANTICS_SOURCES: pl.String,
            TRANSACTION_MATCH_STATUSES: pl.String,
            MISSING_IMPACT_INPUTS: pl.String,
            IMPACT_BASIS: pl.String,
            IMPACT_CONFIDENCE: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )


def _empty_transaction_matching_diagnostics() -> pl.DataFrame:
    """Return empty transaction matching diagnostics with stable columns."""
    return pl.DataFrame(
        schema={
            TRANSACTION_MATCH_STATUS: pl.String,
            FINDING_COUNT: pl.UInt32,
            TRANSACTION_MATCH_CONFIDENCE: pl.String,
            TRANSACTION_MATCH_INTERPRETATION: pl.String,
            TRANSACTION_MATCH_REVIEW_NOTE: pl.String,
        }
    )
