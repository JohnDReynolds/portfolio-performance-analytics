"""Build explanation-oriented tables from performance comparison findings."""

from __future__ import annotations

# Python imports
from collections.abc import Iterable, Mapping

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.findings import (
    CASH_FLOW_SIGN,
    CONTEXT,
    DATASET,
    DELTA_B_MINUS_A,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    MESSAGE,
    PC_PORT_RET,
    PC_SEC_RET,
    PERFORMANCE_FLOW_SIGN,
    PORTFOLIO_ID,
    RELATED_OUTPUT,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
    SECURITY_ID,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SUPPRESSED,
    TARGET_OUTPUT,
    THRU_DATE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_CATEGORY,
    TRANSACTION_SEMANTICS_SOURCE,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_CASH_FLOW_SIGN_NONE,
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_UNKNOWN,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    TRANSACTION_SEMANTICS_SOURCE_MIXED,
    TRANSACTION_SEMANTICS_SOURCE_SOURCE,
    TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
    transaction_impact_semantics_available,
)

PORTFOLIO_RETURN_DELTA = "portfolio_return_delta"
SECURITY_RETURN_DELTA = "security_return_delta"
FINDING_COUNT = "finding_count"
PORTFOLIO_FINDING_COUNT = "portfolio_finding_count"
SECURITY_FINDING_COUNT = "security_finding_count"
DIRECT_INPUT_FINDING_COUNT = "direct_input_finding_count"
RELATED_OUTPUT_FINDING_COUNT = "related_output_finding_count"
CONTEXT_FINDING_COUNT = "context_finding_count"
EVIDENCE_GROUP = "evidence_group"
PRICE_FINDING_COUNT = "price_finding_count"
FX_RATE_FINDING_COUNT = "fx_rate_finding_count"
TRANSACTION_FINDING_COUNT = "transaction_finding_count"
POSITION_FINDING_COUNT = "position_finding_count"
CASH_FINDING_COUNT = "cash_finding_count"
REFERENCE_FINDING_COUNT = "reference_finding_count"
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
    PRICE_FINDING_COUNT,
    FX_RATE_FINDING_COUNT,
    TRANSACTION_FINDING_COUNT,
    POSITION_FINDING_COUNT,
    CASH_FINDING_COUNT,
    REFERENCE_FINDING_COUNT,
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
    PRICE_FINDING_COUNT,
    TRANSACTION_FINDING_COUNT,
    POSITION_FINDING_COUNT,
    REFERENCE_FINDING_COUNT,
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
IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD = "portfolio_source_field"
IMPACT_BASIS_SECURITY_CONTRIBUTION = "security_contribution"
IMPACT_BASIS_SECURITY_RETURN_WEIGHTED = "security_return_weighted"
IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT = "transaction_performance_amount"
IMPACT_CONFIDENCE_LOW = "low"
IMPACT_CONFIDENCE_MEDIUM = "medium"
IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT = (
    "security_return_delta_times_weight"
)
IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV = (
    "source_field_delta_over_begin_market_value"
)
IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA = "vendor_contribution_delta"
IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR = (
    "transaction_amount_delta_over_return_denominator"
)
ROOT_CAUSE_AREA = "root_cause_area"
ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION = "security_return_or_contribution"
ROOT_CAUSE_MARKET_VALUE_OR_POSITION = "market_value_or_position"
ROOT_CAUSE_TRANSACTION_ACTIVITY = "transaction_activity"
ROOT_CAUSE_PRICE = "price"
ROOT_CAUSE_FX_RATE = "fx_rate"
ROOT_CAUSE_CASH = "cash"
ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT = "portfolio_performance_input"
ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE = "classification_or_reference"
ROOT_CAUSE_UNEXPLAINED = "unexplained"
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
    SOURCE_COLUMN,
    TRANSACTION_CATEGORY,
    CASH_FLOW_SIGN,
    PERFORMANCE_FLOW_SIGN,
    TRANSACTION_SEMANTICS_SOURCE,
    TRANSACTION_IMPACT_POLICY,
    DELTA_B_MINUS_A,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
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
ROOT_CAUSE_AREA_COUNT = "root_cause_area_count"
ESTIMATED_CAUSE_AREA_COUNT = "estimated_cause_area_count"
EVIDENCE_ONLY_CAUSE_AREA_COUNT = "evidence_only_cause_area_count"
LOW_CONFIDENCE_ESTIMATE_COUNT = "low_confidence_estimate_count"
MEDIUM_CONFIDENCE_ESTIMATE_COUNT = "medium_confidence_estimate_count"
ESTIMATED_RETURN_IMPACT_TOTAL = "estimated_return_impact_total"
EVIDENCE_ONLY_AREAS = "evidence_only_areas"
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
    MISSING_IMPACT_INPUTS,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
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
    for target in target_findings.iter_rows(named=True):
        related_active = _related_portfolio_period_findings(active_findings, target)
        related_all = _related_portfolio_period_findings(findings, target)
        role_counts = _role_summary_counts(related_active)
        rows.append(
            {
                PORTFOLIO_ID: target[PORTFOLIO_ID],
                FROM_DATE: target[FROM_DATE],
                THRU_DATE: target[THRU_DATE],
                PORTFOLIO_RETURN_DELTA: target[DELTA_B_MINUS_A],
                FINDING_COUNT: related_active.height,
                PORTFOLIO_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.PORTFOLIO_PERFORMANCE,
                ),
                **role_counts,
                PRICE_FINDING_COUNT: _dataset_count(related_active, pc_cols.PRICES),
                FX_RATE_FINDING_COUNT: _dataset_count(related_active, pc_cols.FX_RATES),
                TRANSACTION_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.TRANSACTIONS,
                ),
                POSITION_FINDING_COUNT: _dataset_count(related_active, pc_cols.POSITIONS),
                CASH_FINDING_COUNT: _dataset_count(related_active, pc_cols.CASH),
                REFERENCE_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.SECURITY_MASTER,
                ),
                HAS_SUPPRESSED_FINDINGS: _has_suppressed_findings(related_all),
            }
        )
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_SUMMARY_COLUMNS)


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
    for target in target_findings.iter_rows(named=True):
        related_active = _related_portfolio_period_findings(active_findings, target)
        rows.extend(_evidence_breakdown_rows(target, related_active))
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS)


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
    for target in target_findings.iter_rows(named=True):
        related_active = _related_security_period_findings(active_findings, target)
        related_all = _related_security_period_findings(findings, target)
        role_counts = _role_summary_counts(related_active)
        rows.append(
            {
                PORTFOLIO_ID: target[PORTFOLIO_ID],
                SECURITY_ID: target[SECURITY_ID],
                FROM_DATE: target[FROM_DATE],
                THRU_DATE: target[THRU_DATE],
                SECURITY_RETURN_DELTA: target[DELTA_B_MINUS_A],
                FINDING_COUNT: related_active.height,
                SECURITY_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.SECURITY_PERFORMANCE,
                ),
                **role_counts,
                PRICE_FINDING_COUNT: _dataset_count(related_active, pc_cols.PRICES),
                TRANSACTION_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.TRANSACTIONS,
                ),
                POSITION_FINDING_COUNT: _dataset_count(related_active, pc_cols.POSITIONS),
                REFERENCE_FINDING_COUNT: _dataset_count(
                    related_active,
                    pc_cols.SECURITY_MASTER,
                ),
                HAS_SUPPRESSED_FINDINGS: _has_suppressed_findings(related_all),
            }
        )
    return pl.DataFrame(rows).select(SECURITY_PERIOD_SUMMARY_COLUMNS)


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
    for target in target_findings.iter_rows(named=True):
        related_active = _related_security_period_findings(active_findings, target)
        rows.extend(_security_evidence_breakdown_rows(target, related_active))
    return pl.DataFrame(rows).select(SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS)


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
    if findings.is_empty():
        return _empty_portfolio_period_evidence_ranking()

    active_findings = _active_findings(findings, include_suppressed)
    target_findings = active_findings.filter(
        (pl.col(FINDING_CODE) == PC_PORT_RET)
        & (pl.col(DATASET) == pc_cols.PORTFOLIO_PERFORMANCE)
        & (pl.col(SOURCE_COLUMN) == pc_cols.PORTFOLIO_RETURN)
    )
    if target_findings.is_empty():
        return _empty_portfolio_period_evidence_ranking()

    rows: list[dict[str, object]] = []
    for target in target_findings.iter_rows(named=True):
        related_active = _related_portfolio_period_findings(active_findings, target)
        evidence = related_active.filter(pl.col(EVIDENCE_ROLE) != TARGET_OUTPUT)
        ranked_rows = sorted(
            (
                _ranked_evidence_row(target, finding)
                for finding in evidence.iter_rows(named=True)
            ),
            key=_portfolio_period_evidence_sort_key,
        )
        for review_rank, row in enumerate(ranked_rows, start=1):
            row[REVIEW_RANK] = review_rank
            rows.append(row)

    if not rows:
        return _empty_portfolio_period_evidence_ranking()
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS)


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

    rows = [
        _contribution_candidate_row(row)
        for row in ranking.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS)


def portfolio_period_cause_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
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
    candidates = portfolio_period_contribution_candidates(
        findings,
        include_suppressed=include_suppressed,
    )
    if candidates.is_empty():
        return _empty_portfolio_period_cause_summary()

    buckets: dict[tuple[object, object, object, str], list[dict[str, object]]] = {}
    for row in candidates.iter_rows(named=True):
        cause_area = _root_cause_area(row)
        key = (row[PORTFOLIO_ID], row[FROM_DATE], row[THRU_DATE], cause_area)
        buckets.setdefault(key, []).append(row)

    rows = [
        _portfolio_period_cause_summary_row(key, bucket_rows)
        for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_portfolio_period_cause_summary_sort_key)
    return pl.DataFrame(sorted_rows).select(PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS)


def portfolio_period_impact_coverage_summary(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return estimate-coverage status for each changed portfolio period.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether suppressed findings should be included in
            the underlying portfolio-period, cause-area, and transaction
            summaries.

    Returns:
        One row per changed portfolio period. Counts are cause-area based, not
        finding-row based, because impact estimates are currently aggregated at
        the cause-area level.
    """
    periods = portfolio_period_summary(
        findings,
        include_suppressed=include_suppressed,
    )
    if periods.is_empty():
        return _empty_portfolio_period_impact_coverage_summary()

    causes = portfolio_period_cause_summary(
        findings,
        include_suppressed=include_suppressed,
    )
    transactions = transaction_activity_summary(
        findings,
        include_suppressed=include_suppressed,
    )
    rows = [
        _impact_coverage_summary_row(
            period,
            _matching_period_rows(causes, period),
            _matching_period_rows(transactions, period),
        )
        for period in periods.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS)


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
        _transaction_activity_summary_row(key, bucket_rows)
        for key, bucket_rows in buckets.items()
    ]
    sorted_rows = sorted(rows, key=_transaction_activity_summary_sort_key)
    return pl.DataFrame(sorted_rows).select(TRANSACTION_ACTIVITY_SUMMARY_COLUMNS)


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


def _has_suppressed_findings(findings: pl.DataFrame) -> bool:
    """Return whether a related finding set includes suppressed rows."""
    if findings.is_empty() or SUPPRESSED not in findings.columns:
        return False
    return bool(findings.get_column(SUPPRESSED).any())


def _role_summary_counts(findings: pl.DataFrame) -> dict[str, int]:
    """Return standard role count fields for summary tables."""
    return {
        DIRECT_INPUT_FINDING_COUNT: _direct_input_count(findings),
        RELATED_OUTPUT_FINDING_COUNT: _role_count(findings, RELATED_OUTPUT),
        CONTEXT_FINDING_COUNT: _context_count(findings),
    }


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
            (
                (pl.col(FROM_DATE) == from_date)
                & (pl.col(THRU_DATE) == thru_date)
            )
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
        & (
            (pl.col(PORTFOLIO_ID) == portfolio_id)
            | pl.col(PORTFOLIO_ID).is_null()
        )
        & (
            (
                (pl.col(FROM_DATE) == from_date)
                & (pl.col(THRU_DATE) == thru_date)
            )
            | (pl.col(FROM_DATE).is_null() & pl.col(THRU_DATE).is_null())
        )
    )


def _dataset_count(findings: pl.DataFrame, dataset: str) -> int:
    """Return the number of findings for a normalized dataset."""
    if findings.is_empty():
        return 0
    return findings.filter(pl.col(DATASET) == dataset).height


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
    return findings.filter(pl.col(EVIDENCE_ROLE) == evidence_role).height


def _role_dataset_count(
    findings: pl.DataFrame,
    evidence_role: str,
    dataset: str,
) -> int:
    """Return the number of findings for an evidence role and dataset."""
    if findings.is_empty():
        return 0
    return findings.filter(
        (pl.col(EVIDENCE_ROLE) == evidence_role) & (pl.col(DATASET) == dataset)
    ).height


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
            pc_cols.PRICES,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.PRICES),
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
            pc_cols.POSITIONS,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.POSITIONS),
        ),
        (
            DIRECT_INPUT,
            pc_cols.CASH,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.CASH),
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
        (
            CONTEXT,
            pc_cols.SECURITY_MASTER,
            _role_dataset_count(related_findings, CONTEXT, pc_cols.SECURITY_MASTER),
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
            pc_cols.PRICES,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.PRICES),
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
            pc_cols.POSITIONS,
            _role_dataset_count(related_findings, DIRECT_INPUT, pc_cols.POSITIONS),
        ),
        (
            RELATED_OUTPUT,
            pc_cols.SECURITY_PERFORMANCE,
            _dataset_count(related_output_findings, pc_cols.SECURITY_PERFORMANCE),
        ),
        (
            CONTEXT,
            pc_cols.SECURITY_MASTER,
            _role_dataset_count(related_findings, CONTEXT, pc_cols.SECURITY_MASTER),
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
        SOURCE_COLUMN: finding[SOURCE_COLUMN],
        TRANSACTION_CATEGORY: finding[TRANSACTION_CATEGORY],
        CASH_FLOW_SIGN: finding[CASH_FLOW_SIGN],
        PERFORMANCE_FLOW_SIGN: finding[PERFORMANCE_FLOW_SIGN],
        TRANSACTION_SEMANTICS_SOURCE: finding[TRANSACTION_SEMANTICS_SOURCE],
        TRANSACTION_IMPACT_POLICY: finding[TRANSACTION_IMPACT_POLICY],
        DELTA_B_MINUS_A: delta,
        RETURN_DENOMINATOR: finding[RETURN_DENOMINATOR],
        RETURN_WEIGHT: finding[RETURN_WEIGHT],
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


def _estimated_impact(row: dict[str, object]) -> dict[str, object]:
    """Return the first-pass contribution-impact fields for one evidence row."""
    delta = row[DELTA_B_MINUS_A]
    if (
        row[DATASET] == pc_cols.SECURITY_PERFORMANCE
        and row[SOURCE_COLUMN] == pc_cols.CONTRIBUTION
        and isinstance(delta, (int, float))
        and not isinstance(delta, bool)
    ):
        return {
            ESTIMATED_RETURN_IMPACT: float(delta),
            IMPACT_BASIS: IMPACT_BASIS_SECURITY_CONTRIBUTION,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_MEDIUM,
            IMPACT_METHOD: IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
            IMPACT_MESSAGE: (
                "Uses the vendor-provided security contribution delta as a "
                "related-output impact estimate, not as root-cause attribution."
            ),
        }
    if _is_portfolio_source_field_impact_candidate(row):
        delta_float = float(delta)
        denominator = float(row[RETURN_DENOMINATOR])
        return {
            ESTIMATED_RETURN_IMPACT: delta_float / denominator,
            IMPACT_BASIS: IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
            IMPACT_MESSAGE: (
                "Approximate impact uses the portfolio source-field delta "
                "divided by beginning market value. Treat as a low-confidence "
                "screening estimate."
            ),
        }
    if _is_security_return_weighted_impact_candidate(row):
        delta_float = float(delta)
        weight = float(row[RETURN_WEIGHT])
        return {
            ESTIMATED_RETURN_IMPACT: delta_float * weight,
            IMPACT_BASIS: IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
            IMPACT_MESSAGE: (
                "Approximate impact uses the security return delta multiplied "
                "by snapshot A portfolio weight. Prefer vendor contribution "
                "deltas when available."
            ),
        }
    if _is_transaction_performance_amount_impact_candidate(row):
        delta_float = float(delta)
        denominator = float(row[RETURN_DENOMINATOR])
        return {
            ESTIMATED_RETURN_IMPACT: delta_float / denominator,
            IMPACT_BASIS: IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
            IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
            IMPACT_METHOD: IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR,
            IMPACT_MESSAGE: _transaction_performance_amount_impact_message(row),
        }
    return {
        ESTIMATED_RETURN_IMPACT: None,
        IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_METHOD: None,
        IMPACT_MESSAGE: (
            "No defensible return-impact estimate is available for this "
            "finding yet."
        ),
    }


def _is_portfolio_source_field_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether a portfolio source-field row supports a rough estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    return (
        row[DATASET] == pc_cols.PORTFOLIO_PERFORMANCE
        and row[SOURCE_COLUMN] in {pc_cols.INCOME, pc_cols.GAIN_LOSS}
        and isinstance(delta, (int, float))
        and not isinstance(delta, bool)
        and isinstance(denominator, (int, float))
        and not isinstance(denominator, bool)
        and float(denominator) != 0.0
    )


def _is_security_return_weighted_impact_candidate(row: dict[str, object]) -> bool:
    """Return whether a security return row supports a weighted estimate."""
    delta = row[DELTA_B_MINUS_A]
    weight = row[RETURN_WEIGHT]
    return (
        row[DATASET] == pc_cols.SECURITY_PERFORMANCE
        and row[SOURCE_COLUMN] == pc_cols.SECURITY_RETURN
        and isinstance(delta, (int, float))
        and not isinstance(delta, bool)
        and isinstance(weight, (int, float))
        and not isinstance(weight, bool)
        and float(weight) != 0.0
    )


def _is_transaction_performance_amount_impact_candidate(
    row: dict[str, object],
) -> bool:
    """Return whether a transaction amount row supports a performance estimate."""
    delta = row[DELTA_B_MINUS_A]
    denominator = row[RETURN_DENOMINATOR]
    cash_flow_sign = row.get(CASH_FLOW_SIGN)
    return (
        row[DATASET] == pc_cols.TRANSACTIONS
        and row[SOURCE_COLUMN] == pc_cols.AMOUNT
        and row.get(PERFORMANCE_FLOW_SIGN) == TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
        and cash_flow_sign
        in {
            TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
        }
        and isinstance(delta, (int, float))
        and not isinstance(delta, bool)
        and isinstance(denominator, (int, float))
        and not isinstance(denominator, bool)
        and float(denominator) != 0.0
    )


def _transaction_performance_amount_impact_message(row: dict[str, object]) -> str:
    """Return a provenance-aware transaction amount impact message."""
    semantics_source = row.get(TRANSACTION_SEMANTICS_SOURCE)
    source_text = _readable_transaction_semantics_source(semantics_source)
    return (
        "Approximate impact uses the source-signed transaction amount delta "
        "divided by the return denominator. Applies only when normalized "
        "sign/flow semantics mark the transaction as performance-affecting. "
        f"Transaction semantics source: {source_text}."
    )


def _readable_transaction_semantics_source(value: object) -> str:
    """Return reviewer-facing text for a transaction semantics provenance value."""
    if value == TRANSACTION_SEMANTICS_SOURCE_SOURCE:
        return "source"
    if value == TRANSACTION_SEMANTICS_SOURCE_YAML_RULE:
        return "YAML transaction_rules"
    if value == TRANSACTION_SEMANTICS_SOURCE_MIXED:
        return "mixed source and YAML transaction_rules"
    if value == TRANSACTION_SEMANTICS_SOURCE_UNKNOWN:
        return "unknown"
    if value is None or value == "":
        return "not provided"
    return str(value)


def _has_transaction_impact_method_candidate(rows: list[dict[str, object]]) -> bool:
    """Return whether any transaction row has a currently supported method."""
    return any(_is_transaction_performance_amount_impact_candidate(row) for row in rows)


def _is_usable_number(value: object) -> bool:
    """Return whether a value can be used in impact arithmetic."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0


def _root_cause_area(row: dict[str, object]) -> str:
    """Return the coarse explanation bucket for a contribution candidate."""
    dataset = row[DATASET]
    if dataset == pc_cols.SECURITY_PERFORMANCE:
        return ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
    if dataset == pc_cols.POSITIONS:
        return ROOT_CAUSE_MARKET_VALUE_OR_POSITION
    if dataset == pc_cols.TRANSACTIONS:
        return ROOT_CAUSE_TRANSACTION_ACTIVITY
    if dataset == pc_cols.PRICES:
        return ROOT_CAUSE_PRICE
    if dataset == pc_cols.FX_RATES:
        return ROOT_CAUSE_FX_RATE
    if dataset == pc_cols.CASH:
        return ROOT_CAUSE_CASH
    if dataset == pc_cols.PORTFOLIO_PERFORMANCE:
        return ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT
    if dataset == pc_cols.SECURITY_MASTER:
        return ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE
    return ROOT_CAUSE_UNEXPLAINED


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


def _impact_coverage_summary_row(
    period: dict[str, object],
    causes: list[dict[str, object]],
    transactions: list[dict[str, object]],
) -> dict[str, object]:
    """Return one cause-area estimate-coverage row for a portfolio period."""
    estimate_rows = [
        cause for cause in causes if cause.get(ESTIMATED_RETURN_IMPACT) is not None
    ]
    evidence_only_rows = [
        cause
        for cause in causes
        if cause.get(IMPACT_BASIS) == IMPACT_BASIS_NO_ESTIMATE
    ]

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
        TRANSACTION_SEMANTICS_SOURCES: _period_transaction_semantics_sources(
            transactions
        ),
        MISSING_IMPACT_INPUTS: _coverage_missing_impact_inputs(
            evidence_only_rows,
            transactions,
        ),
        IMPACT_MESSAGE: _impact_coverage_message(
            estimated_count=len(estimate_rows),
            evidence_only_count=len(evidence_only_rows),
        ),
    }


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
            ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
            ROOT_CAUSE_PRICE,
            ROOT_CAUSE_CASH,
            ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE,
        }:
            _extend_unique(missing_inputs, ["return-impact method"])
        elif cause_area == ROOT_CAUSE_FX_RATE:
            _extend_unique(missing_inputs, ["currency exposure linkage"])
        else:
            _extend_unique(missing_inputs, ["defensible impact method"])
    return ", ".join(missing_inputs)


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


def _matching_period_rows(
    table: pl.DataFrame,
    period: dict[str, object],
) -> list[dict[str, object]]:
    """Return rows from a summary table that match a portfolio period."""
    if table.is_empty():
        return []
    period_rows = table.filter(
        (pl.col(PORTFOLIO_ID) == period[PORTFOLIO_ID])
        & (pl.col(FROM_DATE) == period[FROM_DATE])
        & (pl.col(THRU_DATE) == period[THRU_DATE])
    )
    return list(period_rows.iter_rows(named=True))


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
    """Return estimate rows, preferring vendor contribution over weighted return."""
    if any(row[IMPACT_BASIS] == IMPACT_BASIS_SECURITY_CONTRIBUTION for row in rows):
        return [
            row
            for row in rows
            if row[IMPACT_BASIS] == IMPACT_BASIS_SECURITY_CONTRIBUTION
        ]
    return rows


def _summary_impact_basis(rows: list[dict[str, object]]) -> str:
    """Return the aggregate impact basis for a cause-area bucket."""
    bases = {
        str(row[IMPACT_BASIS])
        for row in rows
        if row[IMPACT_BASIS] != IMPACT_BASIS_NO_ESTIMATE
    }
    if IMPACT_BASIS_SECURITY_CONTRIBUTION in bases:
        return IMPACT_BASIS_SECURITY_CONTRIBUTION
    if IMPACT_BASIS_SECURITY_RETURN_WEIGHTED in bases:
        return IMPACT_BASIS_SECURITY_RETURN_WEIGHTED
    if IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD in bases:
        return IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD
    if IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT in bases:
        return IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT
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
    if _has_vendor_contribution_and_weighted_return(rows):
        return (
            "Estimated impact uses vendor contribution deltas. Weighted "
            "security return estimates are available as review cross-checks "
            "but are not summed to avoid double-counting. Representative "
            f"codes: {top_codes}."
        )
    if estimated_impact is not None:
        return (
            "Estimated impact is based on currently supported contribution "
            f"candidate methods. Representative codes: {top_codes}."
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


def _has_vendor_contribution_and_weighted_return(rows: list[dict[str, object]]) -> bool:
    """Return whether a security bucket has both preferred and cross-check estimates."""
    bases = {row[IMPACT_BASIS] for row in rows}
    return (
        IMPACT_BASIS_SECURITY_CONTRIBUTION in bases
        and IMPACT_BASIS_SECURITY_RETURN_WEIGHTED in bases
    )


def _portfolio_period_cause_summary_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    """Return deterministic ordering for cause-area summaries."""
    estimated_impact = row[ESTIMATED_RETURN_IMPACT]
    absolute_impact = (
        abs(float(estimated_impact))
        if isinstance(estimated_impact, (int, float))
        and not isinstance(estimated_impact, bool)
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
        MISSING_IMPACT_INPUTS: missing_impact_inputs,
        IMPACT_BASIS: IMPACT_BASIS_NO_ESTIMATE,
        IMPACT_CONFIDENCE: IMPACT_CONFIDENCE_LOW,
        IMPACT_MESSAGE: impact_message,
    }


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
        for source, count in _parse_transaction_semantics_sources(
            transaction.get(TRANSACTION_SEMANTICS_SOURCES)
        ).items():
            counts[source] = counts.get(source, 0) + count
    return _format_transaction_semantics_source_counts(counts)


def _transaction_semantics_source_counts(rows: list[dict[str, object]]) -> str:
    """Return compact transaction semantics provenance counts for evidence rows."""
    counts: dict[str, int] = {}
    for row in rows:
        source = row.get(TRANSACTION_SEMANTICS_SOURCE)
        if not isinstance(source, str) or not source:
            continue
        counts[source] = counts.get(source, 0) + 1
    return _format_transaction_semantics_source_counts(counts)


def _parse_transaction_semantics_sources(value: object) -> dict[str, int]:
    """Return provenance counts parsed from a transaction summary string."""
    if not isinstance(value, str) or not value:
        return {}

    counts: dict[str, int] = {}
    for part in value.split(","):
        source, separator, count_text = part.strip().partition(":")
        if not source or not separator:
            continue
        try:
            count = int(count_text.strip())
        except ValueError:
            continue
        counts[source.strip()] = counts.get(source.strip(), 0) + count
    return counts


def _format_transaction_semantics_source_counts(counts: Mapping[str, int]) -> str:
    """Return stable readable transaction semantics provenance counts."""
    ordered_sources = (
        TRANSACTION_SEMANTICS_SOURCE_SOURCE,
        TRANSACTION_SEMANTICS_SOURCE_MIXED,
        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
        TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    )
    parts = [
        f"{source}: {counts[source]}"
        for source in ordered_sources
        if counts.get(source, 0) > 0
    ]
    other_sources = sorted(source for source in counts if source not in ordered_sources)
    parts.extend(
        f"{source}: {counts[source]}"
        for source in other_sources
        if counts.get(source, 0) > 0
    )
    return ", ".join(parts)


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
            if (
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


def _changed_transaction_fields(rows: list[dict[str, object]]) -> list[str]:
    """Return changed transaction source fields in stable order."""
    fields = {str(row[SOURCE_COLUMN]) for row in rows if row[SOURCE_COLUMN] is not None}
    return sorted(fields, key=_transaction_field_sort_key)


def _transaction_field_sort_key(field: str) -> tuple[int, str]:
    """Return stable business-oriented transaction field ordering."""
    order = {
        pc_cols.AMOUNT: 0,
        pc_cols.QUANTITY: 1,
        pc_cols.PRICE: 2,
    }
    return (order.get(field, 99), field)


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
        DIRECT_INPUT: 300,
        RELATED_OUTPUT: 200,
        CONTEXT: 100,
        TARGET_OUTPUT: 0,
    }.get(evidence_role, 0)


def _dataset_priority_score(dataset: str) -> int:
    """Return the review-priority weight for a normalized dataset."""
    return {
        pc_cols.PORTFOLIO_PERFORMANCE: 40,
        pc_cols.TRANSACTIONS: 35,
        pc_cols.POSITIONS: 35,
        pc_cols.CASH: 30,
        pc_cols.PRICES: 25,
        pc_cols.FX_RATES: 25,
        pc_cols.SECURITY_PERFORMANCE: 10,
        pc_cols.SECURITY_MASTER: 0,
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
            PRICE_FINDING_COUNT: pl.UInt32,
            FX_RATE_FINDING_COUNT: pl.UInt32,
            TRANSACTION_FINDING_COUNT: pl.UInt32,
            POSITION_FINDING_COUNT: pl.UInt32,
            CASH_FINDING_COUNT: pl.UInt32,
            REFERENCE_FINDING_COUNT: pl.UInt32,
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
            PRICE_FINDING_COUNT: pl.UInt32,
            TRANSACTION_FINDING_COUNT: pl.UInt32,
            POSITION_FINDING_COUNT: pl.UInt32,
            REFERENCE_FINDING_COUNT: pl.UInt32,
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
            SOURCE_COLUMN: pl.String,
            TRANSACTION_CATEGORY: pl.String,
            CASH_FLOW_SIGN: pl.String,
            PERFORMANCE_FLOW_SIGN: pl.String,
            TRANSACTION_SEMANTICS_SOURCE: pl.String,
            TRANSACTION_IMPACT_POLICY: pl.String,
            DELTA_B_MINUS_A: pl.Float64,
            RETURN_DENOMINATOR: pl.Float64,
            RETURN_WEIGHT: pl.Float64,
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
            MISSING_IMPACT_INPUTS: pl.String,
            IMPACT_BASIS: pl.String,
            IMPACT_CONFIDENCE: pl.String,
            IMPACT_MESSAGE: pl.String,
        }
    )
