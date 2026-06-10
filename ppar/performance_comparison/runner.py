"""Public runner functions for performance comparison workflows."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.explain import (
    ABSOLUTE_DELTA,
    AMOUNT_DELTA,
    CASH_FINDING_COUNT,
    CHANGED_FIELDS,
    CONTEXT_FINDING_COUNT,
    DIRECT_INPUT_FINDING_COUNT,
    EVIDENCE_GROUP,
    FINDING_COUNT,
    FX_RATE_FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
    ESTIMATED_RETURN_IMPACT,
    IMPACT_BASIS,
    IMPACT_BASIS_NO_ESTIMATE,
    IMPACT_BASIS_SECURITY_CONTRIBUTION,
    IMPACT_CONFIDENCE,
    IMPACT_CONFIDENCE_LOW,
    IMPACT_CONFIDENCE_MEDIUM,
    IMPACT_MESSAGE,
    IMPACT_METHOD,
    IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
    PORTFOLIO_FINDING_COUNT,
    PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS,
    PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS,
    PORTFOLIO_PERIOD_SUMMARY_COLUMNS,
    PORTFOLIO_RETURN_DELTA,
    POSITION_FINDING_COUNT,
    PRICE_FINDING_COUNT,
    PRIORITY_REASON,
    PRIORITY_SCORE,
    REFERENCE_FINDING_COUNT,
    RELATED_OUTPUT_FINDING_COUNT,
    REVIEW_RANK,
    ROOT_CAUSE_AREA,
    ROOT_CAUSE_CASH,
    ROOT_CAUSE_CLASSIFICATION_OR_REFERENCE,
    ROOT_CAUSE_FX_RATE,
    ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
    ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
    ROOT_CAUSE_PRICE,
    ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION,
    ROOT_CAUSE_TRANSACTION_ACTIVITY,
    ROOT_CAUSE_UNEXPLAINED,
    SECURITY_FINDING_COUNT,
    SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    SECURITY_PERIOD_SUMMARY_COLUMNS,
    SECURITY_RETURN_DELTA,
    TOP_CODES,
    PRICE_DELTA,
    QUANTITY_DELTA,
    TRANSACTION_FINDING_COUNT,
    TRANSACTION_ACTIVITY_SUMMARY_COLUMNS,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_summary,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
)
from ppar.performance_comparison.findings import (
    DATASET,
    DELTA_B_MINUS_A,
    CONTEXT,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    MESSAGE,
    PORTFOLIO_ID,
    RELATED_OUTPUT,
    SECURITY_ID,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SUPPRESSED,
    TARGET_OUTPUT,
    THRU_DATE,
    findings_to_polars,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util


def compare_snapshots(
    specification_path: util.PathLike,
    *,
    include_suppressed: bool = True,
) -> pl.DataFrame:
    """Compare two configured snapshots and return a findings table.

    Args:
        specification_path: Path to a ``ppar_performance_comparison.yaml`` file.
        include_suppressed: Whether to include findings marked suppressed by
            configured suppression rules.

    Returns:
        Polars DataFrame containing one row per finding. If no findings are
        present, the DataFrame is empty but still has the standard finding
        columns.

    Raises:
        PpaError: If the comparison specification is invalid, required files
            are missing, or source columns cannot be resolved.
    """
    specification = PerformanceComparisonSpecification(specification_path)
    findings = PerformanceComparison(specification).compare()
    findings_table = findings_to_polars(findings)
    if include_suppressed:
        return findings_table
    return findings_table.filter(~pl.col(SUPPRESSED))


def summarize_findings(findings: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return compact finding-count summaries.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.

    Returns:
        Dictionary containing count tables keyed by ``"by_code"``,
        ``"by_dataset"``, ``"by_evidence_role"``, ``"by_suppressed"``, and
        ``"by_code_suppressed"``. Empty findings return empty summary tables
        with stable columns.
    """
    return {
        "by_code": _count_by(findings, FINDING_CODE),
        "by_dataset": _count_by(findings, DATASET),
        "by_evidence_role": _count_by(findings, EVIDENCE_ROLE),
        "by_suppressed": _count_by_suppressed(findings),
        "by_code_suppressed": _count_by_code_suppressed(findings),
    }


def compact_findings_table(
    findings: pl.DataFrame,
    *,
    include_suppressed: bool = False,
) -> pl.DataFrame:
    """Return a report-friendly findings table with the most useful columns.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        include_suppressed: Whether to include findings marked suppressed.

    Returns:
        DataFrame containing a compact, stable subset of finding columns.
    """
    compact_columns = [
        FINDING_CODE,
        DATASET,
        EVIDENCE_ROLE,
        PORTFOLIO_ID,
        SECURITY_ID,
        FROM_DATE,
        THRU_DATE,
        SOURCE_FILE,
        SOURCE_COLUMN,
        DELTA_B_MINUS_A,
        MESSAGE,
    ]
    if findings.is_empty():
        return pl.DataFrame(
            schema={
                column: findings.schema.get(column, pl.Null)
                for column in compact_columns
            }
        )

    compact_findings = findings
    if not include_suppressed and SUPPRESSED in compact_findings.columns:
        compact_findings = compact_findings.filter(~pl.col(SUPPRESSED))
    if compact_findings.is_empty():
        return pl.DataFrame(
            schema={
                column: findings.schema.get(column, pl.Null)
                for column in compact_columns
            }
        )
    return compact_findings.select(compact_columns)


def _count_by(findings: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return finding counts grouped by one output column."""
    if findings.is_empty():
        return pl.DataFrame(schema={column: pl.String, "count": pl.UInt32})
    return findings.group_by(column).len(name="count").sort(column)


def _count_by_suppressed(findings: pl.DataFrame) -> pl.DataFrame:
    """Return finding counts grouped by suppression state."""
    if findings.is_empty():
        return pl.DataFrame(schema={SUPPRESSED: pl.Boolean, "count": pl.UInt32})
    return findings.group_by(SUPPRESSED).len(name="count").sort(SUPPRESSED)


def _count_by_code_suppressed(findings: pl.DataFrame) -> pl.DataFrame:
    """Return finding counts grouped by code and suppression state."""
    if findings.is_empty():
        return pl.DataFrame(
            schema={
                FINDING_CODE: pl.String,
                SUPPRESSED: pl.Boolean,
                "count": pl.UInt32,
            }
        )
    return findings.group_by(FINDING_CODE, SUPPRESSED).len(name="count").sort(
        FINDING_CODE,
        SUPPRESSED,
    )
