"""Public runner functions for performance comparison workflows."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.compare import PerformanceComparison
import ppar.performance_comparison.explain as _pc_explain
import ppar.performance_comparison.field_roles as _field_roles
from ppar.performance_comparison.findings import (
    DATASET,
    DELTA_B_MINUS_A,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    IMPACT_POLICY,
    MESSAGE,
    PORTFOLIO_ID,
    SECURITY_ID,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SUPPRESSED,
    THRU_DATE,
    TRANSACTION_IMPACT_POLICY,
    findings_to_polars,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.performance_comparison.schema as pc_cols
import ppar.utilities as util

__all__ = [
    "compact_findings_table",
    "compare_snapshots",
    "summarize_findings",
    "validate_causal_attribution_ready",
    "validate_yaml_setup_complete",
]


def compare_snapshots(
    specification_path: util.PathLike,
    *,
    include_suppressed: bool = True,
    require_causal_attribution: bool = False,
    comparison_level: str | None = None,
) -> pl.DataFrame:
    """Compare two configured snapshots and return a findings table.

    Args:
        specification_path: Path to a performance comparison YAML file.
        include_suppressed: Whether to include findings marked suppressed by
            configured suppression rules.
        require_causal_attribution: Whether changed portfolio periods must have
            all YAML setup needed by supported causal attribution methods before
            results are returned. This does not require every performance
            change to be fully explained.
        comparison_level: Optional primary performance-result level override.
            When omitted, ``comparison.level`` from the YAML is used.

    Returns:
        Polars DataFrame containing one row per finding. If no findings are
        present, the DataFrame is empty but still has the standard finding
        columns.

    Raises:
        PpaError: If the comparison specification is invalid, required files
            are missing, or source columns cannot be resolved.
    """
    specification = PerformanceComparisonSpecification(
        specification_path,
        comparison_level=comparison_level,
    )
    findings = PerformanceComparison(specification).compare()
    findings_table = findings_to_polars(findings)
    validate_findings = findings_table
    if not include_suppressed:
        validate_findings = validate_findings.filter(~pl.col(SUPPRESSED))
    if require_causal_attribution:
        validate_causal_attribution_ready(validate_findings)
    if include_suppressed:
        return findings_table
    return findings_table.filter(~pl.col(SUPPRESSED))


def validate_causal_attribution_ready(findings: pl.DataFrame) -> None:
    """Raise if changed portfolio periods are not ready for causal attribution.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.

    Raises:
        PpaError: If any changed portfolio period still has missing setup or
            unexplained changed rows.
    """
    active_findings = findings
    if active_findings.is_empty():
        return
    if SUPPRESSED in active_findings.columns:
        active_findings = active_findings.filter(~pl.col(SUPPRESSED))
    coverage = _pc_explain.portfolio_period_impact_coverage_summary(active_findings)
    if coverage.is_empty():
        return

    incomplete = coverage.filter(
        ~pl.col(_pc_explain.MISSING_IMPACT_INPUTS).is_in(
            ("", "modified_dietz cross-check only")
        )
    )
    if incomplete.is_empty():
        return

    details = [
        _causal_attribution_issue(row)
        for row in incomplete.iter_rows(named=True)
    ]
    raise PpaError(
        "Causal attribution setup is incomplete: " + "; ".join(details),
        504,
    )


def validate_yaml_setup_complete(findings: pl.DataFrame) -> None:
    """Raise if changed source-data fields lack explicit YAML policy.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.

    Raises:
        PpaError: If a changed source-data field that ppar knows how to classify
            is not explicitly configured as additive, evidence-only, or
            suppressed by YAML.
    """
    active_findings = findings
    if active_findings.is_empty():
        return
    if SUPPRESSED in active_findings.columns:
        active_findings = active_findings.filter(~pl.col(SUPPRESSED))
    if active_findings.is_empty():
        return

    missing = [
        row
        for row in active_findings.iter_rows(named=True)
        if _finding_requires_yaml_policy(row) and not _finding_has_yaml_policy(row)
    ]
    if not missing:
        return

    details = [_missing_yaml_policy_issue(row) for row in missing]
    raise PpaError(
        "YAML setup is incomplete: " + "; ".join(dict.fromkeys(details)),
        504,
    )


def _finding_requires_yaml_policy(row: dict[str, object]) -> bool:
    """Return whether a finding must be explicitly classified in YAML."""
    evidence_role = row.get(EVIDENCE_ROLE)
    if evidence_role not in {"direct_input", "context"}:
        return False

    dataset = row.get(DATASET)
    source_column = row.get(SOURCE_COLUMN)
    if _field_roles.is_reported_performance_component(dataset, source_column):
        return False
    if _field_roles.is_context(dataset, source_column):
        return False
    if dataset == pc_cols.TRANSACTIONS:
        return source_column in {
            pc_cols.AMOUNT,
            pc_cols.QUANTITY,
            pc_cols.PRICE,
            pc_cols.COMMISSION,
        }
    if dataset == pc_cols.HOLDINGS:
        return source_column in {
            pc_cols.MARKET_VALUE,
            pc_cols.ACCRUED,
            pc_cols.QUANTITY,
            pc_cols.COST,
        }
    if dataset == pc_cols.CASH:
        return source_column in {pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE}
    if dataset == pc_cols.FX_RATES:
        return source_column == pc_cols.FX_RATE
    if dataset == pc_cols.PORTFOLIO_PERFORMANCE:
        return source_column in {pc_cols.INCOME, pc_cols.GAIN_LOSS}
    return False


def _finding_has_yaml_policy(row: dict[str, object]) -> bool:
    """Return whether a finding has additive or evidence-only YAML treatment."""
    return any(
        isinstance(row.get(column), str) and bool(str(row.get(column)).strip())
        for column in (IMPACT_POLICY, TRANSACTION_IMPACT_POLICY)
    )


def _missing_yaml_policy_issue(row: dict[str, object]) -> str:
    """Return one concise missing-YAML issue."""
    review_key = f"{row.get(PORTFOLIO_ID)}::{row.get(FROM_DATE)}::{row.get(THRU_DATE)}"
    security_id = row.get(SECURITY_ID)
    if security_id is not None:
        review_key = f"{review_key}::{security_id}"
    return (
        f"{review_key} {row.get(DATASET)}.{row.get(SOURCE_COLUMN)} needs "
        "additive, evidence-only, or suppression YAML"
    )


def _causal_attribution_issue(row: dict[str, object]) -> str:
    """Return one strict-mode attribution issue."""
    review_key = (
        f"{row.get(PORTFOLIO_ID)}::{row.get(FROM_DATE)}::{row.get(THRU_DATE)}"
    )
    missing_inputs = row.get(_pc_explain.MISSING_IMPACT_INPUTS)
    if isinstance(missing_inputs, str) and missing_inputs:
        return f"{review_key} missing YAML setup: {missing_inputs}"
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    return f"{review_key} is {status}; add causal attribution setup"


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
