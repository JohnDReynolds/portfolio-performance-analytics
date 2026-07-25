"""Public runner functions for Audit workflows."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit.performance_comparison.compare import PerformanceComparison
import ppar.audit.performance_comparison.explain as _pc_explain
import ppar.audit.field_roles as _field_roles
import ppar.audit.lineage as _pc_lineage
from ppar.audit.performance_comparison.findings import (
    DATASET,
    DELTA_B_MINUS_A,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    IMPACT_POLICY,
    MESSAGE,
    PERFORMANCE_FLOW_SIGN,
    PORTFOLIO_ID,
    SECURITY_ID,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SUPPRESSED,
    THRU_DATE,
    TRANSACTION_IMPACT_POLICY,
    Finding,
    findings_to_polars,
)
from ppar.audit.specification import (
    COMPARISON_LEVELS,
    AuditSpecification,
)
import ppar.audit.schema as pc_cols
from ppar.audit.transactions import (
    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
)
from ppar.audit.workbook_reconstruction import WorkbookReconstructionCache
import ppar.common as util

__all__ = [
    "compact_findings_table",
    "compare_snapshots",
    "summarize_findings",
    "validate_causal_attribution_ready",
    "validate_yaml_setup_complete",
]


class AuditComparisonViews:
    """Compute shared Audit findings once and expose level-specific views.

    Attributes:
        specification_path: Comparison YAML used by every requested view.
        include_suppressed: Whether returned tables retain suppressed findings.
        require_causal_attribution: Whether each requested view must pass
            supported causal-attribution readiness validation.

    Notes:
        Portfolio and security performance are genuinely different targets and
        remain separate calculations. Holdings, FX rates, splits, and
        transactions describe the same economic changes and are calculated
        once, then given the transaction-policy label appropriate to each view.
    """

    def __init__(
        self,
        specification_path: util.PathLike,
        *,
        include_suppressed: bool = True,
        require_causal_attribution: bool = False,
        reconstruction_cache: WorkbookReconstructionCache | None = None,
    ) -> None:
        """Initialize one lazy, canonical comparison run.

        Args:
            specification_path: Path to an Audit YAML file.
            include_suppressed: Whether returned views include suppressed rows.
            require_causal_attribution: Whether returned views must have all
                setup needed by supported causal-attribution methods.
            reconstruction_cache: Optional run-scoped reconstruction results
                shared with report construction.
        """
        self.specification_path = specification_path
        self.include_suppressed = include_suppressed
        self.require_causal_attribution = require_causal_attribution
        self._reconstruction_cache = (
            reconstruction_cache
            or WorkbookReconstructionCache(specification_path)
        )
        self._comparisons: dict[str, PerformanceComparison] = {}
        self._view_tables: dict[str, pl.DataFrame] = {}
        self._shared_findings: list[Finding] | None = None
        self._financial_inputs_validated = False

    def findings(self, comparison_level: str) -> pl.DataFrame:
        """Return a cached portfolio- or security-level findings view.

        Args:
            comparison_level: Requested primary performance-result level.

        Returns:
            Complete findings table for the requested review level.

        Raises:
            PpaError: If the level is unsupported or its required inputs and
                policies are unavailable.
        """
        if comparison_level not in COMPARISON_LEVELS:
            allowed_values = ", ".join(sorted(COMPARISON_LEVELS))
            raise PpaError(
                f"comparison_level must be one of: {allowed_values}.",
                504,
            )
        cached = self._view_tables.get(comparison_level)
        if cached is not None:
            return cached

        comparison = self._comparison(comparison_level)
        if not self._financial_inputs_validated:
            comparison.validate_inputs()
            self._financial_inputs_validated = True
        primary_findings = comparison.compare_primary_performance()
        if self._shared_findings is None:
            self._shared_findings = comparison.compare_shared_sources()
            shared_findings = self._shared_findings
        else:
            shared_findings = comparison.retarget_shared_findings(
                self._shared_findings
            )
        findings = comparison.apply_view_rules(
            [*primary_findings, *shared_findings]
        )
        findings_table = findings_to_polars(findings)
        _pc_lineage.assert_finding_source_lineage(findings_table)
        if not self.include_suppressed:
            findings_table = findings_table.filter(~pl.col(SUPPRESSED))
        if self.require_causal_attribution:
            validate_causal_attribution_ready(findings_table)
        self._view_tables[comparison_level] = findings_table
        return findings_table

    def _comparison(self, comparison_level: str) -> PerformanceComparison:
        """Return the cached comparison engine for one result level."""
        if comparison_level not in self._comparisons:
            specification = AuditSpecification(
                self.specification_path,
                comparison_level=comparison_level,
            )
            self._comparisons[comparison_level] = PerformanceComparison(
                specification,
                reconstruction_cache=self._reconstruction_cache,
            )
        return self._comparisons[comparison_level]


def compare_snapshots(
    specification_path: util.PathLike,
    *,
    include_suppressed: bool = True,
    require_causal_attribution: bool = False,
    comparison_level: str | None = None,
) -> pl.DataFrame:
    """Compare two configured snapshots and return a findings table.

    Args:
        specification_path: Path to an Audit YAML file.
        include_suppressed: Whether to include findings marked suppressed by
            configured suppression rules.
        require_causal_attribution: Whether changed portfolio periods must have
            all YAML setup needed by supported causal attribution methods before
            results are returned. This does not require every performance
            change to be fully explained.
        comparison_level: Optional primary performance-result level override.
            When omitted, the YAML must provide ``comparison.level``. The
            user-facing ``ppar audit`` command always supplies this explicitly
            for each applicable report level.

    Returns:
        Polars DataFrame containing one row per finding. If no findings are
        present, the DataFrame is empty but still has the standard finding
        columns.

    Raises:
        PpaError: If the comparison specification is invalid, required files
            are missing, or source columns cannot be resolved.
    """
    specification = AuditSpecification(
        specification_path,
        comparison_level=comparison_level,
    )
    findings = PerformanceComparison(specification).compare()
    findings_table = findings_to_polars(findings)
    _pc_lineage.assert_finding_source_lineage(findings_table)
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
            (
                "",
                "modified_dietz cross-check only",
                _pc_explain.NEUTRAL_FLOW_IMPACT_METHOD,
            )
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
        PpaError: If a changed field has no explicit accounting role, or a
            classified performance input lacks additive, evidence-only, or
            suppression YAML.
    """
    if findings.is_empty():
        return
    _validate_finding_field_classifications(findings)
    active_findings = findings
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
    source_column = row.get(SOURCE_COLUMN)
    if source_column is None:
        return False
    dataset = row.get(DATASET)
    return _field_roles.requires_explicit_impact_policy(dataset, source_column)


def _validate_finding_field_classifications(findings: pl.DataFrame) -> None:
    """Raise if any changed field lacks an explicit accounting role.

    Suppression is intentionally ignored here: YAML may decide how a known
    field is treated, but it cannot silently classify a newly introduced field.
    """
    if SOURCE_COLUMN not in findings.columns or DATASET not in findings.columns:
        return
    for dataset, source_column in findings.select(
        DATASET,
        SOURCE_COLUMN,
    ).unique().iter_rows():
        if source_column is None:
            continue
        _field_roles.requires_explicit_impact_policy(dataset, source_column)


def _finding_has_yaml_policy(row: dict[str, object]) -> bool:
    """Return whether a finding has additive or evidence-only YAML treatment."""
    if (
        row.get(DATASET) == pc_cols.TRANSACTIONS
        and row.get(PERFORMANCE_FLOW_SIGN) == TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
    ):
        return True
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
