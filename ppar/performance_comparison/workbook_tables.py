"""Build review workbook tables for performance comparison findings."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import rendering as _pc_rendering
from ppar.performance_comparison import review_keys as _pc_review_keys
from ppar.performance_comparison import workbook as _pc_workbook

__all__ = [
    "write_performance_comparison_review_workbook",
]

_REVIEW_STATUS = "review_status"
_REVIEW_CUES = "review_cues"
_SUGGESTED_NEXT_STEP = "suggested_next_step"
_REVIEW_KEY = _pc_review_keys.REVIEW_KEY
_REVIEW_DETAIL_ARTIFACTS = "review_detail_artifacts"
_DASHBOARD_MISSING_INPUTS = "dashboard_missing_inputs"
_DASHBOARD_OPEN_SECTION = "dashboard_open_section"
_PROBLEM = "problem"
_ACTION_REQUIRED = "action_required"
_WHY_IT_MATTERS = "why_it_matters"
_EVIDENCE_SECTION = "evidence_section"
_PERFORMANCE_CHANGE = "performance_change"
_ESTIMATED_CAUSE_TOTAL = "estimated_cause_total"
_UNEXPLAINED_CHANGE = "unexplained_change"
_USE = "use"
_USE_PRIORITY = "_use_priority"
_CHANGE_LABEL = "change_label"
_CHANGE = "change"
_ESTIMATED_IMPACT = "estimated_impact"
_IMPACT_STATUS = "impact_status"
_NEXT_ACTION = "next_action"
_REQUIRED_YAML_SETUP = "required_yaml_setup"
_USE_EXPLAINS_CHANGE = "Explains Change"
_USE_REVIEW_CONTEXT = "Review Context"
_USE_DIAGNOSTIC = "Diagnostic"
_IMPACT_STATUS_ESTIMATED = "Estimated"
_IMPACT_STATUS_MISSING_METHOD = "Missing impact method"
_IMPACT_STATUS_MISSING_INPUT = "Missing impact input"
_IMPACT_STATUS_REVIEW_ONLY = "Review only"
_NO_UNDERLYING_CAUSE_DATASET = "no_underlying_cause_found"
_WORKBOOK_ROW_KIND_UNDERLYING_CAUSE = "underlying_cause"
_WORKBOOK_ROW_KIND_DERIVED_CHECK = "derived_check"
_WORKBOOK_ROW_KIND_CONTEXT = "context"
_WORKBOOK_ROW_KIND_DIAGNOSTIC = "diagnostic"
_WORKBOOK_ROW_KIND_OTHER = "other"
_STATUS_FULLY_EXPLAINED = "Fully Explained"
_STATUS_NEEDS_SETUP = "Missing YAML Specifications"
_STATUS_PARTLY_EXPLAINED = "Partly Explained"
_STATUS_UNEXPLAINED = "Unexplained"
_CONTEXT_USE = "context_use"
_REVIEW_PRIORITY = "review_priority"
_REVIEW_PRIORITY_REASON = "review_priority_reason"
_RETURN_IMPACT_TREATMENT = "return_impact_treatment"
_format_value = _pc_rendering.format_value
_with_period_review_key = _pc_review_keys.with_period_review_key
_with_security_review_key = _pc_review_keys.with_security_review_key


def write_performance_comparison_review_workbook(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    top_evidence_limit: int = 10,
    comparison_path: util.PathLike | None = None,
) -> Path:
    """Write an XLSX workbook for performance comparison review.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_path: Destination workbook path. Parent directories are created
            when needed.
        top_evidence_limit: Reserved for parity with bundle/report writers.
        comparison_path: Optional path to the comparison YAML. When provided,
            the ``Underlying Causes`` sheet can name the exact file to update
            for missing attribution setup.

    Returns:
        Normalized workbook path.

    Raises:
        PpaError: If the optional Excel dependency group is not installed.

    Notes:
        The workbook is a presentation layer over the same impact coverage,
        top-evidence, and findings output used by the HTML/CSV reports. It does
        not add comparison logic.
    """
    active_findings = _active_findings(findings)
    del top_evidence_limit
    return _pc_workbook.write_review_workbook_sheets(
        _review_workbook_sheets(
            portfolio_changes=_workbook_portfolio_changes_table(active_findings),
            security_changes=_workbook_security_changes_table(active_findings),
            underlying_causes=_workbook_underlying_causes_table(
                active_findings,
                comparison_path=comparison_path,
            ),
            derived_checks=_workbook_derived_checks_table(active_findings),
            context=_workbook_context_table(active_findings),
            findings=findings,
        ),
        output_path,
        column_tooltip=_workbook_column_tooltip,
    )


def _review_workbook_sheets(
    *,
    portfolio_changes: pl.DataFrame,
    security_changes: pl.DataFrame,
    underlying_causes: pl.DataFrame,
    derived_checks: pl.DataFrame,
    context: pl.DataFrame,
    findings: pl.DataFrame,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return workbook sheet specifications in reviewer-first order."""
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="portfolio_changes",
            sheet_name="Portfolio Differences",
            table=portfolio_changes,
            columns=_workbook_portfolio_changes_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="security_changes",
            sheet_name="Security Differences",
            table=security_changes,
            columns=_workbook_security_changes_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="underlying_causes",
            sheet_name="Underlying Causes",
            table=underlying_causes,
            columns=_workbook_underlying_cause_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="derived_checks",
            sheet_name="Reported Performance Checks",
            table=derived_checks,
            columns=_workbook_non_additive_change_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="context",
            sheet_name="Context",
            table=context,
            columns=_workbook_non_additive_change_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name="raw_audit_trail",
            sheet_name="Raw Audit Trail",
            table=_workbook_sorted_table(
                _with_period_review_key(findings),
                _workbook_left_review_sort_columns(),
            ),
            columns=_workbook_findings_columns(findings),
            labels=_workbook_column_labels(),
        ),
    )


def _workbook_portfolio_changes_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return one workbook row per changed portfolio period."""
    coverage = _with_period_review_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    if coverage.is_empty():
        return _workbook_empty_portfolio_changes_table()
    underlying_totals = _workbook_underlying_impact_totals(findings)
    rows = [
        _workbook_performance_change_row(
            {
                **row,
                "_underlying_estimated_total": underlying_totals.get(
                    _workbook_period_key(row),
                    0.0,
                ),
            }
        )
        for row in coverage.iter_rows(named=True)
    ]
    return _workbook_sorted_table(
        pl.DataFrame(rows),
        [_REVIEW_KEY],
    )


def _workbook_underlying_impact_totals(
    findings: pl.DataFrame,
) -> dict[tuple[object, object, object], float]:
    """Return explained difference totals from underlying input rows."""
    totals: dict[tuple[object, object, object], float] = {}
    for row in _workbook_ranked_changed_rows(findings):
        if not _workbook_is_underlying_cause_row(row):
            continue
        estimated_impact = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
        if estimated_impact is None:
            continue
        key = _workbook_period_key(row)
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


def _workbook_period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the workbook period key for a row."""
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
    )


def _workbook_performance_change_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return one plain-English performance-change workbook row."""
    performance_change = _number_or_none(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))
    estimated_total = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL))
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    if underlying_estimated_total is not None:
        estimated_total = underlying_estimated_total
    unexplained_change = None
    if performance_change is not None:
        unexplained_change = performance_change - (estimated_total or 0.0)
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _PERFORMANCE_CHANGE: performance_change,
        _ESTIMATED_CAUSE_TOTAL: estimated_total,
        _UNEXPLAINED_CHANGE: unexplained_change,
        _REVIEW_STATUS: _workbook_explanation_status(row),
        _NEXT_ACTION: _workbook_performance_next_action(row),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_explanation_status(row: Mapping[str, object]) -> str:
    """Return a plain-language explanation status for a portfolio period."""
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    performance_change = _number_or_none(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        if underlying_estimated_total is not None and performance_change is not None:
            residual = performance_change - underlying_estimated_total
            if abs(residual) <= 0.00000001:
                return _STATUS_FULLY_EXPLAINED
            if abs(underlying_estimated_total) > 0:
                return _STATUS_PARTLY_EXPLAINED
            return _STATUS_UNEXPLAINED
        return _STATUS_FULLY_EXPLAINED
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_MISSING_INPUTS:
        return _STATUS_NEEDS_SETUP
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return _STATUS_PARTLY_EXPLAINED
    return _STATUS_UNEXPLAINED


def _workbook_performance_next_action(row: Mapping[str, object]) -> str:
    """Return a plain-language next action for a portfolio period."""
    missing_inputs = row.get(_pc_explain.MISSING_IMPACT_INPUTS)
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    if _has_text(missing_inputs):
        return f"Add missing YAML specifications: {_format_value(missing_inputs)}."
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    performance_change = _number_or_none(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))
    if underlying_estimated_total is not None and performance_change is not None:
        residual = performance_change - underlying_estimated_total
        if abs(residual) <= 0.00000001:
            return "None"
        if abs(underlying_estimated_total) > 0:
            return "Review the Underlying Causes sheet for this portfolio and period."
        return "Review the Underlying Causes sheet for this portfolio and period."
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        return "None"
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return "Review unexplained difference rows and add setup if needed."
    return "Add setup so rows can explain the performance difference."


def _workbook_empty_portfolio_changes_table() -> pl.DataFrame:
    """Return a reviewer-facing Portfolio Differences row for clean comparisons."""
    return pl.DataFrame(
        [
            {
                _pc_findings.PORTFOLIO_ID: "No portfolio performance differences found",
                _pc_findings.FROM_DATE: None,
                _pc_findings.THRU_DATE: None,
                _PERFORMANCE_CHANGE: None,
                _ESTIMATED_CAUSE_TOTAL: None,
                _UNEXPLAINED_CHANGE: None,
                _REVIEW_STATUS: "No differences",
                _NEXT_ACTION: "None",
                _REVIEW_KEY: "NO_PORTFOLIO_PERFORMANCE_DIFFERENCES",
            }
        ],
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _PERFORMANCE_CHANGE: pl.Float64,
            _ESTIMATED_CAUSE_TOTAL: pl.Float64,
            _UNEXPLAINED_CHANGE: pl.Float64,
            _REVIEW_STATUS: pl.String,
            _NEXT_ACTION: pl.String,
            _REVIEW_KEY: pl.String,
        },
    )


def _workbook_security_changes_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return one workbook row per changed security period."""
    summary = _with_security_review_key(_pc_explain.security_period_summary(findings))
    security_totals = _workbook_security_underlying_impact_totals(findings)
    rows: list[dict[str, object]] = []
    if not summary.is_empty():
        rows = [
            _workbook_security_change_row(
                {
                    **row,
                    "_underlying_estimated_total": security_totals.get(
                        _workbook_security_period_key(row),
                        0.0,
                    ),
                }
            )
            for row in summary.iter_rows(named=True)
        ]
    rows.extend(_workbook_missing_security_change_rows(findings, rows))
    if not rows:
        return _workbook_empty_security_changes_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows),
        [_REVIEW_KEY, _pc_findings.SECURITY_ID],
    )


def _workbook_security_underlying_impact_totals(
    findings: pl.DataFrame,
) -> dict[tuple[object, object, object, object], float]:
    """Return security-level explained totals from underlying input rows."""
    totals: dict[tuple[object, object, object, object], float] = {}
    for row in _workbook_ranked_changed_rows(findings):
        if not _workbook_is_underlying_cause_row(row):
            continue
        if not _has_text(row.get(_pc_findings.SECURITY_ID)):
            continue
        estimated_impact = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
        if estimated_impact is None:
            continue
        key = _workbook_security_period_key(row)
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


def _workbook_security_period_key(
    row: Mapping[str, object],
) -> tuple[object, object, object, object]:
    """Return the workbook security-period key for a row."""
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
        row.get(_pc_findings.SECURITY_ID),
    )


def _workbook_security_change_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return one security-level result row for the workbook."""
    performance_change = _number_or_none(row.get(_pc_explain.SECURITY_RETURN_DELTA))
    explained_change = _number_or_none(row.get("_underlying_estimated_total"))
    unexplained_change = None
    if performance_change is not None:
        unexplained_change = performance_change - (explained_change or 0.0)
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _PERFORMANCE_CHANGE: performance_change,
        _ESTIMATED_CAUSE_TOTAL: explained_change,
        _UNEXPLAINED_CHANGE: unexplained_change,
        _REVIEW_STATUS: "Security Difference",
        _NEXT_ACTION: (
            "Review Underlying Causes for this security and period."
        ),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_missing_security_change_rows(
    findings: pl.DataFrame,
    security_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return placeholder rows for portfolio periods without security differences."""
    coverage = _with_period_review_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    if coverage.is_empty():
        return []

    security_period_keys = {
        _workbook_period_key(row)
        for row in security_rows
    }
    rows: list[dict[str, object]] = []
    for row in coverage.iter_rows(named=True):
        if _workbook_period_key(row) in security_period_keys:
            continue
        rows.append(_workbook_missing_security_change_row(row))
    return rows


def _workbook_missing_security_change_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Return a reviewer-facing placeholder for periods with no security differences."""
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _pc_findings.SECURITY_ID: "No security performance differences found",
        _PERFORMANCE_CHANGE: None,
        _ESTIMATED_CAUSE_TOTAL: None,
        _UNEXPLAINED_CHANGE: None,
        _REVIEW_STATUS: "No differences",
        _NEXT_ACTION: "None",
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_empty_security_changes_table() -> pl.DataFrame:
    """Return an empty workbook Security Differences table."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _pc_findings.SECURITY_ID: pl.String,
            _PERFORMANCE_CHANGE: pl.Float64,
            _ESTIMATED_CAUSE_TOTAL: pl.Float64,
            _UNEXPLAINED_CHANGE: pl.Float64,
            _REVIEW_STATUS: pl.String,
            _NEXT_ACTION: pl.String,
            _REVIEW_KEY: pl.String,
        }
    )


def _workbook_ranked_changed_rows(findings: pl.DataFrame) -> list[dict[str, object]]:
    """Return ranked changed rows with selected additive impacts marked."""
    evidence = _with_period_review_key(
        _pc_explain.top_evidence_table(findings, top_evidence_limit=findings.height)
    )
    if evidence.is_empty():
        return []

    selected_impact_bases = _workbook_selected_impact_basis_keys(findings)
    rows: list[dict[str, object]] = []
    for row in evidence.iter_rows(named=True):
        rows.append(_workbook_selected_impact_row(row, selected_impact_bases))
    return rows


def _workbook_underlying_causes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
) -> pl.DataFrame:
    """Return input rows that may directly explain performance differences."""
    rows = [
        _workbook_changed_item_row(row, comparison_path=comparison_path)
        for row in _workbook_ranked_changed_rows(findings)
        if _workbook_is_underlying_cause_row(row)
    ]
    rows.extend(_workbook_missing_underlying_cause_rows(findings, rows))
    if not rows:
        return _workbook_empty_changed_item_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows),
        _workbook_left_review_sort_columns(),
    )


def _workbook_missing_underlying_cause_rows(
    findings: pl.DataFrame,
    underlying_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return placeholder rows for changed periods without input causes."""
    coverage = _with_period_review_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    if coverage.is_empty():
        return []

    underlying_period_keys = {
        _workbook_period_key(row)
        for row in underlying_rows
    }
    rows: list[dict[str, object]] = []
    for row in coverage.iter_rows(named=True):
        if _workbook_period_key(row) in underlying_period_keys:
            continue
        rows.append(_workbook_missing_underlying_cause_row(row))
    return rows


def _workbook_missing_underlying_cause_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Return a reviewer-facing placeholder for periods with no source cause."""
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _USE: _USE_DIAGNOSTIC,
        _CHANGE_LABEL: "No underlying input differences found",
        _pc_findings.SECURITY_ID: None,
        _pc_findings.SNAPSHOT_A_VALUE: None,
        _pc_findings.SNAPSHOT_B_VALUE: None,
        _CHANGE: None,
        _ESTIMATED_IMPACT: None,
        _IMPACT_STATUS: _IMPACT_STATUS_REVIEW_ONLY,
        _NEXT_ACTION: (
            "Review the Reported Performance Checks sheet, Raw Audit Trail sheet, "
            "missing datasets, or vendor methodology."
        ),
        _REQUIRED_YAML_SETUP: (
            "No underlying input differences were found. Review the Reported "
            "Performance Checks sheet, Raw Audit Trail sheet, missing datasets, "
            "or vendor methodology."
        ),
        _pc_findings.DATASET: _NO_UNDERLYING_CAUSE_DATASET,
        _pc_findings.SOURCE_COLUMN: None,
        _pc_findings.FINDING_CODE: None,
        _pc_explain.REVIEW_RANK: 999999,
        _USE_PRIORITY: _workbook_use_priority(_USE_DIAGNOSTIC),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_derived_checks_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return derived performance rows used as checks, not root causes."""
    rows = [
        _workbook_changed_item_row(_workbook_non_additive_row(row))
        for row in _workbook_ranked_changed_rows(findings)
        if _workbook_is_derived_check_row(row)
    ]
    if not rows:
        return _workbook_empty_changed_item_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows),
        _workbook_left_review_sort_columns(),
    )


def _workbook_context_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return review-context rows that are not additive return explanations."""
    rows = [
        _workbook_changed_item_row(_workbook_non_additive_row(row))
        for row in _workbook_ranked_changed_rows(findings)
        if _workbook_is_context_row(row)
    ]
    if not rows:
        return _workbook_empty_changed_item_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows),
        _workbook_left_review_sort_columns(),
    )


def _workbook_left_review_sort_columns() -> tuple[str, ...]:
    """Return the shared left-column sort order for review detail sheets."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.SECURITY_ID,
    )


def _workbook_selected_impact_basis_keys(
    findings: pl.DataFrame,
) -> set[tuple[object, object, object, object]]:
    """Return period/impact-basis keys included in Portfolio Differences totals."""
    causes = _pc_explain.portfolio_period_cause_summary(findings)
    if causes.is_empty():
        return set()

    keys: set[tuple[object, object, object, object]] = set()
    for row in causes.iter_rows(named=True):
        if _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
            continue
        impact_basis = row.get(_pc_explain.IMPACT_BASIS)
        if impact_basis == _pc_explain.IMPACT_BASIS_NO_ESTIMATE:
            continue
        keys.add(
            (
                row.get(_pc_findings.PORTFOLIO_ID),
                row.get(_pc_findings.FROM_DATE),
                row.get(_pc_findings.THRU_DATE),
                impact_basis,
            )
        )
    return keys


def _workbook_selected_impact_row(
    row: Mapping[str, object],
    selected_impact_bases: set[tuple[object, object, object, object]],
) -> dict[str, object]:
    """Return row with unselected candidate estimates cleared for the workbook."""
    row_dict = dict(row)
    if _number_or_none(row_dict.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
        return row_dict

    key = (
        row_dict.get(_pc_findings.PORTFOLIO_ID),
        row_dict.get(_pc_findings.FROM_DATE),
        row_dict.get(_pc_findings.THRU_DATE),
        row_dict.get(_pc_explain.IMPACT_BASIS),
    )
    if key in selected_impact_bases:
        return row_dict

    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[_pc_explain.IMPACT_BASIS] = _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[_pc_explain.IMPACT_METHOD] = None
    row_dict[_pc_explain.IMPACT_MESSAGE] = (
        "Another estimate was selected for this portfolio-period cause area."
    )
    return row_dict


def _workbook_non_additive_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return a workbook row with explained-difference fields cleared."""
    row_dict = dict(row)
    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[_pc_explain.IMPACT_BASIS] = _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[_pc_explain.IMPACT_METHOD] = None
    return row_dict


def _workbook_is_underlying_cause_row(row: Mapping[str, object]) -> bool:
    """Return whether row is an underlying input-cause candidate."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_UNDERLYING_CAUSE


def _workbook_is_derived_check_row(row: Mapping[str, object]) -> bool:
    """Return whether row is a derived performance check, not a root cause."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DERIVED_CHECK


def _workbook_is_context_row(row: Mapping[str, object]) -> bool:
    """Return whether row is context-only evidence."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_CONTEXT


def _workbook_row_kind(row: Mapping[str, object]) -> str:
    """Return the workbook presentation role for a finding row."""
    if row.get(_pc_findings.DATASET) == _NO_UNDERLYING_CAUSE_DATASET:
        return _WORKBOOK_ROW_KIND_DIAGNOSTIC
    if row.get(_pc_findings.EVIDENCE_ROLE) == _pc_findings.CONTEXT.value:
        return _WORKBOOK_ROW_KIND_CONTEXT
    if row.get(_pc_findings.DATASET) in {
        pc_cols.PORTFOLIO_PERFORMANCE,
        pc_cols.SECURITY_PERFORMANCE,
    }:
        return _WORKBOOK_ROW_KIND_DERIVED_CHECK
    if row.get(_pc_findings.EVIDENCE_ROLE) == _pc_findings.DIRECT_INPUT.value:
        return _WORKBOOK_ROW_KIND_UNDERLYING_CAUSE
    return _WORKBOOK_ROW_KIND_OTHER


def _workbook_changed_item_row(
    row: Mapping[str, object],
    *,
    comparison_path: util.PathLike | None = None,
) -> dict[str, object]:
    """Return one plain-English changed-item workbook row."""
    estimated_impact = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
    row_use = _workbook_row_use(row)
    impact_status = _workbook_impact_status(row, estimated_impact)
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _USE: row_use,
        _CHANGE_LABEL: _workbook_change_label(row),
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _pc_findings.SNAPSHOT_A_VALUE: row.get(_pc_findings.SNAPSHOT_A_VALUE),
        _pc_findings.SNAPSHOT_B_VALUE: row.get(_pc_findings.SNAPSHOT_B_VALUE),
        _CHANGE: row.get(_pc_findings.DELTA_B_MINUS_A),
        _pc_findings.IMPACT_INPUT_VALUE: row.get(_pc_findings.IMPACT_INPUT_VALUE),
        _ESTIMATED_IMPACT: estimated_impact,
        _IMPACT_STATUS: impact_status,
        _NEXT_ACTION: _workbook_next_action(row, estimated_impact, row_use, impact_status),
        _REQUIRED_YAML_SETUP: _workbook_required_yaml_setup(
            row,
            estimated_impact,
            comparison_path=comparison_path,
        ),
        _pc_findings.DATASET: row.get(_pc_findings.DATASET),
        _pc_findings.SOURCE_COLUMN: row.get(_pc_findings.SOURCE_COLUMN),
        _pc_findings.FINDING_CODE: row.get(_pc_findings.FINDING_CODE),
        _pc_explain.REVIEW_RANK: row.get(_pc_explain.REVIEW_RANK),
        _USE_PRIORITY: _workbook_use_priority(row_use),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_change_label(row: Mapping[str, object]) -> str:
    """Return a concise changed-item label."""
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    dataset = _format_value(row.get(_pc_findings.DATASET)).replace("_", " ")
    if source_column:
        return f"{dataset} {source_column} changed"
    return _format_value(row.get(_pc_findings.MESSAGE))


def _workbook_row_use(row: Mapping[str, object]) -> str:
    """Return how a changed item should be used during review."""
    if _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC:
        return _USE_DIAGNOSTIC
    evidence_role = row.get(_pc_findings.EVIDENCE_ROLE)
    if evidence_role == _pc_findings.CONTEXT.value:
        return _USE_REVIEW_CONTEXT
    return _USE_EXPLAINS_CHANGE


def _workbook_use_priority(row_use: str) -> int:
    """Return sort priority for reviewer-facing changed-item uses."""
    return {
        _USE_EXPLAINS_CHANGE: 0,
        _USE_REVIEW_CONTEXT: 1,
        _USE_DIAGNOSTIC: 2,
    }.get(row_use, 9)


def _workbook_impact_status(
    row: Mapping[str, object],
    estimated_impact: float | None,
) -> str:
    """Return a compact status for row-level impact treatment."""
    if estimated_impact is not None:
        return _IMPACT_STATUS_ESTIMATED
    if (
        _workbook_is_context_row(row)
        or _workbook_is_derived_check_row(row)
        or _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC
        or _workbook_has_evidence_only_policy(row)
    ):
        return _IMPACT_STATUS_REVIEW_ONLY
    if _workbook_has_additive_policy(row):
        return _IMPACT_STATUS_MISSING_INPUT
    return _IMPACT_STATUS_MISSING_METHOD


def _workbook_next_action(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
) -> str:
    """Return one action-oriented note for a changed workbook row."""
    if estimated_impact is not None:
        return "None"

    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if dataset in {pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.SECURITY_PERFORMANCE}:
        return (
            "This is simply a difference in the raw performance datasets. Check "
            "the Underlying Causes sheet to see what explains it."
        )
    if _workbook_has_evidence_only_policy(row):
        return "Review this input difference; YAML marks it as evidence-only."
    if impact_status == _IMPACT_STATUS_MISSING_INPUT:
        return (
            "Review source inputs needed by the configured YAML method; no "
            "estimate is available for this row."
        )
    if impact_status == _IMPACT_STATUS_MISSING_METHOD:
        return _workbook_missing_impact_method_action(dataset, source_column)
    if row_use == _USE_REVIEW_CONTEXT:
        return "Review context; not included in explained performance difference."
    dataset_actions = {
        pc_cols.PRICES: "Review price change.",
        pc_cols.TRANSACTIONS: _workbook_review_change_action(
            "transaction",
            source_column,
        ),
        pc_cols.POSITIONS: _workbook_review_change_action("position", source_column),
        pc_cols.CASH: _workbook_review_change_action("cash", source_column),
    }
    return dataset_actions.get(
        dataset,
        _workbook_review_change_action("input", source_column),
    )


def _workbook_required_yaml_setup(
    row: Mapping[str, object],
    estimated_impact: float | None,
    *,
    comparison_path: util.PathLike | None,
) -> str:
    """Return the YAML setup required before this row can explain performance."""
    if estimated_impact is not None:
        return "None"
    if (
        _workbook_is_context_row(row)
        or _workbook_is_derived_check_row(row)
        or _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC
    ):
        return "None; this row is review context, not an underlying input difference."
    if _workbook_has_evidence_only_policy(row):
        return "None; configured as evidence-only in comparison YAML."

    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    dataset_column = _workbook_dataset_column_label(dataset, source_column)
    yaml_path = _workbook_yaml_path_label(comparison_path)
    if _workbook_has_additive_policy(row):
        return _workbook_missing_impact_input_setup(dataset, source_column)
    if dataset == pc_cols.TRANSACTIONS:
        if source_column != pc_cols.AMOUNT:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return (
            "Specify the YAML transaction_impact_methods.performance.method, "
            "transaction_impact_methods.performance.denominator_source, and "
            f"transaction_rules for each transaction code in {yaml_path}."
        )
    if dataset == pc_cols.POSITIONS:
        if source_column not in {
            pc_cols.MARKET_VALUE,
            pc_cols.ACCRUED,
            pc_cols.QUANTITY,
        }:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        if source_column == pc_cols.ACCRUED:
            return (
                "Specify the YAML position_impact_methods.accrued.method and "
                "position_impact_methods.accrued.denominator_source in "
                f"{yaml_path}."
            )
        if source_column == pc_cols.QUANTITY:
            return (
                "Specify the YAML position_impact_methods.quantity.method and "
                "position_impact_methods.quantity.denominator_source in "
                f"{yaml_path}."
            )
        return (
            "Specify the YAML position_impact_methods.market_value.method and "
            "position_impact_methods.market_value.denominator_source in "
            f"{yaml_path}."
        )
    if dataset == pc_cols.PRICES:
        if source_column != pc_cols.PRICE:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return (
            "Specify the YAML price_impact_methods.price.method and "
            f"price_impact_methods.price.weight_source in {yaml_path}."
        )
    if dataset == pc_cols.CASH:
        if source_column not in {pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE}:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return (
            f"Specify the YAML cash_impact_methods.{source_column}.method and "
            f"cash_impact_methods.{source_column}.denominator_source in {yaml_path}."
        )
    if dataset == pc_cols.FX_RATES:
        if source_column != pc_cols.FX_RATE:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return f"Specify the YAML fx_rate_impact_methods.fx_rate.method in {yaml_path}."
    if dataset == pc_cols.SECURITY_MASTER:
        return (
            f"Specify the YAML security_master_impact_methods.{source_column}.method "
            f"in {yaml_path}."
        )
    return f"No supported YAML impact method exists yet for {dataset_column}."


def _workbook_yaml_path_label(comparison_path: util.PathLike | None) -> str:
    """Return a compact YAML path label for workbook setup instructions."""
    if comparison_path is None:
        return "comparison YAML"
    return str(Path(comparison_path))


def _workbook_has_evidence_only_policy(row: Mapping[str, object]) -> bool:
    """Return whether a row has explicit YAML evidence-only treatment."""
    policies = (
        row.get(_pc_findings.IMPACT_POLICY),
        row.get(_pc_findings.TRANSACTION_IMPACT_POLICY),
    )
    return any(
        isinstance(policy, str)
        and policy.startswith(_pc_findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)
        for policy in policies
    )


def _workbook_has_additive_policy(row: Mapping[str, object]) -> bool:
    """Return whether a row has a configured non-evidence-only impact policy."""
    policies = (
        row.get(_pc_findings.IMPACT_POLICY),
        row.get(_pc_findings.TRANSACTION_IMPACT_POLICY),
    )
    return any(
        _has_text(policy)
        and not str(policy).startswith(_pc_findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)
        for policy in policies
    )


def _workbook_dataset_column_label(dataset: str, source_column: str) -> str:
    """Return ``dataset.column`` text for impact-method setup messages."""
    if dataset and source_column:
        return f"{dataset}.{source_column}"
    if dataset:
        return dataset
    if source_column:
        return source_column
    return "this input field"


def _workbook_missing_impact_input_setup(dataset: str, source_column: str) -> str:
    """Return setup text when a configured method lacks usable source inputs."""
    if dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.AMOUNT:
        return (
            "Configured transaction impact method is present, but this row still "
            "cannot be estimated. Review return denominator, transaction sign/flow "
            "semantics, and transaction date inputs."
        )
    if dataset == pc_cols.POSITIONS:
        return (
            "Configured position impact method is present, but this row still "
            "cannot be estimated. Review return denominator and position source "
            "values."
        )
    if dataset == pc_cols.PRICES:
        return (
            "Configured price impact method is present, but this row still cannot "
            "be estimated. Review snapshot A price, snapshot A weight, and price "
            "source values."
        )
    if dataset == pc_cols.CASH:
        return (
            "Configured cash impact method is present, but this row still cannot "
            "be estimated. Review return denominator and cash source values."
        )
    return (
        "Configured YAML impact method is present, but this row still cannot be "
        "estimated. Review the source inputs required by that method."
    )


def _workbook_missing_impact_method_action(dataset: str, source_column: str) -> str:
    """Return action text for source rows with no additive impact method."""
    if dataset == pc_cols.PRICES:
        return "Review price change; add price impact method before estimating."
    if dataset == pc_cols.TRANSACTIONS:
        return _workbook_add_method_action("transaction", source_column)
    if dataset == pc_cols.POSITIONS:
        return _workbook_add_method_action("position", source_column)
    if dataset == pc_cols.CASH:
        return _workbook_add_method_action("cash", source_column)
    return _workbook_add_method_action("input", source_column)


def _workbook_review_change_action(dataset_label: str, source_column: str) -> str:
    """Return standardized action text for review-only changed source values."""
    return f"Review {_workbook_source_change_label(dataset_label, source_column)} change."


def _workbook_add_method_action(dataset_label: str, source_column: str) -> str:
    """Return standardized action text for missing impact-method rows."""
    return (
        f"Review {_workbook_source_change_label(dataset_label, source_column)} change; "
        f"add {dataset_label} impact method before estimating."
    )


def _workbook_source_change_label(dataset_label: str, source_column: str) -> str:
    """Return compact dataset/field wording for action text."""
    if source_column:
        return f"{dataset_label} {source_column}"
    return dataset_label


def _workbook_empty_changed_item_table() -> pl.DataFrame:
    """Return an empty workbook changed-item table."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _USE: pl.String,
            _CHANGE_LABEL: pl.String,
            _pc_findings.SECURITY_ID: pl.String,
            _pc_findings.SNAPSHOT_A_VALUE: pl.String,
            _pc_findings.SNAPSHOT_B_VALUE: pl.String,
            _CHANGE: pl.Float64,
            _pc_findings.IMPACT_INPUT_VALUE: pl.Float64,
            _ESTIMATED_IMPACT: pl.Float64,
            _IMPACT_STATUS: pl.String,
            _NEXT_ACTION: pl.String,
            _REQUIRED_YAML_SETUP: pl.String,
            _pc_findings.DATASET: pl.String,
            _pc_findings.SOURCE_COLUMN: pl.String,
            _pc_findings.FINDING_CODE: pl.String,
            _pc_explain.REVIEW_RANK: pl.Int64,
            _USE_PRIORITY: pl.Int64,
            _REVIEW_KEY: pl.String,
        }
    )


def _workbook_portfolio_changes_columns() -> tuple[str, ...]:
    """Return Portfolio Differences worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _PERFORMANCE_CHANGE,
        _ESTIMATED_CAUSE_TOTAL,
        _UNEXPLAINED_CHANGE,
        _REVIEW_STATUS,
        _NEXT_ACTION,
        _REVIEW_KEY,
    )


def _workbook_security_changes_columns() -> tuple[str, ...]:
    """Return Security Differences worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.SECURITY_ID,
        _PERFORMANCE_CHANGE,
        _ESTIMATED_CAUSE_TOTAL,
        _UNEXPLAINED_CHANGE,
        _REVIEW_STATUS,
        _NEXT_ACTION,
        _REVIEW_KEY,
    )


def _workbook_underlying_cause_columns() -> tuple[str, ...]:
    """Return Underlying Causes worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _CHANGE,
        _pc_findings.IMPACT_INPUT_VALUE,
        _ESTIMATED_IMPACT,
        _REQUIRED_YAML_SETUP,
        _REVIEW_KEY,
    )


def _workbook_non_additive_change_columns() -> tuple[str, ...]:
    """Return non-additive reported-performance and context worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _CHANGE,
        _CHANGE_LABEL,
        _NEXT_ACTION,
        _REVIEW_KEY,
    )


def _workbook_findings_columns(findings: pl.DataFrame) -> tuple[str, ...]:
    """Return reviewer-first Findings worksheet columns with review key last."""
    preferred_columns = (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.SECURITY_ID,
    )
    remaining_columns = [
        column
        for column in findings.columns
        if column not in {*preferred_columns, _REVIEW_KEY}
    ]
    return (*preferred_columns, *remaining_columns, _REVIEW_KEY)


def _workbook_sorted_table(table: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Return a workbook table sorted by available reviewer-facing columns."""
    sort_columns = [column for column in columns if column in table.columns]
    if not sort_columns or table.is_empty():
        return table
    return table.sort(sort_columns, nulls_last=True)


def _workbook_column_labels() -> dict[str, str]:
    """Return shared user-facing labels for review workbook columns."""
    return {
        _REVIEW_KEY: "Review Key",
        _pc_findings.PORTFOLIO_ID: "Portfolio",
        _pc_findings.SECURITY_ID: "Security",
        _pc_findings.FROM_DATE: "From Date",
        _pc_findings.THRU_DATE: "Thru Date",
        _PERFORMANCE_CHANGE: "Performance Difference",
        _ESTIMATED_CAUSE_TOTAL: "Explained Difference",
        _UNEXPLAINED_CHANGE: "Unexplained Difference",
        _USE: "Purpose",
        _CHANGE_LABEL: "What Changed",
        _CHANGE: "B - A Difference",
        _ESTIMATED_IMPACT: "Performance Difference Explained",
        _IMPACT_STATUS: "Impact Status",
        _NEXT_ACTION: "Next Action",
        _REQUIRED_YAML_SETUP: "Required YAML Setup",
        _pc_explain.PORTFOLIO_RETURN_DELTA: "Return Delta",
        _REVIEW_STATUS: "Status",
        _PROBLEM: "Problem",
        _ACTION_REQUIRED: "Action Required",
        _WHY_IT_MATTERS: "Why It Matters",
        _EVIDENCE_SECTION: "Evidence Section",
        _DASHBOARD_MISSING_INPUTS: "Missing Inputs",
        _DASHBOARD_OPEN_SECTION: "Open Section",
        _REVIEW_CUES: "Review Cues",
        _SUGGESTED_NEXT_STEP: "Suggested Next Step",
        _REVIEW_DETAIL_ARTIFACTS: "Review Detail Artifacts",
        _CONTEXT_USE: "Context Use",
        _REVIEW_PRIORITY: "Review Priority",
        _REVIEW_PRIORITY_REASON: "Review Priority Reason",
        _RETURN_IMPACT_TREATMENT: "Return Impact Treatment",
        _pc_findings.FINDING_CODE: "Code",
        _pc_findings.DATASET: "Dataset",
        _pc_findings.SOURCE_COLUMN: "Source Column",
        _pc_findings.MESSAGE: "Message",
        _pc_findings.SEVERITY: "Severity",
        _pc_findings.CONFIDENCE: "Confidence",
        _pc_findings.EVIDENCE_ROLE: "Evidence Role",
        _pc_findings.SOURCE_FILE: "Source File",
        _pc_findings.SNAPSHOT_A_VALUE: "Snapshot A Value",
        _pc_findings.SNAPSHOT_B_VALUE: "Snapshot B Value",
        _pc_findings.DELTA_B_MINUS_A: "Delta B Minus A",
        _pc_findings.IMPACT_INPUT_VALUE: "Impact Input Value",
        _pc_findings.SUPPRESSED: "Suppressed",
        _pc_explain.ROOT_CAUSE_AREA: "Cause Area",
        _pc_explain.FINDING_COUNT: "Finding Count",
        _pc_explain.IMPACT_BASIS: "Impact Basis",
        _pc_explain.IMPACT_CONFIDENCE: "Confidence",
        _pc_explain.TOP_CODES: "Top Codes",
        _pc_explain.IMPACT_MESSAGE: "Impact Message",
        _pc_explain.REVIEW_RANK: "Review Rank",
    }


def _workbook_column_tooltip(column: str) -> str:
    """Return explanatory header text for a workbook column comment."""
    tooltips = {
        _REVIEW_KEY: (
            "Stable portfolio-period key used to connect workbook rows."
        ),
        _pc_findings.PORTFOLIO_ID: "Portfolio identifier from the compared source data.",
        _pc_findings.FROM_DATE: "Beginning date of the affected performance period.",
        _pc_findings.THRU_DATE: "Ending date of the affected performance period.",
        _pc_findings.SECURITY_ID: "Security identifier, when the discrepancy is security-level.",
        _pc_findings.SEVERITY: "Materiality/severity assigned to this discrepancy.",
        _PERFORMANCE_CHANGE: (
            "Snapshot B portfolio return minus snapshot A portfolio return."
        ),
        _ESTIMATED_CAUSE_TOTAL: (
            "Total performance difference explained by Underlying Causes sheet rows."
        ),
        _UNEXPLAINED_CHANGE: "Performance difference less explained difference.",
        _USE: "Workbook row category used for sorting and compatibility.",
        _CHANGE_LABEL: "Plain-English changed data item.",
        _CHANGE: "Snapshot B value minus snapshot A value for the compared item.",
        _ESTIMATED_IMPACT: (
            "Decimal portfolio performance difference explained by this underlying "
            "input row."
        ),
        _IMPACT_STATUS: (
            "Whether this row has an additive estimate, is missing an impact method, "
            "or is review-only."
        ),
        _NEXT_ACTION: "Recommended reviewer action for this changed item.",
        _REQUIRED_YAML_SETUP: (
            "YAML setup needed before this input row can receive a performance "
            "difference explanation."
        ),
        _pc_explain.PORTFOLIO_RETURN_DELTA: (
            "Snapshot B portfolio return minus snapshot A portfolio return."
        ),
        _REVIEW_STATUS: "Reviewer triage status for this portfolio-period problem.",
        _PROBLEM: "Plain-English statement of the issue to review.",
        _ACTION_REQUIRED: "Recommended next action for the reviewer or configuration owner.",
        _WHY_IT_MATTERS: "Why this issue affects interpretation of the return change.",
        _DASHBOARD_MISSING_INPUTS: (
            "Configuration or source inputs needed before ppar can estimate impact."
        ),
        _pc_explain.ROOT_CAUSE_AREA: "Coarse explanation bucket for a group of findings.",
        _pc_explain.FINDING_COUNT: "Number of finding rows grouped into this cause.",
        _pc_explain.IMPACT_BASIS: "Method basis used to estimate return impact.",
        _pc_explain.IMPACT_CONFIDENCE: "Confidence level for the estimated impact.",
        _pc_explain.TOP_CODES: "Most relevant finding codes represented by this row.",
        _pc_explain.IMPACT_MESSAGE: "Explanation of the impact estimate or limitation.",
        _pc_explain.REVIEW_RANK: "Priority rank within the portfolio period.",
        _pc_findings.FINDING_CODE: "Stable finding code for the discrepancy type.",
        _pc_findings.CONFIDENCE: "Confidence level for the finding or impact interpretation.",
        _pc_findings.DATASET: "Normalized source dataset where the discrepancy was found.",
        _pc_findings.EVIDENCE_ROLE: (
            "Whether the finding is target output, direct input, related output, or context."
        ),
        _pc_findings.SOURCE_FILE: "Source file path or dataset file where applicable.",
        _pc_findings.SOURCE_COLUMN: "Normalized source column that changed or was relevant.",
        _pc_findings.TRANSACTION_CATEGORY: "Normalized transaction category, when applicable.",
        _pc_findings.CASH_FLOW_SIGN: "Configured or source cash-flow sign, when applicable.",
        _pc_findings.PERFORMANCE_FLOW_SIGN: (
            "Configured or source performance-flow sign, when applicable."
        ),
        _pc_findings.TRANSACTION_SEMANTICS_SOURCE: (
            "Where transaction sign/category semantics came from."
        ),
        _pc_findings.TRANSACTION_MATCH_STATUS: (
            "How transaction rows were matched between snapshots."
        ),
        _pc_findings.IMPACT_POLICY: "Contribution/return impact policy used for this finding.",
        _pc_findings.TRANSACTION_IMPACT_POLICY: (
            "Transaction impact policy used for this finding."
        ),
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC: (
            "Review-only transaction diagnostic name, when available."
        ),
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE: (
            "Review-only transaction diagnostic estimate, when available."
        ),
        _pc_findings.SNAPSHOT_A_VALUE: "Value observed in snapshot A.",
        _pc_findings.SNAPSHOT_B_VALUE: "Value observed in snapshot B.",
        _pc_findings.DELTA_B_MINUS_A: "Numeric difference calculated as snapshot B minus A.",
        _pc_findings.RETURN_DENOMINATOR: (
            "Denominator used for return-impact estimates, when configured."
        ),
        _pc_findings.RETURN_WEIGHT: (
            "Weight used for security return-impact estimates, when available."
        ),
        _pc_findings.IMPACT_INPUT_VALUE: (
            "Additional numeric input used by the selected impact method, when needed."
        ),
        _pc_findings.MESSAGE: "Human-readable finding detail.",
        _pc_findings.SUPPRESSED: "Whether a configured suppression marked this finding hidden.",
    }
    return tooltips.get(
        column,
        f"Workbook column derived from normalized ppar field `{column}`.",
    )


def _number_or_none(value: object) -> float | None:
    """Return a float for numeric values, preserving missing/non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _active_findings(findings: pl.DataFrame) -> pl.DataFrame:
    """Return unsuppressed findings, preserving empty-table behavior."""
    if findings.is_empty() or _pc_findings.SUPPRESSED not in findings.columns:
        return findings
    return findings.filter(~pl.col(_pc_findings.SUPPRESSED))


def _has_text(value: object) -> bool:
    """Return whether a value has non-blank text."""
    return isinstance(value, str) and bool(value.strip())
