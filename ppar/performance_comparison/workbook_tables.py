"""Build review workbook tables for performance comparison findings."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Mapping, Sequence
import datetime as _dt
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import field_roles as _field_roles
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import rendering as _pc_rendering
from ppar.performance_comparison import review_keys as _pc_review_keys
from ppar.performance_comparison import review_model as _pc_review_model
from ppar.performance_comparison import return_reconstruction as _pc_reconstruction
from ppar.performance_comparison import workbook as _pc_workbook
from ppar.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
)

__all__ = [
    "performance_comparison_review_workbook_sheets",
    "workbook_column_tooltip",
    "write_performance_comparison_review_workbook",
]

_REVIEW_STATUS = "review_status"
_REVIEW_CUES = "review_cues"
_SUGGESTED_NEXT_STEP = "suggested_next_step"
_REVIEW_KEY = _pc_review_keys.REVIEW_KEY
_REVIEW_DETAIL_ARTIFACTS = "review_detail_artifacts"
_PERFORMANCE_CHANGE = "performance_change"
_ESTIMATED_CAUSE_TOTAL = "estimated_cause_total"
_UNEXPLAINED_CHANGE = "unexplained_change"
_USE = "use"
_USE_PRIORITY = "_use_priority"
_CHANGE_LABEL = "change_label"
_CHANGE = "change"
_DATASET_FIELD = "dataset_field"
_INPUT_ROLE = "input_role"
_AS_OF_DATE = "as_of_date"
_ESTIMATED_IMPACT = "estimated_impact"
_IMPACT_STATUS = "impact_status"
_REVIEW_NOTE = "review_note"
_REVIEW_GUIDANCE = "review_guidance"
_USE_EXPLAINS_CHANGE = "Explains Change"
_USE_REVIEW_CONTEXT = "Review Context"
_USE_DIAGNOSTIC = "Diagnostic"
_INPUT_ROLE_PERFORMANCE_INPUT = "Performance Input"
_INPUT_ROLE_INPUT_DRIVER = "Input Driver"
_INPUT_ROLE_SUPPORTING_EVIDENCE = "Supporting Evidence"
_INPUT_ROLE_CONTEXT = "Context"
_INPUT_ROLE_DIAGNOSTIC = "Diagnostic"
_IMPACT_STATUS_ESTIMATED = "Estimated"
_IMPACT_STATUS_MISSING_METHOD = "Missing impact method"
_IMPACT_STATUS_MISSING_INPUT = "Missing impact input"
_IMPACT_STATUS_REVIEW_ONLY = "Review only"
_NO_UNDERLYING_CAUSE_DATASET = "no_underlying_causes_found"
_WORKBOOK_ROW_KIND_UNDERLYING_CAUSE = "underlying_cause"
_WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC = "reported_diagnostic"
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
_WORKBOOK_UNSELECTED_RELATED_ESTIMATE = "_workbook_unselected_related_estimate"
_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION = "_workbook_non_additive_portfolio_transaction"
_WORKBOOK_TRANSACTION_FLOW_SUPPORTS_HOLDING = (
    "_workbook_transaction_flow_supports_holding"
)
_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW = (
    "_workbook_transaction_supports_reconstruction_flow"
)
_WORKBOOK_CASH_BALANCE_SECURITY_ID = "_workbook_cash_balance_security_id"
_RECONSTRUCTION_FORMULA_FINDING_CODE = "reconstruction_formula_input"
_RECONSTRUCTION_BEGINNING_VALUE_FIELD = "beginning_market_value"
_RECONSTRUCTION_ENDING_VALUE_FIELD = "ending_market_value"
_RECONSTRUCTION_NET_FLOW_FIELD = "net_flow"
_RECONSTRUCTION_WEIGHTED_FLOW_FIELD = "weighted_flow"
_RECONSTRUCTION_INCOME_FIELD = "income"
_RECONSTRUCTION_ROLE_METADATA = {
    _RECONSTRUCTION_BEGINNING_VALUE_FIELD: (
        pc_cols.HOLDINGS,
        _RECONSTRUCTION_BEGINNING_VALUE_FIELD,
        "Beginning holdings market value",
    ),
    _RECONSTRUCTION_ENDING_VALUE_FIELD: (
        pc_cols.HOLDINGS,
        _RECONSTRUCTION_ENDING_VALUE_FIELD,
        "Ending holdings market value",
    ),
    _RECONSTRUCTION_NET_FLOW_FIELD: (
        pc_cols.TRANSACTIONS,
        _RECONSTRUCTION_NET_FLOW_FIELD,
        "Transaction net flow",
    ),
    _RECONSTRUCTION_WEIGHTED_FLOW_FIELD: (
        pc_cols.TRANSACTIONS,
        _RECONSTRUCTION_WEIGHTED_FLOW_FIELD,
        "Transaction weighted flow",
    ),
    _RECONSTRUCTION_INCOME_FIELD: (
        pc_cols.TRANSACTIONS,
        _RECONSTRUCTION_INCOME_FIELD,
        "Transaction income",
    ),
}
_WORKBOOK_UNEXPLAINED_TOLERANCE = 0.0000005
_WORKBOOK_PROMOTABLE_EVIDENCE_COLUMNS = {
    pc_cols.CASH: {pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE},
    pc_cols.FX_RATES: {pc_cols.FX_RATE},
    pc_cols.HOLDINGS: {pc_cols.ACCRUED, pc_cols.MARKET_VALUE, pc_cols.QUANTITY},
    pc_cols.TRANSACTIONS: {
        pc_cols.AMOUNT,
        pc_cols.COMMISSION,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    },
}
_format_value = _pc_rendering.format_value
_with_period_review_key = _pc_review_keys.with_period_review_key
_with_security_review_key = _pc_review_keys.with_security_review_key


def write_performance_comparison_review_workbook(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    top_evidence_limit: int = 10,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
) -> Path:
    """Write an XLSX workbook for performance comparison review.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_path: Destination workbook path. Parent directories are created
            when needed.
        top_evidence_limit: Reserved for parity with bundle/report writers.
        comparison_path: Optional path to the comparison YAML. When provided,
            the ``Performance Difference Causes`` sheet can name the exact file to update
            for missing attribution setup.
        comparison_level: Primary performance-result level for the workbook.
        include_reconstruction_diagnostics: Whether to include interim
            reconstruction diagnostic sheets in addition to the primary review
            sheets.

    Returns:
        Normalized workbook path.

    Raises:
        PpaError: If the Excel workbook dependency is not installed.

    Notes:
        The workbook is a presentation layer over the same impact coverage,
        top-evidence, and findings output used by the HTML/CSV reports. It does
        not add comparison logic.
    """
    active_findings = _active_findings(findings)
    del top_evidence_limit
    return _pc_workbook.write_review_workbook_sheets(
        performance_comparison_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        ),
        output_path,
        column_tooltip=workbook_column_tooltip,
    )


def performance_comparison_review_workbook_sheets(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return review workbook sheet specifications in reviewer-first order.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        comparison_path: Optional path to the comparison YAML. When provided,
            the ``Performance Difference Causes`` sheet can name the exact file to update
            for missing attribution setup.
        comparison_level: Primary performance-result level for the workbook.
        include_reconstruction_diagnostics: Whether to include interim
            reconstruction diagnostic sheets.

    Returns:
        Ordered sheet specifications used by both the XLSX workbook and the
        browser report.
    """
    active_findings = _active_findings(findings)
    primary_sheet = (
        _security_differences_sheet(
            active_findings,
            comparison_path=comparison_path,
        )
        if comparison_level == SECURITY_COMPARISON_LEVEL
        else _portfolio_differences_sheet(
            active_findings,
            comparison_path=comparison_path,
        )
    )
    diagnostic_sheets = (
        (
            *_return_reconstruction_summary_sheets(comparison_path),
            *_return_reconstruction_sheets(comparison_path),
            *_security_return_reconstruction_sheets(comparison_path),
        )
        if include_reconstruction_diagnostics
        else ()
    )
    return (
        primary_sheet,
        *diagnostic_sheets,
        *_shared_detail_sheets(
            findings,
            active_findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
        ),
    )


def _return_reconstruction_summary_sheets(
    comparison_path: util.PathLike | None,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional return-reconstruction diagnostic summary sheets."""
    summary = _pc_reconstruction.return_reconstruction_summary(comparison_path)
    if summary.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT,
            sheet_name=_pc_review_model.RECONSTRUCTION_SUMMARY_SHEET,
            table=summary,
            columns=_workbook_return_reconstruction_summary_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _return_reconstruction_sheets(
    comparison_path: util.PathLike | None,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional portfolio return-reconstruction diagnostic sheets."""
    reconstruction_checks = (
        _pc_reconstruction.portfolio_return_reconstruction_checks(comparison_path)
    )
    if reconstruction_checks.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
            sheet_name=_pc_review_model.RETURN_RECONSTRUCTION_CHECKS_SHEET,
            table=reconstruction_checks,
            columns=_workbook_return_reconstruction_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _security_return_reconstruction_sheets(
    comparison_path: util.PathLike | None,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional security return-reconstruction diagnostic sheets."""
    reconstruction_checks = (
        _pc_reconstruction.security_return_reconstruction_checks(comparison_path)
    )
    if reconstruction_checks.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=(
                _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT
            ),
            sheet_name=_pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_SHEET,
            table=reconstruction_checks,
            columns=_workbook_security_return_reconstruction_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _portfolio_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
) -> _pc_workbook.ReviewWorkbookSheet:
    """Return the portfolio-level performance differences sheet."""
    labels = _workbook_column_labels()
    labels[_REVIEW_NOTE] = "Comments"
    return _pc_workbook.ReviewWorkbookSheet(
        artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
        sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
        table=_workbook_portfolio_changes_table(
            active_findings,
            comparison_path=comparison_path,
        ),
        columns=_workbook_portfolio_changes_columns(),
        labels=labels,
    )


def _security_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
) -> _pc_workbook.ReviewWorkbookSheet:
    """Return the security-level performance differences sheet."""
    labels = _workbook_column_labels()
    labels[_REVIEW_NOTE] = "Comments"
    return _pc_workbook.ReviewWorkbookSheet(
        artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
        sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
        table=_workbook_security_changes_table(
            active_findings,
            comparison_path=comparison_path,
            comparison_level=SECURITY_COMPARISON_LEVEL,
        ),
        columns=_workbook_security_changes_columns(),
        labels=labels,
    )


def _shared_detail_sheets(
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    comparison_level: str,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return detail sheets shared by portfolio and security workflows."""
    detail_sheets = [
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
            sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
            table=_workbook_underlying_causes_table(
                active_findings,
                comparison_path=comparison_path,
                comparison_level=comparison_level,
            ),
            columns=_workbook_underlying_cause_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.OTHER_DATA_DIFFERENCES_ARTIFACT,
            sheet_name=_pc_review_model.OTHER_DATA_DIFFERENCES_SHEET,
            table=_workbook_context_table(
                active_findings,
                comparison_path=comparison_path,
                comparison_level=comparison_level,
            ),
            columns=_workbook_non_additive_change_columns(),
            labels=_workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.RAW_AUDIT_TRAIL_ARTIFACT,
            sheet_name=_pc_review_model.RAW_AUDIT_TRAIL_SHEET,
            table=_workbook_sorted_table(
                _workbook_with_primary_review_key(findings, comparison_level),
                _workbook_left_review_sort_columns(comparison_level=comparison_level),
            ),
            columns=_workbook_findings_columns(findings),
            labels=_workbook_column_labels(),
        ),
    ]
    return tuple(detail_sheets)


def _workbook_portfolio_changes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
) -> pl.DataFrame:
    """Return one workbook row per changed portfolio period."""
    coverage = _with_period_review_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    if coverage.is_empty():
        return _workbook_empty_portfolio_changes_table()
    underlying_totals = _workbook_underlying_impact_totals(
        findings,
        comparison_path=comparison_path,
    )
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
        pl.DataFrame(rows, infer_schema_length=None),
        [_REVIEW_KEY],
    )


def _workbook_underlying_impact_totals(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
) -> dict[tuple[object, object, object], float]:
    """Return explained difference totals from underlying input rows."""
    totals: dict[tuple[object, object, object], float] = {}
    active_keys = _workbook_active_portfolio_period_keys(findings)
    formula_rows = _workbook_portfolio_reconstruction_formula_rows(
        comparison_path,
        active_keys=active_keys,
    )
    formula_keys = {_workbook_period_key(row) for row in formula_rows}
    for row in formula_rows:
        key = _workbook_period_key(row)
        estimated_impact = _number_or_none(row.get(_ESTIMATED_IMPACT))
        if estimated_impact is not None:
            totals[key] = totals.get(key, 0.0) + estimated_impact

    for row, estimated_impact in _workbook_selected_underlying_impact_rows(
        findings,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    ):
        key = _workbook_period_key(row)
        if key in formula_keys:
            continue
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


def _workbook_period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the workbook period key for a row."""
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
    )


def _workbook_primary_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the workbook grouping key for the configured comparison level."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _workbook_security_period_key(row)
    return _workbook_period_key(row)


def _workbook_with_primary_review_key(
    table: pl.DataFrame,
    comparison_level: str,
) -> pl.DataFrame:
    """Add the review key matching the configured comparison level."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _with_security_review_key(table)
    return _with_period_review_key(table)


def _workbook_top_evidence_table(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> pl.DataFrame:
    """Return top evidence rows for the configured comparison level."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _pc_explain.security_top_evidence_table(
            findings,
            top_evidence_limit=findings.height,
        )
    return _pc_explain.top_evidence_table(findings, top_evidence_limit=findings.height)


def _workbook_primary_cause_summary(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> pl.DataFrame:
    """Return cause summary rows for the configured comparison level."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _pc_explain.security_period_cause_summary(findings)
    return _pc_explain.portfolio_period_cause_summary(findings)


def _workbook_primary_coverage_summary(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> pl.DataFrame:
    """Return coverage summary rows for the configured comparison level."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _pc_explain.security_period_summary(findings)
    return _pc_explain.portfolio_period_impact_coverage_summary(findings)


def _workbook_performance_change_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return one plain-English performance-change workbook row."""
    performance_change = _workbook_performance_difference(row)
    estimated_total = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL))
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    if underlying_estimated_total is not None:
        estimated_total = underlying_estimated_total
    unexplained_change = None
    if performance_change is not None:
        unexplained_change = performance_change - (estimated_total or 0.0)
    review_status = _workbook_explanation_status(row)
    unexplained_display = (
        None
        if review_status == _STATUS_FULLY_EXPLAINED
        else unexplained_change
    )
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _PERFORMANCE_CHANGE: performance_change,
        _ESTIMATED_CAUSE_TOTAL: estimated_total,
        _UNEXPLAINED_CHANGE: unexplained_display,
        _REVIEW_STATUS: review_status,
        _REVIEW_NOTE: _workbook_performance_comments(row),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_performance_difference(row: Mapping[str, object]) -> float | None:
    """Return portfolio or security performance difference for a workbook row."""
    portfolio_difference = _number_or_none(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))
    if portfolio_difference is not None:
        return portfolio_difference
    return _number_or_none(row.get(_pc_explain.SECURITY_RETURN_DELTA))


def _workbook_explanation_status(row: Mapping[str, object]) -> str:
    """Return a plain-language explanation status for a performance difference."""
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    estimated_total = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL))
    performance_change = _workbook_performance_difference(row)
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    if underlying_estimated_total is not None and performance_change is not None:
        estimated_total = underlying_estimated_total
    if estimated_total is not None and performance_change is not None:
        residual = performance_change - estimated_total
        if abs(residual) <= _WORKBOOK_UNEXPLAINED_TOLERANCE:
            return _STATUS_FULLY_EXPLAINED
        if abs(estimated_total) > 0:
            return _STATUS_PARTLY_EXPLAINED
        return _STATUS_UNEXPLAINED
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        return _STATUS_FULLY_EXPLAINED
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_MISSING_INPUTS:
        return _STATUS_NEEDS_SETUP
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return _STATUS_PARTLY_EXPLAINED
    return _STATUS_UNEXPLAINED


def _workbook_performance_comments(row: Mapping[str, object]) -> str:
    """Return plain-language comments for a performance difference."""
    if _workbook_explanation_status(row) == _STATUS_FULLY_EXPLAINED:
        return ""

    missing_inputs = row.get(_pc_explain.MISSING_IMPACT_INPUTS)
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    if _has_text(missing_inputs):
        return f"Missing YAML specifications: {_format_value(missing_inputs)}."
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
    performance_change = _workbook_performance_difference(row)
    if underlying_estimated_total is not None and performance_change is not None:
        residual = performance_change - underlying_estimated_total
        if abs(residual) <= _WORKBOOK_UNEXPLAINED_TOLERANCE:
            return ""
        if abs(underlying_estimated_total) > 0:
            return (
                'Review the "Other Data Differences" sheet and "Raw Audit Trail" sheet for '
                "the Unexplained Difference."
            )
        return (
            'Review the "Other Data Differences" sheet and "Raw Audit Trail" sheet for the '
            "Unexplained Difference."
        )
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        return ""
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return (
            'Review the "Other Data Differences" sheet and "Raw Audit Trail" sheet for the '
            "Unexplained Difference."
        )
    return (
        'Review the "Other Data Differences" sheet and "Raw Audit Trail" sheet for the '
        "Unexplained Difference."
    )


def _workbook_empty_portfolio_changes_table() -> pl.DataFrame:
    """Return a reviewer-facing performance-difference row for clean comparisons."""
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
                _REVIEW_NOTE: "No reported portfolio return differences.",
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
            _REVIEW_NOTE: pl.String,
            _REVIEW_KEY: pl.String,
        },
    )


def _workbook_security_changes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> pl.DataFrame:
    """Return one workbook row per changed security period."""
    summary = _with_security_review_key(_pc_explain.security_period_summary(findings))
    security_totals = _workbook_security_underlying_impact_totals(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
    )
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
        pl.DataFrame(rows, infer_schema_length=None),
        [_REVIEW_KEY, _pc_findings.SECURITY_ID],
    )


def _workbook_security_underlying_impact_totals(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> dict[tuple[object, object, object, object], float]:
    """Return security-level explained totals from underlying input rows."""
    totals: dict[tuple[object, object, object, object], float] = {}
    active_keys = _workbook_active_security_period_keys(findings)
    formula_rows = _workbook_security_reconstruction_formula_rows(
        comparison_path,
        active_keys=active_keys,
    )
    formula_keys = {_workbook_security_period_key(row) for row in formula_rows}
    for row in formula_rows:
        key = _workbook_security_period_key(row)
        estimated_impact = _number_or_none(row.get(_ESTIMATED_IMPACT))
        if estimated_impact is not None:
            totals[key] = totals.get(key, 0.0) + estimated_impact

    for row, estimated_impact in _workbook_selected_underlying_impact_rows(
        findings,
        comparison_level=comparison_level,
    ):
        if not _has_text(row.get(_pc_findings.SECURITY_ID)):
            continue
        key = _workbook_security_period_key(row)
        if key in formula_keys:
            continue
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


def _workbook_portfolio_reconstruction_formula_rows(
    comparison_path: util.PathLike | None,
    *,
    active_keys: set[tuple[object, object, object]] | None = None,
) -> list[dict[str, object]]:
    """Return portfolio reconstruction formula rows for Performance Difference Causes.

    Notes:
        This pilot promotes exact formula-level effects only. The detailed
        ``Return Reconstruction Checks`` sheet remains the source for the
        underlying beginning value, ending value, flow, income, and denominator
        inputs.
    """
    checks = _pc_reconstruction.portfolio_return_reconstruction_checks(comparison_path)
    if checks.is_empty():
        return []

    rows: list[dict[str, object]] = []
    for row in checks.iter_rows(named=True):
        if active_keys is not None and _workbook_period_key(row) not in active_keys:
            continue
        rows.extend(
            _workbook_reconstruction_formula_rows_for_check(
                row,
                row_factory=_workbook_portfolio_reconstruction_formula_row,
            )
        )
    return rows


def _workbook_security_reconstruction_formula_rows(
    comparison_path: util.PathLike | None,
    *,
    active_keys: set[tuple[object, object, object, object]] | None = None,
) -> list[dict[str, object]]:
    """Return security reconstruction formula rows for Performance Difference Causes.

    Notes:
        This pilot promotes exact formula-level effects only. The detailed
        ``Security Return Checks`` sheet remains the source for the underlying
        beginning value, ending value, flow, and income components.
    """
    checks = _pc_reconstruction.security_return_reconstruction_checks(comparison_path)
    if checks.is_empty():
        return []

    rows: list[dict[str, object]] = []
    for row in checks.iter_rows(named=True):
        if (
            active_keys is not None
            and _workbook_security_period_key(row) not in active_keys
        ):
            continue
        rows.extend(
            _workbook_reconstruction_formula_rows_for_check(
                row,
                row_factory=_workbook_security_reconstruction_formula_row,
            )
        )
    return rows


def _workbook_active_portfolio_period_keys(
    findings: pl.DataFrame,
) -> set[tuple[object, object, object]]:
    """Return portfolio-period keys with reported portfolio performance differences."""
    summary = _with_period_review_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    return {
        _workbook_period_key(row)
        for row in summary.iter_rows(named=True)
    }


def _workbook_active_security_period_keys(
    findings: pl.DataFrame,
) -> set[tuple[object, object, object, object]]:
    """Return security-period keys with reported security performance differences."""
    summary = _with_security_review_key(_pc_explain.security_period_summary(findings))
    return {
        _workbook_security_period_key(row)
        for row in summary.iter_rows(named=True)
    }


def _workbook_reconstruction_formula_rows_for_check(
    source_row: Mapping[str, object],
    *,
    row_factory: Callable[..., dict[str, object]],
) -> list[dict[str, object]]:
    numerator_b = _number_or_none(
        source_row.get(_pc_reconstruction.DERIVED_NUMERATOR_B)
    )
    denominator_a = _number_or_none(
        source_row.get(_pc_reconstruction.DERIVED_DENOMINATOR_A)
    )
    denominator_b = _number_or_none(
        source_row.get(_pc_reconstruction.DERIVED_DENOMINATOR_B)
    )
    if (
        numerator_b is None
        or denominator_a is None
        or denominator_b is None
        or denominator_a == 0.0
        or denominator_b == 0.0
    ):
        return []

    denominator_effect = numerator_b * (
        (1.0 / denominator_b) - (1.0 / denominator_a)
    )
    beginning_denominator_effect, weighted_flow_denominator_effect = (
        _workbook_denominator_component_effects(source_row, denominator_effect)
    )
    beginning_value_difference = _number_or_none(
        source_row.get(_pc_reconstruction.BEGIN_VALUE_DIFFERENCE)
    )
    ending_value_difference = _number_or_none(
        source_row.get(_pc_reconstruction.END_VALUE_DIFFERENCE)
    )
    net_flow_difference = _number_or_none(
        source_row.get(_pc_reconstruction.NET_FLOW_DIFFERENCE)
    )
    income_difference = _number_or_none(
        source_row.get(_pc_reconstruction.INCOME_DIFFERENCE)
    )
    rows = [
        row_factory(
            source_row,
            field=_RECONSTRUCTION_BEGINNING_VALUE_FIELD,
            snapshot_a_value=source_row.get(_pc_reconstruction.BEGIN_VALUE_A),
            snapshot_b_value=source_row.get(_pc_reconstruction.BEGIN_VALUE_B),
            difference=beginning_value_difference,
            estimated_impact=(
                _workbook_component_impact(
                    _workbook_negated_difference(beginning_value_difference),
                    denominator_a,
                )
                + beginning_denominator_effect
            ),
            guidance_role="beginning value",
            as_of_date=source_row.get(_pc_reconstruction.BEGIN_VALUE_DATE_B),
        ),
        row_factory(
            source_row,
            field=_RECONSTRUCTION_ENDING_VALUE_FIELD,
            snapshot_a_value=source_row.get(_pc_reconstruction.END_VALUE_A),
            snapshot_b_value=source_row.get(_pc_reconstruction.END_VALUE_B),
            difference=ending_value_difference,
            estimated_impact=_workbook_component_impact(
                ending_value_difference,
                denominator_a,
            ),
            guidance_role="ending value",
            as_of_date=source_row.get(_pc_reconstruction.END_VALUE_DATE_B),
        ),
        row_factory(
            source_row,
            field=_RECONSTRUCTION_NET_FLOW_FIELD,
            snapshot_a_value=source_row.get(_pc_reconstruction.NET_FLOW_A),
            snapshot_b_value=source_row.get(_pc_reconstruction.NET_FLOW_B),
            difference=net_flow_difference,
            estimated_impact=_workbook_component_impact(
                -net_flow_difference if net_flow_difference is not None else None,
                denominator_a,
            ),
            guidance_role="net flow",
            as_of_date=source_row.get(_pc_reconstruction.RECONSTRUCTION_THRU_DATE),
        ),
        row_factory(
            source_row,
            field=_RECONSTRUCTION_WEIGHTED_FLOW_FIELD,
            snapshot_a_value=source_row.get(_pc_reconstruction.WEIGHTED_FLOW_A),
            snapshot_b_value=source_row.get(_pc_reconstruction.WEIGHTED_FLOW_B),
            difference=source_row.get(_pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE),
            estimated_impact=weighted_flow_denominator_effect,
            guidance_role="weighted flow",
            as_of_date=source_row.get(_pc_reconstruction.RECONSTRUCTION_THRU_DATE),
        ),
    ]
    if income_difference is not None:
        rows.append(
            row_factory(
                source_row,
                field=_RECONSTRUCTION_INCOME_FIELD,
                snapshot_a_value=source_row.get(_pc_reconstruction.INCOME_A),
                snapshot_b_value=source_row.get(_pc_reconstruction.INCOME_B),
                difference=income_difference,
                estimated_impact=_workbook_component_impact(
                    income_difference,
                    denominator_a,
                ),
                guidance_role="income",
                as_of_date=source_row.get(_pc_reconstruction.RECONSTRUCTION_THRU_DATE),
            )
        )
    return _workbook_nonzero_formula_rows(rows)


def _workbook_denominator_component_effects(
    source_row: Mapping[str, object],
    denominator_effect: float,
) -> tuple[float, float]:
    """Return denominator effect allocated to beginning value and weighted flow."""
    beginning_value_difference = _number_or_none(
        source_row.get(_pc_reconstruction.BEGIN_VALUE_DIFFERENCE)
    ) or 0.0
    weighted_flow_difference = _number_or_none(
        source_row.get(_pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE)
    ) or 0.0
    denominator_difference = _number_or_none(
        source_row.get(_pc_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE)
    )
    if denominator_difference is None or abs(denominator_difference) <= 0.0000005:
        return 0.0, 0.0
    return (
        denominator_effect * (beginning_value_difference / denominator_difference),
        denominator_effect * (weighted_flow_difference / denominator_difference),
    )


def _workbook_component_impact(
    component_difference: float | None,
    denominator_a: float,
) -> float:
    """Return return impact for a numerator component difference."""
    if component_difference is None:
        return 0.0
    return component_difference / denominator_a


def _workbook_negated_difference(component_difference: float | None) -> float | None:
    """Return the opposite sign for a formula component difference."""
    if component_difference is None:
        return None
    return -component_difference


def _workbook_nonzero_formula_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return formula rows with meaningful value or impact differences."""
    nonzero_rows: list[dict[str, object]] = []
    for row in rows:
        change = _number_or_none(row.get(_CHANGE)) or 0.0
        estimated_impact = _number_or_none(row.get(_ESTIMATED_IMPACT)) or 0.0
        if (
            abs(change) > _WORKBOOK_UNEXPLAINED_TOLERANCE
            or abs(estimated_impact) > _WORKBOOK_UNEXPLAINED_TOLERANCE
        ):
            nonzero_rows.append(row)
    return nonzero_rows


def _workbook_reconstruction_role_metadata(field: str) -> tuple[str, str, str]:
    """Return source-facing dataset, field, and label for a formula role."""
    return _RECONSTRUCTION_ROLE_METADATA[field]


def _workbook_portfolio_reconstruction_formula_row(
    source_row: Mapping[str, object],
    *,
    field: str,
    snapshot_a_value: object,
    snapshot_b_value: object,
    difference: object,
    estimated_impact: float,
    guidance_role: str,
    as_of_date: object,
) -> dict[str, object]:
    """Return one promoted portfolio return-reconstruction formula row."""
    dataset, source_column, role_label = _workbook_reconstruction_role_metadata(field)
    return {
        _pc_findings.PORTFOLIO_ID: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID
        ),
        _pc_findings.FROM_DATE: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_FROM_DATE
        ),
        _pc_findings.THRU_DATE: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_THRU_DATE
        ),
        _AS_OF_DATE: as_of_date,
        _USE: _USE_EXPLAINS_CHANGE,
        _CHANGE_LABEL: f"{role_label} changed",
        _DATASET_FIELD: f"{dataset}.{source_column}",
        _pc_findings.SECURITY_ID: None,
        _pc_findings.SNAPSHOT_A_VALUE: snapshot_a_value,
        _pc_findings.SNAPSHOT_B_VALUE: snapshot_b_value,
        _CHANGE: difference,
        _pc_findings.IMPACT_INPUT_VALUE: snapshot_a_value,
        _ESTIMATED_IMPACT: estimated_impact,
        _INPUT_ROLE: _INPUT_ROLE_PERFORMANCE_INPUT,
        _IMPACT_STATUS: _IMPACT_STATUS_ESTIMATED,
        _REVIEW_NOTE: "",
        _REVIEW_GUIDANCE: _workbook_portfolio_formula_guidance(
            field,
            role_label,
            difference,
        ),
        _pc_findings.DATASET: dataset,
        _pc_findings.SOURCE_COLUMN: source_column,
        _pc_findings.FINDING_CODE: _RECONSTRUCTION_FORMULA_FINDING_CODE,
        _pc_explain.REVIEW_RANK: -100,
        _USE_PRIORITY: _workbook_use_priority(_USE_EXPLAINS_CHANGE),
        _REVIEW_KEY: source_row.get(_pc_reconstruction.RECONSTRUCTION_REVIEW_KEY),
    }


def _workbook_security_reconstruction_formula_row(
    source_row: Mapping[str, object],
    *,
    field: str,
    snapshot_a_value: object,
    snapshot_b_value: object,
    difference: object,
    estimated_impact: float,
    guidance_role: str,
    as_of_date: object,
) -> dict[str, object]:
    """Return one promoted security return-reconstruction formula row."""
    dataset, source_column, role_label = _workbook_reconstruction_role_metadata(field)
    security_id = source_row.get(_pc_reconstruction.RECONSTRUCTION_SECURITY_ID)
    return {
        _pc_findings.PORTFOLIO_ID: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID
        ),
        _pc_findings.FROM_DATE: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_FROM_DATE
        ),
        _pc_findings.THRU_DATE: source_row.get(
            _pc_reconstruction.RECONSTRUCTION_THRU_DATE
        ),
        _AS_OF_DATE: as_of_date,
        _USE: _USE_EXPLAINS_CHANGE,
        _CHANGE_LABEL: f"{role_label} changed",
        _DATASET_FIELD: f"{dataset}.{source_column}",
        _pc_findings.SECURITY_ID: security_id,
        _pc_findings.SNAPSHOT_A_VALUE: snapshot_a_value,
        _pc_findings.SNAPSHOT_B_VALUE: snapshot_b_value,
        _CHANGE: difference,
        _pc_findings.IMPACT_INPUT_VALUE: snapshot_a_value,
        _ESTIMATED_IMPACT: estimated_impact,
        _INPUT_ROLE: _INPUT_ROLE_PERFORMANCE_INPUT,
        _IMPACT_STATUS: _IMPACT_STATUS_ESTIMATED,
        _REVIEW_NOTE: "",
        _REVIEW_GUIDANCE: _workbook_security_formula_guidance(
            field,
            role_label,
            _format_value(security_id),
            difference,
        ),
        _pc_findings.DATASET: dataset,
        _pc_findings.SOURCE_COLUMN: source_column,
        _pc_findings.FINDING_CODE: _RECONSTRUCTION_FORMULA_FINDING_CODE,
        _pc_explain.REVIEW_RANK: -100,
        _USE_PRIORITY: _workbook_use_priority(_USE_EXPLAINS_CHANGE),
        _REVIEW_KEY: source_row.get(_pc_reconstruction.RECONSTRUCTION_REVIEW_KEY),
    }


def _workbook_portfolio_formula_guidance(
    field: str,
    role_label: str,
    difference: object,
) -> str:
    """Return deterministic guidance for portfolio reconstruction formula rows."""
    change_text = _workbook_change_amount_text(difference)
    if field == _RECONSTRUCTION_BEGINNING_VALUE_FIELD:
        return (
            f"Beginning portfolio value {_workbook_increased_or_decreased(difference)} "
            f"by {change_text}. A higher beginning value lowers the "
            "calculated return."
        )
    if field == _RECONSTRUCTION_ENDING_VALUE_FIELD:
        return (
            f"Ending portfolio value {_workbook_increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == _RECONSTRUCTION_NET_FLOW_FIELD:
        return (
            f"Net external flows {_workbook_increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == _RECONSTRUCTION_WEIGHTED_FLOW_FIELD:
        return (
            f"Weighted external flows {_workbook_increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == _RECONSTRUCTION_INCOME_FIELD:
        return f"Income {_workbook_increased_or_decreased(difference)} by {change_text}."
    return (
        f"{role_label} {_workbook_increased_or_decreased(difference)} by "
        f"{change_text}."
    )


def _workbook_security_formula_guidance(
    field: str,
    role_label: str,
    security_id: str,
    difference: object,
) -> str:
    """Return deterministic guidance for security reconstruction formula rows."""
    security_prefix = f"{security_id} " if security_id else ""
    change_text = _workbook_change_amount_text(difference)
    if field == _RECONSTRUCTION_BEGINNING_VALUE_FIELD:
        return (
            f"{security_prefix}beginning value "
            f"{_workbook_increased_or_decreased(difference)} by {change_text}. "
            "A higher beginning value lowers the calculated return."
        )
    if field == _RECONSTRUCTION_ENDING_VALUE_FIELD:
        return (
            f"{security_prefix}ending value "
            f"{_workbook_increased_or_decreased(difference)} by {change_text}."
        )
    if field == _RECONSTRUCTION_NET_FLOW_FIELD:
        return (
            f"{security_prefix}buy/sell flow was {change_text} "
            f"{_workbook_higher_or_lower(difference)}."
        )
    if field == _RECONSTRUCTION_WEIGHTED_FLOW_FIELD:
        return (
            f"{security_prefix}date-weighted buy/sell flow was {change_text} "
            f"{_workbook_higher_or_lower(difference)}."
        )
    if field == _RECONSTRUCTION_INCOME_FIELD:
        return (
            f"{security_prefix}income {_workbook_increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    return (
        f"{security_prefix}{role_label.lower()} "
        f"{_workbook_increased_or_decreased(difference)} by {change_text}."
    )


def _workbook_selected_underlying_impact_rows(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> list[tuple[dict[str, object], float]]:
    """Return additive impact rows selected for workbook explained totals.

    Notes:
        Performance Differences totals must use the same selected impact rows
        as the Performance Difference Causes sheet. Otherwise transaction amount rows can
        be counted in summary totals after the detail sheet has already treated
        them as supporting evidence for changed holdings.
    """
    selected_rows: list[tuple[dict[str, object], float]] = []
    for row in _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=comparison_level,
    ):
        if not _workbook_is_underlying_cause_row(row):
            continue
        estimated_impact = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
        if estimated_impact is None:
            continue
        selected_rows.append((row, estimated_impact))
    return selected_rows


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
    performance_row = _workbook_performance_change_row(row)
    return {
        **performance_row,
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
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
        _REVIEW_NOTE: "None",
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_empty_security_changes_table() -> pl.DataFrame:
    """Return an empty workbook security-level performance differences table."""
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
            _REVIEW_NOTE: pl.String,
            _REVIEW_KEY: pl.String,
        }
    )


def _workbook_ranked_changed_rows(findings: pl.DataFrame) -> list[dict[str, object]]:
    """Return ranked changed rows with selected additive impacts marked."""
    return _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    )


def _workbook_ranked_changed_rows_for_level(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> list[dict[str, object]]:
    """Return ranked changed rows for one primary comparison level."""
    evidence = _workbook_with_primary_review_key(
        _workbook_top_evidence_table(findings, comparison_level=comparison_level),
        comparison_level,
    )
    if evidence.is_empty():
        return []

    selected_impact_bases = _workbook_selected_impact_basis_keys(
        findings,
        comparison_level=comparison_level,
    )
    performance_input_keys = _workbook_performance_input_family_keys(
        findings,
        comparison_level=comparison_level,
    )
    rows: list[dict[str, object]] = []
    for row in evidence.iter_rows(named=True):
        rows.append(
            _workbook_selected_impact_row(
                row,
                selected_impact_bases,
                performance_input_keys,
                comparison_level=comparison_level,
            )
        )
    return rows


def _workbook_underlying_causes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> pl.DataFrame:
    """Return input rows that may directly explain performance differences."""
    unexplained_keys = _workbook_unexplained_primary_keys(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
    )
    performance_input_keys = _workbook_performance_input_family_keys(
        findings,
        comparison_level=comparison_level,
    )
    rows: list[dict[str, object]] = []
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        formula_rows = _workbook_security_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_security_period_keys(findings),
        )
    else:
        formula_rows = _workbook_portfolio_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_portfolio_period_keys(findings),
        )
    formula_keys = {
        _workbook_primary_key(row, comparison_level)
        for row in formula_rows
    }
    ranked_rows = _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=comparison_level,
    )
    cash_security_matches = _workbook_cash_security_matches(
        ranked_rows,
        comparison_level=comparison_level,
    )
    attributed_formula_source_rows = _workbook_formula_source_attributed_rows(
        ranked_rows,
        formula_rows,
        cash_security_matches=cash_security_matches,
        comparison_level=comparison_level,
    )
    attributed_source_keys = {
        _workbook_source_row_key(row, comparison_level)
        for row in attributed_formula_source_rows
    }
    rows.extend(
        _workbook_changed_item_row(row, comparison_path=comparison_path)
        for row in attributed_formula_source_rows
    )

    for row in ranked_rows:
        row = _workbook_with_cash_balance_security(
            row,
            cash_security_matches,
            comparison_level=comparison_level,
        )
        has_formula_role = _workbook_primary_key(row, comparison_level) in formula_keys
        source_row_key = _workbook_source_row_key(row, comparison_level)
        if source_row_key in attributed_source_keys:
            continue
        if has_formula_role and _workbook_is_underlying_cause_row(row):
            workbook_row = _workbook_changed_item_row(
                _workbook_formula_support_row(
                    row,
                    comparison_level=comparison_level,
                ),
                comparison_path=comparison_path,
            )
        elif _workbook_is_underlying_cause_row(row):
            workbook_row = _workbook_changed_item_row(
                row,
                comparison_path=comparison_path,
            )
        elif _workbook_should_promote_context_row(
            row,
            unexplained_keys,
            performance_input_keys,
            comparison_level=comparison_level,
        ):
            workbook_row = _workbook_changed_item_row(
                _workbook_non_additive_row(row),
                comparison_path=comparison_path,
            )
        else:
            continue
        rows.append(workbook_row)
    rows.extend(
        _workbook_missing_underlying_cause_rows(
            findings,
            rows,
            comparison_level=comparison_level,
        )
    )
    if not rows:
        return _workbook_empty_changed_item_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows, infer_schema_length=None),
        _workbook_left_review_sort_columns(comparison_level=comparison_level),
    )


def _workbook_formula_source_attributed_rows(
    source_rows: Sequence[Mapping[str, object]],
    formula_rows: Sequence[Mapping[str, object]],
    *,
    cash_security_matches: Mapping[tuple[object, ...], object],
    comparison_level: str,
) -> list[dict[str, object]]:
    """Return formula impacts allocated onto recognizable source-data rows."""
    rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for formula_row in formula_rows:
        estimated_impact = _number_or_none(formula_row.get(_ESTIMATED_IMPACT))
        if estimated_impact is None:
            continue
        candidate_rows = _workbook_formula_source_candidates(
            source_rows,
            formula_row,
            comparison_level=comparison_level,
        )
        if not candidate_rows:
            continue
        bases = [
            _workbook_formula_source_basis(row, formula_row)
            for row in candidate_rows
        ]
        total_basis = sum(bases)
        if abs(total_basis) <= _WORKBOOK_UNEXPLAINED_TOLERANCE:
            continue
        for row, basis in zip(candidate_rows, bases, strict=True):
            attributed_row = _workbook_source_attributed_row(
                row,
                formula_row,
                estimated_impact * basis / total_basis,
                comparison_level=comparison_level,
            )
            _workbook_attach_cash_balance_security(
                attributed_row,
                cash_security_matches,
                comparison_level=comparison_level,
            )
            key = _workbook_source_row_key(attributed_row, comparison_level)
            existing_row = rows_by_key.get(key)
            if existing_row is None:
                rows_by_key[key] = attributed_row
                continue
            existing_impact = _number_or_none(
                existing_row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)
            )
            additional_impact = _number_or_none(
                attributed_row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)
            )
            existing_row[_pc_explain.ESTIMATED_RETURN_IMPACT] = (
                (existing_impact or 0.0) + (additional_impact or 0.0)
            )
    return list(rows_by_key.values())


def _workbook_formula_source_candidates(
    source_rows: Sequence[Mapping[str, object]],
    formula_row: Mapping[str, object],
    *,
    comparison_level: str,
) -> list[Mapping[str, object]]:
    """Return source rows that make up one reconstruction formula row."""
    formula_field = formula_row.get(_pc_findings.SOURCE_COLUMN)
    formula_date = formula_row.get(_AS_OF_DATE)
    rows = [
        row
        for row in source_rows
        if _workbook_formula_source_owner_matches(
            row,
            formula_row,
            comparison_level=comparison_level,
        )
    ]
    if formula_field in {
        _RECONSTRUCTION_BEGINNING_VALUE_FIELD,
        _RECONSTRUCTION_ENDING_VALUE_FIELD,
    }:
        return [
            row
            for row in rows
            if row.get(_pc_findings.DATASET) == pc_cols.HOLDINGS
            and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.MARKET_VALUE
            and _workbook_as_of_date(row) == formula_date
        ]
    if formula_field in {
        _RECONSTRUCTION_NET_FLOW_FIELD,
        _RECONSTRUCTION_WEIGHTED_FLOW_FIELD,
    }:
        flow_categories = (
            {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
            if comparison_level == SECURITY_COMPARISON_LEVEL
            else {TRANSACTION_CATEGORY_EXTERNAL_FLOW}
        )
        return [
            row
            for row in rows
            if _workbook_same_period(row, formula_row)
            and row.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
            and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.AMOUNT
            and row.get(_pc_findings.TRANSACTION_CATEGORY) in flow_categories
        ]
    if formula_field == _RECONSTRUCTION_INCOME_FIELD:
        return [
            row
            for row in rows
            if _workbook_same_period(row, formula_row)
            and row.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
            and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.AMOUNT
            and row.get(_pc_findings.TRANSACTION_CATEGORY)
            in {TRANSACTION_CATEGORY_FEE_EXPENSE, TRANSACTION_CATEGORY_INCOME}
        ]
    return []


def _workbook_formula_source_owner_matches(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
    *,
    comparison_level: str,
) -> bool:
    """Return whether a source row belongs to the formula row owner."""
    if source_row.get(_pc_findings.PORTFOLIO_ID) != formula_row.get(
        _pc_findings.PORTFOLIO_ID
    ):
        return False
    if comparison_level != SECURITY_COMPARISON_LEVEL:
        return True
    return source_row.get(_pc_findings.SECURITY_ID) == formula_row.get(
        _pc_findings.SECURITY_ID
    )


def _workbook_same_period(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
) -> bool:
    """Return whether two rows use the same inclusive performance period."""
    return (
        source_row.get(_pc_findings.FROM_DATE) == formula_row.get(_pc_findings.FROM_DATE)
        and source_row.get(_pc_findings.THRU_DATE)
        == formula_row.get(_pc_findings.THRU_DATE)
    )


def _workbook_source_attributed_row(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
    estimated_impact: float,
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return a source row cloned into the formula period with allocated impact."""
    row_dict = dict(source_row)
    row_dict[_pc_findings.FROM_DATE] = formula_row.get(_pc_findings.FROM_DATE)
    row_dict[_pc_findings.THRU_DATE] = formula_row.get(_pc_findings.THRU_DATE)
    row_dict[_REVIEW_KEY] = formula_row.get(_REVIEW_KEY)
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        row_dict[_pc_findings.SECURITY_ID] = formula_row.get(_pc_findings.SECURITY_ID)
    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = estimated_impact
    row_dict[_pc_explain.IMPACT_BASIS] = "source_row_reconstruction"
    row_dict[_pc_explain.IMPACT_METHOD] = "return_reconstruction_source_allocation"
    if (
        row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
        and row_dict.get(_pc_findings.SOURCE_COLUMN) == pc_cols.AMOUNT
    ):
        row_dict[_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW] = True
        row_dict["_workbook_reconstruction_comparison_level"] = comparison_level
    return row_dict


def _workbook_with_cash_balance_security(
    row: Mapping[str, object],
    cash_security_matches: Mapping[tuple[object, ...], object],
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return row with matched cash security attached when available."""
    row_dict = dict(row)
    _workbook_attach_cash_balance_security(
        row_dict,
        cash_security_matches,
        comparison_level=comparison_level,
    )
    return row_dict


def _workbook_attach_cash_balance_security(
    attributed_row: dict[str, object],
    cash_security_matches: Mapping[tuple[object, ...], object],
    *,
    comparison_level: str,
) -> None:
    """Attach the changed cash holding security when one row is identifiable."""
    if (
        attributed_row.get(_pc_findings.DATASET) != pc_cols.TRANSACTIONS
        or attributed_row.get(_pc_findings.SOURCE_COLUMN) != pc_cols.AMOUNT
    ):
        return
    cash_security_id = cash_security_matches.get(
        _workbook_source_row_key(attributed_row, comparison_level)
    )
    if cash_security_id:
        attributed_row[_WORKBOOK_CASH_BALANCE_SECURITY_ID] = cash_security_id


def _workbook_cash_security_matches(
    source_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> dict[tuple[object, ...], object]:
    """Return transaction source-row keys mapped to matching cash securities."""
    matches: dict[tuple[object, ...], object] = {}
    for row in source_rows:
        if (
            row.get(_pc_findings.DATASET) != pc_cols.TRANSACTIONS
            or row.get(_pc_findings.SOURCE_COLUMN) != pc_cols.AMOUNT
        ):
            continue
        cash_security_id = _workbook_matching_cash_security_id(row, source_rows)
        if cash_security_id:
            matches[_workbook_source_row_key(row, comparison_level)] = cash_security_id
    return matches


def _workbook_matching_cash_security_id(
    transaction_row: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
) -> object | None:
    """Return the matching cash holding security for a transaction amount row."""
    transaction_delta = _number_or_none(
        transaction_row.get(_pc_findings.DELTA_B_MINUS_A)
    )
    if transaction_delta is None:
        return None
    matches = [
        row
        for row in source_rows
        if row.get(_pc_findings.PORTFOLIO_ID)
        == transaction_row.get(_pc_findings.PORTFOLIO_ID)
        and _workbook_same_period(row, transaction_row)
        and row.get(_pc_findings.DATASET) == pc_cols.HOLDINGS
        and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.MARKET_VALUE
        and _workbook_as_of_date(row) == transaction_row.get(_pc_findings.THRU_DATE)
        and _workbook_is_cash_security(row.get(_pc_findings.SECURITY_ID))
        and _workbook_same_amount(
            _number_or_none(row.get(_pc_findings.DELTA_B_MINUS_A)),
            transaction_delta,
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0].get(_pc_findings.SECURITY_ID)


def _workbook_is_cash_security(security_id: object) -> bool:
    """Return whether an identifier appears to be a cash holding."""
    security_text = _format_value(security_id).upper()
    return security_text.startswith("CASH")


def _workbook_same_amount(first_value: float | None, second_value: float | None) -> bool:
    """Return whether two source amounts are effectively the same amount."""
    if first_value is None or second_value is None:
        return False
    return abs(first_value - second_value) <= 0.005


def _workbook_source_row_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return a stable key for one source-data row in the workbook."""
    return (
        *_workbook_primary_key(row, comparison_level),
        row.get(_pc_findings.DATASET),
        row.get(_pc_findings.SOURCE_COLUMN),
        row.get(_pc_findings.SECURITY_ID),
        _workbook_as_of_date(row),
        row.get(_pc_findings.TRANSACTION_CATEGORY),
        row.get(_pc_findings.SNAPSHOT_A_VALUE),
        row.get(_pc_findings.SNAPSHOT_B_VALUE),
        row.get(_pc_findings.DELTA_B_MINUS_A),
    )

def _workbook_formula_source_basis(
    source_row: Mapping[str, object],
    formula_row: Mapping[str, object],
) -> float:
    """Return source-row basis used to allocate one formula impact."""
    formula_field = formula_row.get(_pc_findings.SOURCE_COLUMN)
    delta = _number_or_none(source_row.get(_pc_findings.DELTA_B_MINUS_A)) or 0.0
    if formula_field == _RECONSTRUCTION_WEIGHTED_FLOW_FIELD:
        return delta * _workbook_source_flow_weight(source_row)
    return delta


def _workbook_source_flow_weight(row: Mapping[str, object]) -> float:
    """Return Modified Dietz flow weight for a transaction source row."""
    from_date = row.get(_pc_findings.FROM_DATE)
    thru_date = row.get(_pc_findings.THRU_DATE)
    flow_date = _workbook_as_of_date(row)
    if not isinstance(from_date, _dt.date) or not isinstance(thru_date, _dt.date):
        return 1.0
    if not isinstance(flow_date, _dt.date):
        return 1.0
    try:
        return modified_dietz_flow_weight(
            from_date=from_date,
            thru_date=thru_date,
            flow_date=flow_date,
            inclusion_rule="beginning_of_day",
        )
    except ValueError:
        return 1.0


def _workbook_missing_underlying_cause_rows(
    findings: pl.DataFrame,
    underlying_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> list[dict[str, object]]:
    """Return placeholder rows for changed periods without input causes."""
    coverage = _workbook_with_primary_review_key(
        _workbook_primary_coverage_summary(findings, comparison_level=comparison_level),
        comparison_level,
    )
    if coverage.is_empty():
        return []

    underlying_period_keys = {
        _workbook_primary_key(row, comparison_level)
        for row in underlying_rows
    }
    rows: list[dict[str, object]] = []
    for row in coverage.iter_rows(named=True):
        if _workbook_primary_key(row, comparison_level) in underlying_period_keys:
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
        _CHANGE_LABEL: "No additive underlying cause found",
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _pc_findings.SNAPSHOT_A_VALUE: None,
        _pc_findings.SNAPSHOT_B_VALUE: None,
        _CHANGE: None,
        _ESTIMATED_IMPACT: None,
        _IMPACT_STATUS: _IMPACT_STATUS_REVIEW_ONLY,
        _REVIEW_NOTE: (
            'Review the "Other Data Differences" sheet, "Raw Audit Trail" sheet, missing '
            "datasets, or vendor methodology."
        ),
        _REVIEW_GUIDANCE: (
            'No identifiable cause was found. Review the "Other Data Differences" '
            'sheet, "Raw Audit Trail" sheet, missing datasets, or vendor methodology.'
        ),
        _pc_findings.DATASET: _NO_UNDERLYING_CAUSE_DATASET,
        _pc_findings.SOURCE_COLUMN: None,
        _pc_findings.FINDING_CODE: None,
        _pc_explain.REVIEW_RANK: 999999,
        _USE_PRIORITY: _workbook_use_priority(_USE_DIAGNOSTIC),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_context_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> pl.DataFrame:
    """Return review-context rows that are not additive return explanations."""
    unexplained_keys = _workbook_unexplained_primary_keys(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
    )
    performance_input_keys = _workbook_performance_input_family_keys(
        findings,
        comparison_level=comparison_level,
    )
    rows = []
    for row in _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=comparison_level,
    ):
        if not _workbook_is_context_row(row):
            continue
        if _workbook_should_promote_context_row(
            row,
            unexplained_keys,
            performance_input_keys,
            comparison_level=comparison_level,
        ):
            continue
        rows.append(_workbook_changed_item_row(_workbook_non_additive_row(row)))
    if not rows:
        return _workbook_empty_changed_item_table()
    return _workbook_sorted_table(
        pl.DataFrame(rows, infer_schema_length=None),
        _workbook_left_review_sort_columns(comparison_level=comparison_level),
    )


def _workbook_unexplained_primary_keys(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str,
) -> set[tuple[object, ...]]:
    """Return primary review keys with a meaningful unexplained remainder."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        summary = _workbook_security_changes_table(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
        )
    else:
        summary = _workbook_portfolio_changes_table(
            findings,
            comparison_path=comparison_path,
        )

    keys: set[tuple[object, ...]] = set()
    for row in summary.iter_rows(named=True):
        unexplained_change = _number_or_none(row.get(_UNEXPLAINED_CHANGE))
        if (
            unexplained_change is None
            or abs(unexplained_change) <= _WORKBOOK_UNEXPLAINED_TOLERANCE
        ):
            continue
        keys.add(_workbook_primary_key(row, comparison_level))
    return keys


def _workbook_should_promote_context_row(
    row: Mapping[str, object],
    unexplained_keys: set[tuple[object, ...]],
    performance_input_keys: set[tuple[object, ...]],
    *,
    comparison_level: str,
) -> bool:
    """Return whether review-only evidence belongs with unresolved causes.

    Notes:
        This is a workbook presentation rule, not an attribution model. It keeps
        fully explained periods clean while surfacing plausible evidence-only
        input changes on the ``Performance Difference Causes`` sheet when a period still has
        a performance difference that additive rows did not explain.
    """
    if not _workbook_is_context_row(row) or not _workbook_has_evidence_only_policy(row):
        return False
    if not _workbook_is_promotable_evidence_only_row(row):
        return False
    if _workbook_is_transaction_component_row(row):
        return _workbook_cause_family_key(row, comparison_level) in performance_input_keys
    if _workbook_primary_key(row, comparison_level) in unexplained_keys:
        return True
    return (
        _field_roles.is_input_component(
            row.get(_pc_findings.DATASET),
            row.get(_pc_findings.SOURCE_COLUMN),
        )
        and _workbook_cause_family_key(row, comparison_level) in performance_input_keys
    )


def _workbook_is_promotable_evidence_only_row(row: Mapping[str, object]) -> bool:
    """Return whether an evidence-only row is plausibly return-explanatory."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    return source_column in _WORKBOOK_PROMOTABLE_EVIDENCE_COLUMNS.get(dataset, set())


def _workbook_left_review_sort_columns(*, comparison_level: str) -> tuple[str, ...]:
    """Return the shared left-column sort order for review detail sheets."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return (
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
            _pc_findings.SECURITY_ID,
            _pc_findings.DATASET,
            _pc_findings.SOURCE_COLUMN,
        )
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.SECURITY_ID,
    )


def _workbook_is_transaction_component_row(row: Mapping[str, object]) -> bool:
    """Return whether a row is support for transaction amount, not an input cause."""
    return row.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS and row.get(
        _pc_findings.SOURCE_COLUMN
    ) in {pc_cols.COMMISSION, pc_cols.PRICE, pc_cols.QUANTITY}


def _workbook_selected_impact_basis_keys(
    findings: pl.DataFrame,
    *,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> set[tuple[object, ...]]:
    """Return period/impact-basis keys included in Performance Differences totals."""
    causes = _workbook_primary_cause_summary(
        findings,
        comparison_level=comparison_level,
    )
    if causes.is_empty():
        return set()

    keys: set[tuple[object, ...]] = set()
    del causes
    rows = _workbook_with_primary_review_key(
        _workbook_top_evidence_table(findings, comparison_level=comparison_level),
        comparison_level,
    )
    grouped_rows: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows.iter_rows(named=True):
        if _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
            continue
        group_key = _workbook_cause_family_key(row, comparison_level)
        grouped_rows.setdefault(group_key, []).append(row)
    for group_rows in grouped_rows.values():
        for selected_row in _workbook_preferred_estimate_rows(group_rows):
            impact_basis = selected_row.get(_pc_explain.IMPACT_BASIS)
            if impact_basis == _pc_explain.IMPACT_BASIS_NO_ESTIMATE:
                continue
            keys.add(
                (
                    *_workbook_cause_family_key(selected_row, comparison_level),
                    impact_basis,
                )
            )
    return keys


def _workbook_performance_input_family_keys(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
) -> set[tuple[object, ...]]:
    """Return cause-family keys with selected performance input rows."""
    keys: set[tuple[object, ...]] = set()
    evidence = _workbook_with_primary_review_key(
        _workbook_top_evidence_table(findings, comparison_level=comparison_level),
        comparison_level,
    )
    for row in evidence.iter_rows(named=True):
        if not _field_roles.is_performance_input(
            row.get(_pc_findings.DATASET),
            row.get(_pc_findings.SOURCE_COLUMN),
        ):
            continue
        if _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
            continue
        keys.add(_workbook_cause_family_key(row, comparison_level))
    return keys


def _workbook_cause_family_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the source-input family where estimates should not double count."""
    family = _workbook_cause_family(row)
    return (
        *_workbook_primary_key(row, comparison_level),
        row.get(_pc_findings.SECURITY_ID),
        family,
    )


def _workbook_cause_family(row: Mapping[str, object]) -> object:
    """Return the broad accounting family for a changed input row."""
    dataset = row.get(_pc_findings.DATASET)
    source_column = row.get(_pc_findings.SOURCE_COLUMN)
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.MARKET_VALUE,
        pc_cols.ACCRUED,
        pc_cols.QUANTITY,
        pc_cols.PRICE,
    }:
        return "holding_value"
    if dataset == pc_cols.TRANSACTIONS:
        return pc_cols.TRANSACTIONS
    return dataset


def _workbook_preferred_estimate_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Return estimate rows selected for workbook additive totals."""
    if any(
        row.get(_pc_explain.IMPACT_BASIS)
        == _pc_explain.IMPACT_BASIS_SECURITY_CONTRIBUTION
        for row in rows
    ):
        return [
            row
            for row in rows
            if row.get(_pc_explain.IMPACT_BASIS)
            == _pc_explain.IMPACT_BASIS_SECURITY_CONTRIBUTION
        ]
    holding_inputs = [
        row
        for row in rows
        if row.get(_pc_findings.DATASET) == pc_cols.HOLDINGS
        and _field_roles.is_performance_input(
            row.get(_pc_findings.DATASET),
            row.get(_pc_findings.SOURCE_COLUMN),
        )
    ]
    if holding_inputs:
        return holding_inputs
    holdings_price_rows = [
        row
        for row in rows
        if row.get(_pc_findings.DATASET) == pc_cols.HOLDINGS
        and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.PRICE
    ]
    if holdings_price_rows:
        return holdings_price_rows
    return list(rows)


def _workbook_selected_impact_row(
    row: Mapping[str, object],
    selected_impact_bases: set[tuple[object, ...]],
    performance_input_keys: set[tuple[object, ...]],
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return row with unselected candidate estimates cleared for the workbook."""
    row_dict = dict(row)
    estimated_impact = _number_or_none(row_dict.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
    if estimated_impact is None:
        return row_dict

    if (
        comparison_level == PORTFOLIO_COMPARISON_LEVEL
        and row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
    ):
        row_dict[_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
        return _workbook_non_additive_row(row_dict)

    holding_value_key = (
        *_workbook_primary_key(row_dict, comparison_level),
        row_dict.get(_pc_findings.SECURITY_ID),
        "holding_value",
    )
    if (
        row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
        and row_dict.get(_pc_findings.TRANSACTION_CATEGORY)
        in {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
        and holding_value_key in performance_input_keys
    ):
        row_dict[_WORKBOOK_UNSELECTED_RELATED_ESTIMATE] = True
        row_dict[_WORKBOOK_TRANSACTION_FLOW_SUPPORTS_HOLDING] = True
        row_dict[_pc_explain.IMPACT_MESSAGE] = (
            "Supporting evidence for changed holdings.market_value."
        )
        return _workbook_non_additive_row(row_dict)

    key = (
        *_workbook_cause_family_key(row_dict, comparison_level),
        row_dict.get(_pc_explain.IMPACT_BASIS),
    )
    if key in selected_impact_bases:
        return row_dict

    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[_pc_explain.IMPACT_BASIS] = _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[_pc_explain.IMPACT_METHOD] = None
    row_dict[_WORKBOOK_UNSELECTED_RELATED_ESTIMATE] = True
    row_dict[_pc_explain.IMPACT_MESSAGE] = (
        "Another estimate was selected for this portfolio-period cause area."
    )
    return row_dict


def _workbook_change_amount_text(value: object) -> str:
    """Return a compact absolute amount for reviewer-facing explanations."""
    number = _number_or_none(value)
    if number is None:
        return "the changed amount"
    return f"{abs(number):,.2f}"


def _workbook_row_change_value(row: Mapping[str, object]) -> object:
    """Return the changed amount from either workbook or finding row shape."""
    change = row.get(_CHANGE)
    if change is not None:
        return change
    return row.get(_pc_findings.DELTA_B_MINUS_A)


def _workbook_increased_or_decreased(value: object) -> str:
    """Return increased/decreased wording for a numeric B-minus-A value."""
    number = _number_or_none(value)
    if number is not None and number < 0:
        return "decreased"
    return "increased"


def _workbook_higher_or_lower(value: object) -> str:
    """Return higher/lower wording for a numeric B-minus-A value."""
    number = _number_or_none(value)
    if number is not None and number < 0:
        return "lower"
    return "higher"


def _workbook_formula_support_row(
    row: Mapping[str, object],
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return a non-additive row marked as support for reconstruction formulas."""
    row_dict = _workbook_non_additive_row(row)
    if (
        row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
        and row_dict.get(_pc_findings.SOURCE_COLUMN) == pc_cols.AMOUNT
    ):
        row_dict[_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW] = True
        row_dict["_workbook_reconstruction_comparison_level"] = comparison_level
    return row_dict


def _workbook_non_additive_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return a workbook row with explained-difference fields cleared."""
    row_dict = dict(row)
    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[_pc_explain.IMPACT_BASIS] = _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[_pc_explain.IMPACT_METHOD] = None
    return row_dict


def _workbook_is_underlying_cause_row(row: Mapping[str, object]) -> bool:
    """Return whether row is an identifiable input-cause candidate."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_UNDERLYING_CAUSE


def _workbook_is_reported_diagnostic_row(row: Mapping[str, object]) -> bool:
    """Return whether row is a reported-performance diagnostic, not a root cause."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC


def _workbook_is_context_row(row: Mapping[str, object]) -> bool:
    """Return whether row is context-only evidence."""
    return _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_CONTEXT


def _workbook_row_kind(row: Mapping[str, object]) -> str:
    """Return the workbook presentation role for a finding row."""
    if row.get(_pc_findings.DATASET) == _NO_UNDERLYING_CAUSE_DATASET:
        return _WORKBOOK_ROW_KIND_DIAGNOSTIC
    if _field_roles.is_reported_performance_component(
        row.get(_pc_findings.DATASET),
        row.get(_pc_findings.SOURCE_COLUMN),
    ):
        return _WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC
    if row.get(_pc_findings.EVIDENCE_ROLE) == _pc_findings.CONTEXT.value:
        return _WORKBOOK_ROW_KIND_CONTEXT
    if _workbook_has_evidence_only_policy(row):
        return _WORKBOOK_ROW_KIND_CONTEXT
    if row.get(_pc_findings.DATASET) in {
        pc_cols.PORTFOLIO_PERFORMANCE,
        pc_cols.SECURITY_PERFORMANCE,
    }:
        return _WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC
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
        _AS_OF_DATE: _workbook_as_of_date(row),
        _USE: row_use,
        _CHANGE_LABEL: _workbook_change_label(row),
        _DATASET_FIELD: _workbook_dataset_field(row),
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _pc_findings.SNAPSHOT_A_VALUE: row.get(_pc_findings.SNAPSHOT_A_VALUE),
        _pc_findings.SNAPSHOT_B_VALUE: row.get(_pc_findings.SNAPSHOT_B_VALUE),
        _CHANGE: row.get(_pc_findings.DELTA_B_MINUS_A),
        _pc_findings.IMPACT_INPUT_VALUE: row.get(_pc_findings.IMPACT_INPUT_VALUE),
        _ESTIMATED_IMPACT: estimated_impact,
        _INPUT_ROLE: _workbook_input_role(row, estimated_impact),
        _IMPACT_STATUS: impact_status,
        _REVIEW_NOTE: _workbook_review_note(row, estimated_impact, row_use, impact_status),
        _REVIEW_GUIDANCE: _workbook_review_guidance(
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


def _workbook_input_role(
    row: Mapping[str, object],
    estimated_impact: float | None,
) -> str:
    """Return the reviewer-facing role for one changed input row."""
    dataset = row.get(_pc_findings.DATASET)
    source_column = row.get(_pc_findings.SOURCE_COLUMN)
    if dataset == _NO_UNDERLYING_CAUSE_DATASET:
        return _INPUT_ROLE_DIAGNOSTIC
    if estimated_impact is not None:
        return _INPUT_ROLE_PERFORMANCE_INPUT
    if dataset == pc_cols.TRANSACTIONS and source_column in {
        pc_cols.COMMISSION,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _INPUT_ROLE_SUPPORTING_EVIDENCE
    if (
        _field_roles.is_input_component(dataset, source_column)
        or _field_roles.is_performance_input(dataset, source_column)
    ):
        return _INPUT_ROLE_INPUT_DRIVER
    if _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC:
        return _INPUT_ROLE_DIAGNOSTIC
    return _INPUT_ROLE_CONTEXT


def _workbook_as_of_date(row: Mapping[str, object]) -> object | None:
    """Return the date represented by a workbook evidence row."""
    input_date = row.get(_pc_findings.INPUT_DATE)
    if input_date is not None:
        return input_date
    return row.get(_pc_findings.THRU_DATE)


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
        row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE)
        or row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION)
        or row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW)
        or _workbook_is_context_row(row)
        or _workbook_is_reported_diagnostic_row(row)
        or _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC
        or _workbook_has_evidence_only_policy(row)
    ):
        return _IMPACT_STATUS_REVIEW_ONLY
    if _workbook_has_additive_policy(row):
        return _IMPACT_STATUS_MISSING_INPUT
    return _IMPACT_STATUS_MISSING_METHOD


def _workbook_review_note(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
) -> str:
    """Return one reviewer-facing note for a changed workbook row."""
    if estimated_impact is not None:
        return ""

    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if dataset in {pc_cols.PORTFOLIO_PERFORMANCE, pc_cols.SECURITY_PERFORMANCE}:
        return (
            "This is simply a difference in the raw performance datasets. Check "
            'the "Performance Difference Causes" sheet to see what explains it.'
        )
    source_explanation = _workbook_source_row_explanation(row, dataset, source_column)
    if source_explanation:
        return source_explanation
    if _workbook_has_evidence_only_policy(row):
        return (
            'Review-only evidence; this row is not counted in '
            '"Performance Differences"."Explained Difference".'
        )
    if row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return (
            "Supporting evidence for Modified Dietz flow rows; not counted "
            "separately."
        )
    if row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return (
            "Supporting evidence for Modified Dietz flow rows; not counted "
            "separately."
        )
    if row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE):
        return "Review this input component; a related performance input is selected."
    if impact_status == _IMPACT_STATUS_MISSING_INPUT:
        return (
            "Review inputs needed by the configured YAML method; no "
            "estimate is available for this row."
        )
    if impact_status == _IMPACT_STATUS_MISSING_METHOD:
        return _workbook_missing_impact_method_action(dataset, source_column)
    if row_use == _USE_REVIEW_CONTEXT:
        return "Review context; not included in explained performance difference."
    dataset_actions = {
        pc_cols.TRANSACTIONS: _workbook_review_change_action(
            "transaction",
            source_column,
        ),
        pc_cols.HOLDINGS: _workbook_review_change_action("holding", source_column),
        pc_cols.CASH: _workbook_review_change_action("cash", source_column),
    }
    return dataset_actions.get(
        dataset,
        _workbook_review_change_action("input", source_column),
    )


def _workbook_source_row_explanation(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return source-data explanation text when a row has a known source shape."""
    if dataset == pc_cols.HOLDINGS:
        return _workbook_holding_detail_explanation(row, source_column)
    if dataset == pc_cols.TRANSACTIONS:
        if source_column == pc_cols.AMOUNT:
            if row.get(_pc_findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
                return _workbook_portfolio_external_flow_transaction_explanation(row)
            return _workbook_transaction_cash_balance_explanation(row)
        return _workbook_transaction_component_explanation(row, source_column)
    if dataset == pc_cols.CASH:
        return _workbook_cash_detail_explanation(row, source_column)
    return ""


def _workbook_review_guidance(
    row: Mapping[str, object],
    estimated_impact: float | None,
    *,
    comparison_path: util.PathLike | None,
) -> str:
    """Return review guidance for why this row does or does not explain performance."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if estimated_impact is not None:
        if row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
            return _workbook_transaction_reconstruction_flow_guidance(row)
        if dataset == pc_cols.HOLDINGS and source_column in {
            pc_cols.ACCRUED,
            pc_cols.MARKET_VALUE,
            pc_cols.PRICE,
            pc_cols.QUANTITY,
        }:
            return _workbook_holding_detail_explanation(row, source_column)
        if dataset == pc_cols.TRANSACTIONS:
            return _workbook_transaction_component_explanation(row, source_column)
        return ""

    if (
        dataset == pc_cols.TRANSACTIONS
        and source_column in {pc_cols.COMMISSION, pc_cols.PRICE, pc_cols.QUANTITY}
    ):
        return _workbook_transaction_component_explanation(row, source_column)
    if row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return _workbook_transaction_cash_balance_explanation(row)
    if row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return _workbook_transaction_reconstruction_flow_guidance(row)
    if row.get(_WORKBOOK_TRANSACTION_FLOW_SUPPORTS_HOLDING):
        security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
        if security_id:
            return (
                f"{security_id} transaction activity changed. The security holding "
                "value row shows the counted effect."
            )
        return (
            "Transaction activity changed. The holding value row shows the counted "
            "effect."
        )
    if row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE):
        return _workbook_related_input_guidance(row, dataset, source_column)
    if _workbook_has_evidence_only_policy(row):
        return (
            'Review-only evidence; this row is not counted in '
            '"Performance Differences"."Explained Difference".'
        )
    if (
        _workbook_is_context_row(row)
        or _workbook_is_reported_diagnostic_row(row)
        or _workbook_row_kind(row) == _WORKBOOK_ROW_KIND_DIAGNOSTIC
    ):
        return "Review context; not an underlying input difference."

    dataset_column = _workbook_dataset_column_label(dataset, source_column)
    yaml_path = _workbook_yaml_path_label(comparison_path)
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.MARKET_VALUE,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _workbook_holding_detail_explanation(row, source_column)
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
    if dataset == pc_cols.HOLDINGS:
        if source_column not in {
            pc_cols.MARKET_VALUE,
            pc_cols.ACCRUED,
            pc_cols.QUANTITY,
        }:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        if source_column == pc_cols.ACCRUED:
            return (
                "Specify the YAML holding_impact_methods.accrued.method and "
                "holding_impact_methods.accrued.denominator_source in "
                f"{yaml_path}."
            )
        if source_column == pc_cols.QUANTITY:
            return (
                "Specify the YAML holding_impact_methods.quantity.method and "
                "holding_impact_methods.quantity.denominator_source in "
                f"{yaml_path}."
            )
        return (
            "Specify the YAML holding_impact_methods.market_value.method and "
            "holding_impact_methods.market_value.denominator_source in "
            f"{yaml_path}."
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


def _workbook_related_input_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return explicit guidance for an input component's related performance field."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.MARKET_VALUE,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _workbook_holding_detail_explanation(row, source_column)
    if dataset == pc_cols.TRANSACTIONS:
        return _workbook_transaction_component_explanation(row, source_column)
    return "This source-data change supports the related counted row."


def _workbook_transaction_reconstruction_flow_guidance(
    row: Mapping[str, object],
) -> str:
    """Return guidance for transaction rows absorbed by reconstruction formulas."""
    comparison_level = row.get("_workbook_reconstruction_comparison_level")
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return _workbook_transaction_component_explanation(row, pc_cols.AMOUNT)
    if row.get(_pc_findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
        return _workbook_portfolio_external_flow_transaction_explanation(row)
    return _workbook_transaction_cash_balance_explanation(row)


def _workbook_portfolio_external_flow_transaction_explanation(
    row: Mapping[str, object],
) -> str:
    """Return source-data wording for a portfolio external-flow transaction."""
    flow_delta = _workbook_row_change_value(row)
    weighted_flow_delta = (
        (_number_or_none(flow_delta) or 0.0) * _workbook_source_flow_weight(row)
    )
    return (
        f"{_workbook_transaction_code_prefix(row)}External flow "
        f"{_workbook_increased_or_decreased(flow_delta)} by "
        f"{_workbook_change_amount_text(flow_delta)}; weighted external flow "
        f"{_workbook_increased_or_decreased(weighted_flow_delta)} by "
        f"{_workbook_change_amount_text(weighted_flow_delta)}."
    )


def _workbook_transaction_cash_balance_explanation(row: Mapping[str, object]) -> str:
    """Return source-data wording for a transaction's ending cash-balance effect."""
    return (
        f"{_workbook_transaction_code_prefix(row)}Caused cash-balance "
        "ending holdings.market_value "
        f"to {_workbook_cash_balance_increased_or_decreased(row)} by "
        f"{_workbook_change_amount_text(_workbook_row_change_value(row))}."
    )


def _workbook_cash_balance_increased_or_decreased(row: Mapping[str, object]) -> str:
    """Return increased/decreased wording for cash effect of a transaction row."""
    if row.get(_pc_findings.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_POSITIVE:
        return "increase"
    return "decrease"


def _workbook_holding_detail_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return plain-language explanation for holding source rows."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    timing_label = _workbook_holding_timing_label(row)
    holdings_label = f"{timing_label} holdings" if timing_label else "holdings"
    change_value = _workbook_row_change_value(row)
    change_text = _workbook_change_amount_text(change_value)
    return (
        f"{security_prefix}{holdings_label}.{source_column} "
        f"{_workbook_increased_or_decreased(change_value)} by "
        f"{change_text}."
    )


def _workbook_cash_detail_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return plain-language explanation for cash source rows."""
    change_value = _workbook_row_change_value(row)
    return (
        f"Cash {source_column} {_workbook_increased_or_decreased(change_value)} "
        f"by {_workbook_change_amount_text(change_value)}."
    )


def _workbook_holding_timing_label(row: Mapping[str, object]) -> str:
    """Return beginning/ending label for inclusive-period holding dates."""
    input_date = row.get(_pc_findings.INPUT_DATE)
    from_date = row.get(_pc_findings.FROM_DATE)
    thru_date = row.get(_pc_findings.THRU_DATE)
    if (
        isinstance(input_date, _dt.date)
        and isinstance(from_date, _dt.date)
        and input_date == from_date - _dt.timedelta(days=1)
    ):
        return "beginning"
    if isinstance(input_date, _dt.date) and input_date == thru_date:
        return "ending"
    return ""


def _workbook_transaction_component_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return plain-language explanation for transaction component rows."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_text = f" for {security_id}" if security_id else ""
    if source_column == pc_cols.COMMISSION:
        change_value = _workbook_row_change_value(row)
        change_number = _number_or_none(change_value)
        change_verb = (
            "decrease"
            if change_number is not None and change_number < 0
            else "increase"
        )
        transaction_amount = (
            f"{security_id} transactions.amount"
            if security_id
            else "transactions.amount"
        )
        return (
            f"{_workbook_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb} "
            f"by {_workbook_change_amount_text(change_value)}."
        )
    if source_column in {pc_cols.PRICE, pc_cols.QUANTITY}:
        change_value = _workbook_row_change_value(row)
        change_number = _number_or_none(change_value)
        change_verb = (
            "decrease"
            if change_number is not None and change_number < 0
            else "increase"
        )
        transaction_amount = (
            f"{security_id} transactions.amount"
            if security_id
            else "transactions.amount"
        )
        return (
            f"{_workbook_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb}."
        )
    return (
        f"{_workbook_transaction_code_prefix(row)}The {source_column}"
        f"{security_text} changed by "
        f"{_workbook_change_amount_text(_workbook_row_change_value(row))}."
    )


def _workbook_transaction_code_prefix(row: Mapping[str, object]) -> str:
    """Return a short transaction-code prefix for transaction review guidance."""
    transaction_code = _format_value(row.get(_pc_findings.TRANSACTION_CODE))
    if not transaction_code:
        transaction_code = _workbook_transaction_code_fallback(row)
    if not transaction_code:
        return ""
    return f"{transaction_code.replace('_', ' ')}: "


def _workbook_transaction_code_fallback(row: Mapping[str, object]) -> str:
    """Return a compact transaction label when the raw code is unavailable."""
    category = _format_value(row.get(_pc_findings.TRANSACTION_CATEGORY))
    if category == TRANSACTION_CATEGORY_EXTERNAL_FLOW:
        if row.get(_pc_findings.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_POSITIVE:
            return "deposit"
        return "withdrawal"
    if category == TRANSACTION_CATEGORY_FEE_EXPENSE:
        return "fee"
    if category == TRANSACTION_CATEGORY_INCOME:
        return "income"
    return category


def _workbook_dataset_field(row: Mapping[str, object]) -> str:
    """Return a compact dataset.field label for workbook rows."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if dataset and source_column:
        return f"{dataset}.{source_column}"
    if dataset:
        return dataset
    return source_column


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
    if dataset == pc_cols.HOLDINGS:
        return (
            "This holding changed. The beginning or ending portfolio value row "
            "shows the counted effect."
        )
    if dataset == pc_cols.CASH:
        return (
            "Configured cash impact method is present, but this row still cannot "
            "be estimated. Review return denominator and cash input values."
        )
    return (
        "Configured YAML impact method is present, but this row still cannot be "
        "estimated. Review the inputs required by that method."
    )


def _workbook_missing_impact_method_action(dataset: str, source_column: str) -> str:
    """Return action text for source rows with no additive impact method."""
    if dataset == pc_cols.TRANSACTIONS:
        return _workbook_add_method_action("transaction", source_column)
    if dataset == pc_cols.HOLDINGS:
        return _workbook_add_method_action("holding", source_column)
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
            _AS_OF_DATE: pl.Date,
            _USE: pl.String,
            _CHANGE_LABEL: pl.String,
            _pc_findings.SECURITY_ID: pl.String,
            _pc_findings.SNAPSHOT_A_VALUE: pl.String,
            _pc_findings.SNAPSHOT_B_VALUE: pl.String,
            _CHANGE: pl.Float64,
            _pc_findings.IMPACT_INPUT_VALUE: pl.Float64,
            _ESTIMATED_IMPACT: pl.Float64,
            _INPUT_ROLE: pl.String,
            _IMPACT_STATUS: pl.String,
            _REVIEW_NOTE: pl.String,
            _REVIEW_GUIDANCE: pl.String,
            _pc_findings.DATASET: pl.String,
            _pc_findings.SOURCE_COLUMN: pl.String,
            _pc_findings.FINDING_CODE: pl.String,
            _pc_explain.REVIEW_RANK: pl.Int64,
            _USE_PRIORITY: pl.Int64,
            _REVIEW_KEY: pl.String,
        }
    )


def _workbook_portfolio_changes_columns() -> tuple[str, ...]:
    """Return portfolio-level Performance Differences worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _PERFORMANCE_CHANGE,
        _ESTIMATED_CAUSE_TOTAL,
        _UNEXPLAINED_CHANGE,
        _REVIEW_STATUS,
        _REVIEW_NOTE,
        _REVIEW_KEY,
    )


def _workbook_security_changes_columns() -> tuple[str, ...]:
    """Return security-level Performance Differences worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.SECURITY_ID,
        _PERFORMANCE_CHANGE,
        _ESTIMATED_CAUSE_TOTAL,
        _UNEXPLAINED_CHANGE,
        _REVIEW_STATUS,
        _REVIEW_NOTE,
        _REVIEW_KEY,
    )


def _workbook_underlying_cause_columns() -> tuple[str, ...]:
    """Return Performance Difference Causes worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _AS_OF_DATE,
        _DATASET_FIELD,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _CHANGE,
        _ESTIMATED_IMPACT,
        _REVIEW_GUIDANCE,
        _REVIEW_KEY,
    )


def _workbook_non_additive_change_columns() -> tuple[str, ...]:
    """Return non-additive reported-performance and context worksheet columns."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _AS_OF_DATE,
        _DATASET_FIELD,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _CHANGE,
        _REVIEW_NOTE,
        _REVIEW_KEY,
    )


def _workbook_return_reconstruction_columns() -> tuple[str, ...]:
    """Return Return Reconstruction Checks worksheet columns."""
    return (
        _pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID,
        _pc_reconstruction.RECONSTRUCTION_FROM_DATE,
        _pc_reconstruction.RECONSTRUCTION_THRU_DATE,
        _pc_reconstruction.REPORTED_RETURN_A,
        _pc_reconstruction.REPORTED_RETURN_B,
        _pc_reconstruction.REPORTED_RETURN_DIFFERENCE,
        _pc_reconstruction.DERIVED_RETURN_A,
        _pc_reconstruction.DERIVED_RETURN_B,
        _pc_reconstruction.DERIVED_RETURN_DIFFERENCE,
        _pc_reconstruction.RECONSTRUCTION_DIFFERENCE,
        _pc_reconstruction.RECONSTRUCTION_STATUS,
        _pc_reconstruction.RECONSTRUCTION_CATEGORY,
        _pc_reconstruction.RECONSTRUCTION_COMMENTS,
        _pc_reconstruction.DERIVED_NUMERATOR_A,
        _pc_reconstruction.DERIVED_NUMERATOR_B,
        _pc_reconstruction.DERIVED_NUMERATOR_DIFFERENCE,
        _pc_reconstruction.DERIVED_DENOMINATOR_A,
        _pc_reconstruction.DERIVED_DENOMINATOR_B,
        _pc_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE,
        _pc_reconstruction.BEGIN_VALUE_A,
        _pc_reconstruction.BEGIN_VALUE_B,
        _pc_reconstruction.BEGIN_VALUE_DIFFERENCE,
        _pc_reconstruction.END_VALUE_A,
        _pc_reconstruction.END_VALUE_B,
        _pc_reconstruction.END_VALUE_DIFFERENCE,
        _pc_reconstruction.NET_FLOW_A,
        _pc_reconstruction.NET_FLOW_B,
        _pc_reconstruction.NET_FLOW_DIFFERENCE,
        _pc_reconstruction.WEIGHTED_FLOW_A,
        _pc_reconstruction.WEIGHTED_FLOW_B,
        _pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE,
        _pc_reconstruction.BEGIN_VALUE_DATE_A,
        _pc_reconstruction.BEGIN_VALUE_DATE_B,
        _pc_reconstruction.END_VALUE_DATE_A,
        _pc_reconstruction.END_VALUE_DATE_B,
        _pc_reconstruction.RECONSTRUCTION_REVIEW_KEY,
    )


def _workbook_security_return_reconstruction_columns() -> tuple[str, ...]:
    """Return Security Return Reconstruction Checks worksheet columns."""
    return (
        _pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID,
        _pc_reconstruction.RECONSTRUCTION_SECURITY_ID,
        _pc_reconstruction.RECONSTRUCTION_FROM_DATE,
        _pc_reconstruction.RECONSTRUCTION_THRU_DATE,
        _pc_reconstruction.REPORTED_RETURN_A,
        _pc_reconstruction.REPORTED_RETURN_B,
        _pc_reconstruction.REPORTED_RETURN_DIFFERENCE,
        _pc_reconstruction.DERIVED_RETURN_A,
        _pc_reconstruction.DERIVED_RETURN_B,
        _pc_reconstruction.DERIVED_RETURN_DIFFERENCE,
        _pc_reconstruction.RECONSTRUCTION_DIFFERENCE,
        _pc_reconstruction.RECONSTRUCTION_STATUS,
        _pc_reconstruction.RECONSTRUCTION_CATEGORY,
        _pc_reconstruction.RECONSTRUCTION_COMMENTS,
        _pc_reconstruction.DERIVED_NUMERATOR_A,
        _pc_reconstruction.DERIVED_NUMERATOR_B,
        _pc_reconstruction.DERIVED_NUMERATOR_DIFFERENCE,
        _pc_reconstruction.DERIVED_DENOMINATOR_A,
        _pc_reconstruction.DERIVED_DENOMINATOR_B,
        _pc_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE,
        _pc_reconstruction.BEGIN_VALUE_A,
        _pc_reconstruction.BEGIN_VALUE_B,
        _pc_reconstruction.BEGIN_VALUE_DIFFERENCE,
        _pc_reconstruction.END_VALUE_A,
        _pc_reconstruction.END_VALUE_B,
        _pc_reconstruction.END_VALUE_DIFFERENCE,
        _pc_reconstruction.NET_FLOW_A,
        _pc_reconstruction.NET_FLOW_B,
        _pc_reconstruction.NET_FLOW_DIFFERENCE,
        _pc_reconstruction.WEIGHTED_FLOW_A,
        _pc_reconstruction.WEIGHTED_FLOW_B,
        _pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE,
        _pc_reconstruction.INCOME_A,
        _pc_reconstruction.INCOME_B,
        _pc_reconstruction.INCOME_DIFFERENCE,
        _pc_reconstruction.BEGIN_VALUE_DATE_A,
        _pc_reconstruction.BEGIN_VALUE_DATE_B,
        _pc_reconstruction.END_VALUE_DATE_A,
        _pc_reconstruction.END_VALUE_DATE_B,
        _pc_reconstruction.RECONSTRUCTION_REVIEW_KEY,
    )


def _workbook_return_reconstruction_summary_columns() -> tuple[str, ...]:
    """Return Reconstruction Summary worksheet columns."""
    return (
        _pc_reconstruction.RECONSTRUCTION_CHECK_TYPE,
        _pc_reconstruction.RECONSTRUCTION_STATUS,
        _pc_reconstruction.RECONSTRUCTION_CATEGORY,
        _pc_reconstruction.RECONSTRUCTION_ROW_COUNT,
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
        _pc_findings.TRANSACTION_CATEGORY,
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
        _AS_OF_DATE: "As Of Date",
        _USE: "Purpose",
        _CHANGE_LABEL: "What Changed",
        _DATASET_FIELD: "Dataset Field",
        _CHANGE: "B - A Difference",
        _ESTIMATED_IMPACT: "Performance Difference Explained",
        _IMPACT_STATUS: "Impact Status",
        _REVIEW_NOTE: "Explanation",
        _REVIEW_GUIDANCE: "Explanation",
        _pc_explain.PORTFOLIO_RETURN_DELTA: "Return Delta",
        _REVIEW_STATUS: "Status",
        _REVIEW_CUES: "Review Cues",
        _SUGGESTED_NEXT_STEP: "Suggested Next Step",
        _REVIEW_DETAIL_ARTIFACTS: "Review Detail Artifacts",
        _CONTEXT_USE: "Context Use",
        _REVIEW_PRIORITY: "Review Priority",
        _REVIEW_PRIORITY_REASON: "Review Priority Reason",
        _RETURN_IMPACT_TREATMENT: "Return Impact Treatment",
        _pc_findings.FINDING_CODE: "Code",
        _pc_findings.DATASET: "Source Dataset",
        _pc_findings.SOURCE_COLUMN: "Input Field",
        _pc_findings.MESSAGE: "Message",
        _pc_findings.SEVERITY: "Severity",
        _pc_findings.CONFIDENCE: "Confidence",
        _pc_findings.EVIDENCE_ROLE: "Evidence Role",
        _pc_findings.SOURCE_FILE: "Source File",
        _pc_findings.TRANSACTION_CATEGORY: "Transaction Category",
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
        _pc_reconstruction.RECONSTRUCTION_REVIEW_KEY: "Review Key",
        _pc_reconstruction.REPORTED_RETURN_A: "Reported Return A",
        _pc_reconstruction.REPORTED_RETURN_B: "Reported Return B",
        _pc_reconstruction.REPORTED_RETURN_DIFFERENCE: "Reported Difference",
        _pc_reconstruction.DERIVED_RETURN_A: "Derived Return A",
        _pc_reconstruction.DERIVED_RETURN_B: "Derived Return B",
        _pc_reconstruction.DERIVED_RETURN_DIFFERENCE: "Derived Difference",
        _pc_reconstruction.RECONSTRUCTION_DIFFERENCE: (
            "Reconstruction Difference"
        ),
        _pc_reconstruction.DERIVED_NUMERATOR_A: "Derived Numerator A",
        _pc_reconstruction.DERIVED_NUMERATOR_B: "Derived Numerator B",
        _pc_reconstruction.DERIVED_NUMERATOR_DIFFERENCE: (
            "Derived Numerator Difference"
        ),
        _pc_reconstruction.DERIVED_DENOMINATOR_A: "Derived Denominator A",
        _pc_reconstruction.DERIVED_DENOMINATOR_B: "Derived Denominator B",
        _pc_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE: (
            "Derived Denominator Difference"
        ),
        _pc_reconstruction.BEGIN_VALUE_A: "Beginning Value A",
        _pc_reconstruction.BEGIN_VALUE_B: "Beginning Value B",
        _pc_reconstruction.BEGIN_VALUE_DIFFERENCE: "Beginning Value Difference",
        _pc_reconstruction.END_VALUE_A: "Ending Value A",
        _pc_reconstruction.END_VALUE_B: "Ending Value B",
        _pc_reconstruction.END_VALUE_DIFFERENCE: "Ending Value Difference",
        _pc_reconstruction.NET_FLOW_A: "Net Flow A",
        _pc_reconstruction.NET_FLOW_B: "Net Flow B",
        _pc_reconstruction.NET_FLOW_DIFFERENCE: "Net Flow Difference",
        _pc_reconstruction.WEIGHTED_FLOW_A: "Weighted Flow A",
        _pc_reconstruction.WEIGHTED_FLOW_B: "Weighted Flow B",
        _pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE: (
            "Weighted Flow Difference"
        ),
        _pc_reconstruction.INCOME_A: "Income A",
        _pc_reconstruction.INCOME_B: "Income B",
        _pc_reconstruction.INCOME_DIFFERENCE: "Income Difference",
        _pc_reconstruction.BEGIN_VALUE_DATE_A: "Beginning Value Date A",
        _pc_reconstruction.BEGIN_VALUE_DATE_B: "Beginning Value Date B",
        _pc_reconstruction.END_VALUE_DATE_A: "Ending Value Date A",
        _pc_reconstruction.END_VALUE_DATE_B: "Ending Value Date B",
        _pc_reconstruction.RECONSTRUCTION_STATUS: "Status",
        _pc_reconstruction.RECONSTRUCTION_CATEGORY: "Diagnostic Category",
        _pc_reconstruction.RECONSTRUCTION_COMMENTS: "Comments",
        _pc_reconstruction.RECONSTRUCTION_CHECK_TYPE: "Check Type",
        _pc_reconstruction.RECONSTRUCTION_ROW_COUNT: "Row Count",
    }


def workbook_column_tooltip(column: str) -> str:
    """Return explanatory header text for a workbook/report column.

    Args:
        column: Internal workbook-table column name.

    Returns:
        Reviewer-facing explanation suitable for XLSX comments and HTML header
        tooltips.
    """
    tooltips = {
        _REVIEW_KEY: (
            "Stable performance-period key used to connect workbook rows."
        ),
        _pc_findings.PORTFOLIO_ID: "Portfolio identifier from the compared source-data.",
        _pc_findings.FROM_DATE: "Beginning date of the affected performance period.",
        _pc_findings.THRU_DATE: "Ending date of the affected performance period.",
        _pc_findings.SECURITY_ID: "Security identifier, when the discrepancy is security-level.",
        _pc_findings.SEVERITY: "Materiality/severity assigned to this discrepancy.",
        _PERFORMANCE_CHANGE: (
            "Snapshot B reported performance minus snapshot A reported performance."
        ),
        _ESTIMATED_CAUSE_TOTAL: (
            'Total performance difference explained by "Performance Difference Causes" sheet rows.'
        ),
        _UNEXPLAINED_CHANGE: "Performance difference less explained difference.",
        _USE: "Workbook row category used for sorting and compatibility.",
        _CHANGE_LABEL: "Plain-English changed data item.",
        _DATASET_FIELD: "Changed input field, shown as dataset.field.",
        _CHANGE: "Snapshot B value minus snapshot A value for the compared item.",
        _AS_OF_DATE: (
            "Date represented by the input row. Holding rows use the period Thru Date."
        ),
        _ESTIMATED_IMPACT: (
            "Decimal performance difference explained by this underlying "
            "input row."
        ),
        _IMPACT_STATUS: (
            "Whether this row has an additive estimate, is missing an impact method, "
            "or is review-only."
        ),
        _REVIEW_NOTE: "Plain-language explanation for this changed item.",
        _REVIEW_GUIDANCE: (
            "Plain-language explanation of what changed and how this row relates "
            "to the performance difference."
        ),
        _pc_explain.PORTFOLIO_RETURN_DELTA: (
            "Snapshot B reported performance minus snapshot A reported performance."
        ),
        _REVIEW_STATUS: "Reviewer triage status for this performance difference.",
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
