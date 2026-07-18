"""Build review workbook tables for performance comparison findings."""

from __future__ import annotations

# Python imports
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit import conservation as _pc_conservation
from ppar.audit import executive_summary as _executive_summary
from ppar.audit import field_roles as _field_roles
from ppar.audit.performance_comparison import explain as _pc_explain
from ppar.audit.performance_comparison import findings as _pc_findings
from ppar.audit import lineage as _pc_lineage
from ppar.audit import rendering as _pc_rendering
from ppar.audit import review_keys as _pc_review_keys
from ppar.audit import review_model as _pc_review_model
from ppar.audit import workbook as _pc_workbook
from ppar.audit import workbook_formula_rows as _formula_rows
from ppar.audit import workbook_guidance as _guidance
from ppar.audit import workbook_layout as _layout
from ppar.audit import workbook_reconstruction as _workbook_reconstruction
from ppar.audit import workbook_rows as _rows
from ppar.audit import workbook_source_allocation as _source_allocation
from ppar.audit.data_issues import checks as _data_issue_checks
from ppar.audit.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.audit.transactions import (
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_SELL,
)

__all__ = [
    "audit_review_workbook_sheets",
    "write_audit_review_workbook",
]

_INPUT_ROLE_PERFORMANCE_INPUT = "Performance Input"
_INPUT_ROLE_INPUT_DRIVER = "Input Driver"
_INPUT_ROLE_SUPPORTING_EVIDENCE = "Supporting Evidence"
_INPUT_ROLE_CONTEXT = "Context"
_INPUT_ROLE_DIAGNOSTIC = "Diagnostic"
_ROW_TYPE_EXPLAINED_CAUSE = "Explained Cause"
_ROW_TYPE_POSSIBLE_CAUSE = "Possible Cause"
_ROW_TYPE_SUPPORTING_EVIDENCE = "Supporting Evidence"
_ROW_TYPE_REVIEW_CONTEXT = "Review Context"
_STATUS_FULLY_EXPLAINED = "Fully Explained"
_STATUS_NEEDS_SETUP = "Missing YAML Specifications"
_STATUS_PARTLY_EXPLAINED = "Partly Explained"
_STATUS_UNEXPLAINED = "Unexplained"
_WORKBOOK_CHANGED_ITEM_IDENTITY_COLUMNS = (
    _pc_findings.PORTFOLIO_ID,
    _pc_findings.SOURCE_RECORD_LOCATOR,
    _layout.REVIEW_KEY,
)
_POSSIBLE_CAUSE_COMMENT = "_possible_cause_comment"
_WORKBOOK_UNEXPLAINED_TOLERANCE = 0.0000005
_WORKBOOK_PROMOTABLE_EVIDENCE_COLUMNS = {
    pc_cols.FX_RATES: {pc_cols.FX_RATE},
    pc_cols.HOLDINGS: {
        pc_cols.ACCRUED,
        pc_cols.BASE_ACCRUED,
        pc_cols.MARKET_VALUE,
        pc_cols.BASE_MARKET_VALUE,
        pc_cols.QUANTITY,
    },
    pc_cols.SPLITS: {pc_cols.SPLIT_FACTOR},
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



class _WorkbookTableCache:
    """Cache derived workbook tables for one workbook/report build."""

    def __init__(self, findings: pl.DataFrame) -> None:
        self._findings = findings
        self._contribution_candidates: dict[str, pl.DataFrame] = {}
        self._cause_summary: dict[str, pl.DataFrame] = {}
        self._primary_coverage: dict[str, pl.DataFrame] = {}
        self._portfolio_period_summary: pl.DataFrame | None = None
        self._top_evidence: dict[tuple[str, int], pl.DataFrame] = {}
        self._ranked_rows: dict[str, list[dict[str, object]]] = {}
        self._selected_impact_basis_keys: dict[str, set[tuple[object, ...]]] = {}
        self._performance_input_family_keys: dict[str, set[tuple[object, ...]]] = {}
        self._reconstruction_formula_rows: dict[
            str,
            list[dict[str, object]],
        ] = {}
        self._active_portfolio_keys: set[tuple[object, object, object]] | None = None
        self._active_security_keys: set[tuple[object, object, object, object]] | None = None

    def contribution_candidates(self, comparison_level: str) -> pl.DataFrame:
        """Return cached contribution candidates for one comparison level."""
        if comparison_level not in self._contribution_candidates:
            candidates = (
                _pc_explain.security_period_contribution_candidates(self._findings)
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _pc_explain.portfolio_period_contribution_candidates(self._findings)
            )
            self._contribution_candidates[comparison_level] = candidates
        return self._contribution_candidates[comparison_level]

    def cause_summary(self, comparison_level: str) -> pl.DataFrame:
        """Return cached cause summary rows for the configured comparison level."""
        if comparison_level not in self._cause_summary:
            candidates = self.contribution_candidates(comparison_level)
            self._cause_summary[comparison_level] = (
                _pc_explain.security_period_cause_summary(
                    self._findings,
                    _candidates=candidates,
                )
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _pc_explain.portfolio_period_cause_summary(
                    self._findings,
                    _candidates=candidates,
                )
            )
        return self._cause_summary[comparison_level]

    def primary_coverage(self, comparison_level: str) -> pl.DataFrame:
        """Return cached primary coverage rows for the configured comparison level."""
        if comparison_level not in self._primary_coverage:
            candidates = self.contribution_candidates(comparison_level)
            self._primary_coverage[comparison_level] = (
                _pc_explain.security_period_summary(self._findings)
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _pc_explain.portfolio_period_impact_coverage_summary(
                    self._findings,
                    _candidates=candidates,
                    _periods=self.portfolio_period_summary(),
                )
            )
        return self._primary_coverage[comparison_level]

    def portfolio_period_summary(self) -> pl.DataFrame:
        """Return cached portfolio-period summary rows."""
        if self._portfolio_period_summary is None:
            self._portfolio_period_summary = _pc_explain.portfolio_period_summary(
                self._findings
            )
        return self._portfolio_period_summary

    def top_evidence(
        self,
        comparison_level: str,
        top_evidence_limit: int | None = None,
    ) -> pl.DataFrame:
        """Return cached top-evidence rows for the configured comparison level."""
        limit = self._findings.height if top_evidence_limit is None else top_evidence_limit
        key = comparison_level, limit
        if key not in self._top_evidence:
            candidates = self.contribution_candidates(comparison_level)
            self._top_evidence[key] = (
                _pc_explain.security_top_evidence_table(
                    self._findings,
                    limit,
                    _candidates=candidates,
                )
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _pc_explain.top_evidence_table(
                    self._findings,
                    limit,
                    _candidates=candidates,
                )
            )
        return self._top_evidence[key]

    def ranked_rows(self, comparison_level: str) -> list[dict[str, object]]:
        """Return cached ranked workbook evidence rows."""
        if comparison_level not in self._ranked_rows:
            self._ranked_rows[comparison_level] = _workbook_ranked_changed_rows_for_level(
                self._findings,
                comparison_level=comparison_level,
                table_cache=self,
            )
        return self._ranked_rows[comparison_level]

    def selected_impact_basis_keys(self, comparison_level: str) -> set[tuple[object, ...]]:
        """Return cached period/impact-basis keys included in explained totals."""
        if comparison_level not in self._selected_impact_basis_keys:
            self._selected_impact_basis_keys[comparison_level] = (
                _workbook_selected_impact_basis_keys(
                    self._findings,
                    comparison_level=comparison_level,
                    table_cache=self,
                )
            )
        return self._selected_impact_basis_keys[comparison_level]

    def performance_input_family_keys(
        self,
        comparison_level: str,
    ) -> set[tuple[object, ...]]:
        """Return cached cause-family keys with selected performance input rows."""
        if comparison_level not in self._performance_input_family_keys:
            self._performance_input_family_keys[comparison_level] = (
                _workbook_performance_input_family_keys(
                    self._findings,
                    comparison_level=comparison_level,
                    table_cache=self,
                )
            )
        return self._performance_input_family_keys[comparison_level]

    def active_portfolio_keys(self) -> set[tuple[object, object, object]]:
        """Return cached portfolio-period keys with reported performance changes."""
        if self._active_portfolio_keys is None:
            self._active_portfolio_keys = _workbook_active_portfolio_period_keys(
                self._findings,
                table_cache=self,
            )
        return self._active_portfolio_keys

    def active_security_keys(self) -> set[tuple[object, object, object, object]]:
        """Return cached security-period keys with reported performance changes."""
        if self._active_security_keys is None:
            self._active_security_keys = _workbook_active_security_period_keys(
                self._findings,
                table_cache=self,
            )
        return self._active_security_keys

    def reconstruction_formula_rows(
        self,
        comparison_level: str,
        *,
        comparison_path: util.PathLike | None,
        reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
    ) -> list[dict[str, object]]:
        """Return cached Modified Dietz formula rows for one review level."""
        if comparison_level not in self._reconstruction_formula_rows:
            rows = (
                _formula_rows.security_reconstruction_formula_rows(
                    comparison_path,
                    active_keys=self.active_security_keys(),
                    reconstruction_cache=reconstruction_cache,
                )
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _formula_rows.portfolio_reconstruction_formula_rows(
                    comparison_path,
                    active_keys=self.active_portfolio_keys(),
                    reconstruction_cache=reconstruction_cache,
                )
            )
            self._reconstruction_formula_rows[comparison_level] = rows
        return self._reconstruction_formula_rows[comparison_level]



def write_audit_review_workbook(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    top_evidence_limit: int = 10,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
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
    del top_evidence_limit
    return _pc_workbook.write_review_workbook_sheets(
        audit_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=_reconstruction_cache,
        ),
        output_path,
        column_tooltip=_layout.workbook_column_tooltip,
    )


def audit_review_workbook_sheets(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
    _table_cache: _WorkbookTableCache | None = None,
    _data_issues: pl.DataFrame | None = None,
    _finding_audit_trail: pl.DataFrame | None = None,
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
    reconstruction_cache = (
        _reconstruction_cache
        or _workbook_reconstruction.WorkbookReconstructionCache(comparison_path)
    )
    table_cache = _table_cache or _WorkbookTableCache(active_findings)
    primary_sheet = (
        _security_differences_sheet(
            active_findings,
            comparison_path=comparison_path,
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        )
        if comparison_level == SECURITY_COMPARISON_LEVEL
        else _portfolio_differences_sheet(
            active_findings,
            comparison_path=comparison_path,
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        )
    )
    diagnostic_sheets = (
        (
            *_return_reconstruction_summary_sheets(reconstruction_cache),
            *_return_reconstruction_sheets(reconstruction_cache),
            *_security_return_reconstruction_sheets(reconstruction_cache),
        )
        if include_reconstruction_diagnostics
        else ()
    )
    detail_sheets = _shared_detail_sheets(
        findings,
        active_findings,
        primary_changes_table=primary_sheet.table,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        table_cache=table_cache,
        reconstruction_cache=reconstruction_cache,
        data_issues=_data_issues,
        finding_audit_trail=_finding_audit_trail,
    )
    primary_sheet = replace(
        primary_sheet,
        table=_workbook_reconcile_displayed_primary_values(primary_sheet.table),
    )
    data_issues_sheet = next(
        sheet
        for sheet in detail_sheets
        if sheet.artifact_name == _pc_review_model.DATA_ISSUES_ARTIFACT
    )
    executive_summary_sheet = _pc_workbook.ReviewWorkbookSheet(
        artifact_name=_pc_review_model.EXECUTIVE_SUMMARY_ARTIFACT,
        sheet_name=_pc_review_model.EXECUTIVE_SUMMARY_SHEET,
        table=_executive_summary.executive_summary_table(
            primary_sheet.table,
            data_issues_sheet.table,
            context=_executive_summary.executive_summary_context(
                comparison_path,
                comparison_level,
            ),
        ),
        columns=_executive_summary.EXECUTIVE_SUMMARY_COLUMNS,
        labels=_layout.workbook_column_labels(),
    )
    return (
        executive_summary_sheet,
        primary_sheet,
        *detail_sheets,
        *diagnostic_sheets,
    )


def _return_reconstruction_summary_sheets(
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional return-reconstruction diagnostic summary sheets."""
    summary = reconstruction_cache.summary()
    if summary.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT,
            sheet_name=_pc_review_model.RECONSTRUCTION_SUMMARY_SHEET,
            table=summary,
            columns=_layout.workbook_return_reconstruction_summary_columns(),
            labels=_layout.workbook_column_labels(),
        ),
    )


def _return_reconstruction_sheets(
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional portfolio return-reconstruction diagnostic sheets."""
    reconstruction_checks = reconstruction_cache.portfolio_checks()
    if reconstruction_checks.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
            sheet_name=_pc_review_model.RETURN_RECONSTRUCTION_CHECKS_SHEET,
            table=reconstruction_checks,
            columns=_layout.workbook_return_reconstruction_columns(),
            labels=_layout.workbook_column_labels(),
        ),
    )


def _security_return_reconstruction_sheets(
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return optional security return-reconstruction diagnostic sheets."""
    reconstruction_checks = reconstruction_cache.security_checks()
    if reconstruction_checks.is_empty():
        return ()
    return (
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=(_pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT),
            sheet_name=_pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_SHEET,
            table=reconstruction_checks,
            columns=_layout.workbook_security_return_reconstruction_columns(),
            labels=_layout.workbook_column_labels(),
        ),
    )


def _portfolio_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    table_cache: _WorkbookTableCache,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
) -> _pc_workbook.ReviewWorkbookSheet:
    """Return the portfolio-level performance differences sheet."""
    labels = _layout.workbook_column_labels()
    labels[_layout.REVIEW_NOTE] = "Comments"
    return _pc_workbook.ReviewWorkbookSheet(
        artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
        sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
        table=_workbook_portfolio_changes_table(
            active_findings,
            comparison_path=comparison_path,
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        ),
        columns=_layout.workbook_portfolio_changes_columns(),
        labels=labels,
    )


def _security_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    table_cache: _WorkbookTableCache,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
) -> _pc_workbook.ReviewWorkbookSheet:
    """Return the security-level performance differences sheet."""
    labels = _layout.workbook_column_labels()
    labels[_layout.REVIEW_NOTE] = "Comments"
    return _pc_workbook.ReviewWorkbookSheet(
        artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
        sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
        table=_workbook_security_changes_table(
            active_findings,
            comparison_path=comparison_path,
            comparison_level=SECURITY_COMPARISON_LEVEL,
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        ),
        columns=_layout.workbook_security_changes_columns(),
        labels=labels,
    )


def _shared_detail_sheets(
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    *,
    primary_changes_table: pl.DataFrame,
    comparison_path: util.PathLike | None,
    comparison_level: str,
    table_cache: _WorkbookTableCache,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache,
    data_issues: pl.DataFrame | None,
    finding_audit_trail: pl.DataFrame | None,
) -> tuple[_pc_workbook.ReviewWorkbookSheet, ...]:
    """Return detail sheets shared by portfolio and security workflows."""
    causes_table = _workbook_underlying_causes_table(
        active_findings,
        lineage_findings=findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        primary_changes_table=primary_changes_table,
        table_cache=table_cache,
        reconstruction_cache=reconstruction_cache,
        finding_audit_trail=finding_audit_trail,
    )
    formula_rows = table_cache.reconstruction_formula_rows(
        comparison_level,
        comparison_path=comparison_path,
        reconstruction_cache=reconstruction_cache,
    )
    _assert_explanation_invariants(
        primary_changes_table,
        causes_table,
        formula_rows,
        comparison_level=comparison_level,
    )
    causes_table = _workbook_reconcile_displayed_explained_values(
        primary_changes_table,
        causes_table,
        comparison_level=comparison_level,
    )
    _assert_displayed_explanation_reconciliation(
        primary_changes_table,
        causes_table,
        comparison_level=comparison_level,
    )
    detail_sheets = [
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
            sheet_name=_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
            table=causes_table,
            columns=_layout.workbook_underlying_cause_columns(),
            labels=_layout.workbook_column_labels(),
        ),
        _pc_workbook.ReviewWorkbookSheet(
            artifact_name=_pc_review_model.DATA_ISSUES_ARTIFACT,
            sheet_name=_pc_review_model.DATA_ISSUES_SHEET,
            table=(
                _data_issue_checks.data_issues_table(comparison_path)
                if data_issues is None
                else data_issues
            ),
            columns=_data_issue_checks.DATA_ISSUE_COLUMNS,
            labels=_layout.workbook_column_labels(),
        ),
    ]
    return tuple(detail_sheets)


def _assert_portfolio_explanation_invariants(
    primary_changes: pl.DataFrame,
    causes: pl.DataFrame,
    formula_rows: Sequence[Mapping[str, object]],
) -> None:
    """Raise when portfolio explanation arithmetic or formula coverage diverges."""
    _assert_explanation_invariants(
        primary_changes,
        causes,
        formula_rows,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    )


def _assert_explanation_invariants(
    primary_changes: pl.DataFrame,
    causes: pl.DataFrame,
    formula_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> None:
    """Raise when explanation arithmetic or formula coverage diverges.

    Raises:
        PpaError: If cause impacts do not reconcile to the summary, a fully
            explained period does not reconcile to its performance difference,
            or a Modified Dietz formula component is absent from the causes.
    """
    cause_totals = _workbook_cause_impact_totals(
        causes,
        displayed=False,
        comparison_level=comparison_level,
    )
    for row in primary_changes.iter_rows(named=True):
        key = _rows.primary_review_period_key(row, comparison_level)
        if not _workbook_is_real_primary_key(key, comparison_level):
            continue
        explained = _rows.number_or_none(row.get(_layout.ESTIMATED_CAUSE_TOTAL)) or 0.0
        cause_total = cause_totals.get(key, 0.0)
        if abs(explained - cause_total) > 0.000000000001:
            raise PpaError(
                "SN-03 explanation invariant failed for "
                f"{_workbook_primary_key_text(key)}: causes total {cause_total:.12f} "
                f"does not equal Explained Difference {explained:.12f}.",
                999,
            )
        if row.get(_layout.REVIEW_STATUS) != _STATUS_FULLY_EXPLAINED:
            continue
        performance_difference = _rows.number_or_none(row.get(_layout.PERFORMANCE_CHANGE)) or 0.0
        if abs(performance_difference - explained) > _WORKBOOK_UNEXPLAINED_TOLERANCE:
            raise PpaError(
                "SN-03 Fully Explained invariant failed for "
                f"{_workbook_primary_key_text(key)}: Performance Difference "
                f"{performance_difference:.12f} does not equal Explained Difference "
                f"{explained:.12f}.",
                999,
            )
        unexplained = _rows.number_or_none(row.get(_layout.UNEXPLAINED_CHANGE))
        if unexplained is not None and abs(unexplained) > _WORKBOOK_UNEXPLAINED_TOLERANCE:
            raise PpaError(
                "SN-03 Fully Explained invariant failed for "
                f"{_workbook_primary_key_text(key)}: Unexplained Difference "
                f"{unexplained:.12f} is not zero.",
                999,
            )

    expected_components = {
        (
            *_rows.primary_review_period_key(row, comparison_level),
            _format_value(row.get(_pc_findings.SOURCE_COLUMN)),
        )
        for row in formula_rows
        if _rows.number_or_none(row.get(_layout.ESTIMATED_IMPACT)) is not None
    }
    observed_components: set[tuple[object, ...]] = set()
    for row in causes.iter_rows(named=True):
        components = _format_value(row.get(_formula_rows.RECONSTRUCTION_COMPONENTS))
        observed_components.update(
            (*_rows.primary_review_period_key(row, comparison_level), component)
            for component in components.split("|")
            if component
        )
    missing_components = sorted(
        expected_components - observed_components,
        key=lambda item: tuple(str(value) for value in item),
    )
    if missing_components:
        missing = missing_components[0]
        key_length = 4 if comparison_level == SECURITY_COMPARISON_LEVEL else 3
        raise PpaError(
            "Modified Dietz evidence invariant failed for "
            f"{_workbook_primary_key_text(missing[:key_length])}: formula component "
            f"{missing[key_length]!r} is absent from Performance Difference Causes.",
            999,
        )


def _workbook_reconcile_displayed_explained_values(
    primary_changes: pl.DataFrame,
    causes: pl.DataFrame,
    *,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> pl.DataFrame:
    """Allocate six-decimal presentation residuals to a counted cause row."""
    if causes.is_empty():
        return causes
    target_by_key = {
        _rows.primary_review_period_key(row, comparison_level): (
            _workbook_displayed_explained_target(row)
        )
        for row in primary_changes.iter_rows(named=True)
        if _workbook_is_real_primary_key(
            _rows.primary_review_period_key(row, comparison_level),
            comparison_level,
        )
    }
    rows = causes.to_dicts()
    indexes_by_key: dict[tuple[object, ...], list[int]] = {}
    for index, row in enumerate(rows):
        if _rows.number_or_none(row.get(_layout.ESTIMATED_IMPACT)) is None:
            continue
        indexes_by_key.setdefault(
            _rows.primary_review_period_key(row, comparison_level),
            [],
        ).append(index)
    for key, target in target_by_key.items():
        indexes = indexes_by_key.get(key, [])
        displayed_total = round(
            sum(round(_rows.number_or_none(rows[index].get(_layout.ESTIMATED_IMPACT)) or 0.0, 6)
                for index in indexes),
            6,
        )
        residual = round(target - displayed_total, 6)
        if residual == 0.0 or not indexes:
            continue
        allocation_index = max(
            indexes,
            key=lambda index: abs(
                _rows.number_or_none(rows[index].get(_layout.ESTIMATED_IMPACT)) or 0.0
            ),
        )
        displayed_impact = round(
            _rows.number_or_none(rows[allocation_index].get(_layout.ESTIMATED_IMPACT)) or 0.0,
            6,
        )
        rows[allocation_index][_layout.ESTIMATED_IMPACT] = round(displayed_impact + residual, 6)
    return pl.DataFrame(rows, schema=causes.schema, infer_schema_length=None)


def _workbook_reconcile_displayed_primary_values(
    primary_changes: pl.DataFrame,
) -> pl.DataFrame:
    """Make Fully Explained summary values agree at workbook precision.

    The raw explanation invariant permits a sub-half-micro residual. If the
    performance and explanation values straddle a six-decimal rounding boundary,
    use the authoritative performance value for the displayed explanation. The
    cause table's presentation-only residual allocator independently reconciles
    its visible rows to the same target; raw cause lineage remains unchanged.
    """
    required_columns = {
        _layout.PERFORMANCE_CHANGE,
        _layout.ESTIMATED_CAUSE_TOTAL,
        _layout.REVIEW_STATUS,
    }
    if not required_columns.issubset(primary_changes.columns):
        return primary_changes
    performance = pl.col(_layout.PERFORMANCE_CHANGE)
    explained = pl.col(_layout.ESTIMATED_CAUSE_TOTAL)
    needs_reconciliation = (
        (pl.col(_layout.REVIEW_STATUS) == _STATUS_FULLY_EXPLAINED)
        & performance.is_not_null()
        & explained.is_not_null()
        & (performance.round(6) != explained.round(6))
    )
    return primary_changes.with_columns(
        pl.when(needs_reconciliation)
        .then(performance.round(6))
        .otherwise(explained)
        .alias(_layout.ESTIMATED_CAUSE_TOTAL)
    )


def _workbook_displayed_explained_target(row: Mapping[str, object]) -> float:
    """Return the authoritative six-decimal explanation target for one row."""
    explained = _rows.number_or_none(row.get(_layout.ESTIMATED_CAUSE_TOTAL)) or 0.0
    performance = _rows.number_or_none(row.get(_layout.PERFORMANCE_CHANGE))
    if row.get(_layout.REVIEW_STATUS) == _STATUS_FULLY_EXPLAINED and performance is not None:
        return round(performance, 6)
    return round(explained, 6)


def _assert_displayed_portfolio_explanation_reconciliation(
    primary_changes: pl.DataFrame,
    causes: pl.DataFrame,
) -> None:
    """Raise unless portfolio six-decimal cause totals equal summary totals."""
    _assert_displayed_explanation_reconciliation(
        primary_changes,
        causes,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    )


def _assert_displayed_explanation_reconciliation(
    primary_changes: pl.DataFrame,
    causes: pl.DataFrame,
    *,
    comparison_level: str,
) -> None:
    """Raise unless serialized six-decimal cause totals equal summary totals."""
    cause_totals = _workbook_cause_impact_totals(
        causes,
        displayed=True,
        comparison_level=comparison_level,
    )
    for row in primary_changes.iter_rows(named=True):
        key = _rows.primary_review_period_key(row, comparison_level)
        if not _workbook_is_real_primary_key(key, comparison_level):
            continue
        explained = _workbook_displayed_explained_target(row)
        cause_total = cause_totals.get(key, 0.0)
        if cause_total != explained:
            raise PpaError(
                "SN-03 displayed explanation invariant failed for "
                f"{_workbook_primary_key_text(key)}: workbook causes total "
                f"{cause_total:.6f} does not equal workbook Explained Difference "
                f"{explained:.6f}.",
                999,
            )


def _workbook_cause_impact_totals(
    causes: pl.DataFrame,
    *,
    displayed: bool,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> dict[tuple[object, ...], float]:
    """Return cause impact totals at the configured report grain."""
    totals: dict[tuple[object, ...], float] = {}
    for row in causes.iter_rows(named=True):
        impact = _rows.number_or_none(row.get(_layout.ESTIMATED_IMPACT))
        if impact is None:
            continue
        value = round(impact, 6) if displayed else impact
        key = _rows.primary_review_period_key(row, comparison_level)
        totals[key] = totals.get(key, 0.0) + value
    if displayed:
        return {key: round(value, 6) for key, value in totals.items()}
    return totals


def _workbook_is_real_primary_key(
    key: Sequence[object],
    comparison_level: str,
) -> bool:
    """Return whether a workbook key represents an actual review period."""
    if len(key) < 3 or key[1] is None or key[2] is None:
        return False
    return not (
        comparison_level == SECURITY_COMPARISON_LEVEL
        and (len(key) < 4 or key[3] is None)
    )


def _workbook_primary_key_text(key: Sequence[object]) -> str:
    """Return a concise review-period label for invariant failures."""
    security = f" security {key[3]}" if len(key) > 3 else ""
    return f"{key[0]}{security} {key[1]} through {key[2]}"


def _workbook_portfolio_changes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> pl.DataFrame:
    """Return one workbook row per changed portfolio period."""
    coverage = _with_period_review_key(
        table_cache.primary_coverage(PORTFOLIO_COMPARISON_LEVEL)
        if table_cache is not None
        else _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    if coverage.is_empty():
        return _workbook_empty_portfolio_changes_table()
    ranked_rows = (
        table_cache.ranked_rows(PORTFOLIO_COMPARISON_LEVEL)
        if table_cache is not None
        else _workbook_ranked_changed_rows(findings)
    )
    underlying_totals = _workbook_underlying_impact_totals(
        findings,
        comparison_path=comparison_path,
        ranked_rows=ranked_rows,
        table_cache=table_cache,
        reconstruction_cache=_workbook_reconstruction.resolved_reconstruction_cache(
            comparison_path,
            reconstruction_cache,
        ),
    )
    unresolved_keys = _workbook_unresolved_primary_keys_from_rows(
        coverage.iter_rows(named=True),
        cast(Mapping[tuple[object, ...], float], underlying_totals),
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
    )
    possible_cause_comments = _workbook_possible_cause_comments(
        findings,
        unresolved_keys=unresolved_keys,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
        ranked_rows=ranked_rows,
    )
    rows = [
        _workbook_performance_change_row(
            {
                **row,
                "_underlying_estimated_total": underlying_totals.get(
                    _rows.review_period_key(row),
                    0.0,
                ),
                _POSSIBLE_CAUSE_COMMENT: possible_cause_comments.get(
                    _rows.review_period_key(row),
                    "",
                ),
            }
        )
        for row in coverage.iter_rows(named=True)
    ]
    return _layout.workbook_sorted_table(
        pl.DataFrame(rows, infer_schema_length=None),
        [_layout.REVIEW_KEY],
    )


def _workbook_underlying_impact_totals(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> dict[tuple[object, object, object], float]:
    """Return explained difference totals from underlying input rows."""
    totals: dict[tuple[object, object, object], float] = {}
    active_keys = (
        table_cache.active_portfolio_keys()
        if table_cache is not None
        else _workbook_active_portfolio_period_keys(findings)
    )
    reconstruction_cache = _workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    formula_rows = (
        table_cache.reconstruction_formula_rows(
            PORTFOLIO_COMPARISON_LEVEL,
            comparison_path=comparison_path,
            reconstruction_cache=reconstruction_cache,
        )
        if table_cache is not None
        else _formula_rows.portfolio_reconstruction_formula_rows(
            comparison_path,
            active_keys=active_keys,
            reconstruction_cache=reconstruction_cache,
        )
    )
    formula_keys = {_rows.review_period_key(row) for row in formula_rows}
    for row in formula_rows:
        key = _rows.review_period_key(row)
        estimated_impact = _rows.number_or_none(row.get(_layout.ESTIMATED_IMPACT))
        if estimated_impact is not None:
            totals[key] = totals.get(key, 0.0) + estimated_impact

    for row, estimated_impact in _workbook_selected_underlying_impact_rows(
        findings,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
        ranked_rows=ranked_rows,
    ):
        key = _rows.review_period_key(row)
        if key in formula_keys:
            continue
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


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


def _workbook_performance_change_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Return one plain-English performance-change workbook row."""
    performance_change = _workbook_performance_difference(row)
    estimated_total = _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL))
    underlying_estimated_total = _rows.number_or_none(row.get("_underlying_estimated_total"))
    if underlying_estimated_total is not None:
        estimated_total = underlying_estimated_total
    unexplained_change = None
    if performance_change is not None:
        unexplained_change = performance_change - (estimated_total or 0.0)
    review_status = _workbook_explanation_status(row)
    unexplained_display = None if review_status == _STATUS_FULLY_EXPLAINED else unexplained_change
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _layout.PERFORMANCE_CHANGE: performance_change,
        _layout.ESTIMATED_CAUSE_TOTAL: estimated_total,
        _layout.UNEXPLAINED_CHANGE: unexplained_display,
        _layout.REVIEW_STATUS: review_status,
        _layout.REVIEW_NOTE: _workbook_performance_comments(row),
        _layout.REVIEW_KEY: row.get(_layout.REVIEW_KEY),
    }


def _workbook_performance_difference(row: Mapping[str, object]) -> float | None:
    """Return portfolio or security performance difference for a workbook row."""
    portfolio_difference = _rows.number_or_none(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))
    if portfolio_difference is not None:
        return portfolio_difference
    return _rows.number_or_none(row.get(_pc_explain.SECURITY_RETURN_DELTA))


def _workbook_explanation_status(row: Mapping[str, object]) -> str:
    """Return a plain-language explanation status for a performance difference."""
    underlying_estimated_total = _rows.number_or_none(row.get("_underlying_estimated_total"))
    estimated_total = _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL))
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


def _workbook_performance_comments(
    row: Mapping[str, object],
) -> str:
    """Return plain-language comments for a performance difference."""
    if _workbook_explanation_status(row) == _STATUS_FULLY_EXPLAINED:
        return ""

    possible_cause_comment = _format_value(row.get(_POSSIBLE_CAUSE_COMMENT))
    if possible_cause_comment:
        return possible_cause_comment

    missing_inputs = row.get(_pc_explain.MISSING_IMPACT_INPUTS)
    status = row.get(_pc_explain.IMPACT_COVERAGE_STATUS)
    if _rows.has_text(missing_inputs):
        return f"Missing YAML specifications: {_format_value(missing_inputs)}."
    underlying_estimated_total = _rows.number_or_none(row.get("_underlying_estimated_total"))
    performance_change = _workbook_performance_difference(row)
    if underlying_estimated_total is not None and performance_change is not None:
        residual = performance_change - underlying_estimated_total
        if abs(residual) <= _WORKBOOK_UNEXPLAINED_TOLERANCE:
            return ""
        if abs(underlying_estimated_total) > 0:
            return _workbook_unexplained_review_comment(
                partly=True,
                residual=residual,
                explained=underlying_estimated_total,
            )
        return _workbook_unexplained_review_comment(
            partly=False,
            residual=residual,
        )
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES:
        return ""
    if status == _pc_explain.IMPACT_COVERAGE_STATUS_PARTIAL_ESTIMATES:
        return _workbook_unexplained_review_comment(partly=True)
    return _workbook_unexplained_review_comment(partly=False)


def _workbook_unexplained_review_comment(
    *,
    partly: bool,
    residual: float | None = None,
    explained: float | None = None,
) -> str:
    """Return directed comments for unresolved performance differences."""
    if residual is not None and partly and explained is not None:
        del explained
        return (
            "The remaining Unexplained Difference may be due to missing "
            "source-data, source-file timing differences, or vendor methodology that "
            "does not match the YAML specifications."
        )
    if residual is not None:
        del residual
        return (
            "The Unexplained Difference may be due to missing source-data, "
            "source-file timing differences, or vendor methodology that does "
            "not match the YAML specifications."
        )
    if partly:
        return (
            "The remaining Unexplained Difference may be due to missing "
            "source-data, source-file timing differences, or vendor methodology that "
            "does not match the YAML specifications."
        )
    return (
        "The Unexplained Difference may be due to missing source-data, "
        "source-file timing differences, or vendor methodology that does not "
        "match the YAML specifications."
    )


def _workbook_empty_portfolio_changes_table() -> pl.DataFrame:
    """Return a reviewer-facing performance-difference row for clean comparisons."""
    return pl.DataFrame(
        [
            {
                _pc_findings.PORTFOLIO_ID: "No portfolio performance differences found",
                _pc_findings.FROM_DATE: None,
                _pc_findings.THRU_DATE: None,
                _layout.PERFORMANCE_CHANGE: None,
                _layout.ESTIMATED_CAUSE_TOTAL: None,
                _layout.UNEXPLAINED_CHANGE: None,
                _layout.REVIEW_STATUS: "No differences",
                _layout.REVIEW_NOTE: "No reported portfolio return differences.",
                _layout.REVIEW_KEY: "NO_PORTFOLIO_PERFORMANCE_DIFFERENCES",
            }
        ],
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _layout.PERFORMANCE_CHANGE: pl.Float64,
            _layout.ESTIMATED_CAUSE_TOTAL: pl.Float64,
            _layout.UNEXPLAINED_CHANGE: pl.Float64,
            _layout.REVIEW_STATUS: pl.String,
            _layout.REVIEW_NOTE: pl.String,
            _layout.REVIEW_KEY: pl.String,
        },
    )


def _workbook_security_changes_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> pl.DataFrame:
    """Return one workbook row per changed security period."""
    summary = _with_security_review_key(
        table_cache.primary_coverage(comparison_level)
        if table_cache is not None
        else _pc_explain.security_period_summary(findings)
    )
    ranked_rows = (
        table_cache.ranked_rows(comparison_level)
        if table_cache is not None
        else _workbook_ranked_changed_rows_for_level(
            findings,
            comparison_level=comparison_level,
        )
    )
    security_totals = _workbook_security_underlying_impact_totals(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        ranked_rows=ranked_rows,
        table_cache=table_cache,
        reconstruction_cache=_workbook_reconstruction.resolved_reconstruction_cache(
            comparison_path,
            reconstruction_cache,
        ),
    )
    rows: list[dict[str, object]] = []
    if not summary.is_empty():
        unresolved_keys = _workbook_unresolved_primary_keys_from_rows(
            summary.iter_rows(named=True),
            cast(Mapping[tuple[object, ...], float], security_totals),
            comparison_level=comparison_level,
        )
        possible_cause_comments = _workbook_possible_cause_comments(
            findings,
            unresolved_keys=unresolved_keys,
            comparison_level=comparison_level,
            ranked_rows=ranked_rows,
        )
        rows = [
            _workbook_security_change_row(
                {
                    **row,
                    "_underlying_estimated_total": security_totals.get(
                        _rows.security_review_period_key(row),
                        0.0,
                    ),
                    _POSSIBLE_CAUSE_COMMENT: possible_cause_comments.get(
                        _rows.security_review_period_key(row),
                        "",
                    ),
                }
            )
            for row in summary.iter_rows(named=True)
        ]
    rows.extend(_workbook_missing_security_change_rows(findings, rows))
    if not rows:
        return _workbook_empty_security_changes_table()
    return _layout.workbook_sorted_table(
        pl.DataFrame(rows, infer_schema_length=None),
        [_layout.REVIEW_KEY, _pc_findings.SECURITY_ID],
    )


def _workbook_security_underlying_impact_totals(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> dict[tuple[object, object, object, object], float]:
    """Return security-level explained totals from underlying input rows."""
    totals: dict[tuple[object, object, object, object], float] = {}
    active_keys = (
        table_cache.active_security_keys()
        if table_cache is not None
        else _workbook_active_security_period_keys(findings)
    )
    reconstruction_cache = _workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    formula_rows = (
        table_cache.reconstruction_formula_rows(
            comparison_level,
            comparison_path=comparison_path,
            reconstruction_cache=reconstruction_cache,
        )
        if table_cache is not None
        else _formula_rows.security_reconstruction_formula_rows(
            comparison_path,
            active_keys=active_keys,
            reconstruction_cache=reconstruction_cache,
        )
    )
    formula_keys = {_rows.security_review_period_key(row) for row in formula_rows}
    for row in formula_rows:
        key = _rows.security_review_period_key(row)
        estimated_impact = _rows.number_or_none(row.get(_layout.ESTIMATED_IMPACT))
        if estimated_impact is not None:
            totals[key] = totals.get(key, 0.0) + estimated_impact

    for row, estimated_impact in _workbook_selected_underlying_impact_rows(
        findings,
        comparison_level=comparison_level,
        ranked_rows=ranked_rows,
    ):
        if not _rows.has_text(row.get(_pc_findings.SECURITY_ID)):
            continue
        key = _rows.security_review_period_key(row)
        if key in formula_keys:
            continue
        totals[key] = totals.get(key, 0.0) + estimated_impact
    return totals


def _workbook_unresolved_primary_keys_from_rows(
    rows: Iterable[Mapping[str, object]],
    underlying_totals: Mapping[tuple[object, ...], float],
    *,
    comparison_level: str,
) -> set[tuple[object, ...]]:
    """Return changed performance keys that still have an unexplained residual."""
    unresolved_keys: set[tuple[object, ...]] = set()
    for row in rows:
        key = _rows.primary_review_period_key(row, comparison_level)
        performance_change = _workbook_performance_difference(row)
        if performance_change is None:
            continue
        residual = performance_change - underlying_totals.get(key, 0.0)
        if abs(residual) > _WORKBOOK_UNEXPLAINED_TOLERANCE:
            unresolved_keys.add(key)
    return unresolved_keys


def _workbook_possible_cause_comments(
    findings: pl.DataFrame,
    *,
    unresolved_keys: set[tuple[object, ...]],
    comparison_level: str,
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
) -> dict[tuple[object, ...], str]:
    """Return possible-cause review comments for unresolved performance rows."""
    possible_comments_by_key: dict[tuple[object, ...], list[str]] = {}
    for row in _workbook_possible_cause_rows(
        findings,
        unresolved_keys=unresolved_keys,
        comparison_level=comparison_level,
        ranked_rows=ranked_rows,
    ):
        key = _rows.primary_review_period_key(row, comparison_level)
        comment = _guidance.possible_cause_row_comment(row)
        if not comment:
            continue
        comments = possible_comments_by_key.setdefault(key, [])
        if comment not in comments:
            comments.append(comment)
    return {
        key: _guidance.possible_cause_summary(comments)
        for key, comments in possible_comments_by_key.items()
    }


def _workbook_possible_cause_rows(
    findings: pl.DataFrame,
    *,
    unresolved_keys: set[tuple[object, ...]],
    comparison_level: str,
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return unestimated source rows that may explain unresolved differences."""
    if not unresolved_keys:
        return []
    possible_rows: list[dict[str, object]] = []
    source_rows = ranked_rows or _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=comparison_level,
    )
    for row in source_rows:
        if _rows.primary_review_period_key(row, comparison_level) not in unresolved_keys:
            continue
        if not _workbook_is_possible_cause_row(row):
            continue
        possible_rows.append(dict(row))
    return possible_rows


def _workbook_is_possible_cause_row(row: Mapping[str, object]) -> bool:
    """Return whether an unestimated row is a possible residual cause."""
    if _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is not None:
        return False
    if _rows.has_additive_policy(row):
        return False
    return bool(_guidance.possible_cause_field_name(row))


def _workbook_active_portfolio_period_keys(
    findings: pl.DataFrame,
    *,
    table_cache: _WorkbookTableCache | None = None,
) -> set[tuple[object, object, object]]:
    """Return portfolio-period keys with reported portfolio performance differences."""
    summary = _with_period_review_key(
        table_cache.primary_coverage(PORTFOLIO_COMPARISON_LEVEL)
        if table_cache is not None
        else _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    return {_rows.review_period_key(row) for row in summary.iter_rows(named=True)}


def _workbook_active_security_period_keys(
    findings: pl.DataFrame,
    *,
    table_cache: _WorkbookTableCache | None = None,
) -> set[tuple[object, object, object, object]]:
    """Return security-period keys with reported security performance differences."""
    summary = _with_security_review_key(
        table_cache.primary_coverage(SECURITY_COMPARISON_LEVEL)
        if table_cache is not None
        else _pc_explain.security_period_summary(findings)
    )
    return {_rows.security_review_period_key(row) for row in summary.iter_rows(named=True)}


def _workbook_selected_underlying_impact_rows(
    findings: pl.DataFrame,
    *,
    comparison_level: str,
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[tuple[dict[str, object], float]]:
    """Return additive impact rows selected for workbook explained totals.

    Notes:
        Performance Differences totals must use the same selected impact rows
        as the Performance Difference Causes sheet. Otherwise transaction amount rows can
        be counted in summary totals after the detail sheet has already treated
        them as supporting evidence for changed holdings.
    """
    selected_rows: list[tuple[dict[str, object], float]] = []
    source_rows = ranked_rows or _workbook_ranked_changed_rows_for_level(
        findings,
        comparison_level=comparison_level,
    )
    for row in source_rows:
        if not _rows.is_underlying_cause_row(row):
            continue
        estimated_impact = _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
        if estimated_impact is None:
            continue
        selected_rows.append((dict(row), estimated_impact))
    return selected_rows


def _workbook_security_change_row(
    row: Mapping[str, object],
) -> dict[str, object]:
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

    security_period_keys = {_rows.review_period_key(row) for row in security_rows}
    rows: list[dict[str, object]] = []
    for row in coverage.iter_rows(named=True):
        if _rows.review_period_key(row) in security_period_keys:
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
        _layout.PERFORMANCE_CHANGE: None,
        _layout.ESTIMATED_CAUSE_TOTAL: None,
        _layout.UNEXPLAINED_CHANGE: None,
        _layout.REVIEW_STATUS: "No differences",
        _layout.REVIEW_NOTE: "None",
        _layout.REVIEW_KEY: row.get(_layout.REVIEW_KEY),
    }


def _workbook_empty_security_changes_table() -> pl.DataFrame:
    """Return an empty workbook security-level performance differences table."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _pc_findings.SECURITY_ID: pl.String,
            _layout.PERFORMANCE_CHANGE: pl.Float64,
            _layout.ESTIMATED_CAUSE_TOTAL: pl.Float64,
            _layout.UNEXPLAINED_CHANGE: pl.Float64,
            _layout.REVIEW_STATUS: pl.String,
            _layout.REVIEW_NOTE: pl.String,
            _layout.REVIEW_KEY: pl.String,
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
    table_cache: _WorkbookTableCache | None = None,
) -> list[dict[str, object]]:
    """Return ranked changed rows for one primary comparison level."""
    evidence = _workbook_with_primary_review_key(
        (
            table_cache.top_evidence(comparison_level)
            if table_cache is not None
            else _workbook_top_evidence_table(findings, comparison_level=comparison_level)
        ),
        comparison_level,
    )
    if evidence.is_empty():
        return []

    selected_impact_bases = (
        table_cache.selected_impact_basis_keys(comparison_level)
        if table_cache is not None
        else _workbook_selected_impact_basis_keys(
            findings,
            comparison_level=comparison_level,
        )
    )
    performance_input_keys = (
        table_cache.performance_input_family_keys(comparison_level)
        if table_cache is not None
        else _workbook_performance_input_family_keys(
            findings,
            comparison_level=comparison_level,
        )
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
    lineage_findings: pl.DataFrame | None = None,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    primary_changes_table: pl.DataFrame | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
    finding_audit_trail: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return input rows that may directly explain performance differences.

    Args:
        findings: Active findings used to construct visible cause rows.
        lineage_findings: Complete findings, including suppressed rows, used
            to bind cause rows to the persisted audit-trail fingerprints.
        comparison_path: Optional comparison YAML path.
        comparison_level: Portfolio or security comparison grain.
        primary_changes_table: Optional precomputed primary changes table.
        table_cache: Optional shared workbook table cache.
        reconstruction_cache: Optional shared return-reconstruction cache.
        finding_audit_trail: Optional precomputed complete finding audit trail.

    Returns:
        Internal cause table with conservation and source-lineage metadata.
    """
    reconstruction_cache = _workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    unexplained_keys = _workbook_unexplained_primary_keys(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        primary_changes_table=primary_changes_table,
        reconstruction_cache=reconstruction_cache,
    )
    performance_input_keys = (
        table_cache.performance_input_family_keys(comparison_level)
        if table_cache is not None
        else _workbook_performance_input_family_keys(
            findings,
            comparison_level=comparison_level,
        )
    )
    rows: list[dict[str, object]] = []
    if table_cache is not None:
        formula_rows = table_cache.reconstruction_formula_rows(
            comparison_level,
            comparison_path=comparison_path,
            reconstruction_cache=reconstruction_cache,
        )
    elif comparison_level == SECURITY_COMPARISON_LEVEL:
        formula_rows = _formula_rows.security_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_security_period_keys(findings),
            reconstruction_cache=reconstruction_cache,
        )
    else:
        formula_rows = _formula_rows.portfolio_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_portfolio_period_keys(findings),
            reconstruction_cache=reconstruction_cache,
        )
    formula_keys = {_rows.primary_review_period_key(row, comparison_level) for row in formula_rows}
    ranked_rows = (
        table_cache.ranked_rows(comparison_level)
        if table_cache is not None
        else _workbook_ranked_changed_rows_for_level(
            findings,
            comparison_level=comparison_level,
        )
    )
    cash_security_matches = _source_allocation.cash_security_matches(
        ranked_rows,
        comparison_level=comparison_level,
    )
    (
        attributed_formula_source_rows,
        unallocated_formula_rows,
    ) = _source_allocation.allocate_formula_sources(
        ranked_rows,
        formula_rows,
        matched_cash_securities=cash_security_matches,
        comparison_level=comparison_level,
    )
    attributed_source_keys = {
        _source_allocation.source_row_key(row, comparison_level)
        for row in attributed_formula_source_rows
    }
    fx_support_rows = _source_allocation.fx_support_rows(
        ranked_rows,
        attributed_formula_source_rows,
        comparison_level=comparison_level,
    )
    linked_fx_sources = {
        _source_allocation.fx_source_identity(row) for row in fx_support_rows
    }
    possible_cause_source_keys = {
        _source_allocation.source_row_key(row, comparison_level)
        for row in _workbook_possible_cause_rows(
            findings,
            unresolved_keys=unexplained_keys,
            comparison_level=comparison_level,
            ranked_rows=ranked_rows,
        )
    }
    for row in attributed_formula_source_rows:
        if (
            _source_allocation.source_row_key(row, comparison_level)
            in possible_cause_source_keys
        ):
            row = _workbook_mark_possible_cause_row(row, comparison_level)
        rows.append(_workbook_changed_item_row(row, comparison_path=comparison_path))
    rows.extend(dict(row) for row in unallocated_formula_rows)
    rows.extend(
        _workbook_changed_item_row(row, comparison_path=comparison_path) for row in fx_support_rows
    )

    for row in ranked_rows:
        row = _source_allocation.with_cash_balance_security(
            row,
            cash_security_matches,
            comparison_level=comparison_level,
        )
        has_formula_role = _rows.primary_review_period_key(row, comparison_level) in formula_keys
        source_row_key = _source_allocation.source_row_key(row, comparison_level)
        if source_row_key in attributed_source_keys:
            continue
        if _source_allocation.fx_source_identity(row) in linked_fx_sources:
            continue
        if has_formula_role and _rows.is_underlying_cause_row(row):
            support_row = _rows.formula_support_row(
                row,
                comparison_level=comparison_level,
            )
            if source_row_key in possible_cause_source_keys:
                support_row = _workbook_mark_possible_cause_row(
                    support_row,
                    comparison_level,
                )
            workbook_row = _workbook_changed_item_row(
                support_row,
                comparison_path=comparison_path,
            )
        elif _rows.is_underlying_cause_row(row):
            if source_row_key in possible_cause_source_keys:
                row = _workbook_mark_possible_cause_row(row, comparison_level)
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
            if _workbook_is_split_factor_row(row):
                row = _workbook_split_factor_support_row(row)
            if source_row_key in possible_cause_source_keys:
                row = _workbook_mark_possible_cause_row(row, comparison_level)
            workbook_row = _workbook_changed_item_row(
                _rows.non_additive_row(row),
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
            table_cache=table_cache,
        )
    )
    if not rows:
        original_table = _workbook_empty_changed_item_table()
    else:
        original_table = _layout.workbook_sorted_table(
            pl.DataFrame(rows, infer_schema_length=None),
            _workbook_left_review_sort_columns(),
        )
    return _workbook_cause_safety_table(
        original_table,
        findings if lineage_findings is None else lineage_findings,
        comparison_level=comparison_level,
        finding_audit_trail=finding_audit_trail,
    )


def _workbook_cause_safety_table(
    causes: pl.DataFrame,
    findings: pl.DataFrame,
    *,
    comparison_level: str,
    finding_audit_trail: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return cause rows with lineage and conservation invariants enforced."""
    lineage_table = _pc_lineage.cause_lineage_table(
        causes,
        findings,
        finding_audit_trail=finding_audit_trail,
    )
    conservation_table = _pc_conservation.cause_conservation_table(
        lineage_table,
        comparison_level=comparison_level,
    )
    _pc_conservation.assert_cause_conservation(
        lineage_table,
        conservation_table,
        comparison_level=comparison_level,
    )
    return conservation_table


def _workbook_missing_underlying_cause_rows(
    findings: pl.DataFrame,
    underlying_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
    table_cache: _WorkbookTableCache | None = None,
) -> list[dict[str, object]]:
    """Return placeholder rows for changed periods without input causes."""
    coverage = _workbook_with_primary_review_key(
        (
            table_cache.primary_coverage(comparison_level)
            if table_cache is not None
            else _workbook_primary_coverage_summary(
                findings,
                comparison_level=comparison_level,
            )
        ),
        comparison_level,
    )
    if coverage.is_empty():
        return []

    underlying_period_keys = {
        _rows.primary_review_period_key(row, comparison_level) for row in underlying_rows
    }
    rows: list[dict[str, object]] = []
    for row in coverage.iter_rows(named=True):
        if _rows.primary_review_period_key(row, comparison_level) in underlying_period_keys:
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
        _layout.USE: _rows.USE_DIAGNOSTIC,
        _layout.CHANGE_LABEL: "No additive underlying cause found",
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _pc_findings.SNAPSHOT_A_VALUE: None,
        _pc_findings.SNAPSHOT_B_VALUE: None,
        _layout.CHANGE: None,
        _layout.ESTIMATED_IMPACT: None,
        _layout.IMPACT_STATUS: _guidance.IMPACT_STATUS_REVIEW_ONLY,
        _layout.REVIEW_NOTE: (
            "Review `supporting_files/source_detail.csv`. The difference may be due to "
            "missing source-data, source-file timing differences, or vendor "
            "methodology that does not match the YAML specifications."
        ),
        _layout.REVIEW_GUIDANCE: (
            "No identifiable cause was found. Review "
            "`supporting_files/source_detail.csv`. "
            "The difference may be due to missing source-data, source-file timing "
            "differences, or vendor methodology that does not match the YAML "
            "specifications."
        ),
        _pc_findings.DATASET: _rows.NO_UNDERLYING_CAUSE_DATASET,
        _pc_findings.SOURCE_COLUMN: None,
        _pc_findings.FINDING_CODE: None,
        _pc_explain.REVIEW_RANK: 999999,
        _layout.USE_PRIORITY: _rows.use_priority(_rows.USE_DIAGNOSTIC),
        _layout.REVIEW_KEY: row.get(_layout.REVIEW_KEY),
    }


def _workbook_unexplained_primary_keys(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str,
    primary_changes_table: pl.DataFrame | None = None,
    reconstruction_cache: _workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> set[tuple[object, ...]]:
    """Return primary review keys with a meaningful unexplained remainder."""
    reconstruction_cache = _workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    if primary_changes_table is not None:
        summary = primary_changes_table
    elif comparison_level == SECURITY_COMPARISON_LEVEL:
        summary = _workbook_security_changes_table(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            reconstruction_cache=reconstruction_cache,
        )
    else:
        summary = _workbook_portfolio_changes_table(
            findings,
            comparison_path=comparison_path,
            reconstruction_cache=reconstruction_cache,
        )

    keys: set[tuple[object, ...]] = set()
    for row in summary.iter_rows(named=True):
        unexplained_change = _rows.number_or_none(row.get(_layout.UNEXPLAINED_CHANGE))
        if (
            unexplained_change is None
            or abs(unexplained_change) <= _WORKBOOK_UNEXPLAINED_TOLERANCE
        ):
            continue
        keys.add(_rows.primary_review_period_key(row, comparison_level))
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
    if not _rows.is_context_row(row) or not _rows.has_evidence_only_policy(row):
        return False
    if not _workbook_is_promotable_evidence_only_row(row):
        return False
    if _workbook_is_transaction_component_row(row):
        return _workbook_cause_family_key(row, comparison_level) in performance_input_keys
    if _workbook_is_split_factor_row(row):
        return (
            _workbook_split_holding_value_key(
                row,
                comparison_level=comparison_level,
            )
            in performance_input_keys
        )
    if _rows.primary_review_period_key(row, comparison_level) in unexplained_keys:
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


def _workbook_is_split_factor_row(row: Mapping[str, object]) -> bool:
    """Return whether the row is split-factor evidence."""
    return (
        row.get(_pc_findings.DATASET) == pc_cols.SPLITS
        and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.SPLIT_FACTOR
    )


def _workbook_split_holding_value_key(
    row: Mapping[str, object],
    *,
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the holding-value cause family that a split factor can support."""
    return (
        *_rows.primary_review_period_key(row, comparison_level),
        row.get(_pc_findings.SECURITY_ID),
        "holding_value",
    )


def _workbook_split_factor_support_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return split-factor evidence marked as supporting holding value changes."""
    row_dict = dict(row)
    row_dict[_rows.SPLIT_FACTOR_SUPPORTS_HOLDING] = True
    return row_dict


def _workbook_possible_cause_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return evidence marked as a possible cause of an unresolved period."""
    row_dict = dict(row)
    row_dict[_rows.POSSIBLE_CAUSE_ROW] = True
    return row_dict


def _workbook_mark_possible_cause_row(
    row: Mapping[str, object],
    comparison_level: str,
) -> dict[str, object]:
    """Return a possible-cause row with level-specific review context."""
    row_dict = _workbook_possible_cause_row(row)
    if (
        comparison_level == PORTFOLIO_COMPARISON_LEVEL
        and row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
        and row_dict.get(_pc_findings.SOURCE_COLUMN) == pc_cols.AMOUNT
    ):
        row_dict[_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
        row_dict = _rows.non_additive_row(row_dict)
    return row_dict


def _workbook_raw_audit_trail_table(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> pl.DataFrame:
    """Return raw audit rows with reviewer-facing leading columns."""
    keyed_findings = _workbook_with_primary_review_key(findings, comparison_level)
    if keyed_findings.is_empty():
        return keyed_findings

    source_rows = list(keyed_findings.iter_rows(named=True))
    cash_security_matches = _source_allocation.cash_security_matches(
        source_rows,
        comparison_level=comparison_level,
    )
    review_rows: list[dict[str, object]] = []
    changed_item_cache: dict[
        tuple[tuple[str, object], ...],
        dict[str, object],
    ] = {}
    for row in source_rows:
        enriched_row = _workbook_raw_audit_enriched_row(
            row,
            cash_security_matches,
            comparison_level=comparison_level,
        )
        cache_key = _workbook_changed_item_cache_key(enriched_row)
        changed_item_template = changed_item_cache.get(cache_key)
        if changed_item_template is None:
            changed_item_template = _workbook_changed_item_row(
                enriched_row,
                comparison_path=comparison_path,
            )
            changed_item_cache[cache_key] = changed_item_template
        changed_item = dict(changed_item_template)
        for column in _WORKBOOK_CHANGED_ITEM_IDENTITY_COLUMNS:
            changed_item[column] = enriched_row.get(column)
        review_rows.append(changed_item)
    review_table = pl.DataFrame(review_rows, infer_schema_length=None)
    raw_columns = [
        column for column in keyed_findings.columns if column not in review_table.columns
    ]
    combined_table = review_table.hstack(keyed_findings.select(raw_columns))
    return _layout.workbook_sorted_table(
        combined_table,
        _workbook_left_review_sort_columns(),
    )


def _workbook_raw_audit_enriched_row(
    row: Mapping[str, object],
    cash_security_matches: Mapping[tuple[object, ...], object],
    *,
    comparison_level: str,
) -> Mapping[str, object]:
    """Return a source-detail row with concrete cash-balance context."""
    enriched_row = _source_allocation.with_cash_balance_security(
        row,
        cash_security_matches,
        comparison_level=comparison_level,
    )
    if (
        enriched_row.get(_pc_findings.DATASET) != pc_cols.TRANSACTIONS
        or enriched_row.get(_pc_findings.SOURCE_COLUMN) != pc_cols.AMOUNT
    ):
        return enriched_row
    if not enriched_row.get(
        _source_allocation.CASH_BALANCE_SECURITY_ID
    ) and not _workbook_is_possible_cause_row(enriched_row):
        return enriched_row
    row_dict = dict(enriched_row)
    row_dict[_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
    return _rows.non_additive_row(row_dict)


def _workbook_changed_item_cache_key(
    row: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return presentation inputs excluding identity-only output columns.

    Notes:
        Finding tables contain scalar values. Portfolio identity, source locator,
        and review key are copied from every source row after the cached
        presentation wording and classification are retrieved.
    """
    return tuple(
        (column, value)
        for column, value in row.items()
        if column not in _WORKBOOK_CHANGED_ITEM_IDENTITY_COLUMNS
    )


def _workbook_left_review_sort_columns() -> tuple[str, ...]:
    """Return the shared left-column sort order for review detail sheets."""
    return (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _layout.AS_OF_DATE,
        _layout.DATASET_FIELD,
        _pc_findings.SECURITY_ID,
    )


def _workbook_raw_audit_columns(findings: pl.DataFrame) -> tuple[str, ...]:
    """Return source-detail presentation columns with review key last."""
    preferred_columns = (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _layout.AS_OF_DATE,
        _layout.DATASET_FIELD,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _layout.CHANGE,
        _layout.REVIEW_GUIDANCE,
    )
    remaining_columns = [
        column
        for column in findings.columns
        if column
        not in {
            *preferred_columns,
            _pc_findings.DATASET,
            _pc_findings.SOURCE_COLUMN,
            _pc_findings.DELTA_B_MINUS_A,
            _layout.ESTIMATED_IMPACT,
            _layout.REVIEW_KEY,
        }
    ]
    return (*preferred_columns, *remaining_columns, _layout.REVIEW_KEY)


def _workbook_is_transaction_component_row(row: Mapping[str, object]) -> bool:
    """Return whether a row is support for transaction amount, not an input cause."""
    return row.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS and row.get(
        _pc_findings.SOURCE_COLUMN
    ) in {pc_cols.COMMISSION, pc_cols.PRICE, pc_cols.QUANTITY}


def _workbook_selected_impact_basis_keys(
    findings: pl.DataFrame,
    *,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    table_cache: _WorkbookTableCache | None = None,
) -> set[tuple[object, ...]]:
    """Return period/impact-basis keys included in Performance Differences totals."""
    causes = (
        table_cache.cause_summary(comparison_level)
        if table_cache is not None
        else _workbook_primary_cause_summary(
            findings,
            comparison_level=comparison_level,
        )
    )
    if causes.is_empty():
        return set()

    keys: set[tuple[object, ...]] = set()
    del causes
    rows = _workbook_with_primary_review_key(
        (
            table_cache.top_evidence(comparison_level)
            if table_cache is not None
            else _workbook_top_evidence_table(findings, comparison_level=comparison_level)
        ),
        comparison_level,
    )
    grouped_rows: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows.iter_rows(named=True):
        if _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
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
    table_cache: _WorkbookTableCache | None = None,
) -> set[tuple[object, ...]]:
    """Return cause-family keys with selected performance input rows."""
    keys: set[tuple[object, ...]] = set()
    evidence = _workbook_with_primary_review_key(
        (
            table_cache.top_evidence(comparison_level)
            if table_cache is not None
            else _workbook_top_evidence_table(findings, comparison_level=comparison_level)
        ),
        comparison_level,
    )
    for row in evidence.iter_rows(named=True):
        if not _field_roles.is_performance_input(
            row.get(_pc_findings.DATASET),
            row.get(_pc_findings.SOURCE_COLUMN),
        ):
            continue
        if _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is None:
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
        *_rows.primary_review_period_key(row, comparison_level),
        row.get(_pc_findings.SECURITY_ID),
        family,
    )


def _workbook_cause_family(row: Mapping[str, object]) -> object:
    """Return the broad accounting family for a changed input row."""
    dataset = row.get(_pc_findings.DATASET)
    source_column = row.get(_pc_findings.SOURCE_COLUMN)
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.BASE_ACCRUED,
    }:
        return "holding_accrued"
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.MARKET_VALUE,
        pc_cols.BASE_MARKET_VALUE,
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
        row.get(_pc_explain.IMPACT_BASIS) == _pc_explain.IMPACT_BASIS_SECURITY_CONTRIBUTION
        for row in rows
    ):
        return [
            row
            for row in rows
            if row.get(_pc_explain.IMPACT_BASIS) == _pc_explain.IMPACT_BASIS_SECURITY_CONTRIBUTION
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
    estimated_impact = _rows.number_or_none(row_dict.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
    if estimated_impact is None:
        return row_dict

    if (
        comparison_level == PORTFOLIO_COMPARISON_LEVEL
        and row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
    ):
        row_dict[_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
        return _rows.non_additive_row(row_dict)

    holding_value_key = (
        *_rows.primary_review_period_key(row_dict, comparison_level),
        row_dict.get(_pc_findings.SECURITY_ID),
        "holding_value",
    )
    if (
        row_dict.get(_pc_findings.DATASET) == pc_cols.TRANSACTIONS
        and row_dict.get(_pc_findings.TRANSACTION_CATEGORY)
        in {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
        and holding_value_key in performance_input_keys
    ):
        row_dict[_rows.UNSELECTED_RELATED_ESTIMATE] = True
        row_dict[_rows.TRANSACTION_FLOW_SUPPORTS_HOLDING] = True
        row_dict[_pc_explain.IMPACT_MESSAGE] = (
            "Supporting evidence for changed holdings.market_value."
        )
        return _rows.non_additive_row(row_dict)

    key = (
        *_workbook_cause_family_key(row_dict, comparison_level),
        row_dict.get(_pc_explain.IMPACT_BASIS),
    )
    if key in selected_impact_bases:
        return row_dict

    row_dict[_pc_explain.ESTIMATED_RETURN_IMPACT] = None
    row_dict[_pc_explain.IMPACT_BASIS] = _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    row_dict[_pc_explain.IMPACT_METHOD] = None
    row_dict[_rows.UNSELECTED_RELATED_ESTIMATE] = True
    row_dict[_pc_explain.IMPACT_MESSAGE] = (
        "Another estimate was selected for this portfolio-period cause area."
    )
    return row_dict



def _workbook_changed_item_row(
    row: Mapping[str, object],
    *,
    comparison_path: util.PathLike | None = None,
) -> dict[str, object]:
    """Return one plain-English changed-item workbook row."""
    estimated_impact = _rows.number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
    row_kind = _rows.workbook_row_kind(row)
    row_use = _workbook_row_use(row, row_kind)
    impact_status = _workbook_impact_status(row, estimated_impact, row_kind)
    input_role = _workbook_input_role(row, estimated_impact, row_kind)
    review_guidance = _guidance.review_guidance(
        row,
        estimated_impact,
        comparison_path=comparison_path,
        impact_status=impact_status,
        row_kind=row_kind,
    )
    review_guidance = _transaction_prefixed_review_guidance(row, review_guidance)
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _layout.AS_OF_DATE: _rows.evidence_as_of_date(row),
        _layout.USE: row_use,
        _layout.CHANGE_LABEL: _workbook_change_label(row),
        _layout.DATASET_FIELD: _guidance.dataset_field(row),
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _layout.ROW_TYPE: _workbook_row_type(
            row,
            estimated_impact,
            row_use,
            impact_status,
            input_role,
        ),
        _pc_findings.SNAPSHOT_A_VALUE: row.get(_pc_findings.SNAPSHOT_A_VALUE),
        _pc_findings.SNAPSHOT_B_VALUE: row.get(_pc_findings.SNAPSHOT_B_VALUE),
        _layout.CHANGE: row.get(_pc_findings.DELTA_B_MINUS_A),
        _pc_findings.IMPACT_INPUT_VALUE: row.get(_pc_findings.IMPACT_INPUT_VALUE),
        _layout.ESTIMATED_IMPACT: estimated_impact,
        _layout.INPUT_ROLE: input_role,
        _layout.IMPACT_STATUS: impact_status,
        _layout.REVIEW_NOTE: _guidance.review_note(row, estimated_impact, row_use, impact_status),
        _layout.REVIEW_GUIDANCE: review_guidance,
        _pc_findings.DATASET: row.get(_pc_findings.DATASET),
        _pc_findings.SOURCE_RECORD_LOCATOR: row.get(
            _pc_findings.SOURCE_RECORD_LOCATOR
        ),
        _pc_findings.SOURCE_COLUMN: row.get(_pc_findings.SOURCE_COLUMN),
        _pc_findings.FINDING_CODE: row.get(_pc_findings.FINDING_CODE),
        _pc_findings.TRANSACTION_CODE: row.get(_pc_findings.TRANSACTION_CODE),
        _pc_findings.TRANSACTION_CATEGORY: row.get(
            _pc_findings.TRANSACTION_CATEGORY
        ),
        _pc_explain.REVIEW_RANK: row.get(_pc_explain.REVIEW_RANK),
        _layout.USE_PRIORITY: _rows.use_priority(row_use),
        _formula_rows.RECONSTRUCTION_COMPONENTS: row.get(
            _formula_rows.RECONSTRUCTION_COMPONENTS
        ),
        _layout.REVIEW_KEY: row.get(_layout.REVIEW_KEY),
    }


def _transaction_prefixed_review_guidance(
    row: Mapping[str, object],
    review_guidance: str,
) -> str:
    """Prefix transaction-associated guidance with its native source code."""
    transaction_code = _format_value(row.get(_pc_findings.TRANSACTION_CODE)).strip()
    if not transaction_code:
        return review_guidance
    prefix = f"{transaction_code}:"
    if review_guidance.startswith(prefix):
        return review_guidance
    if not review_guidance:
        return prefix
    return f"{prefix} {review_guidance}"


def _workbook_row_type(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
    input_role: str,
) -> str:
    """Return the reviewer-facing row type for Performance Difference Causes."""
    if row.get(_rows.POSSIBLE_CAUSE_ROW):
        return _ROW_TYPE_POSSIBLE_CAUSE
    if row.get(_pc_findings.FINDING_CODE) == _formula_rows.FORMULA_FINDING_CODE:
        return _formula_rows.FORMULA_ROW_TYPE
    if estimated_impact is not None:
        return _ROW_TYPE_EXPLAINED_CAUSE
    if (
        row.get(_rows.SPLIT_FACTOR_SUPPORTS_HOLDING)
        or row.get(_source_allocation.FX_RATE_SUPPORTS_BASE_INPUT)
        or row.get(_rows.TRANSACTION_FLOW_SUPPORTS_HOLDING)
        or row.get(_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION)
        or row.get(_rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW)
        or row.get(_rows.UNSELECTED_RELATED_ESTIMATE)
        or impact_status == _guidance.IMPACT_STATUS_REVIEW_ONLY
        or input_role == _INPUT_ROLE_SUPPORTING_EVIDENCE
    ):
        return _ROW_TYPE_SUPPORTING_EVIDENCE
    if row_use == _rows.USE_REVIEW_CONTEXT:
        return _ROW_TYPE_REVIEW_CONTEXT
    return _ROW_TYPE_REVIEW_CONTEXT


def _workbook_input_role(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_kind: str,
) -> str:
    """Return the reviewer-facing role for one changed input row."""
    dataset = row.get(_pc_findings.DATASET)
    source_column = row.get(_pc_findings.SOURCE_COLUMN)
    if dataset == _rows.NO_UNDERLYING_CAUSE_DATASET:
        return _INPUT_ROLE_DIAGNOSTIC
    if estimated_impact is not None:
        return _INPUT_ROLE_PERFORMANCE_INPUT
    if dataset == pc_cols.TRANSACTIONS and source_column in {
        pc_cols.COMMISSION,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _INPUT_ROLE_SUPPORTING_EVIDENCE
    if row.get(_rows.SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _INPUT_ROLE_SUPPORTING_EVIDENCE
    if _field_roles.is_input_component(
        dataset, source_column
    ) or _field_roles.is_performance_input(dataset, source_column):
        return _INPUT_ROLE_INPUT_DRIVER
    if row_kind == _rows.ROW_KIND_DIAGNOSTIC:
        return _INPUT_ROLE_DIAGNOSTIC
    return _INPUT_ROLE_CONTEXT


def _workbook_change_label(row: Mapping[str, object]) -> str:
    """Return a concise changed-item label."""
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    dataset = _format_value(row.get(_pc_findings.DATASET)).replace("_", " ")
    if source_column:
        return f"{dataset} {source_column} changed"
    return _format_value(row.get(_pc_findings.MESSAGE))


def _workbook_row_use(row: Mapping[str, object], row_kind: str) -> str:
    """Return how a changed item should be used during review."""
    if row_kind == _rows.ROW_KIND_DIAGNOSTIC:
        return _rows.USE_DIAGNOSTIC
    if row.get(_rows.SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _rows.USE_EXPLAINS_CHANGE
    evidence_role = row.get(_pc_findings.EVIDENCE_ROLE)
    if evidence_role == _pc_findings.CONTEXT.value:
        return _rows.USE_REVIEW_CONTEXT
    return _rows.USE_EXPLAINS_CHANGE


def _workbook_impact_status(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_kind: str,
) -> str:
    """Return a compact status for row-level impact treatment."""
    if estimated_impact is not None:
        return _guidance.IMPACT_STATUS_ESTIMATED
    if (
        row.get(_rows.UNSELECTED_RELATED_ESTIMATE)
        or row.get(_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION)
        or row.get(_rows.TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW)
        or row_kind in {
            _rows.ROW_KIND_CONTEXT,
            _rows.ROW_KIND_REPORTED_DIAGNOSTIC,
            _rows.ROW_KIND_DIAGNOSTIC,
        }
        or _rows.has_evidence_only_policy(row)
    ):
        return _guidance.IMPACT_STATUS_REVIEW_ONLY
    if _rows.has_additive_policy(row):
        return _guidance.IMPACT_STATUS_MISSING_INPUT
    return _guidance.IMPACT_STATUS_MISSING_METHOD


def _workbook_empty_changed_item_table() -> pl.DataFrame:
    """Return an empty workbook changed-item table."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _layout.AS_OF_DATE: pl.Date,
            _layout.USE: pl.String,
            _layout.CHANGE_LABEL: pl.String,
            _layout.DATASET_FIELD: pl.String,
            _pc_findings.SECURITY_ID: pl.String,
            _layout.ROW_TYPE: pl.String,
            _pc_findings.SNAPSHOT_A_VALUE: pl.String,
            _pc_findings.SNAPSHOT_B_VALUE: pl.String,
            _layout.CHANGE: pl.Float64,
            _pc_findings.IMPACT_INPUT_VALUE: pl.Float64,
            _layout.ESTIMATED_IMPACT: pl.Float64,
            _layout.INPUT_ROLE: pl.String,
            _layout.IMPACT_STATUS: pl.String,
            _layout.REVIEW_NOTE: pl.String,
            _layout.REVIEW_GUIDANCE: pl.String,
            _pc_findings.DATASET: pl.String,
            _pc_findings.SOURCE_RECORD_LOCATOR: pl.String,
            _pc_findings.SOURCE_COLUMN: pl.String,
            _pc_findings.FINDING_CODE: pl.String,
            _pc_findings.TRANSACTION_CODE: pl.String,
            _pc_findings.TRANSACTION_CATEGORY: pl.String,
            _pc_explain.REVIEW_RANK: pl.Int64,
            _layout.USE_PRIORITY: pl.Int64,
            _layout.REVIEW_KEY: pl.String,
        }
    )




def _active_findings(findings: pl.DataFrame) -> pl.DataFrame:
    """Return unsuppressed findings, preserving empty-table behavior."""
    if findings.is_empty() or _pc_findings.SUPPRESSED not in findings.columns:
        return findings
    return findings.filter(~pl.col(_pc_findings.SUPPRESSED))
