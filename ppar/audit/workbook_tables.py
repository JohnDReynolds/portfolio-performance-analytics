"""Build review workbook tables for performance comparison findings."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import datetime as _dt
from pathlib import Path
from typing import cast

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit import conservation as _pc_conservation
from ppar.audit import field_roles as _field_roles
from ppar.audit.performance_comparison import explain as _pc_explain
from ppar.audit.performance_comparison import findings as _pc_findings
from ppar.audit import lineage as _pc_lineage
from ppar.audit import rendering as _pc_rendering
from ppar.audit import review_keys as _pc_review_keys
from ppar.audit import review_model as _pc_review_model
from ppar.audit.performance_comparison import return_reconstruction as _pc_reconstruction
from ppar.audit import workbook as _pc_workbook
from ppar.audit.data_issues import checks as _data_issue_checks
from ppar.audit.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.audit.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.audit.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
)

__all__ = [
    "audit_review_workbook_sheets",
    "workbook_column_tooltip",
    "write_audit_review_workbook",
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
_ROW_TYPE = "row_type"
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
_ROW_TYPE_EXPLAINED_CAUSE = "Explained Cause"
_ROW_TYPE_POSSIBLE_CAUSE = "Possible Cause"
_ROW_TYPE_SUPPORTING_EVIDENCE = "Supporting Evidence"
_ROW_TYPE_FORMULA_INPUT = "Formula Input"
_ROW_TYPE_REVIEW_CONTEXT = "Review Context"
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
_WORKBOOK_TRANSACTION_FLOW_SUPPORTS_HOLDING = "_workbook_transaction_flow_supports_holding"
_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING = "_workbook_split_factor_supports_holding"
_WORKBOOK_FX_RATE_SUPPORTS_BASE_INPUT = "_workbook_fx_rate_supports_base_input"
_WORKBOOK_FX_RATE_TARGET_FIELD = "_workbook_fx_rate_target_field"
_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW = (
    "_workbook_transaction_supports_reconstruction_flow"
)
_WORKBOOK_CASH_BALANCE_SECURITY_ID = "_workbook_cash_balance_security_id"
_WORKBOOK_POSSIBLE_CAUSE_ROW = "_workbook_possible_cause_row"
_WORKBOOK_RECONSTRUCTION_COMPONENTS = "_workbook_reconstruction_components"
_WORKBOOK_CHANGED_ITEM_IDENTITY_COLUMNS = (
    _pc_findings.PORTFOLIO_ID,
    _pc_findings.SOURCE_RECORD_LOCATOR,
    _REVIEW_KEY,
)
_POSSIBLE_CAUSE_COMMENT = "_possible_cause_comment"
_POSSIBLE_CAUSE_CONFIGURATION_NOTE = "Add YAML configuration to count it as explained."
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
_POSSIBLE_CAUSE_FIELDS = {
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE),
    (pc_cols.TRANSACTIONS, pc_cols.AMOUNT),
}
_format_value = _pc_rendering.format_value
_with_period_review_key = _pc_review_keys.with_period_review_key
_with_security_review_key = _pc_review_keys.with_security_review_key


class _WorkbookReconstructionCache:
    """Cache reconstruction diagnostics for one workbook/report build."""

    def __init__(self, comparison_path: util.PathLike | None) -> None:
        self._comparison_path = comparison_path
        self._input_cache = _pc_reconstruction._SnapshotDataIndexCache()
        self._portfolio_checks: pl.DataFrame | None = None
        self._security_checks: pl.DataFrame | None = None
        self._security_checks_by_active_keys: dict[
            frozenset[tuple[str, str, _dt.date, _dt.date]],
            pl.DataFrame,
        ] = {}
        self._summary: pl.DataFrame | None = None

    def portfolio_checks(self) -> pl.DataFrame:
        """Return cached portfolio return-reconstruction checks."""
        if self._portfolio_checks is None:
            self._portfolio_checks = _pc_reconstruction.portfolio_return_reconstruction_checks(
                self._comparison_path,
                _input_cache=self._input_cache,
            )
        return self._portfolio_checks

    def security_checks(
        self,
        active_keys: Iterable[tuple[object, object, object, object]] | None = None,
    ) -> pl.DataFrame:
        """Return cached security return-reconstruction checks."""
        if active_keys is not None:
            reconstruction_keys = _security_reconstruction_active_keys(active_keys)
            cache_key = frozenset(reconstruction_keys)
            if cache_key not in self._security_checks_by_active_keys:
                self._security_checks_by_active_keys[cache_key] = (
                    _pc_reconstruction.security_return_reconstruction_checks(
                        self._comparison_path,
                        active_keys=reconstruction_keys,
                        _input_cache=self._input_cache,
                    )
                )
            return self._security_checks_by_active_keys[cache_key]
        if self._security_checks is None:
            self._security_checks = _pc_reconstruction.security_return_reconstruction_checks(
                self._comparison_path,
                _input_cache=self._input_cache,
            )
        return self._security_checks

    def summary(self) -> pl.DataFrame:
        """Return cached return-reconstruction summary."""
        if self._summary is None:
            self._summary = _pc_reconstruction.return_reconstruction_summary(
                self._comparison_path,
                _input_cache=self._input_cache,
            )
        return self._summary


# Site-level Audit generation intentionally shares this cache between the
# portfolio and security report views. Retain the private name for compatible
# internal call sites while exposing a non-underscored package-internal name.
WorkbookReconstructionCache = _WorkbookReconstructionCache


@dataclass(frozen=True)
class _FormulaSourceIndex:
    """Index source rows by the exact Modified Dietz component they can support.

    Attributes:
        value_rows: Eligible holding values keyed by formula owner and date.
        flow_rows: Eligible transaction flows keyed by owner and period.
        income_rows: Eligible transaction income/expense rows keyed by owner and
            period.
    """

    value_rows: dict[tuple[object, ...], list[Mapping[str, object]]]
    flow_rows: dict[tuple[object, ...], list[Mapping[str, object]]]
    income_rows: dict[tuple[object, ...], list[Mapping[str, object]]]


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
        reconstruction_cache: _WorkbookReconstructionCache,
    ) -> list[dict[str, object]]:
        """Return cached Modified Dietz formula rows for one review level."""
        if comparison_level not in self._reconstruction_formula_rows:
            rows = (
                _workbook_security_reconstruction_formula_rows(
                    comparison_path,
                    active_keys=self.active_security_keys(),
                    reconstruction_cache=reconstruction_cache,
                )
                if comparison_level == SECURITY_COMPARISON_LEVEL
                else _workbook_portfolio_reconstruction_formula_rows(
                    comparison_path,
                    active_keys=self.active_portfolio_keys(),
                    reconstruction_cache=reconstruction_cache,
                )
            )
            self._reconstruction_formula_rows[comparison_level] = rows
        return self._reconstruction_formula_rows[comparison_level]


def _resolved_reconstruction_cache(
    comparison_path: util.PathLike | None,
    reconstruction_cache: _WorkbookReconstructionCache | None,
) -> _WorkbookReconstructionCache:
    """Return an existing reconstruction cache or create one for direct calls."""
    if reconstruction_cache is not None:
        return reconstruction_cache
    return _WorkbookReconstructionCache(comparison_path)


def _security_reconstruction_active_keys(
    active_keys: Iterable[tuple[object, object, object, object]],
) -> set[tuple[str, str, _dt.date, _dt.date]]:
    """Return return-reconstruction security keys from workbook primary keys."""
    reconstruction_keys: set[tuple[str, str, _dt.date, _dt.date]] = set()
    for portfolio_id, from_date, thru_date, security_id in active_keys:
        if not isinstance(from_date, _dt.date) or not isinstance(thru_date, _dt.date):
            continue
        reconstruction_keys.add(
            (
                str(portfolio_id),
                str(security_id),
                from_date,
                thru_date,
            )
        )
    return reconstruction_keys


def write_audit_review_workbook(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    top_evidence_limit: int = 10,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: _WorkbookReconstructionCache | None = None,
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
        audit_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=_reconstruction_cache,
        ),
        output_path,
        column_tooltip=workbook_column_tooltip,
    )


def audit_review_workbook_sheets(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: _WorkbookReconstructionCache | None = None,
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
    reconstruction_cache = _reconstruction_cache or _WorkbookReconstructionCache(comparison_path)
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
    return (
        primary_sheet,
        *_shared_detail_sheets(
            findings,
            active_findings,
            primary_changes_table=primary_sheet.table,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
            data_issues=_data_issues,
            finding_audit_trail=_finding_audit_trail,
        ),
        *diagnostic_sheets,
    )


def _return_reconstruction_summary_sheets(
    reconstruction_cache: _WorkbookReconstructionCache,
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
            columns=_workbook_return_reconstruction_summary_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _return_reconstruction_sheets(
    reconstruction_cache: _WorkbookReconstructionCache,
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
            columns=_workbook_return_reconstruction_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _security_return_reconstruction_sheets(
    reconstruction_cache: _WorkbookReconstructionCache,
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
            columns=_workbook_security_return_reconstruction_columns(),
            labels=_workbook_column_labels(),
        ),
    )


def _portfolio_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    table_cache: _WorkbookTableCache,
    reconstruction_cache: _WorkbookReconstructionCache,
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
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        ),
        columns=_workbook_portfolio_changes_columns(),
        labels=labels,
    )


def _security_differences_sheet(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    table_cache: _WorkbookTableCache,
    reconstruction_cache: _WorkbookReconstructionCache,
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
            table_cache=table_cache,
            reconstruction_cache=reconstruction_cache,
        ),
        columns=_workbook_security_changes_columns(),
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
    reconstruction_cache: _WorkbookReconstructionCache,
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
            columns=_workbook_underlying_cause_columns(),
            labels=_workbook_column_labels(),
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
            labels=_workbook_column_labels(),
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
        key = _workbook_primary_key(row, comparison_level)
        if not _workbook_is_real_primary_key(key, comparison_level):
            continue
        explained = _number_or_none(row.get(_ESTIMATED_CAUSE_TOTAL)) or 0.0
        cause_total = cause_totals.get(key, 0.0)
        if abs(explained - cause_total) > 0.000000000001:
            raise PpaError(
                "SN-03 explanation invariant failed for "
                f"{_workbook_primary_key_text(key)}: causes total {cause_total:.12f} "
                f"does not equal Explained Difference {explained:.12f}.",
                999,
            )
        if row.get(_REVIEW_STATUS) != _STATUS_FULLY_EXPLAINED:
            continue
        performance_difference = _number_or_none(row.get(_PERFORMANCE_CHANGE)) or 0.0
        if abs(performance_difference - explained) > _WORKBOOK_UNEXPLAINED_TOLERANCE:
            raise PpaError(
                "SN-03 Fully Explained invariant failed for "
                f"{_workbook_primary_key_text(key)}: Performance Difference "
                f"{performance_difference:.12f} does not equal Explained Difference "
                f"{explained:.12f}.",
                999,
            )
        unexplained = _number_or_none(row.get(_UNEXPLAINED_CHANGE))
        if unexplained is not None and abs(unexplained) > _WORKBOOK_UNEXPLAINED_TOLERANCE:
            raise PpaError(
                "SN-03 Fully Explained invariant failed for "
                f"{_workbook_primary_key_text(key)}: Unexplained Difference "
                f"{unexplained:.12f} is not zero.",
                999,
            )

    expected_components = {
        (
            *_workbook_primary_key(row, comparison_level),
            _format_value(row.get(_pc_findings.SOURCE_COLUMN)),
        )
        for row in formula_rows
        if _number_or_none(row.get(_ESTIMATED_IMPACT)) is not None
    }
    observed_components: set[tuple[object, ...]] = set()
    for row in causes.iter_rows(named=True):
        components = _format_value(row.get(_WORKBOOK_RECONSTRUCTION_COMPONENTS))
        observed_components.update(
            (*_workbook_primary_key(row, comparison_level), component)
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
        _workbook_primary_key(row, comparison_level): round(
            _number_or_none(row.get(_ESTIMATED_CAUSE_TOTAL)) or 0.0,
            6,
        )
        for row in primary_changes.iter_rows(named=True)
        if _workbook_is_real_primary_key(
            _workbook_primary_key(row, comparison_level),
            comparison_level,
        )
    }
    rows = causes.to_dicts()
    indexes_by_key: dict[tuple[object, ...], list[int]] = {}
    for index, row in enumerate(rows):
        if _number_or_none(row.get(_ESTIMATED_IMPACT)) is None:
            continue
        indexes_by_key.setdefault(
            _workbook_primary_key(row, comparison_level),
            [],
        ).append(index)
    for key, target in target_by_key.items():
        indexes = indexes_by_key.get(key, [])
        displayed_total = round(
            sum(round(_number_or_none(rows[index].get(_ESTIMATED_IMPACT)) or 0.0, 6)
                for index in indexes),
            6,
        )
        residual = round(target - displayed_total, 6)
        if residual == 0.0 or not indexes:
            continue
        allocation_index = max(
            indexes,
            key=lambda index: abs(
                _number_or_none(rows[index].get(_ESTIMATED_IMPACT)) or 0.0
            ),
        )
        displayed_impact = round(
            _number_or_none(rows[allocation_index].get(_ESTIMATED_IMPACT)) or 0.0,
            6,
        )
        rows[allocation_index][_ESTIMATED_IMPACT] = round(displayed_impact + residual, 6)
    return pl.DataFrame(rows, schema=causes.schema, infer_schema_length=None)


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
        key = _workbook_primary_key(row, comparison_level)
        if not _workbook_is_real_primary_key(key, comparison_level):
            continue
        explained = round(_number_or_none(row.get(_ESTIMATED_CAUSE_TOTAL)) or 0.0, 6)
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
        impact = _number_or_none(row.get(_ESTIMATED_IMPACT))
        if impact is None:
            continue
        value = round(impact, 6) if displayed else impact
        key = _workbook_primary_key(row, comparison_level)
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
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
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
        reconstruction_cache=_resolved_reconstruction_cache(
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
                    _workbook_period_key(row),
                    0.0,
                ),
                _POSSIBLE_CAUSE_COMMENT: possible_cause_comments.get(
                    _workbook_period_key(row),
                    "",
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
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
) -> dict[tuple[object, object, object], float]:
    """Return explained difference totals from underlying input rows."""
    totals: dict[tuple[object, object, object], float] = {}
    active_keys = (
        table_cache.active_portfolio_keys()
        if table_cache is not None
        else _workbook_active_portfolio_period_keys(findings)
    )
    reconstruction_cache = _resolved_reconstruction_cache(
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
        else _workbook_portfolio_reconstruction_formula_rows(
            comparison_path,
            active_keys=active_keys,
            reconstruction_cache=reconstruction_cache,
        )
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
        ranked_rows=ranked_rows,
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


def _workbook_performance_change_row(
    row: Mapping[str, object],
) -> dict[str, object]:
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
    unexplained_display = None if review_status == _STATUS_FULLY_EXPLAINED else unexplained_change
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
    if _has_text(missing_inputs):
        return f"Missing YAML specifications: {_format_value(missing_inputs)}."
    underlying_estimated_total = _number_or_none(row.get("_underlying_estimated_total"))
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
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
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
        reconstruction_cache=_resolved_reconstruction_cache(
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
                        _workbook_security_period_key(row),
                        0.0,
                    ),
                    _POSSIBLE_CAUSE_COMMENT: possible_cause_comments.get(
                        _workbook_security_period_key(row),
                        "",
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
    ranked_rows: Sequence[Mapping[str, object]] | None = None,
    table_cache: _WorkbookTableCache | None = None,
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
) -> dict[tuple[object, object, object, object], float]:
    """Return security-level explained totals from underlying input rows."""
    totals: dict[tuple[object, object, object, object], float] = {}
    active_keys = (
        table_cache.active_security_keys()
        if table_cache is not None
        else _workbook_active_security_period_keys(findings)
    )
    reconstruction_cache = _resolved_reconstruction_cache(
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
        else _workbook_security_reconstruction_formula_rows(
            comparison_path,
            active_keys=active_keys,
            reconstruction_cache=reconstruction_cache,
        )
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
        ranked_rows=ranked_rows,
    ):
        if not _has_text(row.get(_pc_findings.SECURITY_ID)):
            continue
        key = _workbook_security_period_key(row)
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
        key = _workbook_primary_key(row, comparison_level)
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
        key = _workbook_primary_key(row, comparison_level)
        comment = _workbook_possible_cause_row_comment(row)
        if not comment:
            continue
        comments = possible_comments_by_key.setdefault(key, [])
        if comment not in comments:
            comments.append(comment)
    return {
        key: _workbook_possible_cause_comment(comments)
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
        if _workbook_primary_key(row, comparison_level) not in unresolved_keys:
            continue
        if not _workbook_is_possible_cause_row(row):
            continue
        possible_rows.append(dict(row))
    return possible_rows


def _workbook_is_possible_cause_row(row: Mapping[str, object]) -> bool:
    """Return whether an unestimated row is a possible residual cause."""
    if _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT)) is not None:
        return False
    if _workbook_has_additive_policy(row):
        return False
    field_key = (
        _format_value(row.get(_pc_findings.DATASET)),
        _format_value(row.get(_pc_findings.SOURCE_COLUMN)),
    )
    return field_key in _POSSIBLE_CAUSE_FIELDS


def _workbook_possible_cause_field_name(row: Mapping[str, object]) -> str:
    """Return ``dataset.field`` text for a possible-cause row."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if (dataset, source_column) not in _POSSIBLE_CAUSE_FIELDS:
        return ""
    return f"{dataset}.{source_column}"


def _workbook_possible_cause_row_comment(row: Mapping[str, object]) -> str:
    """Return a row-specific possible-cause sentence fragment."""
    field_name = _workbook_possible_cause_field_name(row)
    if not field_name:
        return ""

    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = _workbook_row_change_value(row)
    change_direction = _workbook_increased_or_decreased(change_value)
    change_amount = _workbook_change_amount_text(change_value)
    input_date = _format_value(row.get(_pc_findings.INPUT_DATE))
    if input_date:
        return (
            f"{security_prefix}{field_name} {change_direction} by "
            f"{change_amount} on {input_date}."
        )
    return f"{security_prefix}{field_name} {change_direction} by {change_amount}."


def _workbook_possible_cause_comment(comments: Sequence[str]) -> str:
    """Return summary-sheet possible-cause wording for unresolved periods."""
    if not comments:
        return ""
    if len(comments) == 1:
        return f"Possible cause: {comments[0]} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"
    return f"Possible causes: {' '.join(comments)} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"


def _workbook_portfolio_reconstruction_formula_rows(
    comparison_path: util.PathLike | None,
    *,
    active_keys: set[tuple[object, object, object]] | None = None,
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
) -> list[dict[str, object]]:
    """Return portfolio reconstruction formula rows for Performance Difference Causes.

    Notes:
        This pilot promotes exact formula-level effects only. The detailed
        ``Return Reconstruction Checks`` sheet remains the source for the
        underlying beginning value, ending value, flow, income, and denominator
        inputs.
    """
    reconstruction_cache = _resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    checks = reconstruction_cache.portfolio_checks()
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
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
) -> list[dict[str, object]]:
    """Return security reconstruction formula rows for Performance Difference Causes.

    Notes:
        This pilot promotes exact formula-level effects only. The detailed
        ``Security Return Checks`` sheet remains the source for the underlying
        beginning value, ending value, flow, and income components.
    """
    reconstruction_cache = _resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    checks = reconstruction_cache.security_checks(active_keys=active_keys)
    if checks.is_empty():
        return []

    rows: list[dict[str, object]] = []
    for row in checks.iter_rows(named=True):
        if active_keys is not None and _workbook_security_period_key(row) not in active_keys:
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
    *,
    table_cache: _WorkbookTableCache | None = None,
) -> set[tuple[object, object, object]]:
    """Return portfolio-period keys with reported portfolio performance differences."""
    summary = _with_period_review_key(
        table_cache.primary_coverage(PORTFOLIO_COMPARISON_LEVEL)
        if table_cache is not None
        else _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    return {_workbook_period_key(row) for row in summary.iter_rows(named=True)}


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
    return {_workbook_security_period_key(row) for row in summary.iter_rows(named=True)}


def _workbook_reconstruction_formula_rows_for_check(
    source_row: Mapping[str, object],
    *,
    row_factory: Callable[..., dict[str, object]],
) -> list[dict[str, object]]:
    numerator_b = _number_or_none(source_row.get(_pc_reconstruction.DERIVED_NUMERATOR_B))
    denominator_a = _number_or_none(source_row.get(_pc_reconstruction.DERIVED_DENOMINATOR_A))
    denominator_b = _number_or_none(source_row.get(_pc_reconstruction.DERIVED_DENOMINATOR_B))
    if (
        numerator_b is None
        or denominator_a is None
        or denominator_b is None
        or denominator_a == 0.0
        or denominator_b == 0.0
    ):
        return []

    denominator_effect = numerator_b * ((1.0 / denominator_b) - (1.0 / denominator_a))
    beginning_denominator_effect, weighted_flow_denominator_effect = (
        _workbook_denominator_component_effects(source_row, denominator_effect)
    )
    beginning_value_difference = _number_or_none(
        source_row.get(_pc_reconstruction.BEGIN_VALUE_DIFFERENCE)
    )
    ending_value_difference = _number_or_none(
        source_row.get(_pc_reconstruction.END_VALUE_DIFFERENCE)
    )
    net_flow_difference = _number_or_none(source_row.get(_pc_reconstruction.NET_FLOW_DIFFERENCE))
    income_difference = _number_or_none(source_row.get(_pc_reconstruction.INCOME_DIFFERENCE))
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
    beginning_value_difference = (
        _number_or_none(source_row.get(_pc_reconstruction.BEGIN_VALUE_DIFFERENCE)) or 0.0
    )
    weighted_flow_difference = (
        _number_or_none(source_row.get(_pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE)) or 0.0
    )
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
        _pc_findings.PORTFOLIO_ID: source_row.get(_pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID),
        _pc_findings.FROM_DATE: source_row.get(_pc_reconstruction.RECONSTRUCTION_FROM_DATE),
        _pc_findings.THRU_DATE: source_row.get(_pc_reconstruction.RECONSTRUCTION_THRU_DATE),
        _AS_OF_DATE: as_of_date,
        _USE: _USE_EXPLAINS_CHANGE,
        _CHANGE_LABEL: f"{role_label} changed",
        _DATASET_FIELD: f"{dataset}.{source_column}",
        _pc_findings.SECURITY_ID: None,
        _ROW_TYPE: _ROW_TYPE_FORMULA_INPUT,
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
        _WORKBOOK_RECONSTRUCTION_COMPONENTS: source_column,
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
        _pc_findings.PORTFOLIO_ID: source_row.get(_pc_reconstruction.RECONSTRUCTION_PORTFOLIO_ID),
        _pc_findings.FROM_DATE: source_row.get(_pc_reconstruction.RECONSTRUCTION_FROM_DATE),
        _pc_findings.THRU_DATE: source_row.get(_pc_reconstruction.RECONSTRUCTION_THRU_DATE),
        _AS_OF_DATE: as_of_date,
        _USE: _USE_EXPLAINS_CHANGE,
        _CHANGE_LABEL: f"{role_label} changed",
        _DATASET_FIELD: f"{dataset}.{source_column}",
        _pc_findings.SECURITY_ID: security_id,
        _ROW_TYPE: _ROW_TYPE_FORMULA_INPUT,
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
        _WORKBOOK_RECONSTRUCTION_COMPONENTS: source_column,
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
            "calculated return. This value is retained because it is an input "
            "to Modified Dietz."
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
    return f"{role_label} {_workbook_increased_or_decreased(difference)} by " f"{change_text}."


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
        if not _workbook_is_underlying_cause_row(row):
            continue
        estimated_impact = _number_or_none(row.get(_pc_explain.ESTIMATED_RETURN_IMPACT))
        if estimated_impact is None:
            continue
        selected_rows.append((dict(row), estimated_impact))
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

    security_period_keys = {_workbook_period_key(row) for row in security_rows}
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
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
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
    reconstruction_cache = _resolved_reconstruction_cache(
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
        formula_rows = _workbook_security_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_security_period_keys(findings),
            reconstruction_cache=reconstruction_cache,
        )
    else:
        formula_rows = _workbook_portfolio_reconstruction_formula_rows(
            comparison_path,
            active_keys=_workbook_active_portfolio_period_keys(findings),
            reconstruction_cache=reconstruction_cache,
        )
    formula_keys = {_workbook_primary_key(row, comparison_level) for row in formula_rows}
    ranked_rows = (
        table_cache.ranked_rows(comparison_level)
        if table_cache is not None
        else _workbook_ranked_changed_rows_for_level(
            findings,
            comparison_level=comparison_level,
        )
    )
    cash_security_matches = _workbook_cash_security_matches(
        ranked_rows,
        comparison_level=comparison_level,
    )
    (
        attributed_formula_source_rows,
        unallocated_formula_rows,
    ) = _workbook_formula_source_allocation_rows(
        ranked_rows,
        formula_rows,
        cash_security_matches=cash_security_matches,
        comparison_level=comparison_level,
    )
    attributed_source_keys = {
        _workbook_source_row_key(row, comparison_level) for row in attributed_formula_source_rows
    }
    fx_support_rows = _workbook_fx_support_rows(
        ranked_rows,
        attributed_formula_source_rows,
        comparison_level=comparison_level,
    )
    linked_fx_sources = {_workbook_fx_source_identity(row) for row in fx_support_rows}
    possible_cause_source_keys = {
        _workbook_source_row_key(row, comparison_level)
        for row in _workbook_possible_cause_rows(
            findings,
            unresolved_keys=unexplained_keys,
            comparison_level=comparison_level,
            ranked_rows=ranked_rows,
        )
    }
    for row in attributed_formula_source_rows:
        if _workbook_source_row_key(row, comparison_level) in possible_cause_source_keys:
            row = _workbook_mark_possible_cause_row(row, comparison_level)
        rows.append(_workbook_changed_item_row(row, comparison_path=comparison_path))
    rows.extend(dict(row) for row in unallocated_formula_rows)
    rows.extend(
        _workbook_changed_item_row(row, comparison_path=comparison_path) for row in fx_support_rows
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
        if _workbook_fx_source_identity(row) in linked_fx_sources:
            continue
        if has_formula_role and _workbook_is_underlying_cause_row(row):
            support_row = _workbook_formula_support_row(
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
        elif _workbook_is_underlying_cause_row(row):
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
            table_cache=table_cache,
        )
    )
    if not rows:
        original_table = _workbook_empty_changed_item_table()
    else:
        original_table = _workbook_sorted_table(
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


def _workbook_formula_source_allocation_rows(
    source_rows: Sequence[Mapping[str, object]],
    formula_rows: Sequence[Mapping[str, object]],
    *,
    cash_security_matches: Mapping[tuple[object, ...], object],
    comparison_level: str,
) -> tuple[list[dict[str, object]], list[Mapping[str, object]]]:
    """Return allocated source effects and necessarily visible formula rows.

    Notes:
        Candidate source rows and their bases are derived once per Modified
        Dietz formula input. Inputs without an allocatable source row remain in
        the second result so they cannot disappear from the cause sheet.
    """
    rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    unallocated: list[Mapping[str, object]] = []
    source_index = _workbook_formula_source_index(
        source_rows,
        comparison_level=comparison_level,
    )
    for formula_row in formula_rows:
        candidate_rows = _workbook_formula_source_candidates(
            source_index,
            formula_row,
            comparison_level=comparison_level,
        )
        bases = [
            _workbook_formula_source_basis(row, formula_row)
            for row in candidate_rows
        ]
        total_basis = sum(bases)
        if not candidate_rows or abs(total_basis) <= _WORKBOOK_UNEXPLAINED_TOLERANCE:
            unallocated.append(formula_row)
            continue

        estimated_impact = _number_or_none(formula_row.get(_ESTIMATED_IMPACT))
        if estimated_impact is None:
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
            existing_row[_pc_explain.ESTIMATED_RETURN_IMPACT] = (existing_impact or 0.0) + (
                additional_impact or 0.0
            )
            existing_components = _format_value(
                existing_row.get(_WORKBOOK_RECONSTRUCTION_COMPONENTS)
            )
            additional_components = _format_value(
                attributed_row.get(_WORKBOOK_RECONSTRUCTION_COMPONENTS)
            )
            existing_row[_WORKBOOK_RECONSTRUCTION_COMPONENTS] = "|".join(
                sorted(
                    {
                        component
                        for component in (
                            *existing_components.split("|"),
                            *additional_components.split("|"),
                        )
                        if component
                    }
                )
            )
    return list(rows_by_key.values()), unallocated


def _workbook_formula_source_index(
    source_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> _FormulaSourceIndex:
    """Index source rows once by their eligible formula component."""
    value_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    flow_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    income_rows: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    flow_categories = (
        {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
        if comparison_level == SECURITY_COMPARISON_LEVEL
        else {TRANSACTION_CATEGORY_EXTERNAL_FLOW}
    )
    for row in source_rows:
        owner = _workbook_formula_owner_key(row, comparison_level)
        dataset = row.get(_pc_findings.DATASET)
        source_column = row.get(_pc_findings.SOURCE_COLUMN)
        if (
            dataset == pc_cols.HOLDINGS
            and source_column
            in {
                pc_cols.MARKET_VALUE,
                pc_cols.BASE_MARKET_VALUE,
                pc_cols.ACCRUED,
                pc_cols.BASE_ACCRUED,
            }
            and not _workbook_has_evidence_only_policy(row)
        ):
            value_key = (*owner, _workbook_as_of_date(row))
            value_rows.setdefault(value_key, []).append(row)
        if (
            dataset != pc_cols.TRANSACTIONS
            or not _workbook_is_effective_transaction_amount(row)
        ):
            continue
        period_key = (
            *owner,
            row.get(_pc_findings.FROM_DATE),
            row.get(_pc_findings.THRU_DATE),
        )
        transaction_category = row.get(_pc_findings.TRANSACTION_CATEGORY)
        if transaction_category in flow_categories:
            flow_rows.setdefault(period_key, []).append(row)
        if transaction_category in {
            TRANSACTION_CATEGORY_FEE_EXPENSE,
            TRANSACTION_CATEGORY_INCOME,
        }:
            income_rows.setdefault(period_key, []).append(row)
    return _FormulaSourceIndex(value_rows, flow_rows, income_rows)


def _workbook_formula_owner_key(
    row: Mapping[str, object],
    comparison_level: str,
) -> tuple[object, ...]:
    """Return the ownership key used to connect source and formula rows."""
    portfolio_id = row.get(_pc_findings.PORTFOLIO_ID)
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return portfolio_id, row.get(_pc_findings.SECURITY_ID)
    return (portfolio_id,)


def _workbook_formula_source_candidates(
    source_index: _FormulaSourceIndex,
    formula_row: Mapping[str, object],
    *,
    comparison_level: str,
) -> Sequence[Mapping[str, object]]:
    """Return source rows that make up one reconstruction formula row."""
    formula_field = formula_row.get(_pc_findings.SOURCE_COLUMN)
    owner = _workbook_formula_owner_key(formula_row, comparison_level)
    if formula_field in {
        _RECONSTRUCTION_BEGINNING_VALUE_FIELD,
        _RECONSTRUCTION_ENDING_VALUE_FIELD,
    }:
        return source_index.value_rows.get(
            (*owner, formula_row.get(_AS_OF_DATE)),
            (),
        )
    period_key = (
        *owner,
        formula_row.get(_pc_findings.FROM_DATE),
        formula_row.get(_pc_findings.THRU_DATE),
    )
    if formula_field in {
        _RECONSTRUCTION_NET_FLOW_FIELD,
        _RECONSTRUCTION_WEIGHTED_FLOW_FIELD,
    }:
        return source_index.flow_rows.get(period_key, ())
    if formula_field == _RECONSTRUCTION_INCOME_FIELD:
        return source_index.income_rows.get(period_key, ())
    return ()


def _workbook_is_effective_transaction_amount(row: Mapping[str, object]) -> bool:
    """Return whether a row is the transaction amount used in base returns."""
    source_column = row.get(_pc_findings.SOURCE_COLUMN)
    if source_column == pc_cols.BASE_AMOUNT:
        return True
    return source_column == pc_cols.AMOUNT and not _workbook_has_evidence_only_policy(row)


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
    row_dict[_WORKBOOK_RECONSTRUCTION_COMPONENTS] = formula_row.get(
        _pc_findings.SOURCE_COLUMN
    )
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
    cash_holdings_by_period = _workbook_cash_holdings_by_period(source_rows)
    for row in source_rows:
        if (
            row.get(_pc_findings.DATASET) != pc_cols.TRANSACTIONS
            or row.get(_pc_findings.SOURCE_COLUMN) != pc_cols.AMOUNT
        ):
            continue
        cash_security_id = _workbook_matching_cash_security_id(
            row,
            cash_holdings_by_period.get(_workbook_cash_transaction_key(row), ()),
        )
        if cash_security_id:
            matches[_workbook_source_row_key(row, comparison_level)] = cash_security_id
    return matches


def _workbook_cash_holdings_by_period(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    """Index eligible cash holding rows once for transaction matching."""
    holdings_by_period: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in source_rows:
        if (
            row.get(_pc_findings.DATASET) != pc_cols.HOLDINGS
            or row.get(_pc_findings.SOURCE_COLUMN) != pc_cols.MARKET_VALUE
            or not _workbook_is_cash_security(row.get(_pc_findings.SECURITY_ID))
        ):
            continue
        holdings_by_period.setdefault(_workbook_cash_period_key(row), []).append(row)
    return holdings_by_period


def _workbook_cash_period_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the ownership, period, and as-of key for cash matching."""
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
        _workbook_as_of_date(row),
    )


def _workbook_cash_transaction_key(
    row: Mapping[str, object],
) -> tuple[object, ...]:
    """Return the cash-holding lookup key for a transaction source row."""
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
        row.get(_pc_findings.THRU_DATE),
    )


def _workbook_matching_cash_security_id(
    transaction_row: Mapping[str, object],
    cash_holding_rows: Sequence[Mapping[str, object]],
) -> object | None:
    """Return the matching cash holding security for a transaction amount row."""
    transaction_delta = _number_or_none(transaction_row.get(_pc_findings.DELTA_B_MINUS_A))
    if transaction_delta is None:
        return None
    matches = [
        row
        for row in cash_holding_rows
        if _workbook_same_amount(
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


def _workbook_fx_support_rows(
    source_rows: Sequence[Mapping[str, object]],
    attributed_rows: Sequence[Mapping[str, object]],
    *,
    comparison_level: str,
) -> list[dict[str, object]]:
    """Return changed FX rates linked to counted base-currency inputs."""
    support_rows: list[dict[str, object]] = []
    for row in source_rows:
        if not (
            row.get(_pc_findings.DATASET) == pc_cols.FX_RATES
            and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.FX_RATE
        ):
            continue
        rate_delta = _number_or_none(row.get(_pc_findings.DELTA_B_MINUS_A))
        local_exposure = _number_or_none(row.get(_pc_findings.IMPACT_INPUT_VALUE))
        if rate_delta is None or local_exposure is None:
            continue
        base_value_delta = rate_delta * local_exposure
        matching_rows = [
            candidate
            for candidate in attributed_rows
            if candidate.get(_pc_findings.PORTFOLIO_ID) == row.get(_pc_findings.PORTFOLIO_ID)
            and _workbook_as_of_date(candidate) == _workbook_as_of_date(row)
            and candidate.get(_pc_findings.DATASET) in {pc_cols.HOLDINGS, pc_cols.TRANSACTIONS}
            and candidate.get(_pc_findings.SOURCE_COLUMN)
            in {pc_cols.BASE_MARKET_VALUE, pc_cols.BASE_AMOUNT}
            and abs(
                (_number_or_none(candidate.get(_pc_findings.DELTA_B_MINUS_A)) or 0.0)
                - base_value_delta
            )
            <= 0.005
        ]
        targets_by_period: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
        for target in matching_rows:
            targets_by_period.setdefault(
                _workbook_primary_key(target, comparison_level), []
            ).append(target)
        for targets in targets_by_period.values():
            if len(targets) != 1:
                continue
            support_rows.append(_workbook_fx_support_row(row, targets[0]))
    return support_rows


def _workbook_fx_source_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return a period-independent identity for one changed FX-rate row."""
    if not (
        row.get(_pc_findings.DATASET) == pc_cols.FX_RATES
        and row.get(_pc_findings.SOURCE_COLUMN) == pc_cols.FX_RATE
    ):
        return ()
    return (
        row.get(_pc_findings.PORTFOLIO_ID),
        _workbook_as_of_date(row),
        row.get(_pc_findings.SNAPSHOT_A_VALUE),
        row.get(_pc_findings.SNAPSHOT_B_VALUE),
    )


def _workbook_fx_support_row(
    row: Mapping[str, object],
    target: Mapping[str, object],
) -> dict[str, object]:
    """Return a non-additive FX row linked to its counted base-currency input."""
    row_dict = _workbook_non_additive_row(row)
    row_dict[_pc_findings.FROM_DATE] = target.get(_pc_findings.FROM_DATE)
    row_dict[_pc_findings.THRU_DATE] = target.get(_pc_findings.THRU_DATE)
    row_dict[_REVIEW_KEY] = target.get(_REVIEW_KEY)
    row_dict[_WORKBOOK_FX_RATE_SUPPORTS_BASE_INPUT] = True
    row_dict[_WORKBOOK_FX_RATE_TARGET_FIELD] = (
        f"{target.get(_pc_findings.DATASET)}." f"{target.get(_pc_findings.SOURCE_COLUMN)}"
    )
    row_dict[_pc_findings.SECURITY_ID] = target.get(_pc_findings.SECURITY_ID)
    row_dict["_workbook_fx_rate_base_value_change"] = target.get(_pc_findings.DELTA_B_MINUS_A)
    return row_dict


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
        _workbook_primary_key(row, comparison_level) for row in underlying_rows
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
            "Review `supporting_files/source_detail.csv`. The difference may be due to "
            "missing source-data, source-file timing differences, or vendor "
            "methodology that does not match the YAML specifications."
        ),
        _REVIEW_GUIDANCE: (
            "No identifiable cause was found. Review "
            "`supporting_files/source_detail.csv`. "
            "The difference may be due to missing source-data, source-file timing "
            "differences, or vendor methodology that does not match the YAML "
            "specifications."
        ),
        _pc_findings.DATASET: _NO_UNDERLYING_CAUSE_DATASET,
        _pc_findings.SOURCE_COLUMN: None,
        _pc_findings.FINDING_CODE: None,
        _pc_explain.REVIEW_RANK: 999999,
        _USE_PRIORITY: _workbook_use_priority(_USE_DIAGNOSTIC),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_unexplained_primary_keys(
    findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str,
    primary_changes_table: pl.DataFrame | None = None,
    reconstruction_cache: _WorkbookReconstructionCache | None = None,
) -> set[tuple[object, ...]]:
    """Return primary review keys with a meaningful unexplained remainder."""
    reconstruction_cache = _resolved_reconstruction_cache(
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
    if _workbook_is_split_factor_row(row):
        return (
            _workbook_split_holding_value_key(
                row,
                comparison_level=comparison_level,
            )
            in performance_input_keys
        )
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
        *_workbook_primary_key(row, comparison_level),
        row.get(_pc_findings.SECURITY_ID),
        "holding_value",
    )


def _workbook_split_factor_support_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return split-factor evidence marked as supporting holding value changes."""
    row_dict = dict(row)
    row_dict[_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING] = True
    return row_dict


def _workbook_possible_cause_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return evidence marked as a possible cause of an unresolved period."""
    row_dict = dict(row)
    row_dict[_WORKBOOK_POSSIBLE_CAUSE_ROW] = True
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
        row_dict[_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
        row_dict = _workbook_non_additive_row(row_dict)
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
    cash_security_matches = _workbook_cash_security_matches(
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
    return _workbook_sorted_table(
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
    enriched_row = _workbook_with_cash_balance_security(
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
        _WORKBOOK_CASH_BALANCE_SECURITY_ID
    ) and not _workbook_is_possible_cause_row(enriched_row):
        return enriched_row
    row_dict = dict(enriched_row)
    row_dict[_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION] = True
    return _workbook_non_additive_row(row_dict)


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
        _AS_OF_DATE,
        _DATASET_FIELD,
        _pc_findings.SECURITY_ID,
    )


def _workbook_raw_audit_columns(findings: pl.DataFrame) -> tuple[str, ...]:
    """Return source-detail presentation columns with review key last."""
    preferred_columns = (
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _AS_OF_DATE,
        _DATASET_FIELD,
        _pc_findings.SECURITY_ID,
        _pc_findings.SNAPSHOT_A_VALUE,
        _pc_findings.SNAPSHOT_B_VALUE,
        _CHANGE,
        _REVIEW_GUIDANCE,
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
            _ESTIMATED_IMPACT,
            _REVIEW_KEY,
        }
    ]
    return (*preferred_columns, *remaining_columns, _REVIEW_KEY)


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
    row_kind = _workbook_row_kind(row)
    row_use = _workbook_row_use(row, row_kind)
    impact_status = _workbook_impact_status(row, estimated_impact, row_kind)
    input_role = _workbook_input_role(row, estimated_impact, row_kind)
    return {
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _AS_OF_DATE: _workbook_as_of_date(row),
        _USE: row_use,
        _CHANGE_LABEL: _workbook_change_label(row),
        _DATASET_FIELD: _workbook_dataset_field(row),
        _pc_findings.SECURITY_ID: row.get(_pc_findings.SECURITY_ID),
        _ROW_TYPE: _workbook_row_type(
            row,
            estimated_impact,
            row_use,
            impact_status,
            input_role,
        ),
        _pc_findings.SNAPSHOT_A_VALUE: row.get(_pc_findings.SNAPSHOT_A_VALUE),
        _pc_findings.SNAPSHOT_B_VALUE: row.get(_pc_findings.SNAPSHOT_B_VALUE),
        _CHANGE: row.get(_pc_findings.DELTA_B_MINUS_A),
        _pc_findings.IMPACT_INPUT_VALUE: row.get(_pc_findings.IMPACT_INPUT_VALUE),
        _ESTIMATED_IMPACT: estimated_impact,
        _INPUT_ROLE: input_role,
        _IMPACT_STATUS: impact_status,
        _REVIEW_NOTE: _workbook_review_note(row, estimated_impact, row_use, impact_status),
        _REVIEW_GUIDANCE: _workbook_review_guidance(
            row,
            estimated_impact,
            comparison_path=comparison_path,
            impact_status=impact_status,
            row_kind=row_kind,
        ),
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
        _USE_PRIORITY: _workbook_use_priority(row_use),
        _WORKBOOK_RECONSTRUCTION_COMPONENTS: row.get(
            _WORKBOOK_RECONSTRUCTION_COMPONENTS
        ),
        _REVIEW_KEY: row.get(_REVIEW_KEY),
    }


def _workbook_row_type(
    row: Mapping[str, object],
    estimated_impact: float | None,
    row_use: str,
    impact_status: str,
    input_role: str,
) -> str:
    """Return the reviewer-facing row type for Performance Difference Causes."""
    if row.get(_WORKBOOK_POSSIBLE_CAUSE_ROW):
        return _ROW_TYPE_POSSIBLE_CAUSE
    if row.get(_pc_findings.FINDING_CODE) == _RECONSTRUCTION_FORMULA_FINDING_CODE:
        return _ROW_TYPE_FORMULA_INPUT
    if estimated_impact is not None:
        return _ROW_TYPE_EXPLAINED_CAUSE
    if (
        row.get(_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING)
        or row.get(_WORKBOOK_FX_RATE_SUPPORTS_BASE_INPUT)
        or row.get(_WORKBOOK_TRANSACTION_FLOW_SUPPORTS_HOLDING)
        or row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION)
        or row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW)
        or row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE)
        or impact_status == _IMPACT_STATUS_REVIEW_ONLY
        or input_role == _INPUT_ROLE_SUPPORTING_EVIDENCE
    ):
        return _ROW_TYPE_SUPPORTING_EVIDENCE
    if row_use == _USE_REVIEW_CONTEXT:
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
    if row.get(_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _INPUT_ROLE_SUPPORTING_EVIDENCE
    if _field_roles.is_input_component(
        dataset, source_column
    ) or _field_roles.is_performance_input(dataset, source_column):
        return _INPUT_ROLE_INPUT_DRIVER
    if row_kind == _WORKBOOK_ROW_KIND_DIAGNOSTIC:
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


def _workbook_row_use(row: Mapping[str, object], row_kind: str) -> str:
    """Return how a changed item should be used during review."""
    if row_kind == _WORKBOOK_ROW_KIND_DIAGNOSTIC:
        return _USE_DIAGNOSTIC
    if row.get(_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _USE_EXPLAINS_CHANGE
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
    row_kind: str,
) -> str:
    """Return a compact status for row-level impact treatment."""
    if estimated_impact is not None:
        return _IMPACT_STATUS_ESTIMATED
    if (
        row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE)
        or row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION)
        or row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW)
        or row_kind in {
            _WORKBOOK_ROW_KIND_CONTEXT,
            _WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC,
            _WORKBOOK_ROW_KIND_DIAGNOSTIC,
        }
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
        return _workbook_performance_dataset_review_note(source_column)
    source_explanation = _workbook_source_row_explanation(row, dataset, source_column)
    if source_explanation:
        return source_explanation
    if _workbook_has_evidence_only_policy(row):
        return (
            "Review-only evidence; this row is not counted in "
            '"Performance Differences" or "Explained Difference".'
        )
    if row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        return "Supporting evidence for Modified Dietz flow rows; not counted " "separately."
    if row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
        return "Supporting evidence for Modified Dietz flow rows; not counted " "separately."
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
    if dataset == pc_cols.SPLITS and source_column == pc_cols.SPLIT_FACTOR:
        return _workbook_split_factor_explanation(row)
    return ""


def _workbook_performance_dataset_review_note(source_column: str) -> str:
    """Return review guidance for reported performance-extract rows."""
    if source_column in {pc_cols.PORTFOLIO_RETURN, pc_cols.SECURITY_RETURN}:
        return (
            "Reported return residual; no supported source-data row explains " "this difference."
        )
    if source_column in {
        pc_cols.BEGIN_MARKET_VALUE,
        pc_cols.END_MARKET_VALUE,
        pc_cols.FLOW,
        pc_cols.INCOME,
    }:
        return "Performance-extract input; not a separate additive cause."
    return "Performance-extract diagnostic; not a separate additive cause."


def _workbook_review_guidance(
    row: Mapping[str, object],
    estimated_impact: float | None,
    *,
    comparison_path: util.PathLike | None,
    impact_status: str,
    row_kind: str,
) -> str:
    """Return review guidance for why this row does or does not explain performance."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if estimated_impact is not None:
        if row.get(_WORKBOOK_TRANSACTION_SUPPORTS_RECONSTRUCTION_FLOW):
            return _workbook_transaction_reconstruction_flow_guidance(row)
        if dataset == pc_cols.HOLDINGS and source_column in {
            pc_cols.ACCRUED,
            pc_cols.BASE_ACCRUED,
            pc_cols.MARKET_VALUE,
            pc_cols.BASE_MARKET_VALUE,
            pc_cols.PRICE,
            pc_cols.QUANTITY,
        }:
            return _workbook_holding_detail_explanation(row, source_column)
        if dataset == pc_cols.TRANSACTIONS:
            return _workbook_transaction_component_explanation(row, source_column)
        return ""

    if dataset == pc_cols.TRANSACTIONS and source_column in {
        pc_cols.COMMISSION,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _workbook_transaction_component_explanation(row, source_column)
    if row.get(_WORKBOOK_POSSIBLE_CAUSE_ROW):
        return _workbook_possible_cause_review_guidance(row, dataset, source_column)
    if (
        _workbook_has_additive_policy(row)
        and impact_status == _IMPACT_STATUS_MISSING_INPUT
    ):
        return _workbook_missing_impact_input_setup(dataset, source_column)
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
        return "Transaction activity changed. The holding value row shows the counted " "effect."
    if row.get(_WORKBOOK_SPLIT_FACTOR_SUPPORTS_HOLDING):
        return _workbook_split_factor_explanation(row)
    if row.get(_WORKBOOK_FX_RATE_SUPPORTS_BASE_INPUT):
        return _workbook_fx_rate_support_explanation(row)
    if row.get(_WORKBOOK_UNSELECTED_RELATED_ESTIMATE):
        return _workbook_related_input_guidance(row, dataset, source_column)
    if _workbook_has_evidence_only_policy(row):
        return (
            "Review-only evidence; this row is not counted in "
            '"Performance Differences" or "Explained Difference".'
        )
    if row_kind in {
        _WORKBOOK_ROW_KIND_CONTEXT,
        _WORKBOOK_ROW_KIND_REPORTED_DIAGNOSTIC,
        _WORKBOOK_ROW_KIND_DIAGNOSTIC,
    }:
        return "Review context; not an underlying input difference."

    dataset_column = _workbook_dataset_column_label(dataset, source_column)
    yaml_path = _workbook_yaml_path_label(comparison_path)
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.BASE_ACCRUED,
        pc_cols.MARKET_VALUE,
        pc_cols.BASE_MARKET_VALUE,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _workbook_holding_detail_explanation(row, source_column)
    if dataset == pc_cols.TRANSACTIONS:
        if source_column == pc_cols.AMOUNT:
            return _workbook_source_row_explanation(row, dataset, source_column)
        return f"No supported YAML impact method exists yet for {dataset_column}."
    if _workbook_has_additive_policy(row):
        return _workbook_missing_impact_input_setup(dataset, source_column)
    if dataset == pc_cols.HOLDINGS:
        if source_column not in {
            pc_cols.MARKET_VALUE,
            pc_cols.ACCRUED,
            pc_cols.BASE_ACCRUED,
            pc_cols.QUANTITY,
        }:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        if source_column in {pc_cols.ACCRUED, pc_cols.BASE_ACCRUED}:
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
    if dataset == pc_cols.FX_RATES:
        if source_column != pc_cols.FX_RATE:
            return f"No supported YAML impact method exists yet for {dataset_column}."
        return f"Specify the YAML fx_rate_impact_methods.fx_rate.method in {yaml_path}."
    return f"No supported YAML impact method exists yet for {dataset_column}."


def _workbook_related_input_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return explicit guidance for an input component's related performance field."""
    if dataset == pc_cols.HOLDINGS and source_column in {
        pc_cols.ACCRUED,
        pc_cols.BASE_ACCRUED,
        pc_cols.MARKET_VALUE,
        pc_cols.BASE_MARKET_VALUE,
        pc_cols.PRICE,
        pc_cols.QUANTITY,
    }:
        return _workbook_holding_detail_explanation(row, source_column)
    if dataset == pc_cols.TRANSACTIONS:
        return _workbook_transaction_component_explanation(row, source_column)
    return "Review-only supporting evidence for the related counted row."


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


def _workbook_possible_cause_review_guidance(
    row: Mapping[str, object],
    dataset: str,
    source_column: str,
) -> str:
    """Return concise guidance for evidence that may explain a residual."""
    if row.get(_WORKBOOK_NON_ADDITIVE_PORTFOLIO_TRANSACTION):
        explanation = _workbook_transaction_cash_balance_explanation(row)
    elif dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.AMOUNT:
        explanation = _workbook_transaction_amount_possible_cause_explanation(row)
    else:
        explanation = _workbook_source_row_explanation(row, dataset, source_column)
    if not explanation:
        explanation = _workbook_possible_cause_row_comment(row)
    return f"{explanation} {_POSSIBLE_CAUSE_CONFIGURATION_NOTE}"


def _workbook_transaction_amount_possible_cause_explanation(
    row: Mapping[str, object],
) -> str:
    """Return compact possible-cause wording for transaction amount changes."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    change_value = _workbook_row_change_value(row)
    return (
        f"{_workbook_transaction_code_prefix(row)}{security_prefix}"
        "transactions.amount "
        f"{_workbook_increased_or_decreased(change_value)} by "
        f"{_workbook_change_amount_text(change_value)}."
    )


def _workbook_portfolio_external_flow_transaction_explanation(
    row: Mapping[str, object],
) -> str:
    """Return source-data wording for a portfolio external-flow transaction."""
    flow_delta = _workbook_row_change_value(row)
    weighted_flow_delta = (_number_or_none(flow_delta) or 0.0) * _workbook_source_flow_weight(row)
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
    if timing_label == "beginning":
        return (
            "Inherited beginning-value difference from the preceding period: "
            f"{security_prefix}{holdings_label}.{source_column} "
            f"{_workbook_increased_or_decreased(change_value)} by {change_text}. "
            "This value is retained because it is an input to Modified Dietz."
        )
    return (
        f"{security_prefix}{holdings_label}.{source_column} "
        f"{_workbook_increased_or_decreased(change_value)} by "
        f"{change_text}."
    )


def _workbook_fx_rate_support_explanation(row: Mapping[str, object]) -> str:
    """Return an FX-rate explanation linked to the counted base value."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    target_field = _format_value(row.get(_WORKBOOK_FX_RATE_TARGET_FIELD))
    base_value_change = row.get("_workbook_fx_rate_base_value_change")
    snapshot_a = _number_or_none(row.get(_pc_findings.SNAPSHOT_A_VALUE))
    snapshot_b = _number_or_none(row.get(_pc_findings.SNAPSHOT_B_VALUE))
    from_currency = _format_value(row.get(_pc_findings.FROM_CURRENCY))
    to_currency = _format_value(row.get(_pc_findings.TO_CURRENCY))
    pair_prefix = (
        f"{from_currency}-to-{to_currency} FX rate"
        if from_currency and to_currency
        else "FX rate"
    )
    quote_suffix = (
        f" {to_currency} per {from_currency}"
        if from_currency and to_currency
        else ""
    )
    if snapshot_a is None or snapshot_b is None:
        rate_change = "changed"
    else:
        rate_change = f"changed from {snapshot_a:g} to {snapshot_b:g}"
    return (
        f"{pair_prefix} {rate_change}{quote_suffix}; "
        f"{security_prefix}{target_field} shows the counted {to_currency or 'base-currency'} "
        f"effect of {_workbook_change_amount_text(base_value_change)}."
    )


def _workbook_split_factor_explanation(row: Mapping[str, object]) -> str:
    """Return plain-language explanation for split-factor support rows."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_prefix = f"{security_id} " if security_id else ""
    split_factor = _workbook_row_change_value(row)
    return (
        f"split: Caused {security_prefix}holdings.quantity and related "
        "holdings.market_value to increase using a "
        f"{_workbook_split_factor_text(split_factor)} split factor."
    )


def _workbook_split_factor_text(value: object) -> str:
    """Return compact split-factor text for workbook explanations."""
    number = _number_or_none(value)
    if number is None:
        return "changed"
    if float(number).is_integer():
        return f"{abs(number):.1f}"
    return f"{abs(number):g}"


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


def _workbook_is_carry_forward_holding_input(row: Mapping[str, object]) -> bool:
    """Return whether a row is a prior-period holding value carried forward."""
    return (
        row.get(_pc_findings.DATASET) == pc_cols.HOLDINGS
        and row.get(_pc_findings.SOURCE_COLUMN)
        in {pc_cols.MARKET_VALUE, pc_cols.BASE_MARKET_VALUE}
        and _workbook_holding_timing_label(row) == "beginning"
    )


def _workbook_transaction_component_explanation(
    row: Mapping[str, object],
    source_column: str,
) -> str:
    """Return plain-language explanation for transaction component rows."""
    security_id = _format_value(row.get(_pc_findings.SECURITY_ID))
    security_text = f" for {security_id}" if security_id else ""
    field_text = f"transactions.{source_column}"
    if source_column == pc_cols.COMMISSION:
        change_value = _workbook_row_change_value(row)
        change_number = _number_or_none(change_value)
        change_verb = "decrease" if change_number is not None and change_number < 0 else "increase"
        transaction_amount = (
            f"{security_id} transactions.amount" if security_id else "transactions.amount"
        )
        return (
            f"{_workbook_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb} "
            f"by {_workbook_change_amount_text(change_value)}."
        )
    if source_column in {pc_cols.PRICE, pc_cols.QUANTITY}:
        change_value = _workbook_row_change_value(row)
        change_number = _number_or_none(change_value)
        change_verb = "decrease" if change_number is not None and change_number < 0 else "increase"
        transaction_amount = (
            f"{security_id} transactions.amount" if security_id else "transactions.amount"
        )
        if source_column == pc_cols.QUANTITY:
            quantity_effect = _workbook_transaction_quantity_holding_effect(row)
            if quantity_effect:
                holdings_quantity = (
                    f"{security_id} holdings.quantity" if security_id else "holdings.quantity"
                )
                return (
                    f"{_workbook_transaction_code_prefix(row)}Caused "
                    f"{transaction_amount} to {change_verb} and "
                    f"{holdings_quantity} to {quantity_effect}."
                )
        return (
            f"{_workbook_transaction_code_prefix(row)}Caused {transaction_amount} "
            f"to {change_verb}."
        )
    if (
        source_column == pc_cols.AMOUNT
        and row.get(_pc_findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_EXTERNAL_FLOW
    ):
        field_text = "external flow"
    elif (
        source_column == pc_cols.AMOUNT
        and row.get(_pc_findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_FEE_EXPENSE
    ):
        field_text = "fee/expense"
    elif (
        source_column == pc_cols.AMOUNT
        and row.get(_pc_findings.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_INCOME
    ):
        field_text = "income"
    return (
        f"{_workbook_transaction_code_prefix(row)}The {field_text}"
        f"{security_text} changed by "
        f"{_workbook_change_amount_text(_workbook_row_change_value(row))}."
    )


def _workbook_transaction_quantity_holding_effect(
    row: Mapping[str, object],
) -> str:
    """Return buy/sell holding quantity direction for a transaction quantity row."""
    change_number = _number_or_none(_workbook_row_change_value(row))
    if change_number is None:
        return ""
    transaction_code = _format_value(row.get(_pc_findings.TRANSACTION_CODE)).lower()
    if transaction_code in {"ss", "cs"}:
        return ""
    transaction_category = row.get(_pc_findings.TRANSACTION_CATEGORY)
    if transaction_category == TRANSACTION_CATEGORY_BUY:
        return "decrease" if change_number < 0 else "increase"
    if transaction_category == TRANSACTION_CATEGORY_SELL:
        return "increase" if change_number < 0 else "decrease"
    return ""


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
            _DATASET_FIELD: pl.String,
            _pc_findings.SECURITY_ID: pl.String,
            _ROW_TYPE: pl.String,
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
            _pc_findings.SOURCE_RECORD_LOCATOR: pl.String,
            _pc_findings.SOURCE_COLUMN: pl.String,
            _pc_findings.FINDING_CODE: pl.String,
            _pc_findings.TRANSACTION_CODE: pl.String,
            _pc_findings.TRANSACTION_CATEGORY: pl.String,
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
        _data_issue_checks.SNAPSHOT: "Snapshot",
        _data_issue_checks.ISSUE_TYPE: "Issue Type",
        _data_issue_checks.VALUE_A: "Reference Value",
        _data_issue_checks.VALUE_B: "Observed Value",
        _data_issue_checks.DIFFERENCE: "Difference",
        _data_issue_checks.TOLERANCE: "Tolerance",
        _pc_findings.FROM_DATE: "From Date",
        _pc_findings.THRU_DATE: "Thru Date",
        _PERFORMANCE_CHANGE: "Performance Difference",
        _ESTIMATED_CAUSE_TOTAL: "Explained Difference",
        _UNEXPLAINED_CHANGE: "Unexplained Difference",
        _AS_OF_DATE: "As Of Date",
        _USE: "Purpose",
        _CHANGE_LABEL: "What Changed",
        _DATASET_FIELD: "Dataset.Field",
        _ROW_TYPE: "Row Type",
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
        _pc_findings.TRANSACTION_MATCH_STATUS: "Transaction Match Status",
        _pc_explain.TRANSACTION_MATCH_CONFIDENCE: "Match Confidence",
        _pc_explain.TRANSACTION_MATCH_INTERPRETATION: "Match Interpretation",
        _pc_explain.TRANSACTION_MATCH_REVIEW_NOTE: "Review Note",
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
        _pc_reconstruction.RECONSTRUCTION_DIFFERENCE: ("Reconstruction Difference"),
        _pc_reconstruction.DERIVED_NUMERATOR_A: "Derived Numerator A",
        _pc_reconstruction.DERIVED_NUMERATOR_B: "Derived Numerator B",
        _pc_reconstruction.DERIVED_NUMERATOR_DIFFERENCE: ("Derived Numerator Difference"),
        _pc_reconstruction.DERIVED_DENOMINATOR_A: "Derived Denominator A",
        _pc_reconstruction.DERIVED_DENOMINATOR_B: "Derived Denominator B",
        _pc_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE: ("Derived Denominator Difference"),
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
        _pc_reconstruction.WEIGHTED_FLOW_DIFFERENCE: ("Weighted Flow Difference"),
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
        _REVIEW_KEY: ("Stable performance-period key used to connect workbook rows."),
        _pc_findings.PORTFOLIO_ID: "Portfolio identifier from the compared source-data.",
        _pc_findings.FROM_DATE: "Beginning date of the affected performance period.",
        _pc_findings.THRU_DATE: "Ending date of the affected performance period.",
        _pc_findings.SECURITY_ID: "Security identifier, when the discrepancy is security-level.",
        _data_issue_checks.SNAPSHOT: "Snapshot whose internal source-data is being checked.",
        _data_issue_checks.ISSUE_TYPE: "Type of cross-reference consistency issue.",
        _data_issue_checks.VALUE_A: ("Expected value or minimum rate found for this consistency check."),
        _data_issue_checks.VALUE_B: ("Observed value or maximum rate found for this consistency check."),
        _data_issue_checks.DIFFERENCE: (
            "Observed value minus expected value, or maximum rate minus minimum rate."
        ),
        _data_issue_checks.TOLERANCE: (
            "Configured threshold before the consistency check raises an issue."
        ),
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
        _DATASET_FIELD: (
            "Changed input field, shown as dataset.field. In detailed datasets, "
            "unqualified monetary fields use the row currency and base_ fields "
            "use portfolio base currency. Portfolio-performance monetary fields "
            "are inherently base-currency values."
        ),
        _ROW_TYPE: ("Internal reviewer role used for row coloring and sorting."),
        _CHANGE: "Snapshot B value minus snapshot A value for the compared item.",
        _AS_OF_DATE: ("Date represented by the input row. Holding rows use the period Thru Date."),
        _ESTIMATED_IMPACT: (
            "Decimal performance difference explained by this underlying " "input row."
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
        _pc_findings.DATASET: ("Normalized dataset where the source-data discrepancy was found."),
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
        _pc_explain.TRANSACTION_MATCH_CONFIDENCE: (
            "Reviewer-facing confidence tier for interpreting transaction row identity."
        ),
        _pc_explain.TRANSACTION_MATCH_INTERPRETATION: (
            "Short description of what the match status permits reviewers to infer."
        ),
        _pc_explain.TRANSACTION_MATCH_REVIEW_NOTE: (
            "Plain-language note explaining the transaction match status."
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
