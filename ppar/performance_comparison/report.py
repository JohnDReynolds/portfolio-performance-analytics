"""Render performance comparison findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.errors import PpaError
from ppar.performance_comparison import bundle as _pc_bundle
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import rendering as _pc_rendering
from ppar.performance_comparison import review_keys as _pc_review_keys
from ppar.performance_comparison import review_model as _pc_review_model
from ppar.performance_comparison import runner as _pc_runner
from ppar.performance_comparison import workbook as _pc_workbook
from ppar.performance_comparison import workbook_tables as _pc_workbook_tables
from ppar.performance_comparison.specification import PORTFOLIO_COMPARISON_LEVEL

__all__ = [
    "write_performance_comparison_report_bundle",
    "write_performance_comparison_review_workbook",
]

_COUNT = "count"
_ESTIMATED_IMPACT_AREAS = "estimated_impact_areas"
_RESIDUAL_STATUS = "residual_status"
_RESIDUAL_REASON = "residual_reason"
_RESIDUAL_REVIEW_NOTE = "residual_review_note"
_RESIDUAL_WITHHELD_PREFIX = "withheld"
_RESIDUAL_WITHHELD_NO_ESTIMATES = "withheld_no_estimates"
_RESIDUAL_WITHHELD_PARTIAL_ESTIMATES = "withheld_partial_estimates"
_RESIDUAL_WITHHELD_CROSS_CHECKS_ONLY = "withheld_cross_checks_only"
_REVIEW_STATUS = "review_status"
_REVIEW_CUES = "review_cues"
_SUGGESTED_NEXT_STEP = "suggested_next_step"
_REVIEW_KEY = _pc_review_keys.REVIEW_KEY
_REVIEW_DETAIL_ARTIFACTS = "review_detail_artifacts"
_REVIEW_STATUS_NEEDS_REVIEW = "needs_review"
_REVIEW_STATUS_MONITOR = "monitor"
_REVIEW_STATUS_CLEAR = "clear"
_NEEDS_REVIEW_COLUMNS = (
    _REVIEW_KEY,
    _pc_findings.PORTFOLIO_ID,
    _pc_findings.FROM_DATE,
    _pc_findings.THRU_DATE,
    _pc_explain.PORTFOLIO_RETURN_DELTA,
    _REVIEW_STATUS,
    _REVIEW_CUES,
    _SUGGESTED_NEXT_STEP,
    _REVIEW_DETAIL_ARTIFACTS,
)
_CONTEXT_USE = "context_use"
_REVIEW_PRIORITY = "review_priority"
_REVIEW_PRIORITY_REASON = "review_priority_reason"
_RETURN_IMPACT_TREATMENT = "return_impact_treatment"
_FINDING_COUNT = "finding_count"
_PORTFOLIO_COUNT = "portfolio_count"
_SECURITY_COUNT = "security_count"
_AFFECTED_PORTFOLIOS = "affected_portfolios"
_AFFECTED_SECURITIES = "affected_securities"
_CONTEXT_EVIDENCE_SUMMARY_COLUMNS = (
    _pc_findings.DATASET,
    _pc_findings.SOURCE_COLUMN,
    _CONTEXT_USE,
    _REVIEW_PRIORITY,
    _REVIEW_PRIORITY_REASON,
    _FINDING_COUNT,
    _PORTFOLIO_COUNT,
    _SECURITY_COUNT,
    _AFFECTED_PORTFOLIOS,
    _AFFECTED_SECURITIES,
)
_CONTEXT_EVIDENCE_COLUMNS = (
    _pc_findings.PORTFOLIO_ID,
    _pc_findings.SECURITY_ID,
    _pc_findings.FROM_DATE,
    _pc_findings.THRU_DATE,
    _pc_findings.DATASET,
    _pc_findings.FINDING_CODE,
    _pc_findings.SOURCE_COLUMN,
    _pc_findings.DELTA_B_MINUS_A,
    _CONTEXT_USE,
    _REVIEW_PRIORITY,
    _REVIEW_PRIORITY_REASON,
    _RETURN_IMPACT_TREATMENT,
    _pc_findings.MESSAGE,
)
_CONTEXT_NO_IMPACT_TREATMENT = "context only; not included in return-impact estimates"
_REVIEW_WORKBOOK_ARTIFACT = _pc_workbook.REVIEW_WORKBOOK_ARTIFACT
_REVIEW_WORKBOOK_FILE_NAME = _pc_workbook.REVIEW_WORKBOOK_FILE_NAME
_html_section = _pc_rendering.html_section
_html_review_key_row_id = _pc_rendering.html_review_key_row_id
_html_id_token = _pc_rendering.html_id_token
_html_empty = _pc_rendering.html_empty
_html_section_id = _pc_rendering.html_section_id
_display_header = _pc_rendering.display_header
_format_value = _pc_rendering.format_value
_comma_separated = _pc_rendering.comma_separated
_unique_nonblank_values = _pc_rendering.unique_nonblank_values
_escape_html = _pc_rendering.escape_html
_html_style_block = _pc_rendering.html_style_block
_period_key = _pc_review_keys.period_key
_period_review_key = _pc_review_keys.period_review_key
_row_review_key = _pc_review_keys.row_review_key
_with_period_review_key = _pc_review_keys.with_period_review_key
_with_security_review_key = _pc_review_keys.with_security_review_key


def _html_table(
    table: pl.DataFrame,
    columns: Sequence[str],
    *,
    empty_message: str = "No rows.",
    row_id_prefix: str | None = None,
) -> str:
    """Return an HTML table for selected columns."""
    return _pc_rendering.html_table(
        table,
        columns,
        empty_message=empty_message,
        row_id_prefix=row_id_prefix,
        row_id_callback=lambda row, prefix, counts: _html_table_row_id(
            row,
            row_id_prefix=prefix,
            row_id_counts=counts,
        ),
    )


def _performance_comparison_html_report(
    findings: pl.DataFrame,
    *,
    title: str = "Performance Comparison Report",
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: (
        _pc_workbook_tables._WorkbookReconstructionCache | None
    ) = None,
) -> str:
    """Return the workbook-style HTML report used inside review bundles.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        title: HTML document title and visible H1 text.
        comparison_path: Optional path to the comparison YAML. When provided,
            the ``Performance Difference Causes`` section can name the exact file to update
            for missing attribution setup.
        comparison_level: Primary performance-result level for presentation.
        include_reconstruction_diagnostics: Whether to include interim
            reconstruction diagnostic sections.

    Returns:
        Complete HTML document string suitable for writing to disk or opening
        in a browser.
    """
    sheets = _pc_workbook_tables.performance_comparison_review_workbook_sheets(
        findings,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=_reconstruction_cache,
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8"/>',
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
            f"<title>{_escape_html(title)}</title>",
            _html_style_block(),
            "</head>",
            "<body>",
            '<main class="pc-report">',
            '<header class="pc-header">',
            f"<h1>{_escape_html(title)}</h1>",
            "<p>Browser view for reviewing this performance-comparison bundle.</p>",
            "</header>",
            _html_workbook_contents_section(sheets),
            *[_html_workbook_sheet_section(sheet) for sheet in sheets],
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _html_workbook_contents_section(
    sheets: Sequence[_pc_workbook.ReviewWorkbookSheet],
) -> str:
    """Return navigation for workbook-style HTML sections."""
    primary_sheet_name = sheets[0].sheet_name if sheets else "the first sheet"
    items = [
        f'<li><a href="#{_html_section_id(sheet.sheet_name)}">'
        f"{_escape_html(sheet.sheet_name)}</a></li>"
        for sheet in sheets
    ]
    return _html_section(
        _pc_review_model.REVIEW_ORDER_SECTION,
        "\n".join(
            [
                f"<p>Start with {_escape_html(primary_sheet_name)}, then use "
                f"{_escape_html(_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET)} "
                "to see which source-data differences explain each period.</p>",
                '<ol class="pc-contents-list">',
                *items,
                "</ol>",
            ]
        ),
    )


def _html_workbook_sheet_section(sheet: _pc_workbook.ReviewWorkbookSheet) -> str:
    """Return one HTML section matching a review workbook sheet."""
    return _html_section(
        sheet.sheet_name,
        _html_workbook_sheet_table(sheet),
    )


def _html_workbook_sheet_table(sheet: _pc_workbook.ReviewWorkbookSheet) -> str:
    """Return an HTML table for one workbook sheet specification."""
    columns = _workbook_sheet_available_columns(sheet)
    if sheet.table.is_empty() or not columns:
        return _html_empty("No rows.")

    labels = sheet.labels or {}
    header_cells = [
        _html_workbook_header_cell(
            label=labels.get(column, _display_header(column)),
            tooltip=_pc_workbook_tables.workbook_column_tooltip(column),
        )
        for column in columns
    ]
    body_rows = [
        _html_workbook_body_row(row, columns)
        for row in sheet.table.select(columns).iter_rows(named=True)
    ]
    return "\n".join(
        [
            '<div class="pc-table-wrap">',
            f'<p class="pc-table-meta">Rows: {_escape_html(sheet.table.height)}</p>',
            '<table class="pc-table">',
            f"<caption>{_escape_html(sheet.sheet_name)}</caption>",
            "<thead>",
            "<tr>" + "".join(header_cells) + "</tr>",
            "</thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )


def _workbook_sheet_available_columns(
    sheet: _pc_workbook.ReviewWorkbookSheet,
) -> list[str]:
    """Return display columns present in a workbook sheet table."""
    requested_columns = sheet.columns or tuple(sheet.table.columns)
    return [column for column in requested_columns if column in sheet.table.columns]


def _html_workbook_header_cell(*, label: str, tooltip: str) -> str:
    """Return one workbook-style HTML header cell."""
    title_attribute = f' title="{_escape_html(tooltip)}"' if tooltip else ""
    return f'<th scope="col"{title_attribute}>{_escape_html(label)}</th>'


def _html_workbook_body_row(row: Mapping[str, object], columns: Sequence[str]) -> str:
    """Return one workbook-style HTML table row."""
    cells = [
        _pc_rendering.html_table_cell(row[column], column)
        for column in columns
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def _write_performance_comparison_html_report(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    title: str = "Performance Comparison Report",
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: (
        _pc_workbook_tables._WorkbookReconstructionCache | None
    ) = None,
) -> Path:
    """Write the bundle HTML performance comparison report to disk.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_path: Destination HTML report path. Parent directories are
            created when needed.
        title: HTML document title and visible H1 text.
        comparison_path: Optional path to the comparison YAML. When provided,
            the ``Performance Difference Causes`` section can name the exact file to update
            for missing attribution setup.
        comparison_level: Primary performance-result level for presentation.
        include_reconstruction_diagnostics: Whether to include interim
            reconstruction diagnostic sections.

    Returns:
        Normalized ``Path`` to the written report file.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _performance_comparison_html_report(
        findings,
        title=title,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=_reconstruction_cache,
    )
    report_path.write_text(report, encoding=util.ENCODING)
    return report_path


def write_performance_comparison_report_bundle(
    findings: pl.DataFrame,
    output_directory: util.PathLike,
    *,
    title: str = "Performance Comparison Report",
    top_evidence_limit: int = 10,
    include_workbook: bool = False,
    require_complete_yaml_setup: bool = True,
    require_causal_attribution: bool = False,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
) -> dict[str, Path]:
    """Write a reproducible report bundle.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_directory: Destination directory. It is created when needed.
        title: Report title for generated review artifacts.
        top_evidence_limit: Maximum number of top-evidence rows to include per
            portfolio period in ``top_evidence.csv``.
        include_workbook: Whether to include an XLSX review workbook.
        require_complete_yaml_setup: Whether every changed source-data field
            that ppar knows how to classify must have explicit additive,
            evidence-only, or suppression YAML before bundle artifacts are
            written.
        require_causal_attribution: Whether changed portfolio periods must have
            all YAML setup needed by supported attribution methods before
            writing bundle artifacts. This does not require every performance
            change to be fully explained.
        comparison_path: Optional path to the comparison YAML. When provided,
            the XLSX workbook can name the exact YAML file to update for
            missing attribution setup.
        comparison_level: Primary performance-result level for presentation.
        include_reconstruction_diagnostics: Whether to add the optional
            ``Reconstruction Summary``, ``Return Reconstruction Checks``, and
            ``Security Return Checks`` workbook/report sections plus matching
            CSV artifacts.

    Returns:
        Mapping from bundle artifact name to normalized written path.
    """
    if include_workbook:
        _pc_workbook.ensure_openpyxl_installed()

    active_findings = _active_findings(findings)
    if require_complete_yaml_setup:
        _pc_runner.validate_yaml_setup_complete(active_findings)
    if require_causal_attribution:
        _pc_runner.validate_causal_attribution_ready(active_findings)

    bundle_directory = Path(output_directory)
    bundle_directory.mkdir(parents=True, exist_ok=True)
    reconstruction_cache = _pc_workbook_tables._WorkbookReconstructionCache(
        comparison_path
    )
    tables = _report_bundle_tables(
        active_findings,
        top_evidence_limit,
        comparison_path=comparison_path,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=reconstruction_cache,
    )

    paths: dict[str, Path] = {}
    html_report_path = _write_performance_comparison_html_report(
        findings,
        bundle_directory / "report.html",
        title=title,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=reconstruction_cache,
    )
    paths["html_report"] = html_report_path
    paths["findings"] = _pc_bundle.write_csv_artifact(
        findings,
        bundle_directory / "findings.csv",
    )
    for name, table in tables.items():
        paths[name] = _pc_bundle.write_csv_artifact(
            table,
            bundle_directory / f"{name}.csv",
        )
    if include_workbook:
        paths[_REVIEW_WORKBOOK_ARTIFACT] = write_performance_comparison_review_workbook(
            findings,
            bundle_directory / _REVIEW_WORKBOOK_FILE_NAME,
            top_evidence_limit=top_evidence_limit,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=reconstruction_cache,
        )
    paths["readme"] = _pc_bundle.write_report_bundle_readme(
        bundle_directory / "README.md",
        title=title,
        tables=tables,
        include_workbook=include_workbook,
        comparison_level=comparison_level,
    )
    manifest_path = bundle_directory / "manifest.json"
    paths["manifest"] = manifest_path
    paths["review_summary"] = bundle_directory / "review_summary.json"
    _pc_bundle.write_report_bundle_manifest(
        manifest_path,
        findings=findings,
        active_findings=active_findings,
        title=title,
        top_evidence_limit=top_evidence_limit,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        comparison_path=comparison_path,
        artifact_paths=paths,
        tables=tables,
    )
    manifest_data = json.loads(manifest_path.read_text(encoding=util.ENCODING))
    manifest = {str(key): value for key, value in manifest_data.items()}
    _pc_bundle.write_report_bundle_review_summary(
        paths["review_summary"],
        manifest=manifest,
    )
    validation_issues = _pc_bundle.report_bundle_validation_issues(bundle_directory)
    if validation_issues:
        raise PpaError(
            "Report bundle validation failed: " + "; ".join(validation_issues),
            None,
        )
    return paths


def _report_bundle_tables(
    active_findings: pl.DataFrame,
    top_evidence_limit: int,
    *,
    comparison_path: util.PathLike | None = None,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: (
        _pc_workbook_tables._WorkbookReconstructionCache | None
    ) = None,
) -> dict[str, pl.DataFrame]:
    """Return report-bundle tables keyed by artifact stem."""
    reconstruction_cache = _reconstruction_cache or (
        _pc_workbook_tables._WorkbookReconstructionCache(comparison_path)
    )
    tables = {
        "needs_review_summary": _needs_review_summary_table(active_findings),
        "portfolio_period_summary": _pc_explain.portfolio_period_summary(
            active_findings
        ),
        "cause_summary": _pc_explain.portfolio_period_cause_summary(active_findings),
        "impact_estimates": _impact_estimate_summary_table(active_findings),
        "impact_coverage": _pc_explain.portfolio_period_impact_coverage_summary(
            active_findings
        ),
        "context_evidence_summary": _context_evidence_summary_table(active_findings),
        "context_evidence": _context_evidence_table(active_findings),
        "transaction_cross_checks": (
            _pc_explain.portfolio_period_transaction_cross_checks(active_findings)
        ),
        "flow_cross_check_reconciliation": (
            _pc_explain.portfolio_period_flow_cross_check_reconciliation(active_findings)
        ),
        "residual_status": _residual_status_table(active_findings),
        "transaction_activity": _pc_explain.transaction_activity_summary(active_findings),
        "transaction_matching_diagnostics": (
            _pc_explain.transaction_matching_diagnostics(active_findings)
        ),
        "top_evidence": _pc_explain.top_evidence_table(
            active_findings,
            top_evidence_limit,
        ),
    }
    if include_reconstruction_diagnostics:
        reconstruction_checks = reconstruction_cache.portfolio_checks()
        reconstruction_summary = reconstruction_cache.summary()
        if not reconstruction_summary.is_empty():
            tables[_pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT] = (
                reconstruction_summary
            )
        if not reconstruction_checks.is_empty():
            tables[_pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT] = (
                reconstruction_checks
            )
        security_reconstruction_checks = reconstruction_cache.security_checks()
        if not security_reconstruction_checks.is_empty():
            tables[
                _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT
            ] = security_reconstruction_checks
    return {
        name: _with_period_review_key(table)
        for name, table in tables.items()
    }


def write_performance_comparison_review_workbook(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    top_evidence_limit: int = 10,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: (
        _pc_workbook_tables._WorkbookReconstructionCache | None
    ) = None,
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
        comparison_level: Primary performance-result level for presentation.
        include_reconstruction_diagnostics: Whether to include interim
            reconstruction diagnostic sheets.

    Returns:
        Normalized workbook path.
    """
    return _pc_workbook_tables.write_performance_comparison_review_workbook(
        findings,
        output_path,
        top_evidence_limit=top_evidence_limit,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=_reconstruction_cache,
    )


def _needs_review_summary_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return reviewer-facing period cues derived from existing summary tables."""
    periods = _pc_explain.portfolio_period_summary(findings)
    if periods.is_empty():
        return _empty_needs_review_summary()

    coverage_by_period = _period_rows_by_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    residual_by_period = _period_rows_by_key(_residual_status_table(findings))
    cross_checks_by_period = _period_rows_by_key(
        _pc_explain.portfolio_period_transaction_cross_checks(findings)
    )
    context_by_period = _period_rows_by_key(_context_evidence_table(findings))
    rows = [
        _needs_review_summary_row(
            period=period,
            coverage=coverage_by_period.get(_period_key(period), []),
            residual=residual_by_period.get(_period_key(period), []),
            cross_checks=cross_checks_by_period.get(_period_key(period), []),
            context=context_by_period.get(_period_key(period), []),
        )
        for period in periods.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).select(_NEEDS_REVIEW_COLUMNS)


def _empty_needs_review_summary() -> pl.DataFrame:
    """Return an empty needs-review summary with stable columns."""
    return pl.DataFrame(
        schema={
            _REVIEW_KEY: pl.String,
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _pc_explain.PORTFOLIO_RETURN_DELTA: pl.Float64,
            _REVIEW_STATUS: pl.String,
            _REVIEW_CUES: pl.String,
            _SUGGESTED_NEXT_STEP: pl.String,
            _REVIEW_DETAIL_ARTIFACTS: pl.String,
        }
    )


def _needs_review_summary_row(
    *,
    period: dict[str, object],
    coverage: list[dict[str, object]],
    residual: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
    context: list[dict[str, object]],
) -> dict[str, object]:
    """Return one reviewer-cue row for a changed portfolio period."""
    cues = _needs_review_cues(
        coverage=coverage,
        residual=residual,
        cross_checks=cross_checks,
        context=context,
    )
    return {
        _REVIEW_KEY: _period_review_key(period),
        _pc_findings.PORTFOLIO_ID: period[_pc_findings.PORTFOLIO_ID],
        _pc_findings.FROM_DATE: period[_pc_findings.FROM_DATE],
        _pc_findings.THRU_DATE: period[_pc_findings.THRU_DATE],
        _pc_explain.PORTFOLIO_RETURN_DELTA: period[_pc_explain.PORTFOLIO_RETURN_DELTA],
        _REVIEW_STATUS: _needs_review_status(cues),
        _REVIEW_CUES: _comma_separated(cues),
        _SUGGESTED_NEXT_STEP: _suggested_next_step(cues),
        _REVIEW_DETAIL_ARTIFACTS: _review_detail_artifacts(
            coverage=coverage,
            residual=residual,
            cross_checks=cross_checks,
            context=context,
        ),
    }


def _needs_review_cues(
    *,
    coverage: list[dict[str, object]],
    residual: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
    context: list[dict[str, object]],
) -> list[str]:
    """Return deterministic reviewer cues for a portfolio period."""
    cues: list[str] = []
    coverage_row = coverage[0] if coverage else {}
    residual_row = residual[0] if residual else {}

    if _positive_count(coverage_row.get(_pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT)):
        cues.append(
            f"{_format_value(coverage_row[_pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT])} "
            "evidence-only area(s)"
        )
    if _has_text(coverage_row.get(_pc_explain.MISSING_IMPACT_INPUTS)):
        cues.append(
            f"missing inputs: "
            f"{_format_value(coverage_row[_pc_explain.MISSING_IMPACT_INPUTS])}"
        )
    if _positive_count(coverage_row.get(_pc_explain.LOW_CONFIDENCE_ESTIMATE_COUNT)):
        cues.append(
            f"{_format_value(coverage_row[_pc_explain.LOW_CONFIDENCE_ESTIMATE_COUNT])} "
            "low-confidence estimate(s)"
        )
    if cross_checks:
        total_count = sum(
            _count_value(row.get(_pc_explain.CROSS_CHECK_COUNT))
            for row in cross_checks
        )
        policies = _comma_separated(
            [
                str(row[_pc_explain.TRANSACTION_IMPACT_POLICIES])
                for row in cross_checks
                if _has_text(row.get(_pc_explain.TRANSACTION_IMPACT_POLICIES))
            ]
        )
        cross_check_cue = f"{_format_value(total_count)} transaction cross-check(s)"
        if policies:
            cross_check_cue = f"{cross_check_cue}: {policies}"
        cues.append(cross_check_cue)
    context_cue = _high_priority_context_cue(context)
    if context_cue:
        cues.append(context_cue)
    residual_status = residual_row.get(_RESIDUAL_STATUS)
    if _is_residual_withheld_status(residual_status):
        cues.append(f"residual {residual_status}")
    return cues


def _high_priority_context_cue(context: list[dict[str, object]]) -> str:
    """Return a period-level cue for context rows linked to changed periods."""
    if not context:
        return ""
    labels = sorted(
        {
            _context_source_label(row)
            for row in context
            if _has_text(row.get(_pc_findings.DATASET))
        }
    )
    if not labels:
        return ""
    return f"high-priority context: {_comma_separated(labels)}"


def _context_source_label(row: Mapping[str, object]) -> str:
    """Return a compact dataset/source label for context triage cues."""
    dataset = _format_value(row.get(_pc_findings.DATASET))
    source_column = _format_value(row.get(_pc_findings.SOURCE_COLUMN))
    if source_column:
        return f"{dataset}/{source_column}"
    return dataset


def _needs_review_status(cues: Sequence[str]) -> str:
    """Return the triage status for a period's reviewer cues."""
    if not cues:
        return _REVIEW_STATUS_CLEAR
    if any(
        cue.startswith(("missing inputs:", "residual withheld"))
        or cue.startswith("high-priority context:")
        or "evidence-only" in cue
        for cue in cues
    ):
        return _REVIEW_STATUS_NEEDS_REVIEW
    return _REVIEW_STATUS_MONITOR


def _suggested_next_step(cues: Sequence[str]) -> str:
    """Return conservative review guidance for a period's reviewer cues."""
    if any(cue.startswith("missing inputs:") for cue in cues):
        return "Resolve missing impact inputs before interpreting estimates."
    if any(cue.startswith("high-priority context:") for cue in cues):
        return "Review high-priority context evidence linked to the changed period."
    if any("evidence-only" in cue for cue in cues):
        return "Review evidence-only areas before relying on impact totals."
    if any("transaction cross-check" in cue for cue in cues):
        return "Review transaction cross-checks separately from impact totals."
    if any("low-confidence" in cue for cue in cues):
        return "Review low-confidence estimates and supporting evidence."
    if any(cue.startswith("residual withheld") for cue in cues):
        return "Keep residual caveat visible until estimates are complete."
    return "No changed portfolio-period review cue."


def _review_detail_artifacts(
    *,
    coverage: list[dict[str, object]],
    residual: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
    context: list[dict[str, object]],
) -> str:
    """Return bundle artifacts that help review a period-level triage row."""
    artifacts = ["portfolio_period_summary.csv", "cause_summary.csv"]
    coverage_row = coverage[0] if coverage else {}

    if coverage:
        artifacts.extend(["impact_coverage.csv", "impact_estimates.csv"])
    if _coverage_points_to_transaction_activity(coverage_row):
        artifacts.append("transaction_activity.csv")
    if cross_checks:
        artifacts.extend(
            [
                "transaction_cross_checks.csv",
                "flow_cross_check_reconciliation.csv",
            ]
        )
    if residual:
        artifacts.append("residual_status.csv")
    if context:
        artifacts.extend(["context_evidence_summary.csv", "context_evidence.csv"])
    artifacts.append("findings.csv")
    return _comma_separated(list(dict.fromkeys(artifacts)))


def _coverage_points_to_transaction_activity(coverage_row: Mapping[str, object]) -> bool:
    """Return whether impact coverage points reviewers to transaction activity."""
    evidence_only_areas = str(coverage_row.get(_pc_explain.EVIDENCE_ONLY_AREAS, ""))
    missing_inputs = str(coverage_row.get(_pc_explain.MISSING_IMPACT_INPUTS, ""))
    return (
        _pc_explain.ROOT_CAUSE_TRANSACTION_ACTIVITY in evidence_only_areas
        or "transaction" in missing_inputs
    )


def _period_rows_by_key(
    table: pl.DataFrame,
) -> dict[tuple[object, object, object], list[dict[str, object]]]:
    """Return table rows keyed by portfolio period."""
    rows_by_key: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    if table.is_empty():
        return rows_by_key
    for row in table.iter_rows(named=True):
        rows_by_key.setdefault(_period_key(row), []).append(row)
    return rows_by_key


def _positive_count(value: object) -> bool:
    """Return whether a value is a positive non-boolean number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _count_value(value: object) -> int:
    """Return a positive integer count for numeric report values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return int(value)
    return 0


def _has_text(value: object) -> bool:
    """Return whether a value has non-empty display text."""
    return bool(value is not None and str(value).strip())


def _is_residual_withheld_status(value: object) -> bool:
    """Return whether a residual status represents a withheld residual."""
    return isinstance(value, str) and value.startswith(_RESIDUAL_WITHHELD_PREFIX)


def _period_cause_rows(
    causes: pl.DataFrame,
    period: dict[str, object],
) -> list[dict[str, object]]:
    """Return cause-summary rows matching one portfolio period."""
    if causes.is_empty():
        return []
    period_causes = causes.filter(
        (pl.col(_pc_findings.PORTFOLIO_ID) == period[_pc_findings.PORTFOLIO_ID])
        & (pl.col(_pc_findings.FROM_DATE) == period[_pc_findings.FROM_DATE])
        & (pl.col(_pc_findings.THRU_DATE) == period[_pc_findings.THRU_DATE])
    )
    return list(period_causes.iter_rows(named=True))


def _residual_status_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return residual-status rows for portfolio-period return changes."""
    periods = _pc_explain.portfolio_period_summary(findings)
    if periods.is_empty():
        return pl.DataFrame(
            schema={
                _pc_findings.PORTFOLIO_ID: pl.String,
                _pc_findings.FROM_DATE: pl.Date,
                _pc_findings.THRU_DATE: pl.Date,
                _pc_explain.PORTFOLIO_RETURN_DELTA: pl.Float64,
                _ESTIMATED_IMPACT_AREAS: pl.String,
                _RESIDUAL_STATUS: pl.String,
                _RESIDUAL_REASON: pl.String,
                _RESIDUAL_REVIEW_NOTE: pl.String,
            }
        )

    causes = _pc_explain.portfolio_period_cause_summary(findings)
    cross_checks = _pc_explain.portfolio_period_transaction_cross_checks(findings)
    cross_checks_by_period = _period_rows_by_key(cross_checks)
    return pl.DataFrame(
        [
            _residual_status_row(
                period,
                _period_cause_rows(causes, period),
                cross_checks_by_period.get(_period_key(period), []),
            )
            for period in periods.iter_rows(named=True)
        ]
    )


def _residual_status_row(
    period: dict[str, object],
    causes: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
) -> dict[str, object]:
    """Return one residual-status row for a portfolio period."""
    estimated_areas = [
        str(cause[_pc_explain.ROOT_CAUSE_AREA])
        for cause in causes
        if cause.get(_pc_explain.ESTIMATED_RETURN_IMPACT) is not None
    ]
    cross_check_count = sum(
        _count_value(row.get(_pc_explain.CROSS_CHECK_COUNT))
        for row in cross_checks
    )
    if estimated_areas:
        status = _RESIDUAL_WITHHELD_PARTIAL_ESTIMATES
        reason = "partial or overlapping estimates"
    elif cross_check_count > 0:
        status = _RESIDUAL_WITHHELD_CROSS_CHECKS_ONLY
        reason = "transaction cross-checks only; no contribution estimates"
    else:
        status = _RESIDUAL_WITHHELD_NO_ESTIMATES
        reason = "no defensible impact estimates"

    return {
        _pc_findings.PORTFOLIO_ID: period[_pc_findings.PORTFOLIO_ID],
        _pc_findings.FROM_DATE: period[_pc_findings.FROM_DATE],
        _pc_findings.THRU_DATE: period[_pc_findings.THRU_DATE],
        _pc_explain.PORTFOLIO_RETURN_DELTA: period[_pc_explain.PORTFOLIO_RETURN_DELTA],
        _ESTIMATED_IMPACT_AREAS: _comma_separated(estimated_areas),
        _RESIDUAL_STATUS: status,
        _RESIDUAL_REASON: reason,
        _RESIDUAL_REVIEW_NOTE: _residual_review_note(status),
    }


def _residual_review_note(status: str) -> str:
    """Return reviewer-facing guidance for a residual status."""
    if status == _RESIDUAL_WITHHELD_NO_ESTIMATES:
        return (
            "No supported impact estimates exist, so any residual would equal "
            "the whole return delta and imply false precision."
        )
    if status == _RESIDUAL_WITHHELD_PARTIAL_ESTIMATES:
        return (
            "Some estimates exist, but coverage is incomplete or overlapping; "
            "do not reconcile the remaining difference as residual."
        )
    if status == _RESIDUAL_WITHHELD_CROSS_CHECKS_ONLY:
        return (
            "Only review-only cross-check estimates exist; they are excluded "
            "from impact totals."
        )
    return "Residual status requires review before drawing conclusions."


def _context_evidence_summary_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return grouped context evidence counts and affected identifiers."""
    context_evidence = _context_evidence_table(findings)
    if context_evidence.is_empty():
        return _empty_context_evidence_summary_table()

    grouped_rows: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in context_evidence.iter_rows(named=True):
        key = (
            row[_pc_findings.DATASET],
            row[_pc_findings.SOURCE_COLUMN],
            row[_CONTEXT_USE],
        )
        grouped_rows.setdefault(key, []).append(row)

    rows = [
        _context_evidence_summary_row(key, rows_for_key)
        for key, rows_for_key in sorted(grouped_rows.items(), key=_context_summary_key)
    ]
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col(_REVIEW_PRIORITY)
            .replace_strict({"high": 0, "medium": 1, "low": 2}, default=3)
            .alias("_review_priority_rank")
        )
        .sort(
            [
                "_review_priority_rank",
                _pc_findings.DATASET,
                _pc_findings.SOURCE_COLUMN,
                _CONTEXT_USE,
            ],
            nulls_last=True,
        )
        .select(_CONTEXT_EVIDENCE_SUMMARY_COLUMNS)
    )


def _empty_context_evidence_summary_table() -> pl.DataFrame:
    """Return an empty context-evidence summary with stable columns."""
    return pl.DataFrame(
        schema={
            _pc_findings.DATASET: pl.String,
            _pc_findings.SOURCE_COLUMN: pl.String,
            _CONTEXT_USE: pl.String,
            _REVIEW_PRIORITY: pl.String,
            _REVIEW_PRIORITY_REASON: pl.String,
            _FINDING_COUNT: pl.UInt32,
            _PORTFOLIO_COUNT: pl.UInt32,
            _SECURITY_COUNT: pl.UInt32,
            _AFFECTED_PORTFOLIOS: pl.String,
            _AFFECTED_SECURITIES: pl.String,
        }
    )


def _context_summary_key(
    item: tuple[tuple[object, object, object], list[dict[str, object]]],
) -> tuple[str, str, str]:
    """Return stable sort text for context-summary groups."""
    key, _ = item
    dataset, source_column, context_use = key
    return (
        _format_value(dataset),
        _format_value(source_column),
        _format_value(context_use),
    )


def _context_evidence_summary_row(
    key: tuple[object, object, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return one grouped context-evidence summary row."""
    dataset, source_column, context_use = key
    portfolios = _unique_nonblank_values(
        row.get(_pc_findings.PORTFOLIO_ID) for row in rows
    )
    securities = _unique_nonblank_values(
        row.get(_pc_findings.SECURITY_ID) for row in rows
    )
    priority, priority_reason = _context_review_priority(
        dataset=dataset,
        source_column=source_column,
        portfolio_count=len(portfolios),
        security_count=len(securities),
    )
    return {
        _pc_findings.DATASET: dataset,
        _pc_findings.SOURCE_COLUMN: source_column,
        _CONTEXT_USE: context_use,
        _REVIEW_PRIORITY: priority,
        _REVIEW_PRIORITY_REASON: priority_reason,
        _FINDING_COUNT: len(rows),
        _PORTFOLIO_COUNT: len(portfolios),
        _SECURITY_COUNT: len(securities),
        _AFFECTED_PORTFOLIOS: _comma_separated(portfolios),
        _AFFECTED_SECURITIES: _comma_separated(securities),
    }


def _context_review_priority(
    *,
    dataset: object,
    source_column: object,
    portfolio_count: int,
    security_count: int,
) -> tuple[str, str]:
    """Return a reviewer priority label for grouped context evidence."""
    if portfolio_count > 0:
        return (
            "high",
            "Linked to one or more changed portfolio periods.",
        )
    if dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.COMMISSION:
        return (
            "high",
            "Commission context can explain fee or net-amount differences.",
        )
    if security_count > 0:
        return (
            "medium",
            "Security-level context may help identify reference-data changes.",
        )
    return (
        "low",
        "Context is not linked to a changed portfolio period or security.",
    )


def _context_evidence_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return context evidence with explicit no-impact treatment."""
    if findings.is_empty():
        return _empty_context_evidence_table(findings)

    context_findings = findings.filter(
        pl.col(_pc_findings.EVIDENCE_ROLE) == _pc_findings.CONTEXT
    )
    if context_findings.is_empty():
        return _empty_context_evidence_table(findings)

    rows = [
        _context_evidence_row(row)
        for row in context_findings.iter_rows(named=True)
    ]
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col(_REVIEW_PRIORITY)
            .replace_strict({"high": 0, "medium": 1, "low": 2}, default=3)
            .alias("_review_priority_rank")
        )
        .sort(
            [
                "_review_priority_rank",
                _pc_findings.PORTFOLIO_ID,
                _pc_findings.FROM_DATE,
                _pc_findings.THRU_DATE,
                _pc_findings.DATASET,
                _pc_findings.SOURCE_COLUMN,
                _pc_findings.SECURITY_ID,
                _pc_findings.FINDING_CODE,
            ],
            nulls_last=True,
        )
        .select(_CONTEXT_EVIDENCE_COLUMNS)
    )


def _context_evidence_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return one row-level context evidence record with review metadata."""
    priority, priority_reason = _context_review_priority(
        dataset=row.get(_pc_findings.DATASET),
        source_column=row.get(_pc_findings.SOURCE_COLUMN),
        portfolio_count=1 if _has_text(row.get(_pc_findings.PORTFOLIO_ID)) else 0,
        security_count=1 if _has_text(row.get(_pc_findings.SECURITY_ID)) else 0,
    )
    return {
        **{
            column: row.get(column)
            for column in _CONTEXT_EVIDENCE_COLUMNS
            if column in row
        },
        _CONTEXT_USE: _context_use(row),
        _REVIEW_PRIORITY: priority,
        _REVIEW_PRIORITY_REASON: priority_reason,
        _RETURN_IMPACT_TREATMENT: _CONTEXT_NO_IMPACT_TREATMENT,
    }


def _empty_context_evidence_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return an empty context-evidence table with stable columns."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: findings.schema.get(
                _pc_findings.PORTFOLIO_ID,
                pl.String,
            ),
            _pc_findings.SECURITY_ID: findings.schema.get(
                _pc_findings.SECURITY_ID,
                pl.String,
            ),
            _pc_findings.FROM_DATE: findings.schema.get(_pc_findings.FROM_DATE, pl.Date),
            _pc_findings.THRU_DATE: findings.schema.get(_pc_findings.THRU_DATE, pl.Date),
            _pc_findings.DATASET: findings.schema.get(_pc_findings.DATASET, pl.String),
            _pc_findings.FINDING_CODE: findings.schema.get(
                _pc_findings.FINDING_CODE,
                pl.String,
            ),
            _pc_findings.SOURCE_COLUMN: findings.schema.get(
                _pc_findings.SOURCE_COLUMN,
                pl.String,
            ),
            _pc_findings.DELTA_B_MINUS_A: findings.schema.get(
                _pc_findings.DELTA_B_MINUS_A,
                pl.Float64,
            ),
            _CONTEXT_USE: pl.String,
            _REVIEW_PRIORITY: pl.String,
            _REVIEW_PRIORITY_REASON: pl.String,
            _RETURN_IMPACT_TREATMENT: pl.String,
            _pc_findings.MESSAGE: findings.schema.get(_pc_findings.MESSAGE, pl.String),
        }
    )


def _context_use(finding: Mapping[str, object]) -> str:
    """Return reviewer-facing use text for a context finding."""
    dataset = finding.get(_pc_findings.DATASET)
    source_column = finding.get(_pc_findings.SOURCE_COLUMN)
    if dataset == pc_cols.HOLDINGS and source_column == pc_cols.COST:
        return "cost-basis review context; not a performance input"
    if dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.COMMISSION:
        return "commission and fee review context; not modeled without explicit policy"
    if dataset == pc_cols.SECURITY_MASTER:
        return "security-reference review context"
    return "review context"


def _impact_estimate_summary_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return currently quantified cause-summary rows."""
    summary = _pc_explain.portfolio_period_cause_summary(findings)
    if summary.is_empty():
        return summary
    return summary.filter(pl.col(_pc_explain.ESTIMATED_RETURN_IMPACT).is_not_null())


def _active_findings(findings: pl.DataFrame) -> pl.DataFrame:
    """Return unsuppressed findings, preserving empty-table behavior."""
    if findings.is_empty() or _pc_findings.SUPPRESSED not in findings.columns:
        return findings
    return findings.filter(~pl.col(_pc_findings.SUPPRESSED))


def _html_table_row_id(
    row: Mapping[str, object],
    *,
    row_id_prefix: str | None,
    row_id_counts: dict[str, int],
) -> str:
    """Return an optional stable row id for period-level HTML table rows."""
    if not row_id_prefix:
        return ""
    review_key = _row_review_key(row)
    if not review_key:
        return ""

    base_row_id = _html_review_key_row_id(row_id_prefix, review_key)
    row_id_count = row_id_counts.get(base_row_id, 0) + 1
    row_id_counts[base_row_id] = row_id_count
    if row_id_count == 1:
        return base_row_id
    return f"{base_row_id}-{row_id_count}"
