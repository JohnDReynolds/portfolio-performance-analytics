"""Render performance comparison findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import datetime as dt
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
from ppar.performance_comparison import runner as _pc_runner
from ppar.performance_comparison import workbook as _pc_workbook
from ppar.performance_comparison import workbook_tables as _pc_workbook_tables

__all__ = [
    "performance_comparison_html_report",
    "performance_comparison_markdown_report",
    "write_performance_comparison_html_report",
    "write_performance_comparison_markdown_report",
    "write_performance_comparison_report_bundle",
    "write_performance_comparison_review_workbook",
]

_COUNT = "count"
_ACTIVE_ONLY_NOTE = (
    "Unless noted otherwise, report sections exclude suppressed findings. "
    "Suppressed findings are summarized in the audit appendix."
)
_NO_ESTIMATE_NOTE = (
    "Impact estimates are intentionally conservative. Blank impact values mean "
    "the comparison has evidence, but no defensible return-impact estimate yet."
)
_ESTIMATED_IMPACT_AREAS = "estimated_impact_areas"
_RESIDUAL_STATUS = "residual_status"
_RESIDUAL_REASON = "residual_reason"
_RESIDUAL_REVIEW_NOTE = "residual_review_note"
_RESIDUAL_WITHHELD_PREFIX = "withheld"
_RESIDUAL_WITHHELD_NO_ESTIMATES = "withheld_no_estimates"
_RESIDUAL_WITHHELD_PARTIAL_ESTIMATES = "withheld_partial_estimates"
_RESIDUAL_WITHHELD_CROSS_CHECKS_ONLY = "withheld_cross_checks_only"
_RESIDUAL_STATUS_NOTE = (
    "Residual amounts are intentionally withheld until the attribution model has "
    "enough defensible, non-overlapping impact estimates."
)
_REVIEW_STATUS = "review_status"
_REVIEW_CUES = "review_cues"
_SUGGESTED_NEXT_STEP = "suggested_next_step"
_REVIEW_KEY = _pc_review_keys.REVIEW_KEY
_REVIEW_DETAIL_ARTIFACTS = "review_detail_artifacts"
_PRIMARY_REVIEW_CUE = "primary_review_cue"
_DASHBOARD_COVERAGE_COUNTS = "dashboard_coverage_counts"
_DASHBOARD_MISSING_INPUTS = "dashboard_missing_inputs"
_DASHBOARD_CONTEXT_CUE = "dashboard_context_cue"
_DASHBOARD_MAIN_ISSUE = "dashboard_main_issue"
_DASHBOARD_OPEN_SECTION = "dashboard_open_section"
_PROBLEM = "problem"
_ACTION_REQUIRED = "action_required"
_WHY_IT_MATTERS = "why_it_matters"
_EVIDENCE_SECTION = "evidence_section"
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
_TRIAGE_CHANGED_PERIODS = "Changed periods"
_TRIAGE_NEEDS_REVIEW_PERIODS = "Needs-review periods"
_TRIAGE_EVIDENCE_ONLY_AREAS = "Evidence-only cause areas"
_TRIAGE_CONTEXT_GROUPS = "Context evidence groups"
_TRIAGE_HIGH_PRIORITY_CONTEXT_GROUPS = "High-priority context groups"
_TRIAGE_TRANSACTION_CROSS_CHECK_ROWS = "Transaction cross-check rows"
_TRIAGE_RESIDUAL_WITHHELD_PERIODS = "Residual-withheld periods"
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
_markdown_table = _pc_rendering.markdown_table
_html_section = _pc_rendering.html_section
_html_summary_card = _pc_rendering.html_summary_card
_html_paragraph = _pc_rendering.html_paragraph
_html_review_key_row_id = _pc_rendering.html_review_key_row_id
_html_id_token = _pc_rendering.html_id_token
_css_token = _pc_rendering.css_token
_html_empty = _pc_rendering.html_empty
_html_list = _pc_rendering.html_list
_html_section_id = _pc_rendering.html_section_id
_display_header = _pc_rendering.display_header
_format_markdown_cell = _pc_rendering.format_markdown_cell
_format_value = _pc_rendering.format_value
_comma_separated = _pc_rendering.comma_separated
_unique_nonblank_values = _pc_rendering.unique_nonblank_values
_escape_markdown_text = _pc_rendering.escape_markdown_text
_escape_html = _pc_rendering.escape_html
_html_style_block = _pc_rendering.html_style_block
_html_dashboard_script = _pc_rendering.html_dashboard_script
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


def performance_comparison_markdown_report(
    findings: pl.DataFrame,
    *,
    title: str = "Performance Comparison Report",
    include_suppressed_appendix: bool = True,
    top_evidence_limit: int = 10,
) -> str:
    """Return a Markdown report for performance comparison findings.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        title: Markdown H1 text for the report.
        include_suppressed_appendix: Whether to include a compact table of
            suppressed findings at the end of the report.
        top_evidence_limit: Maximum number of contribution-candidate evidence
            rows to show per portfolio period.

    Returns:
        Markdown string suitable for console output, files, notebooks, or
        generated review bundles.
    """
    active_findings = _active_findings(findings)
    summaries = _pc_runner.summarize_findings(findings)
    active_summaries = _pc_runner.summarize_findings(active_findings)

    sections = [
        f"# {_escape_markdown_text(title)}",
        _ACTIVE_ONLY_NOTE,
        _NO_ESTIMATE_NOTE,
        _report_contents_section(
            include_suppressed_appendix=include_suppressed_appendix,
        ),
        _run_summary_section(findings, active_findings, summaries, active_summaries),
        _needs_review_summary_section(active_findings),
        _portfolio_period_narrative_section(active_findings),
        _review_notes_section(active_findings),
        _impact_estimate_summary_section(active_findings),
        _impact_coverage_section(active_findings),
        _context_evidence_summary_section(active_findings),
        _context_evidence_section(active_findings),
        _transaction_cross_checks_section(active_findings),
        _flow_cross_check_reconciliation_section(active_findings),
        _residual_status_section(active_findings),
        _transaction_activity_section(active_findings),
        _transaction_matching_diagnostics_section(active_findings),
        _portfolio_period_section(active_findings),
        _cause_summary_section(active_findings),
        _top_evidence_section(active_findings, top_evidence_limit),
    ]
    if include_suppressed_appendix:
        sections.append(_suppressed_appendix_section(findings, summaries))
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def performance_comparison_html_report(
    findings: pl.DataFrame,
    *,
    title: str = "Performance Comparison Report",
    include_suppressed_appendix: bool = True,
    top_evidence_limit: int = 10,
) -> str:
    """Return a standalone HTML report for performance comparison findings.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        title: HTML document title and visible H1 text.
        include_suppressed_appendix: Whether to include a compact table of
            suppressed findings at the end of the report.
        top_evidence_limit: Maximum number of contribution-candidate evidence
            rows to show per portfolio period.

    Returns:
        Complete HTML document string suitable for writing to disk or opening
        in a browser.
    """
    active_findings = _active_findings(findings)
    summaries = _pc_runner.summarize_findings(findings)
    active_summaries = _pc_runner.summarize_findings(active_findings)
    evidence_sections = [
        (
            "Portfolio-Period Narrative",
            _html_portfolio_period_narrative_section(active_findings),
        ),
        ("Needs Review Summary", _html_needs_review_summary_section(active_findings)),
        ("Impact Coverage", _html_impact_coverage_section(active_findings)),
        ("Context Evidence", _html_context_evidence_section(active_findings)),
        ("Top Evidence", _html_top_evidence_section(active_findings, top_evidence_limit)),
        ("Review Notes", _html_review_notes_section(active_findings)),
        ("Impact Estimate Summary", _html_impact_estimate_summary_section(active_findings)),
        ("Transaction Activity", _html_transaction_activity_section(active_findings)),
        ("Residual Status", _html_residual_status_section(active_findings)),
        ("Context Evidence Summary", _html_context_evidence_summary_section(active_findings)),
        ("Transaction Cross-Checks", _html_transaction_cross_checks_section(active_findings)),
        (
            "Flow Cross-Check Reconciliation",
            _html_flow_cross_check_reconciliation_section(active_findings),
        ),
        (
            "Transaction Matching Diagnostics",
            _html_transaction_matching_diagnostics_section(active_findings),
        ),
        ("Portfolio-Period Changes", _html_portfolio_period_section(active_findings)),
        ("Cause Summary", _html_cause_summary_section(active_findings)),
        (
            "Run Summary",
            _html_run_summary_section(
                findings,
                active_findings,
                summaries,
                active_summaries,
            ),
        ),
    ]
    if include_suppressed_appendix:
        evidence_sections.append(
            (
                "Suppressed Findings Appendix",
                _html_suppressed_appendix_section(findings, summaries),
            )
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
            "</header>",
            _html_problems_section(active_findings),
            _html_evidence_appendix_section(evidence_sections),
            "</main>",
            _html_dashboard_script(),
            "</body>",
            "</html>",
            "",
        ]
    )


def write_performance_comparison_markdown_report(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    title: str = "Performance Comparison Report",
    include_suppressed_appendix: bool = True,
    top_evidence_limit: int = 10,
) -> Path:
    """Write a Markdown performance comparison report to disk.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_path: Destination Markdown file path. Parent directories are
            created when needed.
        title: Markdown H1 text for the report.
        include_suppressed_appendix: Whether to include a compact table of
            suppressed findings at the end of the report.
        top_evidence_limit: Maximum number of contribution-candidate evidence
            rows to show per portfolio period.

    Returns:
        Normalized ``Path`` to the written report file.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = performance_comparison_markdown_report(
        findings,
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
    )
    report_path.write_text(report, encoding=util.ENCODING)
    return report_path


def write_performance_comparison_html_report(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    title: str = "Performance Comparison Report",
    include_suppressed_appendix: bool = True,
    top_evidence_limit: int = 10,
) -> Path:
    """Write an HTML performance comparison report to disk.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_path: Destination HTML report path. Parent directories are
            created when needed.
        title: HTML document title and visible H1 text.
        include_suppressed_appendix: Whether to include the suppressed findings
            appendix section.
        top_evidence_limit: Maximum number of top-evidence rows to show per
            portfolio period.

    Returns:
        Normalized ``Path`` to the written report file.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = performance_comparison_html_report(
        findings,
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
    )
    report_path.write_text(report, encoding=util.ENCODING)
    return report_path


def write_performance_comparison_report_bundle(
    findings: pl.DataFrame,
    output_directory: util.PathLike,
    *,
    title: str = "Performance Comparison Report",
    include_suppressed_appendix: bool = True,
    top_evidence_limit: int = 10,
    include_workbook: bool = False,
    require_causal_attribution: bool = False,
    comparison_path: util.PathLike | None = None,
) -> dict[str, Path]:
    """Write a reproducible report bundle.

    Args:
        findings: Findings table returned by ``compare_snapshots`` or
            ``findings_to_polars``.
        output_directory: Destination directory. It is created when needed.
        title: Markdown H1 text for ``report.md``.
        include_suppressed_appendix: Whether ``report.md`` should include the
            suppressed findings appendix section.
        top_evidence_limit: Maximum number of top-evidence rows to include per
            portfolio period in both ``report.md`` and ``top_evidence.csv``.
        include_workbook: Whether to include an XLSX review workbook. Requires
            installing the optional ``ppar[excel]`` dependency group.
        require_causal_attribution: Whether changed portfolio periods must have
            all YAML setup needed by supported attribution methods before
            writing bundle artifacts. This does not require every performance
            change to be fully explained.
        comparison_path: Optional path to the comparison YAML. When provided,
            the XLSX workbook can name the exact YAML file to update for
            missing attribution setup.

    Returns:
        Mapping from bundle artifact name to normalized written path.
    """
    if include_workbook:
        _pc_workbook.ensure_openpyxl_installed()

    bundle_directory = Path(output_directory)
    bundle_directory.mkdir(parents=True, exist_ok=True)
    active_findings = _active_findings(findings)
    if require_causal_attribution:
        _pc_runner.validate_causal_attribution_ready(active_findings)
    tables = _report_bundle_tables(active_findings, top_evidence_limit)

    paths: dict[str, Path] = {}
    report_path = write_performance_comparison_markdown_report(
        findings,
        bundle_directory / "report.md",
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
    )
    paths["report"] = report_path
    html_report_path = write_performance_comparison_html_report(
        findings,
        bundle_directory / "report.html",
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
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
        )
    paths["readme"] = _pc_bundle.write_report_bundle_readme(
        bundle_directory / "README.md",
        title=title,
        tables=tables,
        include_workbook=include_workbook,
    )
    manifest_path = bundle_directory / "manifest.json"
    paths["manifest"] = manifest_path
    _pc_bundle.write_report_bundle_manifest(
        manifest_path,
        findings=findings,
        active_findings=active_findings,
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
        artifact_paths=paths,
        tables=tables,
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
) -> dict[str, pl.DataFrame]:
    """Return report-bundle tables keyed by artifact stem."""
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
    """
    return _pc_workbook_tables.write_performance_comparison_review_workbook(
        findings,
        output_path,
        top_evidence_limit=top_evidence_limit,
        comparison_path=comparison_path,
    )


def _run_summary_section(
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    summaries: dict[str, pl.DataFrame],
    active_summaries: dict[str, pl.DataFrame],
) -> str:
    """Return the run summary Markdown section."""
    lines = [
        "## Run Summary",
        f"- Total findings: {_format_value(findings.height)}",
        f"- Active findings: {_format_value(active_findings.height)}",
        f"- Suppressed findings: {_format_value(findings.height - active_findings.height)}",
        "",
        "### Reviewer Triage",
        *[
            f"- {label}: {_format_value(value)}"
            for label, value in _reviewer_triage_counts(active_findings)
        ],
        "",
        "### Active Findings By Code",
        _markdown_table(active_summaries["by_code"], [_pc_findings.FINDING_CODE, _COUNT]),
        "",
        "### Active Findings By Dataset",
        _markdown_table(active_summaries["by_dataset"], [_pc_findings.DATASET, _COUNT]),
        "",
        "### Findings By Suppression State",
        _markdown_table(summaries["by_suppressed"], [_pc_findings.SUPPRESSED, _COUNT]),
    ]
    return "\n".join(lines)


def _reviewer_triage_counts(findings: pl.DataFrame) -> list[tuple[str, int]]:
    """Return top-of-report reviewer triage counts."""
    periods = _pc_explain.portfolio_period_summary(findings)
    needs_review = _needs_review_summary_table(findings)
    impact_coverage = _pc_explain.portfolio_period_impact_coverage_summary(findings)
    context_summary = _context_evidence_summary_table(findings)
    transaction_cross_checks = _pc_explain.portfolio_period_transaction_cross_checks(
        findings
    )
    residual_status = _residual_status_table(findings)
    return [
        (_TRIAGE_CHANGED_PERIODS, periods.height),
        (_TRIAGE_NEEDS_REVIEW_PERIODS, _needs_review_period_count(needs_review)),
        (_TRIAGE_EVIDENCE_ONLY_AREAS, _evidence_only_area_count(impact_coverage)),
        (_TRIAGE_CONTEXT_GROUPS, context_summary.height),
        (
            _TRIAGE_HIGH_PRIORITY_CONTEXT_GROUPS,
            _context_priority_group_count(context_summary, "high"),
        ),
        (_TRIAGE_TRANSACTION_CROSS_CHECK_ROWS, transaction_cross_checks.height),
        (_TRIAGE_RESIDUAL_WITHHELD_PERIODS, _residual_withheld_period_count(residual_status)),
    ]


def _changed_period_count(findings: pl.DataFrame) -> int:
    """Return count of changed portfolio periods."""
    return _pc_explain.portfolio_period_summary(findings).height


def _needs_review_period_count(needs_review: pl.DataFrame) -> int:
    """Return count of periods whose review status needs attention."""
    if needs_review.is_empty():
        return 0
    return needs_review.filter(
        pl.col(_REVIEW_STATUS) == _REVIEW_STATUS_NEEDS_REVIEW
    ).height


def _evidence_only_area_count(impact_coverage: pl.DataFrame) -> int:
    """Return total evidence-only cause-area count across changed periods."""
    if impact_coverage.is_empty():
        return 0
    total = impact_coverage.select(
        pl.col(_pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT).sum()
    ).item()
    return _count_value(total)


def _context_priority_group_count(context_summary: pl.DataFrame, priority: str) -> int:
    """Return count of context evidence groups for a priority label."""
    if context_summary.is_empty():
        return 0
    return context_summary.filter(pl.col(_REVIEW_PRIORITY) == priority).height


def _residual_withheld_period_count(residual_status: pl.DataFrame) -> int:
    """Return count of periods whose residual status is withheld."""
    if residual_status.is_empty():
        return 0
    return residual_status.filter(
        pl.col(_RESIDUAL_STATUS).str.starts_with(_RESIDUAL_WITHHELD_PREFIX)
    ).height


def _needs_review_summary_section(findings: pl.DataFrame) -> str:
    """Return a Markdown triage summary for changed portfolio periods."""
    return "\n".join(
        [
            "## Needs Review Summary",
            _markdown_table(
                _needs_review_summary_table(findings),
                _NEEDS_REVIEW_COLUMNS,
                empty_message="No changed portfolio periods need review.",
            ),
        ]
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
    """Return a conservative next action for a period's reviewer cues."""
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


def _report_contents_section(*, include_suppressed_appendix: bool) -> str:
    """Return the report contents section."""
    section_names = _report_section_names(
        include_suppressed_appendix=include_suppressed_appendix
    )
    lines = ["## Report Contents", *[f"- {name}" for name in section_names]]
    return "\n".join(lines)


def _html_contents_section(*, include_suppressed_appendix: bool) -> str:
    """Return the HTML report contents navigation."""
    section_names = _report_section_names(
        include_suppressed_appendix=include_suppressed_appendix
    )
    links = [
        f'<li><a href="#{_html_section_id(name)}">{_escape_html(name)}</a></li>'
        for name in section_names
    ]
    return "\n".join(
        [
            '<nav class="pc-section" aria-labelledby="report-contents">',
            '<h2 id="report-contents">Report Contents</h2>',
            '<ol class="pc-contents-list">',
            *links,
            "</ol>",
            "</nav>",
        ]
    )


def _html_review_basis_section(findings: pl.DataFrame) -> str:
    """Return a compact HTML strip describing the report's review basis."""
    needs_review = _needs_review_summary_table(findings)
    values = [
        ("Scope", "active findings"),
        ("Impact", "conservative estimates"),
        ("Residual", "withheld until supported"),
        ("Changed periods", _changed_period_count(findings)),
        ("Needs review", _needs_review_period_count(needs_review)),
    ]
    items = [
        "\n".join(
            [
                '<div class="pc-basis-item">',
                f"<span>{_escape_html(label)}</span>",
                f"<strong>{_escape_html(_format_value(value))}</strong>",
                "</div>",
            ]
        )
        for label, value in values
    ]
    return "\n".join(
        [
            '<section class="pc-review-basis" aria-label="Review basis">',
            *items,
            "</section>",
        ]
    )


def _report_section_names(*, include_suppressed_appendix: bool) -> list[str]:
    """Return report section names in display order."""
    section_names = [
        "Run Summary",
        "Needs Review Summary",
        "Portfolio-Period Narrative",
        "Review Notes",
        "Impact Estimate Summary",
        "Impact Coverage",
        "Context Evidence Summary",
        "Context Evidence",
        "Transaction Cross-Checks",
        "Flow Cross-Check Reconciliation",
        "Residual Status",
        "Transaction Activity",
        "Transaction Matching Diagnostics",
        "Portfolio-Period Changes",
        "Cause Summary",
        "Top Evidence",
    ]
    if include_suppressed_appendix:
        section_names.append("Suppressed Findings Appendix")
    return section_names


def _portfolio_period_narrative_section(findings: pl.DataFrame) -> str:
    """Return conservative narrative summaries for portfolio-period changes."""
    summary = _pc_explain.portfolio_period_summary(findings)
    if summary.is_empty():
        return "\n".join(
            [
                "## Portfolio-Period Narrative",
                "_No portfolio return changes to narrate._",
            ]
        )

    causes = _pc_explain.portfolio_period_cause_summary(findings)
    paragraphs = ["## Portfolio-Period Narrative"]
    for period in summary.iter_rows(named=True):
        period_causes = _period_cause_rows(causes, period)
        paragraphs.append(_portfolio_period_narrative(period, period_causes))
    return "\n\n".join(paragraphs)


def _review_notes_section(findings: pl.DataFrame) -> str:
    """Return review notes for current model limits visible in the report."""
    causes = _pc_explain.portfolio_period_cause_summary(findings)
    if causes.is_empty():
        return "\n".join(
            [
                "## Review Notes",
                "_No portfolio-period review notes._",
            ]
        )

    cause_rows = list(causes.iter_rows(named=True))
    notes = _review_notes_for_cause_rows(cause_rows)
    if not notes:
        notes = [
            "No model-limit review notes were generated for the current evidence mix.",
        ]

    lines = ["## Review Notes", *[f"- {note}" for note in notes]]
    return "\n".join(lines)


def _review_notes_for_cause_rows(causes: list[dict[str, object]]) -> list[str]:
    """Return deterministic review notes for cause areas present in a report."""
    notes: list[str] = []
    cause_areas = {cause[_pc_explain.ROOT_CAUSE_AREA] for cause in causes}
    if _pc_explain.ROOT_CAUSE_TRANSACTION_ACTIVITY in cause_areas:
        notes.append(_transaction_activity_review_note(causes))
    if _pc_explain.ROOT_CAUSE_MARKET_VALUE_OR_POSITION in cause_areas:
        notes.append(
            "Market value or position evidence has no return-impact estimate yet."
        )
    if _pc_explain.ROOT_CAUSE_PRICE in cause_areas:
        notes.append(
            "Price evidence is linked to affected portfolio periods, but no "
            "portfolio-period impact estimate is calculated yet."
        )
    if _pc_explain.ROOT_CAUSE_CASH in cause_areas:
        notes.append("Cash evidence has no return-impact estimate yet.")
    if _pc_explain.ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT in cause_areas:
        notes.append(_portfolio_source_field_review_note(causes))
    if _pc_explain.ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION in cause_areas:
        notes.append(_security_return_weighted_review_note(causes))
    notes.append(
        "No residual amount is calculated because not enough defensible impact "
        "estimates exist yet."
    )
    return notes


def _transaction_activity_review_note(causes: list[dict[str, object]]) -> str:
    """Return a review note for transaction activity estimates."""
    has_estimate = any(
        cause[_pc_explain.ROOT_CAUSE_AREA] == _pc_explain.ROOT_CAUSE_TRANSACTION_ACTIVITY
        and cause[_pc_explain.IMPACT_BASIS] != _pc_explain.IMPACT_BASIS_NO_ESTIMATE
        for cause in causes
    )
    if has_estimate:
        return (
            "Transaction activity estimates are low-confidence and limited to "
            "source-signed performance-treated amount deltas over the return "
            "denominator."
        )
    return (
        "Transaction activity is evidence-only until portfolio period, return "
        "denominator, transaction sign and flow semantics, and an applicable "
        "impact method are available."
    )


def _portfolio_source_field_review_note(causes: list[dict[str, object]]) -> str:
    """Return a review note for portfolio performance source-field estimates."""
    has_estimate = any(
        cause[_pc_explain.ROOT_CAUSE_AREA] == _pc_explain.ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT
        and cause[_pc_explain.IMPACT_BASIS] == _pc_explain.IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD
        for cause in causes
    )
    if has_estimate:
        return (
            "Portfolio performance source-field estimates are low-confidence "
            "approximations based on source-field deltas over beginning "
            "market value."
        )
    return (
        "Portfolio performance source-field changes are direct evidence, but "
        "denominator-based impact formulas are not modeled for these rows yet."
    )


def _security_return_weighted_review_note(causes: list[dict[str, object]]) -> str:
    """Return a review note for weighted security return estimates."""
    has_vendor_contribution = any(
        cause[_pc_explain.ROOT_CAUSE_AREA]
        == _pc_explain.ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
        and cause[_pc_explain.IMPACT_BASIS] == _pc_explain.IMPACT_BASIS_SECURITY_CONTRIBUTION
        for cause in causes
    )
    if has_vendor_contribution:
        return (
            "Weighted security return estimates are available for review, but "
            "vendor contribution deltas are preferred in the cause summary to "
            "avoid double-counting."
        )
    return (
        "Weighted security return estimates are low-confidence approximations "
        "using security return deltas times portfolio weight."
    )


def _portfolio_period_narrative(
    period: dict[str, object],
    causes: list[dict[str, object]],
) -> str:
    """Return one portfolio-period narrative paragraph."""
    portfolio_id = _format_value(period[_pc_findings.PORTFOLIO_ID])
    from_date = _format_value(period[_pc_findings.FROM_DATE])
    thru_date = _format_value(period[_pc_findings.THRU_DATE])
    return_delta = _format_value(period[_pc_explain.PORTFOLIO_RETURN_DELTA])
    sentences = [
        (
            f"{portfolio_id} changed by {return_delta} for {from_date} to "
            f"{thru_date}."
        )
    ]

    estimated_causes = [
        cause
        for cause in causes
        if cause.get(_pc_explain.ESTIMATED_RETURN_IMPACT) is not None
    ]
    if estimated_causes:
        strongest = max(
            estimated_causes,
            key=_absolute_estimated_return_impact,
        )
        sentences.append(_estimated_impact_sentence(strongest))
    else:
        sentences.append(
            "No currently supported impact estimates are available for this period."
        )

    evidence_only_areas = [
        str(cause[_pc_explain.ROOT_CAUSE_AREA])
        for cause in causes
        if cause.get(_pc_explain.IMPACT_BASIS) == _pc_explain.IMPACT_BASIS_NO_ESTIMATE
    ]
    if evidence_only_areas:
        sentences.append(
            "Evidence-only areas are "
            f"{_comma_separated(evidence_only_areas)}; these rows remain "
            f"{_pc_explain.IMPACT_BASIS_NO_ESTIMATE}."
        )

    if period[_pc_explain.HAS_SUPPRESSED_FINDINGS]:
        sentences.append("Suppressed findings exist for this portfolio period.")

    return " ".join(_escape_markdown_text(sentence) for sentence in sentences)


def _absolute_estimated_return_impact(cause: Mapping[str, object]) -> float:
    """Return the absolute estimated impact for sorting narrative causes."""
    value = cause.get(_pc_explain.ESTIMATED_RETURN_IMPACT)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    return 0.0


def _estimated_impact_sentence(cause: dict[str, object]) -> str:
    """Return a conservative sentence for the strongest estimated impact."""
    cause_area = _format_value(cause[_pc_explain.ROOT_CAUSE_AREA])
    estimated_impact = _format_value(cause[_pc_explain.ESTIMATED_RETURN_IMPACT])
    impact_basis = _format_value(cause[_pc_explain.IMPACT_BASIS])
    confidence = _format_value(cause[_pc_explain.IMPACT_CONFIDENCE])
    return (
        "The strongest currently estimated impact is "
        f"{cause_area} at {estimated_impact}, based on {impact_basis} "
        f"with {confidence} confidence."
    )


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


def _portfolio_period_section(findings: pl.DataFrame) -> str:
    """Return the portfolio-period return changes Markdown section."""
    summary = _pc_explain.portfolio_period_summary(findings)
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _pc_explain.FINDING_COUNT,
        _pc_explain.HAS_SUPPRESSED_FINDINGS,
    ]
    return "\n".join(
        [
            "## Portfolio-Period Changes",
            _markdown_table(summary, columns, empty_message="No portfolio return changes."),
        ]
    )


def _transaction_activity_section(findings: pl.DataFrame) -> str:
    """Return changed transaction activity and impact-eligibility gaps."""
    summary = _pc_explain.transaction_activity_summary(findings)
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.SECURITY_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.TRANSACTION_CATEGORY,
        _pc_explain.CHANGED_FIELDS,
        _pc_explain.AMOUNT_DELTA,
        _pc_explain.QUANTITY_DELTA,
        _pc_explain.PRICE_DELTA,
        _pc_explain.TRANSACTION_SEMANTICS_SOURCES,
        _pc_explain.TRANSACTION_MATCH_STATUSES,
        _pc_explain.MISSING_IMPACT_INPUTS,
    ]
    return "\n".join(
        [
            "## Transaction Activity",
            _markdown_table(
                summary,
                columns,
                empty_message="No changed transaction activity.",
            ),
        ]
    )


def _transaction_matching_diagnostics_section(findings: pl.DataFrame) -> str:
    """Return transaction matching status diagnostics."""
    columns = [
        _pc_findings.TRANSACTION_MATCH_STATUS,
        _pc_explain.FINDING_COUNT,
        _pc_explain.TRANSACTION_MATCH_REVIEW_NOTE,
    ]
    return "\n".join(
        [
            "## Transaction Matching Diagnostics",
            _markdown_table(
                _pc_explain.transaction_matching_diagnostics(findings),
                columns,
                empty_message="No transaction matching diagnostics.",
            ),
        ]
    )


def _residual_status_section(findings: pl.DataFrame) -> str:
    """Return a Markdown section explaining whether residuals are calculated."""
    residuals = _residual_status_table(findings)
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _ESTIMATED_IMPACT_AREAS,
        _RESIDUAL_STATUS,
        _RESIDUAL_REASON,
        _RESIDUAL_REVIEW_NOTE,
    ]
    return "\n".join(
        [
            "## Residual Status",
            _RESIDUAL_STATUS_NOTE,
            "",
            _markdown_table(
                residuals,
                columns,
                empty_message="No portfolio return changes need residual review.",
            ),
        ]
    )


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


def _impact_coverage_section(findings: pl.DataFrame) -> str:
    """Return estimate-coverage status by portfolio period."""
    coverage = _pc_explain.portfolio_period_impact_coverage_summary(findings)
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _pc_explain.ROOT_CAUSE_AREA_COUNT,
        _pc_explain.ESTIMATED_CAUSE_AREA_COUNT,
        _pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT,
        _pc_explain.LOW_CONFIDENCE_ESTIMATE_COUNT,
        _pc_explain.MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
        _pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL,
        _pc_explain.EVIDENCE_ONLY_AREAS,
        _pc_explain.TRANSACTION_SEMANTICS_SOURCES,
        _pc_explain.MISSING_IMPACT_INPUTS,
        _pc_explain.IMPACT_COVERAGE_STATUS,
        _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE,
        _pc_explain.IMPACT_MESSAGE,
    ]
    return "\n".join(
        [
            "## Impact Coverage",
            _markdown_table(
                coverage,
                columns,
                empty_message="No portfolio return changes need impact coverage review.",
            ),
        ]
    )


def _context_evidence_section(findings: pl.DataFrame) -> str:
    """Return context-only evidence excluded from impact estimates."""
    return "\n".join(
        [
            "## Context Evidence",
            _markdown_table(
                _context_evidence_table(findings),
                _CONTEXT_EVIDENCE_COLUMNS,
                empty_message="No context-only evidence.",
            ),
        ]
    )


def _context_evidence_summary_section(findings: pl.DataFrame) -> str:
    """Return context-only evidence counts by source field."""
    return "\n".join(
        [
            "## Context Evidence Summary",
            _markdown_table(
                _context_evidence_summary_table(findings),
                _CONTEXT_EVIDENCE_SUMMARY_COLUMNS,
                empty_message="No context-only evidence summary.",
            ),
        ]
    )


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
    if dataset == pc_cols.POSITIONS and source_column == pc_cols.COST:
        return "cost-basis review context; not a performance input"
    if dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.COMMISSION:
        return "commission and fee review context; not modeled without explicit policy"
    if dataset == pc_cols.SECURITY_MASTER:
        return "security-reference review context"
    return "review context"


def _transaction_cross_checks_section(findings: pl.DataFrame) -> str:
    """Return transaction cross-check diagnostics by portfolio period."""
    return "\n".join(
        [
            "## Transaction Cross-Checks",
            _markdown_table(
                _pc_explain.portfolio_period_transaction_cross_checks(findings),
                list(_pc_explain.PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS),
                empty_message="No transaction cross-check estimates are available.",
            ),
        ]
    )


def _flow_cross_check_reconciliation_section(findings: pl.DataFrame) -> str:
    """Return flow/cross-check reconciliation diagnostics."""
    return "\n".join(
        [
            "## Flow Cross-Check Reconciliation",
            _markdown_table(
                _pc_explain.portfolio_period_flow_cross_check_reconciliation(findings),
                list(_pc_explain.PORTFOLIO_PERIOD_FLOW_CROSS_CHECK_RECONCILIATION_COLUMNS),
                empty_message="No flow/cross-check reconciliation rows are available.",
            ),
        ]
    )


def _impact_estimate_summary_section(findings: pl.DataFrame) -> str:
    """Return a concise Markdown section for currently quantified impacts."""
    estimated_summary = _impact_estimate_summary_table(findings)
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.ROOT_CAUSE_AREA,
        _pc_explain.ESTIMATED_RETURN_IMPACT,
        _pc_explain.IMPACT_BASIS,
        _pc_explain.IMPACT_CONFIDENCE,
        _pc_explain.IMPACT_MESSAGE,
    ]
    return "\n".join(
        [
            "## Impact Estimate Summary",
            _markdown_table(
                estimated_summary,
                columns,
                empty_message="No impact estimates are currently available.",
            ),
        ]
    )


def _impact_estimate_summary_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return currently quantified cause-summary rows."""
    summary = _pc_explain.portfolio_period_cause_summary(findings)
    if summary.is_empty():
        return summary
    return summary.filter(pl.col(_pc_explain.ESTIMATED_RETURN_IMPACT).is_not_null())


def _cause_summary_section(findings: pl.DataFrame) -> str:
    """Return the cause summary Markdown section."""
    summary = _pc_explain.portfolio_period_cause_summary(findings)
    columns = [
        column
        for column in _pc_explain.PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS
        if column
        in {
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
            _pc_explain.ROOT_CAUSE_AREA,
            _pc_explain.FINDING_COUNT,
            _pc_explain.ESTIMATED_RETURN_IMPACT,
            _pc_explain.IMPACT_BASIS,
            _pc_explain.IMPACT_CONFIDENCE,
            _pc_explain.TOP_CODES,
            _pc_explain.IMPACT_MESSAGE,
        }
    ]
    return "\n".join(
        [
            "## Cause Summary",
            _markdown_table(summary, columns, empty_message="No cause summary available."),
        ]
    )


def _top_evidence_section(findings: pl.DataFrame, top_evidence_limit: int) -> str:
    """Return the top contribution-candidate evidence Markdown section."""
    table = _pc_explain.top_evidence_table(findings, top_evidence_limit)
    if table.is_empty():
        return "\n".join(
            [
                "## Top Evidence",
                "_No ranked evidence is available for portfolio return changes._",
            ]
        )

    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        "review_rank",
        _pc_findings.FINDING_CODE,
        _pc_findings.DATASET,
        _pc_findings.EVIDENCE_ROLE,
        _pc_findings.SECURITY_ID,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.TRANSACTION_SEMANTICS_SOURCE,
        _pc_findings.TRANSACTION_MATCH_STATUS,
        _pc_findings.IMPACT_POLICY,
        _pc_findings.TRANSACTION_IMPACT_POLICY,
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC,
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
        _pc_findings.DELTA_B_MINUS_A,
        _pc_explain.ESTIMATED_RETURN_IMPACT,
        _pc_explain.IMPACT_BASIS,
        _pc_explain.IMPACT_CONFIDENCE,
        _pc_explain.IMPACT_METHOD,
        _pc_explain.IMPACT_MESSAGE,
        _pc_findings.MESSAGE,
    ]
    return "\n".join(
        [
            "## Top Evidence",
            _markdown_table(table, columns, empty_message="No ranked evidence is available."),
        ]
    )


def _suppressed_appendix_section(
    findings: pl.DataFrame,
    summaries: dict[str, pl.DataFrame],
) -> str:
    """Return the suppressed findings audit appendix."""
    suppressed = (
        findings.filter(pl.col(_pc_findings.SUPPRESSED))
        if not findings.is_empty()
        else findings
    )
    lines = [
        "## Suppressed Findings Appendix",
        "### Suppressed Counts By Code",
        _markdown_table(
            summaries["by_code_suppressed"].filter(pl.col(_pc_findings.SUPPRESSED))
            if not summaries["by_code_suppressed"].is_empty()
            else summaries["by_code_suppressed"],
            [_pc_findings.FINDING_CODE, _pc_findings.SUPPRESSED, _COUNT],
            empty_message="No suppressed findings.",
        ),
        "",
        "### Suppressed Finding Detail",
        _markdown_table(
            _pc_runner.compact_findings_table(suppressed, include_suppressed=True),
            [
                _pc_findings.FINDING_CODE,
                _pc_findings.DATASET,
                _pc_findings.EVIDENCE_ROLE,
                _pc_findings.PORTFOLIO_ID,
                _pc_findings.SECURITY_ID,
                _pc_findings.FROM_DATE,
                _pc_findings.THRU_DATE,
                _pc_findings.SOURCE_COLUMN,
                _pc_findings.DELTA_B_MINUS_A,
                _pc_findings.MESSAGE,
            ],
            empty_message="No suppressed finding detail.",
        ),
    ]
    return "\n".join(lines)


def _html_run_summary_section(
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    summaries: dict[str, pl.DataFrame],
    active_summaries: dict[str, pl.DataFrame],
) -> str:
    """Return the run summary HTML section."""
    cards = [
        ("Total findings", findings.height),
        ("Active findings", active_findings.height),
        ("Suppressed findings", findings.height - active_findings.height),
    ]
    card_html = "\n".join(_html_summary_card(label, value) for label, value in cards)
    triage_card_html = "\n".join(
        _html_summary_card(label, value)
        for label, value in _reviewer_triage_counts(active_findings)
    )
    content = "\n".join(
        [
            '<div class="pc-card-row">',
            card_html,
            "</div>",
            "<h3>Reviewer Triage</h3>",
            '<div class="pc-card-row pc-triage-row">',
            triage_card_html,
            "</div>",
            "<h3>Active Findings By Code</h3>",
            _html_table(active_summaries["by_code"], [_pc_findings.FINDING_CODE, _COUNT]),
            "<h3>Active Findings By Dataset</h3>",
            _html_table(active_summaries["by_dataset"], [_pc_findings.DATASET, _COUNT]),
            "<h3>Findings By Suppression State</h3>",
            _html_table(summaries["by_suppressed"], [_pc_findings.SUPPRESSED, _COUNT]),
        ]
    )
    return _html_section("Run Summary", content)


def _html_problems_section(findings: pl.DataFrame) -> str:
    """Return the first-screen actionable problem grid."""
    problems = _problem_table(findings)
    if problems.is_empty():
        return _html_section(
            "Problems",
            _html_empty("No actionable performance comparison problems."),
        )

    content = "\n".join(
        [
            _html_problems_summary(problems),
            (
                '<p class="pc-note">Start here. Each row is one problem; the '
                'Action Required cell is the recommended next step.</p>'
            ),
            _html_dashboard_filters(),
            _html_problems_table(problems),
            '<p class="pc-dashboard-no-results" hidden>No problem rows match the filters.</p>',
        ]
    )
    return _html_section("Problems", content)


def _problem_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return one actionable problem row per changed portfolio period."""
    dashboard = _review_dashboard_table(findings)
    if dashboard.is_empty():
        return pl.DataFrame(
            schema={
                _REVIEW_KEY: pl.String,
                _pc_findings.PORTFOLIO_ID: pl.String,
                _pc_findings.FROM_DATE: pl.Date,
                _pc_findings.THRU_DATE: pl.Date,
                _pc_explain.PORTFOLIO_RETURN_DELTA: pl.Float64,
                _REVIEW_STATUS: pl.String,
                _PROBLEM: pl.String,
                _ACTION_REQUIRED: pl.String,
                _WHY_IT_MATTERS: pl.String,
                _EVIDENCE_SECTION: pl.String,
                _DASHBOARD_MISSING_INPUTS: pl.String,
                _DASHBOARD_OPEN_SECTION: pl.String,
            }
        )

    rows = [
        _problem_row(row)
        for row in dashboard.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).select(
        [
            _REVIEW_KEY,
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
            _pc_explain.PORTFOLIO_RETURN_DELTA,
            _REVIEW_STATUS,
            _PROBLEM,
            _ACTION_REQUIRED,
            _WHY_IT_MATTERS,
            _EVIDENCE_SECTION,
            _DASHBOARD_MISSING_INPUTS,
            _DASHBOARD_OPEN_SECTION,
        ]
    )


def _problem_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return one action-oriented problem row from dashboard data."""
    missing_inputs = row.get(_DASHBOARD_MISSING_INPUTS)
    primary_cue = _format_value(row.get(_PRIMARY_REVIEW_CUE))
    return {
        _REVIEW_KEY: row.get(_REVIEW_KEY),
        _pc_findings.PORTFOLIO_ID: row.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: row.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: row.get(_pc_findings.THRU_DATE),
        _pc_explain.PORTFOLIO_RETURN_DELTA: row.get(
            _pc_explain.PORTFOLIO_RETURN_DELTA
        ),
        _REVIEW_STATUS: row.get(_REVIEW_STATUS),
        _PROBLEM: _problem_text(row),
        _ACTION_REQUIRED: _problem_action_required(
            missing_inputs=missing_inputs,
            primary_cue=primary_cue,
        ),
        _WHY_IT_MATTERS: _problem_why_it_matters(
            missing_inputs=missing_inputs,
            primary_cue=primary_cue,
        ),
        _EVIDENCE_SECTION: _dashboard_section_label(
            _format_value(row.get(_DASHBOARD_OPEN_SECTION))
        ),
        _DASHBOARD_MISSING_INPUTS: missing_inputs,
        _DASHBOARD_OPEN_SECTION: row.get(_DASHBOARD_OPEN_SECTION),
    }


def _problem_text(row: Mapping[str, object]) -> str:
    """Return a plain-English problem statement."""
    missing_inputs = row.get(_DASHBOARD_MISSING_INPUTS)
    if _has_text(missing_inputs):
        return (
            "Return-impact estimate is blocked by missing configuration: "
            f"{_format_value(missing_inputs)}."
        )

    primary_cue = _format_value(row.get(_PRIMARY_REVIEW_CUE))
    if "low-confidence" in primary_cue:
        return "Return difference relies on a low-confidence screening estimate."
    if _has_text(row.get(_DASHBOARD_CONTEXT_CUE)):
        return f"Context evidence needs review: {row[_DASHBOARD_CONTEXT_CUE]}."
    if primary_cue and primary_cue != "No review cue.":
        return primary_cue
    return _format_value(row.get(_DASHBOARD_MAIN_ISSUE))


def _problem_action_required(
    *,
    missing_inputs: object,
    primary_cue: str,
) -> str:
    """Return the concrete next action for a problem row."""
    if _has_text(missing_inputs):
        return _missing_inputs_action_required(missing_inputs)
    if "low-confidence" in primary_cue:
        return (
            "Decide whether the screening estimate is acceptable; if not, "
            "provide vendor contribution or mark the estimate review-only."
        )
    if "context:" in primary_cue:
        return (
            "Confirm whether the context-only change affects reviewer judgment; "
            "no YAML change is required unless it should affect return impact."
        )
    if "evidence-only" in primary_cue:
        return (
            "Add an explicit impact method for this evidence type or document "
            "that it should remain evidence-only."
        )
    return "Review the problem statement and update source data or YAML policy as needed."


def _missing_inputs_action_required(missing_inputs: object) -> str:
    """Return a YAML-oriented action for missing impact inputs."""
    text = _format_value(missing_inputs).lower()
    actions: list[str] = []
    if "market value" in text or "position" in text:
        actions.append("configure `position_impact_methods` for market value")
    if "price" in text:
        actions.append("configure `price_impact_methods` for price")
    if "transaction impact method" in text or "return-impact method" in text:
        actions.append("configure `transaction_impact_methods` with an explicit method")
    if "defensible impact method" in text:
        actions.append(
            "select the relevant `contribution_impact_methods` policy for "
            "evidence-only cause areas"
        )
    if "denominator" in text:
        actions.append("set `denominator_source` or map beginning market value")
    if "transaction sign" in text or "flow semantics" in text:
        actions.append("define transaction sign and external-flow semantics")
    if not actions:
        actions.append(f"provide {_format_value(missing_inputs)}")
    return f"Update the comparison YAML to {_join_action_phrases(actions)}; then rerun."


def _join_action_phrases(actions: Sequence[str]) -> str:
    """Return a compact natural-language action list."""
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} and {actions[1]}"
    return f"{', '.join(actions[:-1])}, and {actions[-1]}"


def _problem_why_it_matters(
    *,
    missing_inputs: object,
    primary_cue: str,
) -> str:
    """Return why a problem matters to the reviewer."""
    if _has_text(missing_inputs):
        return (
            "ppar can show evidence, but it cannot calculate a defensible "
            "return-impact estimate until the policy is explicit."
        )
    if "low-confidence" in primary_cue:
        return "The estimate is useful for screening, not final attribution."
    if "evidence-only" in primary_cue:
        return "The report has evidence but no accepted impact calculation."
    return "This row may explain or qualify the reported return difference."


def _review_dashboard_table(findings: pl.DataFrame) -> pl.DataFrame:
    """Return compact portfolio-period rows for the HTML review dashboard."""
    needs_review = _needs_review_summary_table(findings)
    if needs_review.is_empty():
        return pl.DataFrame(
            schema={
                _REVIEW_KEY: pl.String,
                _pc_findings.PORTFOLIO_ID: pl.String,
                _pc_findings.FROM_DATE: pl.Date,
                _pc_findings.THRU_DATE: pl.Date,
                _pc_explain.PORTFOLIO_RETURN_DELTA: pl.Float64,
                _REVIEW_STATUS: pl.String,
                _PRIMARY_REVIEW_CUE: pl.String,
                _SUGGESTED_NEXT_STEP: pl.String,
                _DASHBOARD_COVERAGE_COUNTS: pl.String,
                _DASHBOARD_MISSING_INPUTS: pl.String,
                _DASHBOARD_CONTEXT_CUE: pl.String,
                _DASHBOARD_MAIN_ISSUE: pl.String,
                _DASHBOARD_OPEN_SECTION: pl.String,
                _pc_explain.IMPACT_COVERAGE_STATUS: pl.String,
                _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE: pl.String,
                "_missing_input_rank": pl.Int64,
            }
        )

    coverage_by_period = _period_rows_by_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
    )
    rows = [
        _review_dashboard_row(
            period=row,
            coverage=coverage_by_period.get(_period_key(row), []),
        )
        for row in needs_review.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).sort(
        [
            "_review_status_rank",
            "_missing_input_rank",
            "_absolute_return_delta",
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
        ],
        descending=[False, False, True, False, False, False],
    ).select(
        [
            _REVIEW_KEY,
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
            _pc_explain.PORTFOLIO_RETURN_DELTA,
            _REVIEW_STATUS,
            _PRIMARY_REVIEW_CUE,
            _SUGGESTED_NEXT_STEP,
            _DASHBOARD_COVERAGE_COUNTS,
            _DASHBOARD_MISSING_INPUTS,
            _DASHBOARD_CONTEXT_CUE,
            _DASHBOARD_MAIN_ISSUE,
            _DASHBOARD_OPEN_SECTION,
            _pc_explain.IMPACT_COVERAGE_STATUS,
            _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE,
        ]
    )


def _review_dashboard_row(
    *,
    period: Mapping[str, object],
    coverage: list[dict[str, object]],
) -> dict[str, object]:
    """Return one compact dashboard row for a portfolio period."""
    coverage_row = coverage[0] if coverage else {}
    return_delta = period.get(_pc_explain.PORTFOLIO_RETURN_DELTA)
    primary_cue = _primary_review_cue(period.get(_REVIEW_CUES))
    missing_inputs = coverage_row.get(_pc_explain.MISSING_IMPACT_INPUTS, "")
    context_cue = _dashboard_context_cue(period.get(_REVIEW_CUES))
    return {
        _REVIEW_KEY: period.get(_REVIEW_KEY),
        _pc_findings.PORTFOLIO_ID: period.get(_pc_findings.PORTFOLIO_ID),
        _pc_findings.FROM_DATE: period.get(_pc_findings.FROM_DATE),
        _pc_findings.THRU_DATE: period.get(_pc_findings.THRU_DATE),
        _pc_explain.PORTFOLIO_RETURN_DELTA: return_delta,
        _REVIEW_STATUS: period.get(_REVIEW_STATUS),
        _PRIMARY_REVIEW_CUE: primary_cue,
        _SUGGESTED_NEXT_STEP: period.get(_SUGGESTED_NEXT_STEP),
        _DASHBOARD_COVERAGE_COUNTS: _dashboard_coverage_counts(coverage_row),
        _DASHBOARD_MISSING_INPUTS: missing_inputs,
        _DASHBOARD_CONTEXT_CUE: context_cue,
        _DASHBOARD_MAIN_ISSUE: _dashboard_main_issue(
            primary_cue=primary_cue,
            missing_inputs=missing_inputs,
            context_cue=context_cue,
            coverage_row=coverage_row,
        ),
        _DASHBOARD_OPEN_SECTION: _dashboard_open_section(
            primary_cue=primary_cue,
            missing_inputs=missing_inputs,
            context_cue=context_cue,
        ),
        _pc_explain.IMPACT_COVERAGE_STATUS: coverage_row.get(
            _pc_explain.IMPACT_COVERAGE_STATUS,
            "",
        ),
        _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE: coverage_row.get(
            _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE,
            "",
        ),
        "_review_status_rank": _review_status_rank(period.get(_REVIEW_STATUS)),
        "_missing_input_rank": _missing_input_rank(
            coverage_row.get(_pc_explain.MISSING_IMPACT_INPUTS)
        ),
        "_absolute_return_delta": _absolute_numeric_value(return_delta),
    }


def _html_problems_summary(problems: pl.DataFrame) -> str:
    """Return a compact problem-grid scope summary."""
    problem_count = problems.height
    portfolio_count = problems.select(
        pl.col(_pc_findings.PORTFOLIO_ID).n_unique()
    ).item()
    needs_review_count = problems.filter(
        pl.col(_REVIEW_STATUS) == _REVIEW_STATUS_NEEDS_REVIEW
    ).height
    return (
        '<p class="pc-dashboard-summary">'
        f"{_escape_html(needs_review_count)} of {_escape_html(problem_count)} "
        f"problem(s) need review across {_escape_html(portfolio_count)} "
        "portfolio(s).</p>"
    )


def _html_dashboard_filters() -> str:
    """Return lightweight dashboard filter controls."""
    return "\n".join(
        [
            '<form class="pc-dashboard-filters" data-dashboard-filters>',
            '<label for="pc-dashboard-search">Search</label>',
            (
                '<input id="pc-dashboard-search" type="search" '
                'placeholder="Portfolio, problem, or action" data-dashboard-search/>'
            ),
            '<label for="pc-dashboard-status">Status</label>',
            '<select id="pc-dashboard-status" data-dashboard-status>',
            '<option value="">All statuses</option>',
            '<option value="needs_review">Needs review</option>',
            '<option value="monitor">Monitor</option>',
            '<option value="clear">Clear</option>',
            "</select>",
            '<label class="pc-dashboard-checkbox">',
            '<input type="checkbox" data-dashboard-missing-only/>',
            "Missing inputs only",
            "</label>",
            '<button type="reset">Reset</button>',
            "</form>",
        ]
    )


def _primary_review_cue(cues: object) -> str:
    """Return the leading cue from a comma-separated cue list."""
    if not _has_text(cues):
        return "No review cue."
    return str(cues).split(",", maxsplit=1)[0].strip()


def _dashboard_coverage_counts(coverage_row: Mapping[str, object]) -> str:
    """Return estimated versus evidence-only coverage text for a dashboard row."""
    estimated = _count_value(coverage_row.get(_pc_explain.ESTIMATED_CAUSE_AREA_COUNT))
    evidence_only = _count_value(
        coverage_row.get(_pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT)
    )
    if estimated == 0 and evidence_only == 0:
        return ""
    return f"{estimated} estimated / {evidence_only} evidence-only"


def _dashboard_context_cue(cues: object) -> str:
    """Return high-priority context cue text when present."""
    cue = _cue_with_prefix(cues, "high-priority context:")
    if not cue:
        return ""
    return cue.removeprefix("high-priority context:").strip()


def _dashboard_main_issue(
    *,
    primary_cue: str,
    missing_inputs: object,
    context_cue: str,
    coverage_row: Mapping[str, object],
) -> str:
    """Return one plain-English issue for the dashboard row."""
    if _has_text(missing_inputs):
        return f"Missing inputs: {_format_value(missing_inputs)}"
    if context_cue:
        return f"Review context: {context_cue}"
    if primary_cue and primary_cue != "No review cue.":
        return primary_cue

    coverage_status = _format_value(coverage_row.get(_pc_explain.IMPACT_COVERAGE_STATUS))
    if coverage_status:
        return coverage_status.replace("_", " ")
    return "No specific issue identified."


def _dashboard_open_section(
    *,
    primary_cue: str,
    missing_inputs: object,
    context_cue: str,
) -> str:
    """Return the best first drilldown section for a dashboard row."""
    if _has_text(missing_inputs) or "evidence-only" in primary_cue:
        return "impact-coverage"
    if context_cue:
        return "context-evidence"
    if "low-confidence" in primary_cue:
        return "top-evidence"
    return "needs-review-summary"


def _cue_with_prefix(cues: object, prefix: str) -> str:
    """Return the first comma-separated cue with a specific prefix."""
    if not _has_text(cues):
        return ""
    for cue in str(cues).split(","):
        cleaned_cue = cue.strip()
        if cleaned_cue.startswith(prefix):
            return cleaned_cue
    return ""


def _review_status_rank(status: object) -> int:
    """Return dashboard sort priority for review status values."""
    if status == _REVIEW_STATUS_NEEDS_REVIEW:
        return 0
    if status == _REVIEW_STATUS_MONITOR:
        return 1
    return 2


def _missing_input_rank(value: object) -> int:
    """Return dashboard sort priority for missing impact inputs."""
    return 0 if _has_text(value) else 1


def _absolute_numeric_value(value: object) -> float:
    """Return absolute numeric value for dashboard sorting."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    return 0.0


def _html_problems_table(problems: pl.DataFrame) -> str:
    """Return the first-pass actionable problems grid."""
    rows = [
        _html_problem_table_row(row)
        for row in problems.iter_rows(named=True)
    ]
    return "\n".join(
        [
            '<div class="pc-dashboard-table-wrap">',
            '<table class="pc-dashboard-table">',
            "<caption>Actionable performance comparison problems.</caption>",
            "<thead>",
            "<tr>",
            _html_dashboard_sort_header("Severity", "severity"),
            _html_dashboard_sort_header("Portfolio", "portfolio"),
            _html_dashboard_sort_header("Period", "period"),
            _html_dashboard_sort_header("Return Delta", "return-delta"),
            _html_dashboard_sort_header("Problem", "problem"),
            _html_dashboard_sort_header("Action Required", "action"),
            _html_dashboard_sort_header("Why It Matters", "why"),
            _html_dashboard_sort_header("Evidence", "evidence"),
            "</tr>",
            "</thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )


def _html_problem_table_row(row: Mapping[str, object]) -> str:
    """Return one compact problem-grid row."""
    status = _format_value(row.get(_REVIEW_STATUS))
    missing_inputs = row.get(_DASHBOARD_MISSING_INPUTS)
    period = (
        f"{_format_value(row.get(_pc_findings.FROM_DATE))} to "
        f"{_format_value(row.get(_pc_findings.THRU_DATE))}"
    )
    search_text = _html_dashboard_search_text(row)
    missing_inputs_token = _boolean_token(_has_text(missing_inputs))
    row_id = _html_dashboard_link_target(
        "problems",
        _format_value(row.get(_REVIEW_KEY)),
    )
    article_attributes = " ".join(
        [
            f'class="pc-dashboard-row pc-dashboard-{_css_token(status)}"',
            f'id="{row_id}"',
            "data-dashboard-row",
            f'data-review-status="{_escape_html(status)}"',
            f'data-missing-inputs="{_escape_html(missing_inputs_token)}"',
            f'data-dashboard-search="{_escape_html(search_text)}"',
            f'data-sort-severity="{_escape_html(status)}"',
            f'data-sort-portfolio="{_escape_html(row.get(_pc_findings.PORTFOLIO_ID))}"',
            f'data-sort-period="{_escape_html(period)}"',
            (
                'data-sort-return-delta="'
                f'{_escape_html(row.get(_pc_explain.PORTFOLIO_RETURN_DELTA))}"'
            ),
            f'data-sort-problem="{_escape_html(row.get(_PROBLEM))}"',
            f'data-sort-action="{_escape_html(row.get(_ACTION_REQUIRED))}"',
            f'data-sort-why="{_escape_html(row.get(_WHY_IT_MATTERS))}"',
            f'data-sort-evidence="{_escape_html(row.get(_EVIDENCE_SECTION))}"',
        ]
    )
    return "\n".join(
        [
            f"<tr {article_attributes}>",
            _html_dashboard_status_cell(status),
            _html_dashboard_table_cell(row.get(_pc_findings.PORTFOLIO_ID)),
            _html_dashboard_table_cell(period),
            _html_dashboard_table_cell(
                row.get(_pc_explain.PORTFOLIO_RETURN_DELTA),
                numeric=True,
            ),
            _html_dashboard_table_cell(row.get(_PROBLEM)),
            _html_dashboard_table_cell(row.get(_ACTION_REQUIRED)),
            _html_dashboard_table_cell(row.get(_WHY_IT_MATTERS)),
            f"<td>{_html_problem_evidence_link(row)}</td>",
            "</tr>",
        ]
    )


def _html_dashboard_sort_header(label: str, sort_key: str) -> str:
    """Return a sortable dashboard table header."""
    return (
        '<th scope="col">'
        f'<button type="button" data-dashboard-sort="{sort_key}">'
        f"{_escape_html(label)}</button></th>"
    )


def _html_dashboard_search_text(row: Mapping[str, object]) -> str:
    """Return searchable dashboard row text."""
    values = [
        row.get(_pc_findings.PORTFOLIO_ID),
        row.get(_pc_findings.FROM_DATE),
        row.get(_pc_findings.THRU_DATE),
        row.get(_REVIEW_STATUS),
        row.get(_PROBLEM),
        row.get(_ACTION_REQUIRED),
        row.get(_WHY_IT_MATTERS),
        row.get(_DASHBOARD_MISSING_INPUTS),
        row.get(_EVIDENCE_SECTION),
    ]
    return " ".join(_format_value(value) for value in values if _has_text(value)).lower()


def _html_dashboard_table_cell(value: object, *, numeric: bool = False) -> str:
    """Return one compact dashboard table cell."""
    alignment_class = "pc-right" if numeric else "pc-left"
    return f'<td class="{alignment_class}">{_escape_html(_format_value(value))}</td>'


def _html_dashboard_status_cell(status: str) -> str:
    """Return the dashboard status cell with stable visual class."""
    return (
        f'<td class="pc-left pc-status-{_css_token(status)}">'
        f"{_escape_html(status)}</td>"
    )


def _boolean_token(value: bool) -> str:
    """Return a lower-case JavaScript-friendly boolean token."""
    return "true" if value else "false"


def _html_problem_evidence_link(row: Mapping[str, object]) -> str:
    """Return the optional evidence link for one problem row."""
    review_key = _format_value(row.get(_REVIEW_KEY))
    section_id = _format_value(row.get(_DASHBOARD_OPEN_SECTION))
    target = _html_dashboard_link_target(section_id, review_key)
    label = _format_value(row.get(_EVIDENCE_SECTION))
    return (
        '<a class="pc-problem-evidence-link" '
        f'href="#{target}">{_escape_html(label)}</a>'
    )


def _dashboard_section_label(section_id: str) -> str:
    """Return a reviewer-facing label for a dashboard detail section."""
    labels = {
        "context-evidence": "Context Evidence",
        "impact-coverage": "Impact Coverage",
        "needs-review-summary": "Needs Review Summary",
        "top-evidence": "Top Evidence",
    }
    return labels.get(section_id, section_id.replace("-", " ").title())


def _html_dashboard_link_target(section_id: str, review_key: str) -> str:
    """Return a dashboard link target for a section and review key."""
    if not review_key:
        return section_id
    return _html_review_key_row_id(section_id, review_key)


def _html_needs_review_summary_section(findings: pl.DataFrame) -> str:
    """Return changed-period reviewer cues as an HTML section."""
    return _html_section(
        "Needs Review Summary",
        _html_table(
            _needs_review_summary_table(findings),
            _NEEDS_REVIEW_COLUMNS,
            empty_message="No changed portfolio periods need review.",
            row_id_prefix="needs-review-summary",
        ),
    )


def _html_portfolio_period_narrative_section(findings: pl.DataFrame) -> str:
    """Return conservative narrative summaries as HTML."""
    summary = _pc_explain.portfolio_period_summary(findings)
    if summary.is_empty():
        return _html_section(
            "Portfolio-Period Narrative",
            _html_empty("No portfolio return changes to narrate."),
        )

    causes = _pc_explain.portfolio_period_cause_summary(findings)
    paragraphs = [
        _html_paragraph(
            _portfolio_period_narrative(period, _period_cause_rows(causes, period))
        )
        for period in summary.iter_rows(named=True)
    ]
    return _html_section("Portfolio-Period Narrative", "\n".join(paragraphs))


def _html_review_notes_section(findings: pl.DataFrame) -> str:
    """Return current model-limit review notes as HTML."""
    notes = [_ACTIVE_ONLY_NOTE, _NO_ESTIMATE_NOTE]
    causes = _pc_explain.portfolio_period_cause_summary(findings)
    if causes.is_empty():
        return _html_section("Review Notes", _html_list(notes))

    model_notes = _review_notes_for_cause_rows(list(causes.iter_rows(named=True)))
    if not model_notes:
        model_notes = [
            "No model-limit review notes were generated for the current evidence mix.",
        ]
    notes.extend(model_notes)
    return _html_section("Review Notes", _html_list(notes))


def _html_evidence_appendix_section(sections: Sequence[tuple[str, str]]) -> str:
    """Return optional backing evidence sections."""
    return _html_detail_group_section(
        "Evidence Appendix",
        (
            "Use this appendix only when you need to audit a Problems-grid row. "
            "The Problems grid is the primary review workflow."
        ),
        sections,
    )


def _html_detail_group_section(
    title: str,
    note: str,
    sections: Sequence[tuple[str, str]],
) -> str:
    """Return a grouped set of collapsible HTML detail sections."""
    detail_items = [
        _html_detail_group_item(section_title, content)
        for section_title, content in sections
    ]
    content = "\n".join(
        [
            f'<p class="pc-note">{_escape_html(note)}</p>',
            *detail_items,
        ]
    )
    return _html_section(title, content)


def _html_detail_group_item(title: str, content: str) -> str:
    """Return one native disclosure item containing a report section."""
    return "\n".join(
        [
            '<details class="pc-detail">',
            f"<summary>{_escape_html(title)}</summary>",
            content,
            "</details>",
        ]
    )


def _html_impact_estimate_summary_section(findings: pl.DataFrame) -> str:
    """Return quantified impact estimates as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.ROOT_CAUSE_AREA,
        _pc_explain.ESTIMATED_RETURN_IMPACT,
        _pc_explain.IMPACT_BASIS,
        _pc_explain.IMPACT_CONFIDENCE,
        _pc_explain.IMPACT_MESSAGE,
    ]
    return _html_section(
        "Impact Estimate Summary",
        _html_table(
            _impact_estimate_summary_table(findings),
            columns,
            empty_message="No impact estimates are currently available.",
        ),
    )


def _html_impact_coverage_section(findings: pl.DataFrame) -> str:
    """Return estimate-coverage status as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _pc_explain.ROOT_CAUSE_AREA_COUNT,
        _pc_explain.ESTIMATED_CAUSE_AREA_COUNT,
        _pc_explain.EVIDENCE_ONLY_CAUSE_AREA_COUNT,
        _pc_explain.LOW_CONFIDENCE_ESTIMATE_COUNT,
        _pc_explain.MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
        _pc_explain.ESTIMATED_RETURN_IMPACT_TOTAL,
        _pc_explain.EVIDENCE_ONLY_AREAS,
        _pc_explain.TRANSACTION_SEMANTICS_SOURCES,
        _pc_explain.MISSING_IMPACT_INPUTS,
        _pc_explain.IMPACT_COVERAGE_STATUS,
        _pc_explain.IMPACT_COVERAGE_REVIEW_NOTE,
        _pc_explain.IMPACT_MESSAGE,
    ]
    return _html_section(
        "Impact Coverage",
        _html_table(
            _pc_explain.portfolio_period_impact_coverage_summary(findings),
            columns,
            empty_message="No portfolio return changes need impact coverage review.",
            row_id_prefix="impact-coverage",
        ),
    )


def _html_context_evidence_summary_section(findings: pl.DataFrame) -> str:
    """Return context-only evidence counts as an HTML section."""
    return _html_section(
        "Context Evidence Summary",
        _html_table(
            _context_evidence_summary_table(findings),
            _CONTEXT_EVIDENCE_SUMMARY_COLUMNS,
            empty_message="No context-only evidence summary.",
        ),
    )


def _html_context_evidence_section(findings: pl.DataFrame) -> str:
    """Return context-only evidence as an HTML section."""
    return _html_section(
        "Context Evidence",
        _html_table(
            _context_evidence_table(findings),
            _CONTEXT_EVIDENCE_COLUMNS,
            empty_message="No context-only evidence.",
            row_id_prefix="context-evidence",
        ),
    )


def _html_transaction_cross_checks_section(findings: pl.DataFrame) -> str:
    """Return transaction cross-check diagnostics as an HTML section."""
    return _html_section(
        "Transaction Cross-Checks",
        _html_table(
            _pc_explain.portfolio_period_transaction_cross_checks(findings),
            list(_pc_explain.PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS),
            empty_message="No transaction cross-check estimates are available.",
            row_id_prefix="transaction-cross-checks",
        ),
    )


def _html_flow_cross_check_reconciliation_section(findings: pl.DataFrame) -> str:
    """Return flow/cross-check reconciliation diagnostics as HTML."""
    return _html_section(
        "Flow Cross-Check Reconciliation",
        _html_table(
            _pc_explain.portfolio_period_flow_cross_check_reconciliation(findings),
            list(_pc_explain.PORTFOLIO_PERIOD_FLOW_CROSS_CHECK_RECONCILIATION_COLUMNS),
            empty_message="No flow/cross-check reconciliation rows are available.",
            row_id_prefix="flow-cross-check-reconciliation",
        ),
    )


def _html_residual_status_section(findings: pl.DataFrame) -> str:
    """Return residual-status caveats and rows as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _ESTIMATED_IMPACT_AREAS,
        _RESIDUAL_STATUS,
        _RESIDUAL_REASON,
        _RESIDUAL_REVIEW_NOTE,
    ]
    content = "\n".join(
        [
            f'<p class="pc-note">{_escape_html(_RESIDUAL_STATUS_NOTE)}</p>',
            _html_table(
                _residual_status_table(findings),
                columns,
                empty_message="No portfolio return changes need residual review.",
                row_id_prefix="residual-status",
            ),
        ]
    )
    return _html_section("Residual Status", content)


def _html_transaction_activity_section(findings: pl.DataFrame) -> str:
    """Return changed transaction activity as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.SECURITY_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_findings.TRANSACTION_CATEGORY,
        _pc_explain.CHANGED_FIELDS,
        _pc_explain.AMOUNT_DELTA,
        _pc_explain.QUANTITY_DELTA,
        _pc_explain.PRICE_DELTA,
        _pc_explain.TRANSACTION_SEMANTICS_SOURCES,
        _pc_explain.TRANSACTION_MATCH_STATUSES,
        _pc_explain.MISSING_IMPACT_INPUTS,
    ]
    return _html_section(
        "Transaction Activity",
        _html_table(
            _pc_explain.transaction_activity_summary(findings),
            columns,
            empty_message="No changed transaction activity.",
            row_id_prefix="transaction-activity",
        ),
    )


def _html_transaction_matching_diagnostics_section(findings: pl.DataFrame) -> str:
    """Return transaction matching status diagnostics as an HTML section."""
    columns = [
        _pc_findings.TRANSACTION_MATCH_STATUS,
        _pc_explain.FINDING_COUNT,
        _pc_explain.TRANSACTION_MATCH_REVIEW_NOTE,
    ]
    return _html_section(
        "Transaction Matching Diagnostics",
        _html_table(
            _pc_explain.transaction_matching_diagnostics(findings),
            columns,
            empty_message="No transaction matching diagnostics.",
        ),
    )


def _html_portfolio_period_section(findings: pl.DataFrame) -> str:
    """Return portfolio-period return changes as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        _pc_explain.PORTFOLIO_RETURN_DELTA,
        _pc_explain.FINDING_COUNT,
        _pc_explain.HAS_SUPPRESSED_FINDINGS,
    ]
    return _html_section(
        "Portfolio-Period Changes",
        _html_table(
            _pc_explain.portfolio_period_summary(findings),
            columns,
            empty_message="No portfolio return changes.",
            row_id_prefix="portfolio-period-changes",
        ),
    )


def _html_cause_summary_section(findings: pl.DataFrame) -> str:
    """Return cause-area summaries as an HTML section."""
    columns = [
        column
        for column in _pc_explain.PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS
        if column
        in {
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
            _pc_explain.ROOT_CAUSE_AREA,
            _pc_explain.FINDING_COUNT,
            _pc_explain.ESTIMATED_RETURN_IMPACT,
            _pc_explain.IMPACT_BASIS,
            _pc_explain.IMPACT_CONFIDENCE,
            _pc_explain.TOP_CODES,
            _pc_explain.IMPACT_MESSAGE,
        }
    ]
    return _html_section(
        "Cause Summary",
        _html_table(
            _pc_explain.portfolio_period_cause_summary(findings),
            columns,
            empty_message="No cause summary available.",
            row_id_prefix="cause-summary",
        ),
    )


def _html_top_evidence_section(
    findings: pl.DataFrame,
    top_evidence_limit: int,
) -> str:
    """Return top contribution-candidate evidence as an HTML section."""
    columns = [
        _pc_findings.PORTFOLIO_ID,
        _pc_findings.FROM_DATE,
        _pc_findings.THRU_DATE,
        "review_rank",
        _pc_findings.FINDING_CODE,
        _pc_findings.DATASET,
        _pc_findings.EVIDENCE_ROLE,
        _pc_findings.SECURITY_ID,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.TRANSACTION_SEMANTICS_SOURCE,
        _pc_findings.TRANSACTION_MATCH_STATUS,
        _pc_findings.IMPACT_POLICY,
        _pc_findings.TRANSACTION_IMPACT_POLICY,
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC,
        _pc_findings.TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
        _pc_findings.DELTA_B_MINUS_A,
        _pc_explain.ESTIMATED_RETURN_IMPACT,
        _pc_explain.IMPACT_BASIS,
        _pc_explain.IMPACT_CONFIDENCE,
        _pc_explain.IMPACT_METHOD,
        _pc_explain.IMPACT_MESSAGE,
        _pc_findings.MESSAGE,
    ]
    return _html_section(
        "Top Evidence",
        _html_table(
            _pc_explain.top_evidence_table(findings, top_evidence_limit),
            columns,
            empty_message="No ranked evidence is available.",
            row_id_prefix="top-evidence",
        ),
    )


def _html_suppressed_appendix_section(
    findings: pl.DataFrame,
    summaries: dict[str, pl.DataFrame],
) -> str:
    """Return suppressed findings audit appendix as HTML."""
    suppressed = (
        findings.filter(pl.col(_pc_findings.SUPPRESSED))
        if not findings.is_empty()
        else findings
    )
    content = "\n".join(
        [
            "<h3>Suppressed Counts By Code</h3>",
            _html_table(
                summaries["by_code_suppressed"].filter(pl.col(_pc_findings.SUPPRESSED))
                if not summaries["by_code_suppressed"].is_empty()
                else summaries["by_code_suppressed"],
                [_pc_findings.FINDING_CODE, _pc_findings.SUPPRESSED, _COUNT],
                empty_message="No suppressed findings.",
            ),
            "<h3>Suppressed Finding Detail</h3>",
            _html_table(
                _pc_runner.compact_findings_table(suppressed, include_suppressed=True),
                [
                    _pc_findings.FINDING_CODE,
                    _pc_findings.DATASET,
                    _pc_findings.EVIDENCE_ROLE,
                    _pc_findings.PORTFOLIO_ID,
                    _pc_findings.SECURITY_ID,
                    _pc_findings.FROM_DATE,
                    _pc_findings.THRU_DATE,
                    _pc_findings.SOURCE_COLUMN,
                    _pc_findings.DELTA_B_MINUS_A,
                    _pc_findings.MESSAGE,
                ],
                empty_message="No suppressed finding detail.",
            ),
        ]
    )
    return _html_section("Suppressed Findings Appendix", content)


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
