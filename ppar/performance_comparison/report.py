"""Render performance comparison findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping, Sequence
import datetime as dt
import html as html_lib
import json
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.errors import PpaError
from ppar.performance_comparison import explain as _pc_explain
from ppar.performance_comparison import findings as _pc_findings
from ppar.performance_comparison import runner as _pc_runner

__all__ = [
    "performance_comparison_html_report",
    "performance_comparison_markdown_report",
    "write_performance_comparison_html_report",
    "write_performance_comparison_markdown_report",
    "write_performance_comparison_report_bundle",
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
_RESIDUAL_WITHHELD = "withheld"
_RESIDUAL_STATUS_NOTE = (
    "Residual amounts are intentionally withheld until the attribution model has "
    "enough defensible, non-overlapping impact estimates."
)
_HTML_REPORT_KICKER = "Performance Comparison Review"
_HTML_REPORT_SUBTITLE = "Exception Review Worksheet"
_REVIEW_STATUS = "review_status"
_REVIEW_CUES = "review_cues"
_SUGGESTED_NEXT_STEP = "suggested_next_step"
_REVIEW_STATUS_NEEDS_REVIEW = "needs_review"
_REVIEW_STATUS_MONITOR = "monitor"
_REVIEW_STATUS_CLEAR = "clear"
_NEEDS_REVIEW_COLUMNS = (
    _pc_findings.PORTFOLIO_ID,
    _pc_findings.FROM_DATE,
    _pc_findings.THRU_DATE,
    _pc_explain.PORTFOLIO_RETURN_DELTA,
    _REVIEW_STATUS,
    _REVIEW_CUES,
    _SUGGESTED_NEXT_STEP,
)
_REPORT_BUNDLE_REQUIRED_ARTIFACTS = (
    "report",
    "html_report",
    "readme",
    "manifest",
    "findings",
    "needs_review_summary",
    "portfolio_period_summary",
    "cause_summary",
    "impact_estimates",
    "impact_coverage",
    "transaction_cross_checks",
    "flow_cross_check_reconciliation",
    "residual_status",
    "transaction_activity",
    "transaction_matching_diagnostics",
    "top_evidence",
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
    sections = [
        _html_run_summary_section(findings, active_findings, summaries, active_summaries),
        _html_needs_review_summary_section(active_findings),
        _html_portfolio_period_narrative_section(active_findings),
        _html_review_notes_section(active_findings),
        _html_impact_estimate_summary_section(active_findings),
        _html_impact_coverage_section(active_findings),
        _html_transaction_cross_checks_section(active_findings),
        _html_flow_cross_check_reconciliation_section(active_findings),
        _html_residual_status_section(active_findings),
        _html_transaction_activity_section(active_findings),
        _html_transaction_matching_diagnostics_section(active_findings),
        _html_portfolio_period_section(active_findings),
        _html_cause_summary_section(active_findings),
        _html_top_evidence_section(active_findings, top_evidence_limit),
    ]
    if include_suppressed_appendix:
        sections.append(_html_suppressed_appendix_section(findings, summaries))

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
            f'<p class="pc-kicker">{_escape_html(_HTML_REPORT_KICKER)}</p>',
            f"<h1>{_escape_html(title)}</h1>",
            f'<p class="pc-subtitle">{_escape_html(_HTML_REPORT_SUBTITLE)}</p>',
            '<div class="pc-header-notes">',
            f"<p>{_escape_html(_ACTIVE_ONLY_NOTE)}</p>",
            f"<p>{_escape_html(_NO_ESTIMATE_NOTE)}</p>",
            "</div>",
            "</header>",
            _html_contents_section(include_suppressed_appendix=include_suppressed_appendix),
            *sections,
            "</main>",
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

    Returns:
        Mapping from bundle artifact name to normalized written path.
    """
    bundle_directory = Path(output_directory)
    bundle_directory.mkdir(parents=True, exist_ok=True)
    active_findings = _active_findings(findings)
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
    paths["findings"] = _write_csv(findings, bundle_directory / "findings.csv")
    for name, table in tables.items():
        paths[name] = _write_csv(table, bundle_directory / f"{name}.csv")
    paths["readme"] = _write_report_bundle_readme(
        bundle_directory / "README.md",
        title=title,
        tables=tables,
    )

    manifest_path = bundle_directory / "manifest.json"
    paths["manifest"] = manifest_path
    manifest = _report_bundle_manifest(
        findings=findings,
        active_findings=active_findings,
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
        artifact_paths=paths,
        tables=tables,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding=util.ENCODING,
    )
    validation_issues = _report_bundle_validation_issues(bundle_directory)
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
    return {
        "needs_review_summary": _needs_review_summary_table(active_findings),
        "portfolio_period_summary": _pc_explain.portfolio_period_summary(
            active_findings
        ),
        "cause_summary": _pc_explain.portfolio_period_cause_summary(active_findings),
        "impact_estimates": _impact_estimate_summary_table(active_findings),
        "impact_coverage": _pc_explain.portfolio_period_impact_coverage_summary(
            active_findings
        ),
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
        "top_evidence": _top_evidence_table(active_findings, top_evidence_limit),
    }


def _write_csv(table: pl.DataFrame, output_path: Path) -> Path:
    """Write a CSV table and return the normalized path."""
    table.write_csv(output_path)
    return output_path


def _write_report_bundle_readme(
    output_path: Path,
    *,
    title: str,
    tables: Mapping[str, pl.DataFrame],
) -> Path:
    """Write a short bundle README and return the normalized path."""
    lines = [
        f"# {_escape_markdown_text(title)}",
        "",
        "This directory is a portable performance-comparison review bundle.",
        "Open `report.html` for the browser report, or `report.md` for a plain-text review.",
        "",
        "## Primary Artifacts",
        "",
        "- `report.html`: standalone browser report with reviewer cues and tables.",
        "- `report.md`: Markdown version of the same review narrative.",
        "- `manifest.json`: machine-readable artifact and row-count metadata.",
        "- `findings.csv`: complete finding-level comparison output.",
        "",
        "## Review Tables",
        "",
        *_report_bundle_readme_table_lines(tables),
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding=util.ENCODING)
    return output_path


def _report_bundle_readme_table_lines(tables: Mapping[str, pl.DataFrame]) -> list[str]:
    """Return README bullets for report-bundle table artifacts."""
    descriptions = {
        "needs_review_summary": (
            "top triage table for changed portfolio periods and suggested next steps"
        ),
        "portfolio_period_summary": "portfolio-period return-change summary",
        "cause_summary": "cause-area summary with selected impact estimates",
        "impact_estimates": "currently quantified impact estimates",
        "impact_coverage": "period-level estimate coverage and missing inputs",
        "transaction_cross_checks": "review-only transaction impact cross-checks",
        "flow_cross_check_reconciliation": "flow/cross-check reconciliation diagnostics",
        "residual_status": "residual caveat status by changed portfolio period",
        "transaction_activity": "changed transaction activity and missing inputs",
        "transaction_matching_diagnostics": (
            "transaction matching status counts and review notes"
        ),
        "top_evidence": "ranked evidence rows shown in the report",
    }
    return [
        f"- `{name}.csv`: {descriptions.get(name, 'report helper table')} "
        f"({table.height} row(s))."
        for name, table in sorted(tables.items())
    ]


def _report_bundle_manifest(
    *,
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    title: str,
    include_suppressed_appendix: bool,
    top_evidence_limit: int,
    artifact_paths: dict[str, Path],
    tables: dict[str, pl.DataFrame],
) -> dict[str, object]:
    """Return JSON-serializable metadata for a report bundle."""
    suppressed_count = findings.height - active_findings.height
    return {
        "bundle_type": "performance_comparison_report",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "title": title,
        "options": {
            "include_suppressed_appendix": include_suppressed_appendix,
            "top_evidence_limit": top_evidence_limit,
        },
        "counts": {
            "findings": findings.height,
            "active_findings": active_findings.height,
            "suppressed_findings": suppressed_count,
        },
        "artifacts": {
            name: path.name
            for name, path in sorted(artifact_paths.items())
        },
        "tables": {
            "findings": {"rows": findings.height},
            **{
                name: {"rows": table.height}
                for name, table in sorted(tables.items())
            },
        },
    }


def _report_bundle_validation_issues(bundle_directory: util.PathLike) -> list[str]:
    """Return validation issues for a generated report bundle."""
    bundle_path = Path(bundle_directory)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json is missing"]

    manifest = _read_report_bundle_manifest(manifest_path)
    if manifest is None:
        return ["manifest.json is not a JSON object"]

    issues: list[str] = []
    artifacts = _manifest_mapping(manifest, "artifacts")
    tables = _manifest_mapping(manifest, "tables")
    issues.extend(_report_bundle_artifact_issues(bundle_path, artifacts))
    issues.extend(_report_bundle_table_issues(bundle_path, artifacts, tables))
    return issues


def _read_report_bundle_manifest(manifest_path: Path) -> dict[str, object] | None:
    """Read a bundle manifest JSON object."""
    try:
        manifest_data: object = json.loads(manifest_path.read_text(encoding=util.ENCODING))
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest_data, dict):
        return None
    return {str(key): value for key, value in manifest_data.items()}


def _manifest_mapping(
    manifest: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    """Return a nested manifest mapping with string keys."""
    value = manifest.get(key)
    if not isinstance(value, dict):
        return {}
    return {str(inner_key): inner_value for inner_key, inner_value in value.items()}


def _report_bundle_artifact_issues(
    bundle_path: Path,
    artifacts: Mapping[str, object],
) -> list[str]:
    """Return missing or malformed artifact issues."""
    issues: list[str] = []
    for artifact_name in _REPORT_BUNDLE_REQUIRED_ARTIFACTS:
        artifact_file = artifacts.get(artifact_name)
        if not isinstance(artifact_file, str) or not artifact_file:
            issues.append(f"manifest artifact {artifact_name!r} is missing")
            continue
        if not (bundle_path / artifact_file).is_file():
            issues.append(f"artifact file {artifact_file!r} is missing")
    return issues


def _report_bundle_table_issues(
    bundle_path: Path,
    artifacts: Mapping[str, object],
    tables: Mapping[str, object],
) -> list[str]:
    """Return CSV table row-count and header validation issues."""
    issues: list[str] = []
    for table_name, metadata in tables.items():
        row_count = _manifest_table_row_count(metadata)
        if row_count is None:
            issues.append(f"manifest table {table_name!r} has no integer row count")
            continue
        artifact_name = "findings" if table_name == "findings" else table_name
        artifact_file = artifacts.get(artifact_name)
        if not isinstance(artifact_file, str) or not artifact_file:
            issues.append(f"manifest artifact {artifact_name!r} is missing")
            continue
        csv_path = bundle_path / artifact_file
        if not csv_path.exists():
            continue
        issues.extend(_csv_table_validation_issues(csv_path, table_name, row_count))
    return issues


def _manifest_table_row_count(metadata: object) -> int | None:
    """Return the manifest row count for a table."""
    if not isinstance(metadata, dict):
        return None
    row_count = metadata.get("rows")
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        return None
    if row_count < 0:
        return None
    return row_count


def _csv_table_validation_issues(
    csv_path: Path,
    table_name: str,
    expected_rows: int,
) -> list[str]:
    """Return validation issues for one CSV table artifact."""
    try:
        table = pl.read_csv(csv_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        return [f"table {table_name!r} could not be read: {error}"]

    issues: list[str] = []
    if table.height != expected_rows:
        issues.append(
            f"table {table_name!r} row count is {table.height}, expected {expected_rows}"
        )
    if expected_rows == 0 and not _csv_file_has_header(csv_path):
        issues.append(f"table {table_name!r} is empty and has no header")
    return issues


def _csv_file_has_header(csv_path: Path) -> bool:
    """Return whether a CSV artifact has a non-empty header line."""
    try:
        first_line = csv_path.read_text(encoding=util.ENCODING).splitlines()[0]
    except (IndexError, OSError, UnicodeDecodeError):
        return False
    return bool(first_line.strip())


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
    rows = [
        _needs_review_summary_row(
            period=period,
            coverage=coverage_by_period.get(_period_key(period), []),
            residual=residual_by_period.get(_period_key(period), []),
            cross_checks=cross_checks_by_period.get(_period_key(period), []),
        )
        for period in periods.iter_rows(named=True)
    ]
    return pl.DataFrame(rows).select(_NEEDS_REVIEW_COLUMNS)


def _empty_needs_review_summary() -> pl.DataFrame:
    """Return an empty needs-review summary with stable columns."""
    return pl.DataFrame(
        schema={
            _pc_findings.PORTFOLIO_ID: pl.String,
            _pc_findings.FROM_DATE: pl.Date,
            _pc_findings.THRU_DATE: pl.Date,
            _pc_explain.PORTFOLIO_RETURN_DELTA: pl.Float64,
            _REVIEW_STATUS: pl.String,
            _REVIEW_CUES: pl.String,
            _SUGGESTED_NEXT_STEP: pl.String,
        }
    )


def _needs_review_summary_row(
    *,
    period: dict[str, object],
    coverage: list[dict[str, object]],
    residual: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
) -> dict[str, object]:
    """Return one reviewer-cue row for a changed portfolio period."""
    cues = _needs_review_cues(
        coverage=coverage,
        residual=residual,
        cross_checks=cross_checks,
    )
    return {
        _pc_findings.PORTFOLIO_ID: period[_pc_findings.PORTFOLIO_ID],
        _pc_findings.FROM_DATE: period[_pc_findings.FROM_DATE],
        _pc_findings.THRU_DATE: period[_pc_findings.THRU_DATE],
        _pc_explain.PORTFOLIO_RETURN_DELTA: period[_pc_explain.PORTFOLIO_RETURN_DELTA],
        _REVIEW_STATUS: _needs_review_status(cues),
        _REVIEW_CUES: _comma_separated(cues),
        _SUGGESTED_NEXT_STEP: _suggested_next_step(cues),
    }


def _needs_review_cues(
    *,
    coverage: list[dict[str, object]],
    residual: list[dict[str, object]],
    cross_checks: list[dict[str, object]],
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
    if residual_row.get(_RESIDUAL_STATUS) == _RESIDUAL_WITHHELD:
        cues.append(f"residual {_RESIDUAL_WITHHELD}")
    return cues


def _needs_review_status(cues: Sequence[str]) -> str:
    """Return the triage status for a period's reviewer cues."""
    if not cues:
        return _REVIEW_STATUS_CLEAR
    if any(
        cue.startswith(("missing inputs:", "residual withheld"))
        or "evidence-only" in cue
        for cue in cues
    ):
        return _REVIEW_STATUS_NEEDS_REVIEW
    return _REVIEW_STATUS_MONITOR


def _suggested_next_step(cues: Sequence[str]) -> str:
    """Return a conservative next action for a period's reviewer cues."""
    if any(cue.startswith("missing inputs:") for cue in cues):
        return "Resolve missing impact inputs before interpreting estimates."
    if any("evidence-only" in cue for cue in cues):
        return "Review evidence-only areas before relying on impact totals."
    if any("transaction cross-check" in cue for cue in cues):
        return "Review transaction cross-checks separately from impact totals."
    if any("low-confidence" in cue for cue in cues):
        return "Review low-confidence estimates and supporting evidence."
    if any(cue == "residual withheld" for cue in cues):
        return "Keep residual caveat visible until estimates are complete."
    return "No changed portfolio-period review cue."


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


def _period_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    """Return the portfolio-period grouping key for a report row."""
    return (
        row[_pc_findings.PORTFOLIO_ID],
        row[_pc_findings.FROM_DATE],
        row[_pc_findings.THRU_DATE],
    )


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


def _report_section_names(*, include_suppressed_appendix: bool) -> list[str]:
    """Return report section names in display order."""
    section_names = [
        "Run Summary",
        "Needs Review Summary",
        "Portfolio-Period Narrative",
        "Review Notes",
        "Impact Estimate Summary",
        "Impact Coverage",
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
            }
        )

    causes = _pc_explain.portfolio_period_cause_summary(findings)
    return pl.DataFrame(
        [
            _residual_status_row(period, _period_cause_rows(causes, period))
            for period in periods.iter_rows(named=True)
        ]
    )


def _residual_status_row(
    period: dict[str, object],
    causes: list[dict[str, object]],
) -> dict[str, object]:
    """Return one residual-status row for a portfolio period."""
    estimated_areas = [
        str(cause[_pc_explain.ROOT_CAUSE_AREA])
        for cause in causes
        if cause.get(_pc_explain.ESTIMATED_RETURN_IMPACT) is not None
    ]
    if estimated_areas:
        reason = "partial or overlapping estimates"
    else:
        reason = "no defensible impact estimates"

    return {
        _pc_findings.PORTFOLIO_ID: period[_pc_findings.PORTFOLIO_ID],
        _pc_findings.FROM_DATE: period[_pc_findings.FROM_DATE],
        _pc_findings.THRU_DATE: period[_pc_findings.THRU_DATE],
        _pc_explain.PORTFOLIO_RETURN_DELTA: period[_pc_explain.PORTFOLIO_RETURN_DELTA],
        _ESTIMATED_IMPACT_AREAS: _comma_separated(estimated_areas),
        _RESIDUAL_STATUS: _RESIDUAL_WITHHELD,
        _RESIDUAL_REASON: reason,
    }


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
    table = _top_evidence_table(findings, top_evidence_limit)
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


def _top_evidence_table(findings: pl.DataFrame, top_evidence_limit: int) -> pl.DataFrame:
    """Return top contribution-candidate rows per portfolio period."""
    candidates = _pc_explain.portfolio_period_contribution_candidates(findings)
    if candidates.is_empty():
        return candidates

    rows: list[dict[str, object]] = []
    for _, group in candidates.group_by(
        [
            _pc_findings.PORTFOLIO_ID,
            _pc_findings.FROM_DATE,
            _pc_findings.THRU_DATE,
        ]
    ):
        rows.extend(group.sort("review_rank").head(top_evidence_limit).iter_rows(named=True))
    return pl.DataFrame(rows).select(
        _pc_explain.PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS
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
    content = "\n".join(
        [
            '<div class="pc-card-row">',
            card_html,
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


def _html_needs_review_summary_section(findings: pl.DataFrame) -> str:
    """Return changed-period reviewer cues as an HTML section."""
    return _html_section(
        "Needs Review Summary",
        _html_table(
            _needs_review_summary_table(findings),
            _NEEDS_REVIEW_COLUMNS,
            empty_message="No changed portfolio periods need review.",
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
    causes = _pc_explain.portfolio_period_cause_summary(findings)
    if causes.is_empty():
        return _html_section(
            "Review Notes",
            _html_empty("No portfolio-period review notes."),
        )

    notes = _review_notes_for_cause_rows(list(causes.iter_rows(named=True)))
    if not notes:
        notes = [
            "No model-limit review notes were generated for the current evidence mix.",
        ]
    return _html_section("Review Notes", _html_list(notes))


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
        _pc_explain.IMPACT_MESSAGE,
    ]
    return _html_section(
        "Impact Coverage",
        _html_table(
            _pc_explain.portfolio_period_impact_coverage_summary(findings),
            columns,
            empty_message="No portfolio return changes need impact coverage review.",
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
    ]
    content = "\n".join(
        [
            f'<p class="pc-note">{_escape_html(_RESIDUAL_STATUS_NOTE)}</p>',
            _html_table(
                _residual_status_table(findings),
                columns,
                empty_message="No portfolio return changes need residual review.",
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
            _top_evidence_table(findings, top_evidence_limit),
            columns,
            empty_message="No ranked evidence is available.",
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


def _markdown_table(
    table: pl.DataFrame,
    columns: Sequence[str],
    *,
    empty_message: str = "No rows.",
) -> str:
    """Return a compact Markdown pipe table for selected columns."""
    if table.is_empty():
        return f"_{_escape_markdown_text(empty_message)}_"

    available_columns = [column for column in columns if column in table.columns]
    if not available_columns:
        return f"_{_escape_markdown_text(empty_message)}_"

    header = "| " + " | ".join(_display_header(column) for column in available_columns) + " |"
    separator = "| " + " | ".join("---" for _ in available_columns) + " |"
    body = [
        "| "
        + " | ".join(_format_markdown_cell(row[column]) for column in available_columns)
        + " |"
        for row in table.select(available_columns).iter_rows(named=True)
    ]
    return "\n".join([header, separator, *body])


def _html_section(title: str, content: str) -> str:
    """Return one titled HTML report section."""
    section_id = _html_section_id(title)
    return "\n".join(
        [
            f'<section class="pc-section" id="{section_id}">',
            f"<h2>{_escape_html(title)}</h2>",
            content,
            "</section>",
        ]
    )


def _html_summary_card(label: str, value: object) -> str:
    """Return one compact summary card."""
    return "\n".join(
        [
            '<div class="pc-card">',
            f"<span>{_escape_html(label)}</span>",
            f"<strong>{_escape_html(_format_value(value))}</strong>",
            "</div>",
        ]
    )


def _html_paragraph(value: object) -> str:
    """Return one escaped HTML paragraph."""
    return f"<p>{_escape_html(value)}</p>"


def _html_table(
    table: pl.DataFrame,
    columns: Sequence[str],
    *,
    empty_message: str = "No rows.",
) -> str:
    """Return an HTML table for selected columns."""
    if table.is_empty():
        return _html_empty(empty_message)

    available_columns = [column for column in columns if column in table.columns]
    if not available_columns:
        return _html_empty(empty_message)

    header_cells = [
        f'<th scope="col">{_escape_html(_display_header(column))}</th>'
        for column in available_columns
    ]
    body_rows = []
    for row in table.select(available_columns).iter_rows(named=True):
        cells = [_html_table_cell(row[column], column) for column in available_columns]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(
        [
            '<div class="pc-table-wrap">',
            f'<p class="pc-table-meta">Rows: {_escape_html(table.height)}</p>',
            '<table class="pc-table">',
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


def _html_table_cell(value: object, column: str) -> str:
    """Return one escaped HTML table cell."""
    classes = " ".join(
        [
            _html_cell_alignment(value),
            _html_column_class(column),
            *_html_value_classes(column, value),
        ]
    )
    return f'<td class="{classes}">{_escape_html(_format_value(value))}</td>'


def _html_cell_alignment(value: object) -> str:
    """Return a CSS alignment class for an HTML table value."""
    if isinstance(value, bool):
        return "pc-center"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "pc-right"
    return "pc-left"


def _html_column_class(column: str) -> str:
    """Return a stable CSS class for a report table column."""
    normalized = column.replace("_", "-")
    return f"pc-col-{normalized}"


def _html_value_classes(column: str, value: object) -> list[str]:
    """Return CSS classes derived from stable report status values."""
    if column == _REVIEW_STATUS:
        return [f"pc-status-{_css_token(_format_value(value))}"]
    if column == _RESIDUAL_STATUS and value == _RESIDUAL_WITHHELD:
        return ["pc-status-withheld"]
    return []


def _css_token(value: str) -> str:
    """Return a simple CSS token for controlled status strings."""
    return value.replace("_", "-").lower()


def _html_empty(message: str) -> str:
    """Return a styled empty-state paragraph."""
    return f'<p class="pc-empty">{_escape_html(message)}</p>'


def _html_list(items: Sequence[str]) -> str:
    """Return an escaped unordered HTML list."""
    list_items = [f"<li>{_escape_html(item)}</li>" for item in items]
    return "\n".join(["<ul>", *list_items, "</ul>"])


def _html_section_id(title: str) -> str:
    """Return a deterministic HTML section id."""
    return title.lower().replace(" ", "-")


def _display_header(column: str) -> str:
    """Return a report-friendly column label."""
    return column.replace("_", " ").title()


def _format_markdown_cell(value: object) -> str:
    """Return one escaped Markdown table cell."""
    return _escape_markdown_text(_format_value(value))


def _format_value(value: object) -> str:
    """Return a compact display value for report cells."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def _comma_separated(values: Sequence[str]) -> str:
    """Return a readable comma-separated list."""
    return ", ".join(values)


def _escape_markdown_text(value: object) -> str:
    """Escape Markdown table delimiters and normalize whitespace."""
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def _escape_html(value: object) -> str:
    """Escape text for HTML element content."""
    text = " ".join(str(value).split())
    return html_lib.escape(text, quote=True)


def _html_style_block() -> str:
    """Return CSS for the standalone performance comparison HTML report."""
    return """
<style>
:root {
  color-scheme: light;
  --pc-bg: #f3f4f2;
  --pc-panel: #ffffff;
  --pc-border: #bfc7c9;
  --pc-border-strong: #687477;
  --pc-text: #202426;
  --pc-muted: #5c6669;
  --pc-accent: #245f70;
  --pc-table-stripe: #f7f8f7;
  --pc-table-head: #e5eaeb;
  --pc-status-review: #8a3f10;
  --pc-status-monitor: #51610f;
  --pc-status-clear: #24613d;
}
body {
  margin: 0;
  background: var(--pc-bg);
  color: var(--pc-text);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  line-height: 1.4;
}
.pc-report {
  max-width: 1320px;
  margin: 0 auto;
  padding: 20px 24px 28px;
}
.pc-header,
.pc-section {
  background: var(--pc-panel);
  border: 1px solid var(--pc-border);
  border-radius: 2px;
  margin: 0 0 16px;
  padding: 14px 16px;
}
.pc-header {
  border-top: 4px solid var(--pc-border-strong);
}
.pc-header h1,
.pc-section h2,
.pc-section h3 {
  margin: 0 0 10px;
}
.pc-header p,
.pc-section p {
  margin: 6px 0;
}
.pc-kicker {
  color: var(--pc-muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  margin: 0 0 4px;
  text-transform: uppercase;
}
.pc-subtitle {
  color: var(--pc-muted);
  font-size: 13px;
  font-weight: 700;
}
.pc-header-notes {
  border-top: 1px solid var(--pc-border);
  margin-top: 12px;
  padding-top: 8px;
}
.pc-section a {
  color: var(--pc-accent);
}
.pc-contents-list {
  column-gap: 28px;
  columns: 2;
  margin: 0;
  padding-left: 20px;
}
.pc-contents-list li {
  break-inside: avoid;
  margin: 0 0 4px;
}
.pc-card-row {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin-bottom: 14px;
}
.pc-card {
  border: 1px solid var(--pc-border);
  border-radius: 2px;
  padding: 8px 10px;
}
.pc-card span {
  color: var(--pc-muted);
  display: block;
  font-size: 12px;
}
.pc-card strong {
  display: block;
  font-size: 22px;
  margin-top: 2px;
}
.pc-note,
.pc-empty {
  color: var(--pc-muted);
}
.pc-table-wrap {
  overflow-x: auto;
  margin-top: 8px;
}
.pc-table-meta {
  color: var(--pc-muted);
  font-size: 12px;
  font-weight: 700;
  margin: 0 0 4px;
  text-transform: uppercase;
}
table {
  border-collapse: collapse;
  min-width: 100%;
  width: 100%;
}
th,
td {
  border: 1px solid var(--pc-border);
  padding: 5px 7px;
  vertical-align: top;
}
th {
  background: var(--pc-table-head);
  border-top: 2px solid var(--pc-border-strong);
  color: #263033;
  font-size: 12px;
  font-weight: 700;
  text-align: left;
  white-space: nowrap;
}
tbody tr:nth-child(even) {
  background: var(--pc-table-stripe);
}
.pc-left {
  text-align: left;
}
.pc-center {
  text-align: center;
}
.pc-right {
  text-align: right;
  white-space: nowrap;
}
.pc-col-portfolio-return-delta,
.pc-col-estimated-return-impact,
.pc-col-estimated-return-impact-total,
.pc-col-transaction-impact-diagnostic-estimate,
.pc-col-delta-b-minus-a,
.pc-col-amount-delta,
.pc-col-quantity-delta,
.pc-col-price-delta {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.pc-col-review-status,
.pc-status-withheld {
  font-weight: 700;
}
.pc-status-needs-review {
  color: var(--pc-status-review);
}
.pc-status-monitor {
  color: var(--pc-status-monitor);
}
.pc-status-clear {
  color: var(--pc-status-clear);
}
#needs-review-summary {
  border-left: 4px solid var(--pc-status-review);
}
@media (max-width: 760px) {
  .pc-report {
    padding: 12px;
  }
  .pc-contents-list {
    columns: 1;
  }
}
@media print {
  body {
    background: #ffffff;
    font-size: 11px;
  }
  .pc-report {
    max-width: none;
    padding: 0;
  }
  .pc-header,
  .pc-section {
    border-color: #888888;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .pc-section {
    margin-bottom: 10px;
  }
  a {
    color: inherit;
    text-decoration: none;
  }
}
</style>""".strip()
