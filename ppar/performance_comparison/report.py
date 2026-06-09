"""Render performance comparison findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Sequence
import datetime as dt
import html as html_lib
import json
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison.explain import (
    AMOUNT_DELTA,
    CHANGED_FIELDS,
    ESTIMATED_CAUSE_AREA_COUNT,
    ESTIMATED_RETURN_IMPACT,
    ESTIMATED_RETURN_IMPACT_TOTAL,
    EVIDENCE_ONLY_AREAS,
    EVIDENCE_ONLY_CAUSE_AREA_COUNT,
    FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
    IMPACT_BASIS,
    IMPACT_BASIS_NO_ESTIMATE,
    IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
    IMPACT_BASIS_SECURITY_CONTRIBUTION,
    IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
    IMPACT_CONFIDENCE,
    IMPACT_MESSAGE,
    LOW_CONFIDENCE_ESTIMATE_COUNT,
    MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
    MISSING_IMPACT_INPUTS,
    PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS,
    PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS,
    PORTFOLIO_RETURN_DELTA,
    PRICE_DELTA,
    QUANTITY_DELTA,
    ROOT_CAUSE_AREA,
    ROOT_CAUSE_CASH,
    ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
    ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
    ROOT_CAUSE_PRICE,
    ROOT_CAUSE_AREA_COUNT,
    ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION,
    ROOT_CAUSE_TRANSACTION_ACTIVITY,
    TOP_CODES,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    transaction_activity_summary,
)
from ppar.performance_comparison.findings import (
    DATASET,
    DELTA_B_MINUS_A,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    MESSAGE,
    PORTFOLIO_ID,
    SECURITY_ID,
    SOURCE_COLUMN,
    SUPPRESSED,
    THRU_DATE,
    TRANSACTION_CATEGORY,
)
from ppar.performance_comparison.runner import (
    compact_findings_table,
    summarize_findings,
)

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
        Markdown string suitable for console output, files, notebooks, or a
        future HTML rendering layer.
    """
    active_findings = _active_findings(findings)
    summaries = summarize_findings(findings)
    active_summaries = summarize_findings(active_findings)

    sections = [
        f"# {_escape_markdown_text(title)}",
        _ACTIVE_ONLY_NOTE,
        _NO_ESTIMATE_NOTE,
        _report_contents_section(
            include_suppressed_appendix=include_suppressed_appendix,
        ),
        _run_summary_section(findings, active_findings, summaries, active_summaries),
        _portfolio_period_narrative_section(active_findings),
        _review_notes_section(active_findings),
        _impact_estimate_summary_section(active_findings),
        _impact_coverage_section(active_findings),
        _residual_status_section(active_findings),
        _transaction_activity_section(active_findings),
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
    summaries = summarize_findings(findings)
    active_summaries = summarize_findings(active_findings)
    sections = [
        _html_run_summary_section(findings, active_findings, summaries, active_summaries),
        _html_portfolio_period_narrative_section(active_findings),
        _html_review_notes_section(active_findings),
        _html_impact_estimate_summary_section(active_findings),
        _html_impact_coverage_section(active_findings),
        _html_residual_status_section(active_findings),
        _html_transaction_activity_section(active_findings),
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
            f"<h1>{_escape_html(title)}</h1>",
            f"<p>{_escape_html(_ACTIVE_ONLY_NOTE)}</p>",
            f"<p>{_escape_html(_NO_ESTIMATE_NOTE)}</p>",
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
    """Write a reproducible Markdown-and-CSV report bundle.

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
    return paths


def _report_bundle_tables(
    active_findings: pl.DataFrame,
    top_evidence_limit: int,
) -> dict[str, pl.DataFrame]:
    """Return report-bundle tables keyed by artifact stem."""
    return {
        "portfolio_period_summary": portfolio_period_summary(active_findings),
        "cause_summary": portfolio_period_cause_summary(active_findings),
        "impact_estimates": _impact_estimate_summary_table(active_findings),
        "impact_coverage": portfolio_period_impact_coverage_summary(active_findings),
        "residual_status": _residual_status_table(active_findings),
        "transaction_activity": transaction_activity_summary(active_findings),
        "top_evidence": _top_evidence_table(active_findings, top_evidence_limit),
    }


def _write_csv(table: pl.DataFrame, output_path: Path) -> Path:
    """Write a CSV table and return the normalized path."""
    table.write_csv(output_path)
    return output_path


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
        _markdown_table(active_summaries["by_code"], [FINDING_CODE, _COUNT]),
        "",
        "### Active Findings By Dataset",
        _markdown_table(active_summaries["by_dataset"], [DATASET, _COUNT]),
        "",
        "### Findings By Suppression State",
        _markdown_table(summaries["by_suppressed"], [SUPPRESSED, _COUNT]),
    ]
    return "\n".join(lines)


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
            "<ul>",
            *links,
            "</ul>",
            "</nav>",
        ]
    )


def _report_section_names(*, include_suppressed_appendix: bool) -> list[str]:
    """Return report section names in display order."""
    section_names = [
        "Run Summary",
        "Portfolio-Period Narrative",
        "Review Notes",
        "Impact Estimate Summary",
        "Impact Coverage",
        "Residual Status",
        "Transaction Activity",
        "Portfolio-Period Changes",
        "Cause Summary",
        "Top Evidence",
    ]
    if include_suppressed_appendix:
        section_names.append("Suppressed Findings Appendix")
    return section_names


def _portfolio_period_narrative_section(findings: pl.DataFrame) -> str:
    """Return conservative narrative summaries for portfolio-period changes."""
    summary = portfolio_period_summary(findings)
    if summary.is_empty():
        return "\n".join(
            [
                "## Portfolio-Period Narrative",
                "_No portfolio return changes to narrate._",
            ]
        )

    causes = portfolio_period_cause_summary(findings)
    paragraphs = ["## Portfolio-Period Narrative"]
    for period in summary.iter_rows(named=True):
        period_causes = _period_cause_rows(causes, period)
        paragraphs.append(_portfolio_period_narrative(period, period_causes))
    return "\n\n".join(paragraphs)


def _review_notes_section(findings: pl.DataFrame) -> str:
    """Return review notes for current model limits visible in the report."""
    causes = portfolio_period_cause_summary(findings)
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
    cause_areas = {cause[ROOT_CAUSE_AREA] for cause in causes}
    if ROOT_CAUSE_TRANSACTION_ACTIVITY in cause_areas:
        notes.append(
            "Transaction activity is evidence-only until portfolio period, "
            "return denominator, and transaction sign and flow semantics are "
            "available."
        )
    if ROOT_CAUSE_MARKET_VALUE_OR_POSITION in cause_areas:
        notes.append(
            "Market value or position evidence has no return-impact estimate yet."
        )
    if ROOT_CAUSE_PRICE in cause_areas:
        notes.append(
            "Price evidence is linked to affected portfolio periods, but no "
            "portfolio-period impact estimate is calculated yet."
        )
    if ROOT_CAUSE_CASH in cause_areas:
        notes.append("Cash evidence has no return-impact estimate yet.")
    if ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT in cause_areas:
        notes.append(_portfolio_source_field_review_note(causes))
    if ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION in cause_areas:
        notes.append(_security_return_weighted_review_note(causes))
    notes.append(
        "No residual amount is calculated because not enough defensible impact "
        "estimates exist yet."
    )
    return notes


def _portfolio_source_field_review_note(causes: list[dict[str, object]]) -> str:
    """Return a review note for portfolio performance source-field estimates."""
    has_estimate = any(
        cause[ROOT_CAUSE_AREA] == ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT
        and cause[IMPACT_BASIS] == IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD
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
        cause[ROOT_CAUSE_AREA] == ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
        and cause[IMPACT_BASIS] == IMPACT_BASIS_SECURITY_CONTRIBUTION
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
    portfolio_id = _format_value(period[PORTFOLIO_ID])
    from_date = _format_value(period[FROM_DATE])
    thru_date = _format_value(period[THRU_DATE])
    return_delta = _format_value(period[PORTFOLIO_RETURN_DELTA])
    sentences = [
        (
            f"{portfolio_id} changed by {return_delta} for {from_date} to "
            f"{thru_date}."
        )
    ]

    estimated_causes = [
        cause
        for cause in causes
        if cause.get(ESTIMATED_RETURN_IMPACT) is not None
    ]
    if estimated_causes:
        strongest = max(
            estimated_causes,
            key=lambda cause: abs(float(cause[ESTIMATED_RETURN_IMPACT])),
        )
        sentences.append(_estimated_impact_sentence(strongest))
    else:
        sentences.append(
            "No currently supported impact estimates are available for this period."
        )

    evidence_only_areas = [
        str(cause[ROOT_CAUSE_AREA])
        for cause in causes
        if cause.get(IMPACT_BASIS) == IMPACT_BASIS_NO_ESTIMATE
    ]
    if evidence_only_areas:
        sentences.append(
            "Evidence-only areas are "
            f"{_comma_separated(evidence_only_areas)}; these rows remain "
            f"{IMPACT_BASIS_NO_ESTIMATE}."
        )

    if period[HAS_SUPPRESSED_FINDINGS]:
        sentences.append("Suppressed findings exist for this portfolio period.")

    return " ".join(_escape_markdown_text(sentence) for sentence in sentences)


def _estimated_impact_sentence(cause: dict[str, object]) -> str:
    """Return a conservative sentence for the strongest estimated impact."""
    cause_area = _format_value(cause[ROOT_CAUSE_AREA])
    estimated_impact = _format_value(cause[ESTIMATED_RETURN_IMPACT])
    impact_basis = _format_value(cause[IMPACT_BASIS])
    confidence = _format_value(cause[IMPACT_CONFIDENCE])
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
        (pl.col(PORTFOLIO_ID) == period[PORTFOLIO_ID])
        & (pl.col(FROM_DATE) == period[FROM_DATE])
        & (pl.col(THRU_DATE) == period[THRU_DATE])
    )
    return list(period_causes.iter_rows(named=True))


def _portfolio_period_section(findings: pl.DataFrame) -> str:
    """Return the portfolio-period return changes Markdown section."""
    summary = portfolio_period_summary(findings)
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
        FINDING_COUNT,
        HAS_SUPPRESSED_FINDINGS,
    ]
    return "\n".join(
        [
            "## Portfolio-Period Changes",
            _markdown_table(summary, columns, empty_message="No portfolio return changes."),
        ]
    )


def _transaction_activity_section(findings: pl.DataFrame) -> str:
    """Return changed transaction activity and impact-eligibility gaps."""
    summary = transaction_activity_summary(findings)
    columns = [
        PORTFOLIO_ID,
        SECURITY_ID,
        FROM_DATE,
        THRU_DATE,
        TRANSACTION_CATEGORY,
        CHANGED_FIELDS,
        AMOUNT_DELTA,
        QUANTITY_DELTA,
        PRICE_DELTA,
        MISSING_IMPACT_INPUTS,
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


def _residual_status_section(findings: pl.DataFrame) -> str:
    """Return a Markdown section explaining whether residuals are calculated."""
    residuals = _residual_status_table(findings)
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
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
    periods = portfolio_period_summary(findings)
    if periods.is_empty():
        return pl.DataFrame(
            schema={
                PORTFOLIO_ID: pl.String,
                FROM_DATE: pl.Date,
                THRU_DATE: pl.Date,
                PORTFOLIO_RETURN_DELTA: pl.Float64,
                _ESTIMATED_IMPACT_AREAS: pl.String,
                _RESIDUAL_STATUS: pl.String,
                _RESIDUAL_REASON: pl.String,
            }
        )

    causes = portfolio_period_cause_summary(findings)
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
        str(cause[ROOT_CAUSE_AREA])
        for cause in causes
        if cause.get(ESTIMATED_RETURN_IMPACT) is not None
    ]
    if estimated_areas:
        reason = "partial or overlapping estimates"
    else:
        reason = "no defensible impact estimates"

    return {
        PORTFOLIO_ID: period[PORTFOLIO_ID],
        FROM_DATE: period[FROM_DATE],
        THRU_DATE: period[THRU_DATE],
        PORTFOLIO_RETURN_DELTA: period[PORTFOLIO_RETURN_DELTA],
        _ESTIMATED_IMPACT_AREAS: _comma_separated(estimated_areas),
        _RESIDUAL_STATUS: _RESIDUAL_WITHHELD,
        _RESIDUAL_REASON: reason,
    }


def _impact_coverage_section(findings: pl.DataFrame) -> str:
    """Return estimate-coverage status by portfolio period."""
    coverage = portfolio_period_impact_coverage_summary(findings)
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
        ROOT_CAUSE_AREA_COUNT,
        ESTIMATED_CAUSE_AREA_COUNT,
        EVIDENCE_ONLY_CAUSE_AREA_COUNT,
        LOW_CONFIDENCE_ESTIMATE_COUNT,
        MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
        ESTIMATED_RETURN_IMPACT_TOTAL,
        EVIDENCE_ONLY_AREAS,
        MISSING_IMPACT_INPUTS,
        IMPACT_MESSAGE,
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


def _impact_estimate_summary_section(findings: pl.DataFrame) -> str:
    """Return a concise Markdown section for currently quantified impacts."""
    estimated_summary = _impact_estimate_summary_table(findings)
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        ROOT_CAUSE_AREA,
        ESTIMATED_RETURN_IMPACT,
        IMPACT_BASIS,
        IMPACT_CONFIDENCE,
        IMPACT_MESSAGE,
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
    summary = portfolio_period_cause_summary(findings)
    if summary.is_empty():
        return summary
    return summary.filter(pl.col(ESTIMATED_RETURN_IMPACT).is_not_null())


def _cause_summary_section(findings: pl.DataFrame) -> str:
    """Return the cause summary Markdown section."""
    summary = portfolio_period_cause_summary(findings)
    columns = [
        column
        for column in PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS
        if column
        in {
            PORTFOLIO_ID,
            FROM_DATE,
            THRU_DATE,
            ROOT_CAUSE_AREA,
            FINDING_COUNT,
            ESTIMATED_RETURN_IMPACT,
            IMPACT_BASIS,
            IMPACT_CONFIDENCE,
            TOP_CODES,
            IMPACT_MESSAGE,
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
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        "review_rank",
        FINDING_CODE,
        DATASET,
        EVIDENCE_ROLE,
        SECURITY_ID,
        SOURCE_COLUMN,
        DELTA_B_MINUS_A,
        ESTIMATED_RETURN_IMPACT,
        IMPACT_BASIS,
        IMPACT_CONFIDENCE,
        MESSAGE,
    ]
    return "\n".join(
        [
            "## Top Evidence",
            _markdown_table(table, columns, empty_message="No ranked evidence is available."),
        ]
    )


def _top_evidence_table(findings: pl.DataFrame, top_evidence_limit: int) -> pl.DataFrame:
    """Return top contribution-candidate rows per portfolio period."""
    candidates = portfolio_period_contribution_candidates(findings)
    if candidates.is_empty():
        return candidates

    rows = []
    for _, group in candidates.group_by([PORTFOLIO_ID, FROM_DATE, THRU_DATE]):
        rows.extend(group.sort("review_rank").head(top_evidence_limit).iter_rows(named=True))
    return pl.DataFrame(rows).select(PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS)


def _suppressed_appendix_section(
    findings: pl.DataFrame,
    summaries: dict[str, pl.DataFrame],
) -> str:
    """Return the suppressed findings audit appendix."""
    suppressed = findings.filter(pl.col(SUPPRESSED)) if not findings.is_empty() else findings
    lines = [
        "## Suppressed Findings Appendix",
        "### Suppressed Counts By Code",
        _markdown_table(
            summaries["by_code_suppressed"].filter(pl.col(SUPPRESSED))
            if not summaries["by_code_suppressed"].is_empty()
            else summaries["by_code_suppressed"],
            [FINDING_CODE, SUPPRESSED, _COUNT],
            empty_message="No suppressed findings.",
        ),
        "",
        "### Suppressed Finding Detail",
        _markdown_table(
            compact_findings_table(suppressed, include_suppressed=True),
            [
                FINDING_CODE,
                DATASET,
                EVIDENCE_ROLE,
                PORTFOLIO_ID,
                SECURITY_ID,
                FROM_DATE,
                THRU_DATE,
                SOURCE_COLUMN,
                DELTA_B_MINUS_A,
                MESSAGE,
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
            _html_table(active_summaries["by_code"], [FINDING_CODE, _COUNT]),
            "<h3>Active Findings By Dataset</h3>",
            _html_table(active_summaries["by_dataset"], [DATASET, _COUNT]),
            "<h3>Findings By Suppression State</h3>",
            _html_table(summaries["by_suppressed"], [SUPPRESSED, _COUNT]),
        ]
    )
    return _html_section("Run Summary", content)


def _html_portfolio_period_narrative_section(findings: pl.DataFrame) -> str:
    """Return conservative narrative summaries as HTML."""
    summary = portfolio_period_summary(findings)
    if summary.is_empty():
        return _html_section(
            "Portfolio-Period Narrative",
            _html_empty("No portfolio return changes to narrate."),
        )

    causes = portfolio_period_cause_summary(findings)
    paragraphs = [
        _html_paragraph(
            _portfolio_period_narrative(period, _period_cause_rows(causes, period))
        )
        for period in summary.iter_rows(named=True)
    ]
    return _html_section("Portfolio-Period Narrative", "\n".join(paragraphs))


def _html_review_notes_section(findings: pl.DataFrame) -> str:
    """Return current model-limit review notes as HTML."""
    causes = portfolio_period_cause_summary(findings)
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
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        ROOT_CAUSE_AREA,
        ESTIMATED_RETURN_IMPACT,
        IMPACT_BASIS,
        IMPACT_CONFIDENCE,
        IMPACT_MESSAGE,
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
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
        ROOT_CAUSE_AREA_COUNT,
        ESTIMATED_CAUSE_AREA_COUNT,
        EVIDENCE_ONLY_CAUSE_AREA_COUNT,
        LOW_CONFIDENCE_ESTIMATE_COUNT,
        MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
        ESTIMATED_RETURN_IMPACT_TOTAL,
        EVIDENCE_ONLY_AREAS,
        MISSING_IMPACT_INPUTS,
        IMPACT_MESSAGE,
    ]
    return _html_section(
        "Impact Coverage",
        _html_table(
            portfolio_period_impact_coverage_summary(findings),
            columns,
            empty_message="No portfolio return changes need impact coverage review.",
        ),
    )


def _html_residual_status_section(findings: pl.DataFrame) -> str:
    """Return residual-status caveats and rows as an HTML section."""
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
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
        PORTFOLIO_ID,
        SECURITY_ID,
        FROM_DATE,
        THRU_DATE,
        TRANSACTION_CATEGORY,
        CHANGED_FIELDS,
        AMOUNT_DELTA,
        QUANTITY_DELTA,
        PRICE_DELTA,
        MISSING_IMPACT_INPUTS,
    ]
    return _html_section(
        "Transaction Activity",
        _html_table(
            transaction_activity_summary(findings),
            columns,
            empty_message="No changed transaction activity.",
        ),
    )


def _html_portfolio_period_section(findings: pl.DataFrame) -> str:
    """Return portfolio-period return changes as an HTML section."""
    columns = [
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        PORTFOLIO_RETURN_DELTA,
        FINDING_COUNT,
        HAS_SUPPRESSED_FINDINGS,
    ]
    return _html_section(
        "Portfolio-Period Changes",
        _html_table(
            portfolio_period_summary(findings),
            columns,
            empty_message="No portfolio return changes.",
        ),
    )


def _html_cause_summary_section(findings: pl.DataFrame) -> str:
    """Return cause-area summaries as an HTML section."""
    columns = [
        column
        for column in PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS
        if column
        in {
            PORTFOLIO_ID,
            FROM_DATE,
            THRU_DATE,
            ROOT_CAUSE_AREA,
            FINDING_COUNT,
            ESTIMATED_RETURN_IMPACT,
            IMPACT_BASIS,
            IMPACT_CONFIDENCE,
            TOP_CODES,
            IMPACT_MESSAGE,
        }
    ]
    return _html_section(
        "Cause Summary",
        _html_table(
            portfolio_period_cause_summary(findings),
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
        PORTFOLIO_ID,
        FROM_DATE,
        THRU_DATE,
        "review_rank",
        FINDING_CODE,
        DATASET,
        EVIDENCE_ROLE,
        SECURITY_ID,
        SOURCE_COLUMN,
        DELTA_B_MINUS_A,
        ESTIMATED_RETURN_IMPACT,
        IMPACT_BASIS,
        IMPACT_CONFIDENCE,
        MESSAGE,
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
    suppressed = findings.filter(pl.col(SUPPRESSED)) if not findings.is_empty() else findings
    content = "\n".join(
        [
            "<h3>Suppressed Counts By Code</h3>",
            _html_table(
                summaries["by_code_suppressed"].filter(pl.col(SUPPRESSED))
                if not summaries["by_code_suppressed"].is_empty()
                else summaries["by_code_suppressed"],
                [FINDING_CODE, SUPPRESSED, _COUNT],
                empty_message="No suppressed findings.",
            ),
            "<h3>Suppressed Finding Detail</h3>",
            _html_table(
                compact_findings_table(suppressed, include_suppressed=True),
                [
                    FINDING_CODE,
                    DATASET,
                    EVIDENCE_ROLE,
                    PORTFOLIO_ID,
                    SECURITY_ID,
                    FROM_DATE,
                    THRU_DATE,
                    SOURCE_COLUMN,
                    DELTA_B_MINUS_A,
                    MESSAGE,
                ],
                empty_message="No suppressed finding detail.",
            ),
        ]
    )
    return _html_section("Suppressed Findings Appendix", content)


def _active_findings(findings: pl.DataFrame) -> pl.DataFrame:
    """Return unsuppressed findings, preserving empty-table behavior."""
    if findings.is_empty() or SUPPRESSED not in findings.columns:
        return findings
    return findings.filter(~pl.col(SUPPRESSED))


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
        cells = [
            _html_table_cell(row[column], _html_cell_alignment(row[column]))
            for column in available_columns
        ]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(
        [
            '<div class="pc-table-wrap">',
            "<table>",
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


def _html_table_cell(value: object, alignment: str) -> str:
    """Return one escaped HTML table cell."""
    return f'<td class="{alignment}">{_escape_html(_format_value(value))}</td>'


def _html_cell_alignment(value: object) -> str:
    """Return a CSS alignment class for an HTML table value."""
    if isinstance(value, bool):
        return "pc-center"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "pc-right"
    return "pc-left"


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
  --pc-bg: #f6f7f9;
  --pc-panel: #ffffff;
  --pc-border: #d7dce2;
  --pc-text: #1e2329;
  --pc-muted: #5d6978;
  --pc-accent: #276f86;
  --pc-table-stripe: #f1f5f8;
}
body {
  margin: 0;
  background: var(--pc-bg);
  color: var(--pc-text);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
.pc-report {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
.pc-header,
.pc-section {
  background: var(--pc-panel);
  border: 1px solid var(--pc-border);
  border-radius: 6px;
  margin: 0 0 16px;
  padding: 16px;
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
.pc-section a {
  color: var(--pc-accent);
}
.pc-card-row {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin-bottom: 14px;
}
.pc-card {
  border: 1px solid var(--pc-border);
  border-radius: 6px;
  padding: 10px;
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
}
table {
  border-collapse: collapse;
  min-width: 100%;
}
th,
td {
  border: 1px solid var(--pc-border);
  padding: 6px 8px;
  vertical-align: top;
}
th {
  background: #e9eef2;
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
</style>""".strip()
