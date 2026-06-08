"""Render performance comparison findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Sequence
import datetime as dt

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison.explain import (
    ESTIMATED_RETURN_IMPACT,
    FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
    IMPACT_BASIS,
    IMPACT_CONFIDENCE,
    IMPACT_MESSAGE,
    PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS,
    PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS,
    PORTFOLIO_RETURN_DELTA,
    ROOT_CAUSE_AREA,
    TOP_CODES,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_summary,
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
        _run_summary_section(findings, active_findings, summaries, active_summaries),
        _portfolio_period_section(active_findings),
        _cause_summary_section(active_findings),
        _top_evidence_section(active_findings, top_evidence_limit),
    ]
    if include_suppressed_appendix:
        sections.append(_suppressed_appendix_section(findings, summaries))
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


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
    candidates = portfolio_period_contribution_candidates(findings)
    if candidates.is_empty():
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
    rows = []
    for _, group in candidates.group_by([PORTFOLIO_ID, FROM_DATE, THRU_DATE]):
        rows.extend(group.sort("review_rank").head(top_evidence_limit).iter_rows(named=True))
    table = pl.DataFrame(rows).select(PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS)
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


def _escape_markdown_text(value: object) -> str:
    """Escape Markdown table delimiters and normalize whitespace."""
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")

