"""Validate generated performance comparison report bundles."""

from __future__ import annotations

# Python imports
import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison import rendering as _pc_rendering
from ppar.performance_comparison import workbook as _pc_workbook

__all__ = [
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "report_bundle_manifest",
    "report_bundle_validation_issues",
    "write_csv_artifact",
    "write_report_bundle_manifest",
    "write_report_bundle_readme",
]

REPORT_BUNDLE_REQUIRED_ARTIFACTS = (
    "html_report",
    "readme",
    "manifest",
    "findings",
    "needs_review_summary",
    "portfolio_period_summary",
    "cause_summary",
    "impact_estimates",
    "impact_coverage",
    "context_evidence_summary",
    "context_evidence",
    "transaction_cross_checks",
    "flow_cross_check_reconciliation",
    "residual_status",
    "transaction_activity",
    "transaction_matching_diagnostics",
    "top_evidence",
)


def write_csv_artifact(table: pl.DataFrame, output_path: Path) -> Path:
    """Write a report-bundle CSV artifact.

    Args:
        table: Table to write.
        output_path: Destination CSV path.

    Returns:
        Normalized destination path.
    """
    table.write_csv(output_path)
    return output_path


def write_report_bundle_readme(
    output_path: Path,
    *,
    title: str,
    tables: Mapping[str, pl.DataFrame],
    include_workbook: bool,
) -> Path:
    """Write a portable report-bundle README.

    Args:
        output_path: Destination README path.
        title: Report title to show as the README heading.
        tables: Named CSV helper tables included in the bundle.
        include_workbook: Whether the bundle includes the XLSX review workbook.

    Returns:
        Normalized destination path.
    """
    excel_line = (
        "- `report.xlsx`: Excel review workbook with the Portfolio Differences sheet, "
        "Security Differences sheet, Underlying Causes sheet, Reported Performance "
        "Checks sheet, Context sheet, and Raw Audit Trail sheet."
    )
    html_line = (
        "- `report.html`: browser review report with the same sections and order as "
        "`report.xlsx`."
    )
    primary_artifact_lines = (
        [excel_line, html_line]
        if include_workbook
        else [html_line]
    )
    opening_line = (
        "Open `report.xlsx` first for Excel review. Use `report.html` when you want "
        "the same review model in a browser."
        if include_workbook
        else "Open `report.html` for the browser review."
    )
    first_review_step = (
        "1. Open `report.xlsx` or `report.html` and start with Portfolio Differences."
        if include_workbook
        else "1. Open `report.html` and start with Portfolio Differences."
    )
    lines = [
        f"# {_pc_rendering.escape_markdown_text(title)}",
        "",
        "This directory is a portable performance-comparison review bundle.",
        opening_line,
        "",
        "## Primary Review Artifact",
        "",
        *primary_artifact_lines,
        "",
        "## Recommended Review Order",
        "",
        first_review_step,
        "2. Use Underlying Causes to see which source-data differences explain each "
        "portfolio period.",
        "3. Use Reported Performance Checks, Context, and Raw Audit Trail as "
        "supporting detail.",
        "4. Use the `review_key` column to follow a period across CSV artifacts.",
        "",
        "## Audit/Export Files",
        "",
        "- `findings.csv`: complete finding-level comparison output.",
        "- `manifest.json`: machine-readable artifact and row-count metadata.",
        *_report_bundle_readme_table_lines(tables),
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding=util.ENCODING)
    return output_path


def write_report_bundle_manifest(
    output_path: Path,
    *,
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    title: str,
    include_suppressed_appendix: bool,
    top_evidence_limit: int,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
) -> Path:
    """Write a report-bundle JSON manifest.

    Args:
        output_path: Destination manifest path.
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        include_suppressed_appendix: Compatibility flag recorded in the
            manifest for standalone report options.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.

    Returns:
        Normalized destination path.
    """
    manifest = report_bundle_manifest(
        findings=findings,
        active_findings=active_findings,
        title=title,
        include_suppressed_appendix=include_suppressed_appendix,
        top_evidence_limit=top_evidence_limit,
        artifact_paths=artifact_paths,
        tables=tables,
    )
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding=util.ENCODING,
    )
    return output_path


def report_bundle_manifest(
    *,
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    title: str,
    include_suppressed_appendix: bool,
    top_evidence_limit: int,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
) -> dict[str, object]:
    """Return JSON-serializable metadata for a report bundle.

    Args:
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        include_suppressed_appendix: Compatibility flag recorded in the
            manifest for standalone report options.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.

    Returns:
        JSON-serializable manifest data.
    """
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


def report_bundle_validation_issues(bundle_directory: util.PathLike) -> list[str]:
    """Return validation issues for a generated report bundle.

    Args:
        bundle_directory: Directory containing a generated report bundle.

    Returns:
        Human-readable validation issues. An empty list means validation passed.
    """
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
    issues.extend(_report_bundle_workbook_issues(bundle_path, artifacts))
    return issues


def _report_bundle_readme_table_lines(tables: Mapping[str, pl.DataFrame]) -> list[str]:
    """Return README bullets for report-bundle table artifacts."""
    descriptions = {
        "needs_review_summary": (
            "top triage table for changed periods, suggested next steps, and "
            "drilldown artifacts"
        ),
        "portfolio_period_summary": "portfolio-period return-change summary",
        "cause_summary": "cause-area summary with explained-change methods",
        "impact_estimates": "currently quantified impact estimates",
        "impact_coverage": "period-level estimate coverage and missing inputs",
        "context_evidence_summary": (
            "context-only evidence counts, reviewer priority, and affected identifiers"
        ),
        "context_evidence": (
            "row-level context evidence, reviewer priority, and no-impact treatment"
        ),
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
    for artifact_name in REPORT_BUNDLE_REQUIRED_ARTIFACTS:
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


def _report_bundle_workbook_issues(
    bundle_path: Path,
    artifacts: Mapping[str, object],
) -> list[str]:
    """Return optional XLSX review workbook validation issues."""
    return _pc_workbook.workbook_artifact_issues(bundle_path, artifacts)
