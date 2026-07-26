"""Render Audit findings as review-oriented reports."""

from __future__ import annotations

# Python imports
from collections.abc import Iterator, Mapping, Sequence
import json
from pathlib import Path
import shutil

# Third-party imports
import polars as pl

# Project imports
import ppar.common as util
from ppar.errors import PpaError
from ppar.audit import atomic_directory as _atomic_directory
from ppar.audit import bundle as _pc_bundle
from ppar.audit import conservation as _pc_conservation
from ppar.audit import executive_summary as _executive_summary
from ppar.audit import schema as pc_cols
from ppar.audit.performance_comparison import explain as _pc_explain
from ppar.audit.performance_comparison import findings as _pc_findings
from ppar.audit import lineage as _pc_lineage
from ppar.audit import output_policy as _pc_output_policy
from ppar.audit import rendering as _pc_rendering
from ppar.audit import review_keys as _pc_review_keys
from ppar.audit import review_model as _pc_review_model
from ppar.audit import runner as _pc_runner
from ppar.audit import workbook as _pc_workbook
from ppar.audit import workbook_layout as _pc_workbook_layout
from ppar.audit import workbook_reconstruction as _pc_workbook_reconstruction
from ppar.audit import workbook_tables as _pc_workbook_tables
from ppar.audit.specification import PORTFOLIO_COMPARISON_LEVEL

__all__ = [
    "write_audit_report_bundle",
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
_OPTIONAL_REPORT_BUNDLE_TABLE_ARTIFACTS = (
    _pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT,
    _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
    _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
)
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
    title: str = "Audit Report",
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _sheets: Sequence[_pc_workbook.ReviewWorkbookSheet] | None = None,
    _reconstruction_cache: (
        _pc_workbook_reconstruction.WorkbookReconstructionCache | None
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
    sheets = _sheets or (
        _pc_workbook_tables.audit_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=_reconstruction_cache,
        )
    )
    return "\n".join(_html_document_parts(title, sheets))


def _html_document_parts(
    title: str,
    sheets: Sequence[_pc_workbook.ReviewWorkbookSheet],
) -> Iterator[str]:
    """Yield a complete report in sections to bound peak output memory."""
    yield "<!DOCTYPE html>"
    yield '<html lang="en">'
    yield "<head>"
    yield '<meta charset="utf-8"/>'
    yield '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
    yield f"<title>{_escape_html(title)}</title>"
    yield _html_style_block()
    yield "</head>"
    yield "<body>"
    yield '<main class="pc-report">'
    yield '<header class="pc-header">'
    yield f"<h1>{_escape_html(title)}</h1>"
    yield "</header>"
    for sheet in sheets:
        yield _html_workbook_sheet_section(sheet)
    yield "</main>"
    yield "</body>"
    yield "</html>"
    yield ""


def _html_workbook_sheet_section(sheet: _pc_workbook.ReviewWorkbookSheet) -> str:
    """Return one HTML section matching a review workbook sheet."""
    if sheet.artifact_name == _pc_review_model.EXECUTIVE_SUMMARY_ARTIFACT:
        content = _html_executive_summary_tables(sheet)
    else:
        content = _html_workbook_sheet_table(sheet)
    return _html_section(
        sheet.sheet_name,
        content,
    )


def _html_executive_summary_tables(
    sheet: _pc_workbook.ReviewWorkbookSheet,
) -> str:
    """Return two simple quantity tables for the Executive Summary."""
    payloads = _executive_summary.executive_summary_display_tables(sheet.table)
    sections: list[str] = []
    for section_name, caption in (
        (
            _executive_summary.PERFORMANCE_SECTION,
            _executive_summary.PERFORMANCE_TABLE_CAPTION,
        ),
        (
            _executive_summary.DATA_ISSUES_SECTION,
            _executive_summary.DATA_ISSUES_TABLE_CAPTION,
        ),
    ):
        payload = payloads[caption]
        headers = "".join(
            f'<th scope="col">{_escape_html(header)}</th>'
            for header in payload["columns"]
        )
        body_rows = [
            "<tr>"
            + "".join(
                f'<td class="{"pc-left" if index == 0 else "pc-right"}">'
                f"{_escape_html(value)}</td>"
                for index, value in enumerate(row)
            )
            + "</tr>"
            for row in payload["rows"]
        ]
        sections.extend(
            [
                f"<h3>{_escape_html(section_name)}</h3>",
                '<div class="pc-table-wrap">',
                '<table class="pc-table">',
                f"<caption>{_escape_html(caption)}</caption>",
                f"<thead><tr>{headers}</tr></thead>",
                "<tbody>",
                *body_rows,
                "</tbody>",
                "</table>",
                "</div>",
            ]
        )
    return "\n".join(sections)


def _html_workbook_sheet_table(sheet: _pc_workbook.ReviewWorkbookSheet) -> str:
    """Return an HTML table for one workbook sheet specification."""
    columns = _workbook_sheet_available_columns(sheet)
    if sheet.table.is_empty() or not columns:
        return _html_empty("No rows.")

    labels = sheet.labels or {}
    header_cells = [
        _html_workbook_header_cell(
            column=column,
            label=labels.get(column, _display_header(column)),
            tooltip=_pc_workbook_layout.workbook_column_tooltip(column),
        )
        for column in columns
    ]
    rendered_columns = tuple(
        (column, _pc_rendering.html_column_class(column)) for column in columns
    )
    body_rows = [
        _html_workbook_body_row(row, rendered_columns)
        for row in sheet.table.iter_rows(named=True)
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
    review_key_columns = {"review_key", "reconstruction_review_key"}
    return [
        column
        for column in requested_columns
        if column in sheet.table.columns and column not in review_key_columns
    ]


def _html_workbook_header_cell(*, column: str, label: str, tooltip: str) -> str:
    """Return one workbook-style HTML header cell."""
    title_attribute = f' title="{_escape_html(tooltip)}"' if tooltip else ""
    class_attribute = f' class="{_pc_rendering.html_column_class(column)}"'
    return (
        f'<th scope="col"{class_attribute}{title_attribute}>'
        f"{_escape_html(label)}</th>"
    )


def _html_workbook_body_row(
    row: Mapping[str, object],
    columns: Sequence[tuple[str, str]],
) -> str:
    """Return one workbook-style HTML table row."""
    row_type = _format_value(row.get("row_type"))
    review_status = _format_value(row.get("review_status"))
    cells = [
        _html_workbook_body_cell(
            row,
            column,
            column_class=column_class,
            row_type=row_type,
            review_status=review_status,
        )
        for column, column_class in columns
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def _html_workbook_body_cell(
    row: Mapping[str, object],
    column: str,
    *,
    column_class: str,
    row_type: str,
    review_status: str,
) -> str:
    """Return one workbook-style HTML table cell with row-aware classes."""
    value = row[column]
    classes = " ".join(
        [
            _pc_rendering.html_cell_alignment(value),
            column_class,
            *_pc_rendering.html_value_classes(column, value),
            *_html_workbook_row_value_classes(row_type, review_status, column),
        ]
    )
    rendered_value = _escape_html(_format_value(value))
    if column == "review_destination" and rendered_value:
        section_id = _pc_rendering.html_section_id(str(value))
        rendered_value = f'<a href="#{section_id}">{rendered_value}</a>'
    return f'<td class="{classes}">{rendered_value}</td>'


def _html_workbook_row_value_classes(
    row_type: str,
    review_status: str,
    column: str,
) -> tuple[str, ...]:
    """Return row-aware classes from values normalized once per table row."""
    if row_type == "Explained Cause" and column in {
        "row_type",
        "estimated_impact",
        "review_guidance",
    }:
        return ("pc-fill-explained-cause",)
    if row_type == "Possible Cause" and column in {"row_type", "review_guidance"}:
        return ("pc-fill-possible-cause",)
    if review_status in {"Partly Explained", "Unexplained"} and column in {
        "unexplained_change",
        "review_status",
        "review_note",
    }:
        return ("pc-fill-review-needed",)
    return ()


def _write_audit_html_report(
    findings: pl.DataFrame,
    output_path: util.PathLike,
    *,
    title: str = "Audit Report",
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _sheets: Sequence[_pc_workbook.ReviewWorkbookSheet] | None = None,
    _reconstruction_cache: (
        _pc_workbook_reconstruction.WorkbookReconstructionCache | None
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
    sheets = _sheets or (
        _pc_workbook_tables.audit_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=_reconstruction_cache,
        )
    )
    with report_path.open("w", encoding=util.ENCODING) as report_file:
        first_part = True
        for part in _html_document_parts(title, sheets):
            if not first_part:
                report_file.write("\n")
            first_part = False
            report_file.write(part)
    return report_path


def write_audit_report_bundle(
    findings: pl.DataFrame,
    output_directory: util.PathLike,
    *,
    title: str = "Audit Report",
    top_evidence_limit: int = 10,
    include_workbook: bool = True,
    include_html_output: bool = True,
    require_complete_yaml_setup: bool = True,
    require_causal_attribution: bool = False,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    expand_all_supporting_files: bool = False,
    _data_issues: pl.DataFrame | None = None,
    _reconstruction_cache: (
        _pc_workbook_reconstruction.WorkbookReconstructionCache | None
    ) = None,
) -> dict[str, Path]:
    """Write and atomically promote one validated Audit report bundle.

    Args:
        findings: Findings table returned by ``compare_snapshots``.
        output_directory: Final destination directory.
        title: Report title for generated review artifacts.
        top_evidence_limit: Maximum top-evidence rows per portfolio period.
        include_workbook: Whether to include an XLSX review workbook.
        include_html_output: Whether to include the browser report.
        require_complete_yaml_setup: Whether incomplete field treatment blocks
            output.
        require_causal_attribution: Whether incomplete supported causal setup
            blocks output.
        comparison_path: Optional Audit YAML path for report context.
        comparison_level: Portfolio or security review level.
        include_reconstruction_diagnostics: Whether to include detailed
            reconstruction artifacts.
        expand_all_supporting_files: Whether supporting files remain expanded.
        _data_issues: Optional run-scoped Data Issues table.
        _reconstruction_cache: Optional run-scoped reconstruction cache.

    Returns:
        Artifact paths below the final promoted directory.

    Raises:
        PpaError: If report construction, validation, or promotion fails.

    Notes:
        Existing output remains unchanged unless the complete staged bundle
        passes validation and is successfully promoted.
    """
    destination = Path(output_directory)
    with _atomic_directory.staged_directory(destination) as staging_directory:
        staged_paths = _write_audit_report_bundle_in_place(
            findings,
            staging_directory,
            title=title,
            top_evidence_limit=top_evidence_limit,
            include_workbook=include_workbook,
            include_html_output=include_html_output,
            require_complete_yaml_setup=require_complete_yaml_setup,
            require_causal_attribution=require_causal_attribution,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            expand_all_supporting_files=expand_all_supporting_files,
            _data_issues=_data_issues,
            _reconstruction_cache=_reconstruction_cache,
        )
    return {
        name: _atomic_directory.remap_staged_path(
            path,
            staging_root=staging_directory,
            destination_root=destination,
        )
        for name, path in staged_paths.items()
    }


def _write_audit_report_bundle_in_place(
    findings: pl.DataFrame,
    output_directory: util.PathLike,
    *,
    title: str = "Audit Report",
    top_evidence_limit: int = 10,
    include_workbook: bool = True,
    include_html_output: bool = True,
    require_complete_yaml_setup: bool = True,
    require_causal_attribution: bool = False,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    expand_all_supporting_files: bool = False,
    _data_issues: pl.DataFrame | None = None,
    _reconstruction_cache: (
        _pc_workbook_reconstruction.WorkbookReconstructionCache | None
    ) = None,
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
        include_html_output: Whether to include the browser HTML review report.
            Defaults to true.
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
        expand_all_supporting_files: Whether to retain the remaining supporting
            CSV and JSON files in ``supporting_files``. ``source_detail.csv`` is
            always written at the report root and is never duplicated in the
            supporting directory or archive. When false, the validated supporting
            directory is stored in ``audit_support.zip``.

    Returns:
        Mapping from bundle artifact name to normalized written path.

    Raises:
        PpaError: If report validation fails.
    """
    csv_only_output = not include_workbook and not include_html_output
    if include_workbook:
        _pc_workbook.ensure_openpyxl_installed()

    _pc_lineage.assert_finding_source_lineage(findings)
    finding_audit_trail = _pc_conservation.finding_audit_trail(findings)
    _pc_conservation.assert_complete_finding_audit_trail(
        findings,
        finding_audit_trail,
    )
    active_findings = _active_findings(findings)
    if require_complete_yaml_setup:
        _pc_runner.validate_yaml_setup_complete(findings)
    if require_causal_attribution:
        _pc_runner.validate_causal_attribution_ready(active_findings)

    reconstruction_cache = (
        _reconstruction_cache
        or _pc_workbook_reconstruction.WorkbookReconstructionCache(comparison_path)
    )
    table_cache = _pc_workbook_tables._WorkbookTableCache(active_findings)
    workbook_sheets = (
        _pc_workbook_tables.audit_review_workbook_sheets(
            findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _reconstruction_cache=reconstruction_cache,
            _table_cache=table_cache,
            _data_issues=_data_issues,
            _finding_audit_trail=finding_audit_trail,
        )
    )
    _pc_output_policy.assert_review_output_row_limit(
        workbook_sheets,
        comparison_level=comparison_level,
    )
    tables = _report_bundle_tables(
        active_findings,
        top_evidence_limit,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _reconstruction_cache=reconstruction_cache,
        _table_cache=table_cache,
    )
    cause_sheet = next(
        sheet
        for sheet in workbook_sheets
        if sheet.artifact_name
        == _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT
    )
    tables[_pc_review_model.CAUSE_LINEAGE_ARTIFACT] = _cause_lineage_export_table(
        cause_sheet.table
    )
    for sheet in workbook_sheets:
        tables[sheet.artifact_name] = sheet.table

    bundle_directory = Path(output_directory)
    bundle_directory.mkdir(parents=True, exist_ok=True)
    supporting_files_directory = (
        bundle_directory / _pc_bundle.SUPPORTING_FILES_DIRECTORY
    )
    supporting_files_directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    html_report_file_name = _pc_review_model.html_report_file_name(comparison_level)
    review_workbook_file_name = _pc_review_model.review_workbook_file_name(
        comparison_level
    )
    if include_html_output:
        paths["html_report"] = _write_audit_html_report(
            findings,
            bundle_directory / html_report_file_name,
            title=title,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            _sheets=workbook_sheets,
            _reconstruction_cache=reconstruction_cache,
        )
    paths["findings"] = _pc_bundle.write_csv_artifact(
        finding_audit_trail,
        supporting_files_directory / "findings.csv",
    )
    for name, table in tables.items():
        artifact_path = (
            bundle_directory / _pc_bundle.PROMOTED_SOURCE_DETAIL
            if name == Path(_pc_bundle.PROMOTED_SOURCE_DETAIL).stem
            else supporting_files_directory / f"{name}.csv"
        )
        paths[name] = _pc_bundle.write_csv_artifact(
            table,
            artifact_path,
        )
    if csv_only_output:
        for artifact_name in _pc_bundle._CSV_PRIMARY_REVIEW_ARTIFACTS:
            promoted_path = bundle_directory / f"{artifact_name}.csv"
            shutil.copy2(paths[artifact_name], promoted_path)
            paths[artifact_name] = promoted_path
    if include_workbook:
        paths[_REVIEW_WORKBOOK_ARTIFACT] = _pc_workbook.write_review_workbook_sheets(
            workbook_sheets or (),
            bundle_directory / review_workbook_file_name,
            column_tooltip=_pc_workbook_layout.workbook_column_tooltip,
        )
    paths["readme"] = _pc_bundle.write_report_bundle_readme(
        bundle_directory / "README.md",
        title=title,
        tables=tables,
        include_workbook=include_workbook,
        include_html_output=include_html_output,
        comparison_level=comparison_level,
        expand_all_supporting_files=expand_all_supporting_files,
    )
    manifest_path = supporting_files_directory / "manifest.json"
    paths["manifest"] = manifest_path
    paths["review_summary"] = supporting_files_directory / "review_summary.json"
    _pc_bundle.write_report_bundle_manifest(
        manifest_path,
        findings=findings,
        active_findings=active_findings,
        title=title,
        top_evidence_limit=top_evidence_limit,
        include_workbook=include_workbook,
        include_html_output=include_html_output,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        expand_all_supporting_files=expand_all_supporting_files,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        artifact_paths=paths,
        tables=tables,
        review_sheets=workbook_sheets,
        finding_audit_trail=finding_audit_trail,
        bundle_root=bundle_directory,
    )
    manifest_data = json.loads(manifest_path.read_text(encoding=util.ENCODING))
    manifest = {str(key): value for key, value in manifest_data.items()}
    _pc_bundle.write_report_bundle_review_summary(
        paths["review_summary"],
        manifest=manifest,
    )
    validation_issues = _pc_bundle.report_bundle_validation_issues(
        bundle_directory,
        include_output_parity=False,
    )
    if validation_issues:
        raise PpaError(
            "Report bundle validation failed: " + "; ".join(validation_issues),
            None,
        )
    if not expand_all_supporting_files:
        promoted_file_names = (
            tuple(f"{name}.csv" for name in _pc_bundle._CSV_PRIMARY_REVIEW_ARTIFACTS)
            if csv_only_output
            else ()
        )
        compact_paths = _pc_bundle.compact_supporting_files(
            bundle_directory,
            promoted_file_names=promoted_file_names,
        )
        paths = {
            name: path
            for name, path in paths.items()
            if path.parent == bundle_directory
        }
        paths.update(compact_paths)
    return paths


def _cause_lineage_export_table(causes: pl.DataFrame) -> pl.DataFrame:
    """Return a compact, traceable projection of internal cause lineage."""
    columns = (
        pc_cols.PORTFOLIO_ID,
        pc_cols.FROM_DATE,
        pc_cols.THRU_DATE,
        "as_of_date",
        pc_cols.SECURITY_ID,
        _pc_findings.DATASET,
        _pc_findings.SOURCE_COLUMN,
        _pc_findings.FINDING_CODE,
        _pc_findings.SOURCE_RECORD_LOCATOR,
        "estimated_impact",
        _pc_lineage.SOURCE_LINEAGE_TYPE,
        _pc_lineage.SOURCE_FINDING_FINGERPRINTS,
        _pc_conservation.SAFETY_DISPOSITION,
        _pc_conservation.ECONOMIC_EFFECT_ID,
        _pc_conservation.COUNTED_CAUSE_OWNER,
    )
    return causes.select([column for column in columns if column in causes.columns])


def _report_bundle_tables(
    active_findings: pl.DataFrame,
    top_evidence_limit: int,
    *,
    comparison_path: util.PathLike | None = None,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_diagnostics: bool = False,
    _reconstruction_cache: (
        _pc_workbook_reconstruction.WorkbookReconstructionCache | None
    ) = None,
    _table_cache: _pc_workbook_tables._WorkbookTableCache | None = None,
) -> dict[str, pl.DataFrame]:
    """Return report-bundle tables keyed by artifact stem."""
    reconstruction_cache = _reconstruction_cache or (
        _pc_workbook_reconstruction.WorkbookReconstructionCache(comparison_path)
    )
    table_cache = _table_cache or _pc_workbook_tables._WorkbookTableCache(
        active_findings
    )
    portfolio_period_summary = table_cache.portfolio_period_summary()
    cause_summary = table_cache.cause_summary(PORTFOLIO_COMPARISON_LEVEL)
    impact_coverage = table_cache.primary_coverage(PORTFOLIO_COMPARISON_LEVEL)
    transaction_cross_checks = (
        _pc_explain.portfolio_period_transaction_cross_checks(active_findings)
    )
    residual_status = _residual_status_table(
        active_findings,
        periods=portfolio_period_summary,
        causes=cause_summary,
        cross_checks=transaction_cross_checks,
    )
    context_evidence = _context_evidence_table(active_findings)
    tables = {
        "needs_review_summary": _needs_review_summary_table(
            active_findings,
            periods=portfolio_period_summary,
            coverage=impact_coverage,
            residual=residual_status,
            cross_checks=transaction_cross_checks,
            context=context_evidence,
        ),
        "portfolio_period_summary": portfolio_period_summary,
        "cause_summary": cause_summary,
        "impact_estimates": _impact_estimate_summary_table(
            active_findings,
            cause_summary=cause_summary,
        ),
        "impact_coverage": impact_coverage,
        "context_evidence_summary": _context_evidence_summary_table(
            active_findings,
            context_evidence=context_evidence,
        ),
        "context_evidence": context_evidence,
        "source_detail": _pc_workbook_tables._workbook_raw_audit_trail_table(
            active_findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
        ),
        "transaction_cross_checks": transaction_cross_checks,
        "residual_status": residual_status,
        "transaction_activity": _pc_explain.transaction_activity_summary(active_findings),
        "transaction_matching_diagnostics": (
            _pc_explain.transaction_matching_diagnostics(active_findings)
        ),
        "top_evidence": table_cache.top_evidence(
            PORTFOLIO_COMPARISON_LEVEL,
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


def _needs_review_summary_table(
    findings: pl.DataFrame,
    *,
    periods: pl.DataFrame | None = None,
    coverage: pl.DataFrame | None = None,
    residual: pl.DataFrame | None = None,
    cross_checks: pl.DataFrame | None = None,
    context: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return reviewer cues, reusing caller-provided summary tables when available."""
    periods = (
        _pc_explain.portfolio_period_summary(findings)
        if periods is None
        else periods
    )
    if periods.is_empty():
        return _empty_needs_review_summary()

    coverage_by_period = _period_rows_by_key(
        _pc_explain.portfolio_period_impact_coverage_summary(findings)
        if coverage is None
        else coverage
    )
    residual_by_period = _period_rows_by_key(
        _residual_status_table(findings) if residual is None else residual
    )
    cross_checks_by_period = _period_rows_by_key(
        _pc_explain.portfolio_period_transaction_cross_checks(findings)
        if cross_checks is None
        else cross_checks
    )
    context_by_period = _period_rows_by_key(
        _context_evidence_table(findings) if context is None else context
    )
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
    return pl.DataFrame(rows, infer_schema_length=None).select(_NEEDS_REVIEW_COLUMNS)


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
        artifacts.append("transaction_cross_checks.csv")
    if residual:
        artifacts.append("residual_status.csv")
    if context:
        artifacts.extend(["context_evidence_summary.csv", "context_evidence.csv"])
    artifacts.append("findings.csv")
    supporting_artifacts = [
        f"{_pc_bundle.SUPPORTING_FILES_DIRECTORY}/{artifact}"
        for artifact in dict.fromkeys(artifacts)
    ]
    return _comma_separated(supporting_artifacts)


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


def _residual_status_table(
    findings: pl.DataFrame,
    *,
    periods: pl.DataFrame | None = None,
    causes: pl.DataFrame | None = None,
    cross_checks: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return residual-status rows for portfolio-period return changes."""
    periods = (
        _pc_explain.portfolio_period_summary(findings)
        if periods is None
        else periods
    )
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

    causes = (
        _pc_explain.portfolio_period_cause_summary(findings)
        if causes is None
        else causes
    )
    cross_checks = (
        _pc_explain.portfolio_period_transaction_cross_checks(findings)
        if cross_checks is None
        else cross_checks
    )
    causes_by_period = _period_rows_by_key(causes)
    cross_checks_by_period = _period_rows_by_key(cross_checks)
    return pl.DataFrame(
        [
            _residual_status_row(
                period,
                causes_by_period.get(_period_key(period), []),
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


def _context_evidence_summary_table(
    findings: pl.DataFrame,
    *,
    context_evidence: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return grouped context evidence counts and affected identifiers."""
    context_evidence = (
        _context_evidence_table(findings)
        if context_evidence is None
        else context_evidence
    )
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
        pl.DataFrame(rows, infer_schema_length=None)
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
        pl.DataFrame(rows, infer_schema_length=None)
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
    return "review context"


def _impact_estimate_summary_table(
    findings: pl.DataFrame,
    *,
    cause_summary: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return currently quantified cause-summary rows."""
    summary = (
        _pc_explain.portfolio_period_cause_summary(findings)
        if cause_summary is None
        else cause_summary
    )
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
