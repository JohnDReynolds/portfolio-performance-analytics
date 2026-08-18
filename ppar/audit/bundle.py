"""Validate generated Audit report bundles."""

from __future__ import annotations

# Python imports
import datetime as dt
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.common as util
from ppar.audit import conservation as _pc_conservation
from ppar.audit import extract_contract as _pc_extract_contract
from ppar.audit import lineage as _pc_lineage
from ppar.audit import output_integrity as _pc_output_integrity
from ppar.audit import review_model as _pc_review_model
from ppar.audit import workbook as _pc_workbook
from ppar.audit.specification import (
    AuditSpecification,
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.audit.portfolio_performance import SnapshotKey
from ppar.audit.transaction_summary import (
    transaction_rule_codes,
    transaction_semantics_summary,
)
from ppar.audit.transactions import TransactionsLoader

__all__ = [
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "SUPPORTING_FILES_DIRECTORY",
    "report_bundle_contract",
    "report_bundle_manifest",
    "report_bundle_validation_issues",
    "write_csv_artifact",
    "write_report_bundle_manifest",
    "write_report_bundle_readme",
    "write_report_bundle_review_summary",
]

SUPPORTING_FILES_DIRECTORY = "supporting_files"
AUDIT_SUPPORT_ARCHIVE = "audit_support.zip"
PROMOTED_SOURCE_DETAIL = "source_detail.csv"
REPORT_BUNDLE_REQUIRED_ARTIFACTS = (
    "readme",
    "manifest",
    "review_summary",
    "findings",
    "source_detail",
    _pc_review_model.EXECUTIVE_SUMMARY_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
    _pc_review_model.DATA_ISSUES_ARTIFACT,
    _pc_review_model.CAUSE_LINEAGE_ARTIFACT,
    "needs_review_summary",
    "portfolio_period_summary",
    "cause_summary",
    "impact_estimates",
    "impact_coverage",
    "context_evidence_summary",
    "context_evidence",
    "transaction_cross_checks",
    "residual_status",
    "transaction_activity",
    "transaction_matching_diagnostics",
    "top_evidence",
)
_CSV_PRIMARY_REVIEW_ARTIFACTS = (
    _pc_review_model.EXECUTIVE_SUMMARY_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
    _pc_review_model.DATA_ISSUES_ARTIFACT,
)
_REPORT_BUNDLE_TYPE = "audit_report"
_REPORT_BUNDLE_MANIFEST_VERSION = 10
_REPORT_BUNDLE_REQUIRED_MANIFEST_KEYS = (
    "bundle_type",
    "manifest_version",
    "created_at",
    "title",
    "options",
    "source_context",
    "counts",
    "transaction_semantics",
    "artifacts",
    "tables",
    "review_entrypoints",
    "output_integrity",
)
_REPORT_BUNDLE_REQUIRED_REVIEW_ENTRYPOINTS = (
    "primary_review",
    "period_triage",
    "formula_input_causes",
    "supporting_context",
    "transaction_diagnostics",
    "audit_trail",
    "review_handoff",
)
_REVIEW_SUMMARY_VERSION = 1
_REVIEW_BASIS = "Modified Dietz evidence pack"
_REVIEW_SUMMARY_REQUIRED_KEYS = (
    "summary_version",
    "review_basis",
    "review_vocabulary",
    "entrypoints",
    "source_context",
    "counts",
    "transaction_semantics",
    "artifacts",
)
_REVIEW_VOCABULARY = {
    "formula_input": (
        "A value that ppar can use directly in the Modified Dietz return "
        "reconstruction or additive explanation."
    ),
    "source_data": (
        "The source-data is the original loaded portfolio, security, transaction, "
        "holding, cash, FX, or reference data compared by the report."
    ),
    "finding_level": (
        "The individual finding-level comparison-row detail before period or "
        "cause-area rollups."
    ),
    "cause_area": (
        "A cause-area grouping used to summarize related finding-level "
        "differences."
    ),
    "supporting_evidence": (
        "Source evidence that helps audit or explain a formula input without "
        "being counted as a separate return input."
    ),
    "context_only": (
        "Reviewer context that can prioritize or interpret a difference but is "
        "not counted in the Modified Dietz formula."
    ),
    "review_only": (
        "Rows marked review-only are shown for reviewer judgment with no automatic "
        "return-impact treatment."
    ),
    "evidence_only": (
        "Rows marked evidence-only are retained for audit or review without being "
        "counted as additive Modified Dietz inputs."
    ),
    "non_additive": (
        "Rows marked non-additive are diagnostics or cross-check rows that must "
        "not be summed into explained performance change."
    ),
    "explained_change": (
        "The explained-change wording names how source-data differences account "
        "for a reported return difference."
    ),
    "backlog_gate": (
        "A transaction or evidence family blocked until policy or source samples "
        "justify a Modified Dietz treatment."
    ),
}


def report_bundle_contract() -> dict[str, object]:
    """Return the stable machine-readable report-bundle contract.

    Returns:
        JSON-serializable contract data for generated report bundles.

    Notes:
        This helper describes the bundle handoff surface; it does not describe
        all internal implementation details or optional diagnostic artifacts.
    """
    return {
        "bundle_type": _REPORT_BUNDLE_TYPE,
        "audit_artifact_files": {
            comparison_level: {
                _pc_review_model.HTML_REPORT_ARTIFACT: (
                    _pc_review_model.html_report_file_name(comparison_level)
                ),
                _pc_review_model.REVIEW_WORKBOOK_ARTIFACT: (
                    _pc_review_model.review_workbook_file_name(comparison_level)
                ),
            }
            for comparison_level in (
                PORTFOLIO_COMPARISON_LEVEL,
                SECURITY_COMPARISON_LEVEL,
            )
        },
        "supporting_files_packaging": {
            "archive": AUDIT_SUPPORT_ARCHIVE,
            "archive_root": SUPPORTING_FILES_DIRECTORY,
            "promoted_reviewer_files": {
                "normal": [PROMOTED_SOURCE_DETAIL],
                "csv_only": [
                    *(f"{name}.csv" for name in _CSV_PRIMARY_REVIEW_ARTIFACTS),
                    PROMOTED_SOURCE_DETAIL,
                ],
            },
        },
        "required_artifacts": list(REPORT_BUNDLE_REQUIRED_ARTIFACTS),
        "primary_review_artifact_modes": {
            "xlsx": _pc_review_model.REVIEW_WORKBOOK_ARTIFACT,
            "html": _pc_review_model.HTML_REPORT_ARTIFACT,
            "csv": list(_CSV_PRIMARY_REVIEW_ARTIFACTS),
        },
        "manifest_version": _REPORT_BUNDLE_MANIFEST_VERSION,
        "normalization_version": _pc_output_integrity.NORMALIZATION_VERSION,
        "volatile_metadata": list(_pc_output_integrity.VOLATILE_METADATA),
        "required_manifest_keys": list(_REPORT_BUNDLE_REQUIRED_MANIFEST_KEYS),
        "required_review_entrypoints": list(
            _REPORT_BUNDLE_REQUIRED_REVIEW_ENTRYPOINTS
        ),
        "review_summary_version": _REVIEW_SUMMARY_VERSION,
        "review_basis": _REVIEW_BASIS,
        "required_review_summary_keys": list(_REVIEW_SUMMARY_REQUIRED_KEYS),
        "review_vocabulary_keys": list(_REVIEW_VOCABULARY),
    }


def write_csv_artifact(table: pl.DataFrame, output_path: Path) -> Path:
    """Write a report-bundle CSV artifact.

    Args:
        table: Table to write.
        output_path: Destination CSV path.

    Returns:
        Normalized destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(output_path)
    return output_path


def compact_supporting_files(
    bundle_directory: Path,
    *,
    promoted_file_names: Sequence[str] = (),
) -> dict[str, Path]:
    """Promote reviewer files and archive a validated supporting directory.

    Args:
        bundle_directory: Root of a fully written report-bundle staging area.
        promoted_file_names: Supporting filenames to copy to the bundle root
            before archiving the supporting directory. Root-level
            ``source_detail.csv`` is written before this function is called and
            is never part of the supporting directory.

    Returns:
        Paths to any promoted reviewer files and the supporting ZIP archive.

    Raises:
        FileNotFoundError: If the staging supporting directory or a requested
            promoted artifact is missing.
    """
    supporting_directory = bundle_directory / SUPPORTING_FILES_DIRECTORY
    if not supporting_directory.is_dir():
        raise FileNotFoundError(f"{supporting_directory} is missing")
    promoted_paths: dict[str, Path] = {}
    for file_name in promoted_file_names:
        supporting_path = supporting_directory / file_name
        if not supporting_path.is_file():
            raise FileNotFoundError(f"{supporting_path} is missing")
        promoted_path = bundle_directory / file_name
        shutil.copy2(supporting_path, promoted_path)
        promoted_paths[Path(file_name).stem] = promoted_path
    archive_path = bundle_directory / AUDIT_SUPPORT_ARCHIVE
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    temporary_archive.unlink(missing_ok=True)
    supporting_paths = sorted(
        path for path in supporting_directory.rglob("*") if path.is_file()
    )
    expected_members = [
        path.relative_to(bundle_directory).as_posix()
        for path in supporting_paths
    ]
    with zipfile.ZipFile(
        temporary_archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        for path, member_name in zip(
            supporting_paths,
            expected_members,
            strict=True,
        ):
            archive.write(path, member_name)
    with zipfile.ZipFile(temporary_archive) as archive:
        if archive.namelist() != expected_members:
            raise OSError("Compact audit archive member inventory does not match")
    temporary_archive.replace(archive_path)
    shutil.rmtree(supporting_directory)
    return {**promoted_paths, "audit_support": archive_path}


def write_report_bundle_readme(
    output_path: Path,
    *,
    title: str,
    include_workbook: bool,
    include_html_output: bool = True,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> Path:
    """Write reviewer guidance for an Audit report bundle.

    Args:
        output_path: Destination README path.
        title: Report title to show as the README heading.
        include_workbook: Whether the bundle includes the XLSX review workbook.
        include_html_output: Whether the bundle includes the browser HTML
            review report.
        comparison_level: Primary performance-result level for presentation.

    Returns:
        Normalized destination path.
    """
    primary_sheet = _pc_review_model.EXECUTIVE_SUMMARY_SHEET
    html_report_file_name = _pc_review_model.html_report_file_name(comparison_level)
    review_workbook_file_name = _pc_review_model.review_workbook_file_name(
        comparison_level
    )
    csv_only_output = not include_workbook and not include_html_output
    if include_workbook and include_html_output:
        first_review_step = (
            f"1. Open `{review_workbook_file_name}` or `{html_report_file_name}`. "
            "They are alternative views of the same report data."
        )
    elif include_workbook:
        first_review_step = f"1. Open `{review_workbook_file_name}`."
    elif include_html_output:
        first_review_step = f"1. Open `{html_report_file_name}`."
    else:
        first_review_step = (
            "1. Open `executive_summary.csv`, then use the other primary CSV "
            "review files in the order below."
        )
    review_unit = _readme_review_unit(comparison_level)
    causes_review_artifact = (
        "`performance_difference_causes.csv`"
        if csv_only_output
        else _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET
    )
    performance_review_artifact = (
        "`performance_differences.csv`"
        if csv_only_output
        else _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET
    )
    data_issues_review_artifact = (
        "`data_issues.csv`"
        if csv_only_output
        else _pc_review_model.DATA_ISSUES_SHEET
    )
    lines = [
        f"# {_escape_readme_text(title)}",
        "",
        "This directory contains an Audit review bundle.",
        "",
        "## Recommended Review Order",
        "",
        first_review_step,
        f"2. Start with {primary_sheet} for an overview of the performance "
        "differences, explanation status, and data issues.",
        f"3. Use {performance_review_artifact} to review the exact {review_unit}s "
        "that changed.",
        *(
            []
            if csv_only_output
            else [
                "   Yellow cells identify partly explained or unexplained "
                "differences that still need review."
            ]
        ),
        f"4. Use {causes_review_artifact} to review explained causes, supporting "
        "evidence, possible causes, and the Modified Dietz inputs used to evaluate "
        "each performance difference.",
        *(
            []
            if csv_only_output
            else [
                "   Yellow cells identify quantified causes included in the "
                "explained difference. Gold cells identify possible causes that "
                "are not counted as explanations."
            ]
        ),
        f"5. Use {data_issues_review_artifact} to review cross-reference "
        "consistency checks across Snapshot A and Snapshot B.",
        "",
        "## Supporting Audit Evidence",
        "",
        f"- `{PROMOTED_SOURCE_DETAIL}` contains the complete finding-level audit "
        "trail, including detail that may not be promoted into the primary review "
        "sheets.",
        f"- `{AUDIT_SUPPORT_ARCHIVE}` contains validated supporting CSV and JSON "
        "files, including findings, lineage, diagnostics, manifest metadata, and "
        "review-handoff information. Extract the archive when these individual "
        "files are needed; extraction does not recalculate or change the report.",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding=util.ENCODING)
    return output_path


def _readme_review_unit(comparison_level: str) -> str:
    """Return reviewer-facing wording for the primary comparison grain."""
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return "security period"
    return "performance period"


def _escape_readme_text(value: object) -> str:
    """Return normalized text safe for generated README Markdown."""
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def write_report_bundle_manifest(
    output_path: Path,
    *,
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    title: str,
    top_evidence_limit: int,
    include_workbook: bool = False,
    include_html_output: bool = True,
    include_reconstruction_diagnostics: bool = False,
    comparison_path: util.PathLike | None = None,
    comparison_level: str,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
    review_sheets: Sequence[_pc_workbook.ReviewWorkbookSheet],
    finding_audit_trail: pl.DataFrame | None = None,
    bundle_root: Path | None = None,
) -> Path:
    """Write a report-bundle JSON manifest.

    Args:
        output_path: Destination manifest path.
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        include_workbook: Whether the XLSX primary review artifact is included.
        include_html_output: Whether the HTML primary review artifact is included.
        include_reconstruction_diagnostics: Whether optional return
            reconstruction diagnostic sections and CSV artifacts are included in
            the bundle.
        comparison_path: Optional comparison YAML path used to generate the
            bundle.
        comparison_level: Explicit portfolio or security result level.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.
        review_sheets: Canonical internal tables shared by HTML, XLSX, and the
            review-sheet CSV artifacts.
        finding_audit_trail: Optional precomputed complete finding audit trail.
        bundle_root: Report bundle root directory used for manifest-relative
            artifact references. Defaults to the manifest output directory.

    Returns:
        Normalized destination path.
    """
    manifest = report_bundle_manifest(
        findings=findings,
        active_findings=active_findings,
        title=title,
        top_evidence_limit=top_evidence_limit,
        include_workbook=include_workbook,
        include_html_output=include_html_output,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
        artifact_paths=artifact_paths,
        tables=tables,
        review_sheets=review_sheets,
        finding_audit_trail=finding_audit_trail,
        bundle_root=bundle_root or output_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding=util.ENCODING,
    )
    return output_path


def write_report_bundle_review_summary(
    output_path: Path,
    *,
    manifest: Mapping[str, object],
) -> Path:
    """Write a compact review-handoff summary for a report bundle.

    Args:
        output_path: Destination summary path.
        manifest: Manifest data for the same report bundle.

    Returns:
        Normalized destination path.
    """
    summary = _report_bundle_review_summary(manifest=manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding=util.ENCODING,
    )
    return output_path


def _report_bundle_review_summary(
    *,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Return compact reviewer-facing metadata for a report bundle.

    Args:
        manifest: Manifest data for the same report bundle.

    Returns:
        JSON-serializable review summary data.

    Notes:
        The summary repeats selected manifest fields intentionally. It is a
        short reviewer handoff file, not the authoritative artifact inventory.
    """
    return {
        "summary_version": _REVIEW_SUMMARY_VERSION,
        "review_basis": _REVIEW_BASIS,
        "review_vocabulary": dict(_REVIEW_VOCABULARY),
        "entrypoints": _manifest_mapping(manifest, "review_entrypoints"),
        "source_context": _manifest_mapping(manifest, "source_context"),
        "counts": _manifest_mapping(manifest, "counts"),
        "transaction_semantics": _manifest_mapping(manifest, "transaction_semantics"),
        "artifacts": _manifest_mapping(manifest, "artifacts"),
    }


def report_bundle_manifest(
    *,
    findings: pl.DataFrame,
    active_findings: pl.DataFrame,
    title: str,
    top_evidence_limit: int,
    include_workbook: bool = False,
    include_html_output: bool = True,
    include_reconstruction_diagnostics: bool = False,
    comparison_path: util.PathLike | None = None,
    comparison_level: str,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
    review_sheets: Sequence[_pc_workbook.ReviewWorkbookSheet],
    finding_audit_trail: pl.DataFrame | None = None,
    bundle_root: Path | None = None,
) -> dict[str, object]:
    """Return JSON-serializable metadata for a report bundle.

    Args:
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        include_workbook: Whether the XLSX primary review artifact is included.
        include_html_output: Whether the HTML primary review artifact is included.
        include_reconstruction_diagnostics: Whether interim reconstruction
            diagnostics are included in the bundle.
        comparison_path: Optional comparison YAML path used to generate the
            bundle.
        comparison_level: Explicit portfolio or security result level.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.
        review_sheets: Canonical internal tables shared by review formats.
        finding_audit_trail: Optional precomputed complete finding audit trail.
        bundle_root: Report bundle root directory used for manifest-relative
            artifact references. Defaults to the current directory.

    Returns:
        JSON-serializable manifest data.
    """
    suppressed_count = findings.height - active_findings.height
    artifact_references = _artifact_references(
        artifact_paths,
        bundle_root=bundle_root or Path("."),
    )
    persisted_findings = (
        _pc_conservation.finding_audit_trail(findings)
        if finding_audit_trail is None
        else finding_audit_trail
    )
    manifest = {
        "bundle_type": _REPORT_BUNDLE_TYPE,
        "manifest_version": _REPORT_BUNDLE_MANIFEST_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "title": title,
        "options": {
            "top_evidence_limit": top_evidence_limit,
            "include_workbook": include_workbook,
            "include_html_output": include_html_output,
            "include_reconstruction_diagnostics": include_reconstruction_diagnostics,
        },
        "source_context": _report_bundle_source_context(
            comparison_path,
            comparison_level=comparison_level,
        ),
        "counts": {
            "findings": findings.height,
            "active_findings": active_findings.height,
            "suppressed_findings": suppressed_count,
        },
        "transaction_semantics": _report_bundle_transaction_semantics(
            active_findings,
            comparison_path=comparison_path,
            comparison_level=comparison_level,
        ),
        "artifacts": artifact_references,
        "tables": {
            "findings": _pc_output_integrity.table_manifest_metadata(
                persisted_findings
            ),
            **{
                name: _pc_output_integrity.table_manifest_metadata(table)
                for name, table in sorted(tables.items())
            },
        },
        "review_entrypoints": _report_bundle_review_entrypoints(
            artifact_references,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        ),
        "output_integrity": _pc_output_integrity.output_integrity_metadata(
            review_sheets
        ),
    }
    return _pc_output_integrity.with_normalized_bundle_fingerprint(manifest)


def _report_bundle_source_context(
    comparison_path: util.PathLike | None,
    *,
    comparison_level: str,
) -> dict[str, object]:
    """Return source metadata that helps reviewers reproduce a bundle."""
    if comparison_path is None:
        return {
            "comparison_path": None,
            "extract_contract": None,
        }
    specification = AuditSpecification(
        comparison_path,
        comparison_level=comparison_level,
    )
    return {
        "comparison_path": str(comparison_path),
        "extract_contract": _pc_extract_contract.extract_contract_summary(
            specification.values,
            specification_path=specification.path,
        ),
    }


def _report_bundle_transaction_semantics(
    active_findings: pl.DataFrame,
    *,
    comparison_path: util.PathLike | None,
    comparison_level: str,
) -> dict[str, object]:
    """Return compact transaction semantics metadata for a report bundle."""
    if comparison_path is None:
        return transaction_semantics_summary([active_findings])
    specification = AuditSpecification(
        comparison_path,
        comparison_level=comparison_level,
    )
    loader = TransactionsLoader(specification)
    snapshot_keys: tuple[SnapshotKey, SnapshotKey] = ("a", "b")
    frames = [
        frame
        for snapshot_key in snapshot_keys
        if (frame := loader.load(snapshot_key)) is not None
    ]
    return transaction_semantics_summary(
        frames,
        rule_codes=transaction_rule_codes(specification.values),
    )


def _artifact_references(
    artifact_paths: Mapping[str, Path],
    *,
    bundle_root: Path,
) -> dict[str, str]:
    """Return manifest artifact references relative to the bundle root."""
    references: dict[str, str] = {}
    root = bundle_root.resolve()
    for name, path in sorted(artifact_paths.items()):
        try:
            reference = path.resolve().relative_to(root)
        except ValueError:
            references[name] = path.name
            continue
        references[name] = reference.as_posix()
    return references


def _report_bundle_review_entrypoints(
    artifact_references: Mapping[str, str],
    *,
    include_reconstruction_diagnostics: bool,
) -> dict[str, object]:
    """Return the intended first-stop artifacts for reviewer navigation."""
    artifacts = dict(artifact_references)
    transaction_diagnostics = [
        artifacts[name]
        for name in (
            "transaction_activity",
            "transaction_cross_checks",
            "transaction_matching_diagnostics",
        )
        if name in artifacts
    ]
    primary_review: object = artifacts.get(
        _pc_review_model.REVIEW_WORKBOOK_ARTIFACT,
        artifacts.get(_pc_review_model.HTML_REPORT_ARTIFACT),
    )
    if primary_review is None:
        primary_review = [
            artifacts[name]
            for name in _CSV_PRIMARY_REVIEW_ARTIFACTS
            if name in artifacts
        ]
    entrypoints: dict[str, object] = {
        "primary_review": primary_review,
        "period_triage": artifacts.get("needs_review_summary"),
        "formula_input_causes": artifacts.get("cause_summary"),
        "supporting_context": artifacts.get("context_evidence_summary"),
        "transaction_diagnostics": transaction_diagnostics,
        "audit_trail": artifacts.get("findings"),
        "review_handoff": artifacts.get("review_summary"),
    }
    if include_reconstruction_diagnostics:
        entrypoints["return_reconstruction"] = [
            artifacts[name]
            for name in (
                _pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT,
                _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
                _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
            )
            if name in artifacts
        ]
    return entrypoints


def report_bundle_validation_issues(
    bundle_directory: util.PathLike,
    *,
    include_output_parity: bool = True,
) -> list[str]:
    """Return validation issues for a generated report bundle.

    Args:
        bundle_directory: Directory containing a generated report bundle.
        include_output_parity: Whether to independently reparse every persisted
            CSV, HTML table, and XLSX sheet. Generation can disable this expensive
            cross-format pass because it already enforces financial and workbook
            invariants before serialization. Standalone validation retains the
            complete parity pass by default.

    Returns:
        Human-readable validation issues. An empty list means validation passed.
    """
    bundle_path = Path(bundle_directory)
    archive_path = bundle_path / AUDIT_SUPPORT_ARCHIVE
    if not archive_path.is_file():
        return [f"{AUDIT_SUPPORT_ARCHIVE} is missing"]
    return _archived_report_bundle_validation_issues(
        bundle_path,
        archive_path,
        include_output_parity=include_output_parity,
    )


def _archived_report_bundle_validation_issues(
    bundle_path: Path,
    archive_path: Path,
    *,
    include_output_parity: bool,
) -> list[str]:
    """Validate a compact bundle through a safely expanded temporary copy."""
    try:
        with tempfile.TemporaryDirectory() as directory:
            expanded_path = Path(directory)
            with zipfile.ZipFile(archive_path) as archive:
                unsafe_member = next(
                    (
                        member.filename
                        for member in archive.infolist()
                        if not _safe_supporting_archive_member(member.filename)
                    ),
                    None,
                )
                if unsafe_member is not None:
                    return [
                        f"{AUDIT_SUPPORT_ARCHIVE} contains unsafe path "
                        f"{unsafe_member!r}"
                    ]
                archive.extractall(expanded_path)
            for path in bundle_path.iterdir():
                if path.is_file() and path.name != AUDIT_SUPPORT_ARCHIVE:
                    shutil.copy2(path, expanded_path / path.name)
            manifest_path = (
                expanded_path / SUPPORTING_FILES_DIRECTORY / "manifest.json"
            )
            if not manifest_path.is_file():
                return [
                    f"{AUDIT_SUPPORT_ARCHIVE} does not contain "
                    f"{SUPPORTING_FILES_DIRECTORY}/manifest.json"
                ]
            if not (bundle_path / PROMOTED_SOURCE_DETAIL).is_file():
                return [f"promoted {PROMOTED_SOURCE_DETAIL} is missing"]
            promoted_issues = _promoted_csv_archive_issues(
                bundle_path,
                expanded_path,
            )
            if promoted_issues:
                return promoted_issues
            return _extracted_report_bundle_validation_issues(
                expanded_path,
                manifest_path,
                include_output_parity=include_output_parity,
            )
    except (OSError, zipfile.BadZipFile) as error:
        return [f"{AUDIT_SUPPORT_ARCHIVE} cannot be read: {error}"]


def _promoted_csv_archive_issues(
    bundle_path: Path,
    expanded_path: Path,
) -> list[str]:
    """Return parity issues between promoted CSVs and their archived copies."""
    for promoted_path in sorted(bundle_path.glob("*.csv")):
        if promoted_path.name == PROMOTED_SOURCE_DETAIL:
            continue
        archived_path = (
            expanded_path / SUPPORTING_FILES_DIRECTORY / promoted_path.name
        )
        if not archived_path.is_file():
            return [
                f"{AUDIT_SUPPORT_ARCHIVE} does not contain "
                f"{SUPPORTING_FILES_DIRECTORY}/{promoted_path.name}"
            ]
        if promoted_path.read_bytes() != archived_path.read_bytes():
            return [
                f"promoted {promoted_path.name} does not match "
                f"{AUDIT_SUPPORT_ARCHIVE}"
            ]
    return []


def _safe_supporting_archive_member(member_name: str) -> bool:
    """Return whether a ZIP member stays within ``supporting_files``."""
    path = Path(member_name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] == SUPPORTING_FILES_DIRECTORY
    )


def _extracted_report_bundle_validation_issues(
    bundle_path: Path,
    manifest_path: Path,
    *,
    include_output_parity: bool = True,
) -> list[str]:
    """Return validation issues for a safely extracted report bundle."""

    manifest = _read_report_bundle_manifest(manifest_path)
    if manifest is None:
        return ["manifest.json is not a JSON object"]

    issues: list[str] = []
    issues.extend(_report_bundle_manifest_shape_issues(manifest))
    artifacts = _manifest_mapping(manifest, "artifacts")
    tables = _manifest_mapping(manifest, "tables")
    issues.extend(_report_bundle_artifact_issues(bundle_path, artifacts))
    issues.extend(_report_bundle_output_mode_issues(manifest, artifacts))
    issues.extend(_report_bundle_review_entrypoint_issues(manifest, artifacts))
    issues.extend(_report_bundle_source_context_issues(manifest))
    issues.extend(_report_bundle_transaction_semantics_issues(manifest))
    issues.extend(_report_bundle_review_summary_issues(bundle_path, manifest, artifacts))
    issues.extend(_report_bundle_table_issues(bundle_path, artifacts, tables))
    issues.extend(_report_bundle_lineage_issues(bundle_path, artifacts))
    issues.extend(_report_bundle_workbook_issues(bundle_path, artifacts))
    if include_output_parity:
        issues.extend(
            _pc_output_integrity.report_bundle_output_integrity_issues(
                bundle_path,
                manifest,
                artifacts,
                tables,
            )
        )
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
    for artifact_name in REPORT_BUNDLE_REQUIRED_ARTIFACTS:
        artifact_file = artifacts.get(artifact_name)
        if not isinstance(artifact_file, str) or not artifact_file:
            issues.append(f"manifest artifact {artifact_name!r} is missing")
            continue
        if not (bundle_path / artifact_file).is_file():
            issues.append(f"artifact file {artifact_file!r} is missing")
    return issues


def _report_bundle_output_mode_issues(
    manifest: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> list[str]:
    """Return manifest option and primary-artifact consistency issues."""
    options = _manifest_mapping(manifest, "options")
    issues: list[str] = []
    resolved_modes: dict[str, bool] = {}
    for option_name, artifact_name in (
        ("include_workbook", _pc_review_model.REVIEW_WORKBOOK_ARTIFACT),
        ("include_html_output", _pc_review_model.HTML_REPORT_ARTIFACT),
    ):
        option_value = options.get(option_name)
        if not isinstance(option_value, bool):
            issues.append(f"manifest option {option_name!r} is missing or malformed")
            continue
        resolved_modes[option_name] = option_value
        artifact_is_declared = bool(artifacts.get(artifact_name))
        if option_value != artifact_is_declared:
            issues.append(
                f"manifest option {option_name!r} does not match artifact "
                f"{artifact_name!r}"
            )
    if len(resolved_modes) != 2:
        return issues

    include_workbook = resolved_modes["include_workbook"]
    include_html_output = resolved_modes["include_html_output"]
    entrypoints = _manifest_mapping(manifest, "review_entrypoints")
    if include_workbook:
        expected_primary_review: object = artifacts.get(
            _pc_review_model.REVIEW_WORKBOOK_ARTIFACT
        )
    elif include_html_output:
        expected_primary_review = artifacts.get(
            _pc_review_model.HTML_REPORT_ARTIFACT
        )
    else:
        expected_primary_review = [
            f"{artifact_name}.csv"
            for artifact_name in _CSV_PRIMARY_REVIEW_ARTIFACTS
        ]
        for artifact_name, expected_reference in zip(
            (*_CSV_PRIMARY_REVIEW_ARTIFACTS, "source_detail"),
            (*expected_primary_review, PROMOTED_SOURCE_DETAIL),
            strict=True,
        ):
            if artifacts.get(artifact_name) != expected_reference:
                issues.append(
                    f"CSV-only artifact {artifact_name!r} must be promoted as "
                    f"{expected_reference!r}"
                )
    if entrypoints.get("primary_review") != expected_primary_review:
        issues.append("manifest primary_review does not match selected output modes")
    return issues


def _report_bundle_manifest_shape_issues(
    manifest: Mapping[str, object],
) -> list[str]:
    """Return top-level manifest schema issues."""
    issues: list[str] = []
    for key in _REPORT_BUNDLE_REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            issues.append(f"manifest top-level key {key!r} is missing")
    if manifest.get("bundle_type") != _REPORT_BUNDLE_TYPE:
        issues.append("manifest bundle_type is malformed")
    version = manifest.get("manifest_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _REPORT_BUNDLE_MANIFEST_VERSION
    ):
        issues.append("manifest manifest_version is unsupported")
    if not isinstance(manifest.get("created_at"), str):
        issues.append("manifest created_at is malformed")
    if not isinstance(manifest.get("title"), str):
        issues.append("manifest title is malformed")
    if not isinstance(manifest.get("options"), dict):
        issues.append("manifest options is missing or malformed")
    if not isinstance(manifest.get("counts"), dict):
        issues.append("manifest counts is missing or malformed")
    return issues


def _report_bundle_review_entrypoint_issues(
    manifest: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> list[str]:
    """Return review-entrypoint references that do not map to artifacts."""
    declared_artifacts = {
        artifact_file
        for artifact_file in artifacts.values()
        if isinstance(artifact_file, str) and artifact_file
    }
    entrypoints = _manifest_mapping(manifest, "review_entrypoints")
    issues: list[str] = []
    for entrypoint_name in _REPORT_BUNDLE_REQUIRED_REVIEW_ENTRYPOINTS:
        if entrypoint_name not in entrypoints:
            issues.append(f"manifest review entrypoint {entrypoint_name!r} is missing")
    for entrypoint_name, entrypoint_value in entrypoints.items():
        issues.extend(
            _review_entrypoint_value_issues(
                entrypoint_name,
                entrypoint_value,
                declared_artifacts,
            )
        )
    return issues


def _report_bundle_source_context_issues(
    manifest: Mapping[str, object],
) -> list[str]:
    """Return malformed source-context metadata issues."""
    source_context = manifest.get("source_context")
    if not isinstance(source_context, dict):
        return ["manifest source_context is missing or malformed"]
    issues: list[str] = []
    if "comparison_path" not in source_context:
        issues.append("manifest source_context.comparison_path is missing")
    extract_contract = source_context.get("extract_contract")
    if extract_contract is None:
        return issues
    if not isinstance(extract_contract, dict):
        return ["manifest source_context.extract_contract is malformed"]
    issues.extend(_extract_contract_summary_issues(extract_contract))
    return issues


def _extract_contract_summary_issues(
    extract_contract: Mapping[str, object],
) -> list[str]:
    """Return malformed extract-contract summary issues."""
    issues: list[str] = []
    if not isinstance(extract_contract.get("path"), str):
        issues.append("manifest extract_contract.path is missing or malformed")
    enforce_value = extract_contract.get("enforce_ambiguous_axys_flows")
    if not isinstance(enforce_value, bool):
        issues.append(
            "manifest extract_contract.enforce_ambiguous_axys_flows is malformed"
        )
    context_columns = extract_contract.get("required_transaction_context_columns")
    if not _is_string_list(context_columns):
        issues.append(
            "manifest extract_contract.required_transaction_context_columns "
            "is malformed"
        )
    return issues


def _report_bundle_transaction_semantics_issues(
    manifest: Mapping[str, object],
) -> list[str]:
    """Return malformed transaction-semantics summary issues."""
    semantics = manifest.get("transaction_semantics")
    if not isinstance(semantics, dict):
        return ["manifest transaction_semantics is missing or malformed"]
    issues: list[str] = []
    for key in ("observed_codes", "codes_without_yaml_rules"):
        if not _is_string_list(semantics.get(key)):
            issues.append(f"manifest transaction_semantics.{key} is malformed")
    for key in ("unknown_category_count", "ambiguous_context_blocked_count"):
        value = semantics.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(f"manifest transaction_semantics.{key} is malformed")
    counts = semantics.get("semantics_source_counts")
    if not isinstance(counts, dict) or not all(
        isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in counts.items()
    ):
        issues.append(
            "manifest transaction_semantics.semantics_source_counts is malformed"
        )
    return issues


def _report_bundle_review_summary_issues(
    bundle_path: Path,
    manifest: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> list[str]:
    """Return malformed review-summary artifact issues."""
    summary_file = artifacts.get("review_summary")
    if not isinstance(summary_file, str) or not summary_file:
        return []
    summary_path = bundle_path / summary_file
    if not summary_path.exists():
        return []

    summary = _read_json_object(summary_path)
    if summary is None:
        return ["review_summary.json is not a JSON object"]

    issues: list[str] = []
    for key in _REVIEW_SUMMARY_REQUIRED_KEYS:
        if key not in summary:
            issues.append(f"review_summary top-level key {key!r} is missing")
    if summary.get("summary_version") != _REVIEW_SUMMARY_VERSION:
        issues.append("review_summary summary_version is unsupported")
    if summary.get("review_basis") != _REVIEW_BASIS:
        issues.append("review_summary review_basis is malformed")

    vocabulary = summary.get("review_vocabulary")
    if not isinstance(vocabulary, dict) or not all(
        isinstance(value, str) and value
        for value in vocabulary.values()
    ):
        issues.append("review_summary review_vocabulary is malformed")
    elif set(vocabulary) != set(_REVIEW_VOCABULARY):
        issues.append("review_summary review_vocabulary keys are malformed")

    manifest_fields = {
        "entrypoints": _manifest_mapping(manifest, "review_entrypoints"),
        "source_context": _manifest_mapping(manifest, "source_context"),
        "counts": _manifest_mapping(manifest, "counts"),
        "transaction_semantics": _manifest_mapping(manifest, "transaction_semantics"),
        "artifacts": _manifest_mapping(manifest, "artifacts"),
    }
    for key, expected_value in manifest_fields.items():
        if summary.get(key) != expected_value:
            issues.append(f"review_summary {key} does not match manifest")
    return issues


def _read_json_object(json_path: Path) -> dict[str, object] | None:
    """Read a JSON object from a generated report-bundle file."""
    try:
        json_data: object = json.loads(json_path.read_text(encoding=util.ENCODING))
    except json.JSONDecodeError:
        return None
    if not isinstance(json_data, dict):
        return None
    return {str(key): value for key, value in json_data.items()}


def _is_string_list(value: object) -> bool:
    """Return whether a value is a list of strings."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _review_entrypoint_value_issues(
    entrypoint_name: str,
    entrypoint_value: object,
    declared_artifacts: set[str],
) -> list[str]:
    """Return issues for one manifest review-entrypoint value."""
    if entrypoint_value is None:
        return []
    if isinstance(entrypoint_value, str):
        if entrypoint_value in declared_artifacts:
            return []
        return [
            (
                f"manifest review entrypoint {entrypoint_name!r} points to "
                f"undeclared artifact {entrypoint_value!r}"
            )
        ]
    if isinstance(entrypoint_value, list):
        issues: list[str] = []
        for item in entrypoint_value:
            issues.extend(
                _review_entrypoint_value_issues(
                    entrypoint_name,
                    item,
                    declared_artifacts,
                )
            )
        return issues
    return [f"manifest review entrypoint {entrypoint_name!r} is malformed"]


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
    if table_name == "findings":
        issues.extend(_finding_audit_trail_issues(table))
    if table_name == _pc_review_model.CAUSE_LINEAGE_ARTIFACT:
        issues.extend(_pc_lineage.persisted_cause_lineage_issues(table))
    return issues


def _finding_audit_trail_issues(table: pl.DataFrame) -> list[str]:
    """Return SN-01 validation issues for the persisted complete audit trail."""
    return _pc_conservation.persisted_finding_audit_trail_issues(table)


def _report_bundle_lineage_issues(bundle_path: Path, artifacts: Mapping[str, object]) -> list[str]:
    """Return cross-artifact finding-to-cause lineage issues."""
    finding_file = artifacts.get("findings")
    cause_file = artifacts.get(_pc_review_model.CAUSE_LINEAGE_ARTIFACT)
    if not isinstance(finding_file, str) or not isinstance(cause_file, str):
        return []
    try:
        findings = pl.read_csv(bundle_path / finding_file)
        causes = pl.read_csv(bundle_path / cause_file)
    except (OSError, pl.exceptions.PolarsError):
        return []
    return _pc_lineage.persisted_cross_artifact_lineage_issues(findings, causes)


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
