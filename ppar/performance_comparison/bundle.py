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
from ppar.performance_comparison import extract_contract as _pc_extract_contract
from ppar.performance_comparison import review_model as _pc_review_model
from ppar.performance_comparison import workbook as _pc_workbook
from ppar.performance_comparison.specification import (
    PerformanceComparisonSpecification,
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
from ppar.performance_comparison.transaction_summary import (
    transaction_rule_codes,
    transaction_semantics_summary,
)
from ppar.performance_comparison.transactions import TransactionsLoader

__all__ = [
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "report_bundle_contract",
    "report_bundle_manifest",
    "report_bundle_validation_issues",
    "write_csv_artifact",
    "write_report_bundle_manifest",
    "write_report_bundle_readme",
    "write_report_bundle_review_summary",
]

REPORT_BUNDLE_REQUIRED_ARTIFACTS = (
    "html_report",
    "readme",
    "manifest",
    "review_summary",
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
_REPORT_BUNDLE_MANIFEST_VERSION = 1
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
        "bundle_type": "performance_comparison_report",
        "required_artifacts": list(REPORT_BUNDLE_REQUIRED_ARTIFACTS),
        "manifest_version": _REPORT_BUNDLE_MANIFEST_VERSION,
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
    table.write_csv(output_path)
    return output_path


def write_report_bundle_readme(
    output_path: Path,
    *,
    title: str,
    tables: Mapping[str, pl.DataFrame],
    include_workbook: bool,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
) -> Path:
    """Write a portable report-bundle README.

    Args:
        output_path: Destination README path.
        title: Report title to show as the README heading.
        tables: Named CSV helper tables included in the bundle.
        include_workbook: Whether the bundle includes the XLSX review workbook.
        comparison_level: Primary performance-result level for presentation.

    Returns:
        Normalized destination path.
    """
    primary_sheet = _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET
    first_review_step = (
        f"1. Open `report.xlsx` when present; use `report.html` for browser "
        f"review. Start with {primary_sheet}."
        if include_workbook
        else f"1. Open `report.html`. Start with {primary_sheet}."
    )
    review_unit = _readme_review_unit(comparison_level)
    lines = [
        f"# {_escape_readme_text(title)}",
        "",
        "This directory is a portable performance-comparison review bundle.",
        "",
        "## Recommended Review Order",
        "",
        first_review_step,
        f"2. Use {_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET} to see which "
        f"source-data differences additively explain each {review_unit}.",
        f"3. Use {_pc_review_model.OTHER_DATA_DIFFERENCES_SHEET} for "
        f"review-only context. Use {_pc_review_model.RAW_AUDIT_TRAIL_SHEET} "
        "for audit and troubleshooting; it is the complete "
        "finding-level audit trail.",
        f"4. Use the `review_key` column to follow a {review_unit} across CSV artifacts.",
        "5. Use `transaction_activity.csv`, `transaction_cross_checks.csv`, and "
        "`flow_cross_check_reconciliation.csv` for supplementary transaction "
        "and external-flow diagnostics.",
        "   Use `transaction_matching_diagnostics.csv` only when auditing "
        "transaction row-identity evidence; it reports conservative matching "
        "status and does not imply fuzzy transaction linkage.",
        "6. Use `review_summary.json` when handing the bundle to another reviewer "
        "or automation. It names the Modified Dietz vocabulary, entrypoints, "
        "source context, and transaction-semantics summary in one compact file.",
        "",
        "## Audit/Export Files",
        "",
        "- `findings.csv`: complete finding-level comparison output.",
        "- `manifest.json`: machine-readable artifact map, source context, "
        "transaction semantics summary, and row-count metadata.",
        "- `review_summary.json`: compact reviewer handoff summary with Modified "
        "Dietz vocabulary, entrypoints, source context, counts, and transaction "
        "semantics.",
        *_report_bundle_readme_table_lines(tables),
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
    include_reconstruction_diagnostics: bool = False,
    comparison_path: util.PathLike | None = None,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
) -> Path:
    """Write a report-bundle JSON manifest.

    Args:
        output_path: Destination manifest path.
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        include_reconstruction_diagnostics: Whether interim reconstruction
            diagnostics are included in the bundle.
        comparison_path: Optional comparison YAML path used to generate the
            bundle.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.

    Returns:
        Normalized destination path.
    """
    manifest = report_bundle_manifest(
        findings=findings,
        active_findings=active_findings,
        title=title,
        top_evidence_limit=top_evidence_limit,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        comparison_path=comparison_path,
        artifact_paths=artifact_paths,
        tables=tables,
    )
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
    include_reconstruction_diagnostics: bool = False,
    comparison_path: util.PathLike | None = None,
    artifact_paths: Mapping[str, Path],
    tables: Mapping[str, pl.DataFrame],
) -> dict[str, object]:
    """Return JSON-serializable metadata for a report bundle.

    Args:
        findings: Complete findings table.
        active_findings: Findings table after suppressed rows are excluded.
        title: Report title.
        top_evidence_limit: Maximum number of evidence rows shown per period.
        include_reconstruction_diagnostics: Whether interim reconstruction
            diagnostics are included in the bundle.
        comparison_path: Optional comparison YAML path used to generate the
            bundle.
        artifact_paths: Bundle artifact paths keyed by artifact name.
        tables: Named helper tables included as CSV artifacts.

    Returns:
        JSON-serializable manifest data.
    """
    suppressed_count = findings.height - active_findings.height
    return {
        "bundle_type": "performance_comparison_report",
        "manifest_version": _REPORT_BUNDLE_MANIFEST_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "title": title,
        "options": {
            "top_evidence_limit": top_evidence_limit,
            "include_reconstruction_diagnostics": include_reconstruction_diagnostics,
        },
        "source_context": _report_bundle_source_context(comparison_path),
        "counts": {
            "findings": findings.height,
            "active_findings": active_findings.height,
            "suppressed_findings": suppressed_count,
        },
        "transaction_semantics": _report_bundle_transaction_semantics(
            active_findings,
            comparison_path=comparison_path,
        ),
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
        "review_entrypoints": _report_bundle_review_entrypoints(
            artifact_paths,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        ),
    }


def _report_bundle_source_context(
    comparison_path: util.PathLike | None,
) -> dict[str, object]:
    """Return source metadata that helps reviewers reproduce a bundle."""
    if comparison_path is None:
        return {
            "comparison_path": None,
            "extract_contract": None,
        }
    specification = PerformanceComparisonSpecification(comparison_path)
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
) -> dict[str, object]:
    """Return compact transaction semantics metadata for a report bundle."""
    if comparison_path is None:
        return transaction_semantics_summary([active_findings])
    specification = PerformanceComparisonSpecification(comparison_path)
    loader = TransactionsLoader(specification)
    frames = [
        frame
        for snapshot_key in ("a", "b")
        if (frame := loader.load(snapshot_key)) is not None
    ]
    return transaction_semantics_summary(
        frames,
        rule_codes=transaction_rule_codes(specification.values),
    )


def _report_bundle_review_entrypoints(
    artifact_paths: Mapping[str, Path],
    *,
    include_reconstruction_diagnostics: bool,
) -> dict[str, object]:
    """Return the intended first-stop artifacts for reviewer navigation."""
    artifacts = {name: path.name for name, path in sorted(artifact_paths.items())}
    transaction_diagnostics = [
        artifacts[name]
        for name in (
            "transaction_activity",
            "transaction_cross_checks",
            "flow_cross_check_reconciliation",
            "transaction_matching_diagnostics",
        )
        if name in artifacts
    ]
    entrypoints: dict[str, object] = {
        "primary_review": artifacts.get(
            _pc_review_model.REVIEW_WORKBOOK_ARTIFACT,
            artifacts.get("html_report"),
        ),
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
    issues.extend(_report_bundle_manifest_shape_issues(manifest))
    artifacts = _manifest_mapping(manifest, "artifacts")
    tables = _manifest_mapping(manifest, "tables")
    issues.extend(_report_bundle_artifact_issues(bundle_path, artifacts))
    issues.extend(_report_bundle_review_entrypoint_issues(manifest, artifacts))
    issues.extend(_report_bundle_source_context_issues(manifest))
    issues.extend(_report_bundle_transaction_semantics_issues(manifest))
    issues.extend(_report_bundle_review_summary_issues(bundle_path, manifest, artifacts))
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
        "transaction_cross_checks": "transaction impact diagnostics",
        "flow_cross_check_reconciliation": "flow/cross-check reconciliation diagnostics",
        "reconstruction_summary": "return reconstruction diagnostic summary",
        "return_reconstruction_checks": "portfolio return reconstruction diagnostics",
        "security_return_reconstruction_checks": (
            "security return reconstruction diagnostics"
        ),
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


def _report_bundle_manifest_shape_issues(
    manifest: Mapping[str, object],
) -> list[str]:
    """Return top-level manifest schema issues."""
    issues: list[str] = []
    for key in _REPORT_BUNDLE_REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            issues.append(f"manifest top-level key {key!r} is missing")
    if manifest.get("bundle_type") != "performance_comparison_report":
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
