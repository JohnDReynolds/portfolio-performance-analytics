"""Validate generated performance comparison report bundles."""

from __future__ import annotations

# Python imports
import json
from collections.abc import Mapping
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison import workbook as _pc_workbook

__all__ = [
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "report_bundle_validation_issues",
]

REPORT_BUNDLE_REQUIRED_ARTIFACTS = (
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
    "context_evidence_summary",
    "context_evidence",
    "transaction_cross_checks",
    "flow_cross_check_reconciliation",
    "residual_status",
    "transaction_activity",
    "transaction_matching_diagnostics",
    "top_evidence",
)


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
