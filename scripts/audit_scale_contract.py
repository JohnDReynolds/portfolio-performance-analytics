"""Lock semantic Audit results and output observations during scale checks."""

from __future__ import annotations

from collections.abc import Sequence
import io
import json
from pathlib import Path
import zipfile

import polars as pl
from polars.testing import assert_frame_equal


_EQUIVALENCE_TABLES = {
    "findings.csv": (
        "finding_sequence",
        "finding_fingerprint",
        "source_record_locator",
    ),
    "performance_differences.csv": (),
    "performance_difference_causes.csv": (
        "source_record_locator",
        "source_finding_fingerprints",
        "economic_effect_id",
    ),
    "data_issues.csv": (),
}
_SCALE_PORTFOLIO_SUFFIX_PATTERN = r"_SCALE_\d{3}"
_SYNTHETIC_AGGREGATION_PHRASE = " and other portfolios"


def read_supporting_csv(
    report_path: Path,
    file_name: str,
    *,
    infer_types: bool = True,
) -> pl.DataFrame:
    """Read one supporting CSV from an expanded or compact audit bundle.

    Args:
        report_path: Root directory of an Audit report bundle.
        file_name: Supporting CSV filename without its directory prefix.
        infer_types: Whether Polars should infer scalar and date types. Disable
            this for exact persisted-value comparisons across differently sized
            files whose all-null columns may infer differently.

    Returns:
        Parsed supporting table with date inference enabled.

    Raises:
        RuntimeError: If the expanded file or compact archive member is unreadable.
    """
    expanded_path = report_path / "supporting_files" / file_name
    if expanded_path.is_file():
        if infer_types:
            return pl.read_csv(expanded_path, try_parse_dates=True)
        return pl.read_csv(expanded_path, infer_schema=False)

    archive_path = report_path / "audit_support.zip"
    member_name = f"supporting_files/{file_name}"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            contents = archive.read(member_name)
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as error:
        raise RuntimeError(
            f"Audit bundle is missing readable supporting file {member_name}."
        ) from error
    if infer_types:
        return pl.read_csv(io.BytesIO(contents), try_parse_dates=True)
    return pl.read_csv(io.BytesIO(contents), infer_schema=False)


def _read_supporting_json(report_path: Path, file_name: str) -> dict[str, object]:
    """Read one supporting JSON object from an expanded or compact bundle."""
    expanded_path = report_path / "supporting_files" / file_name
    try:
        if expanded_path.is_file():
            contents = expanded_path.read_text(encoding="utf-8")
        else:
            archive_path = report_path / "audit_support.zip"
            member_name = f"supporting_files/{file_name}"
            with zipfile.ZipFile(archive_path) as archive:
                contents = archive.read(member_name).decode("utf-8")
        parsed = json.loads(contents)
    except (
        FileNotFoundError,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        raise RuntimeError(
            f"Audit bundle is missing readable supporting file {file_name}."
        ) from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Audit supporting file {file_name} is not a JSON object.")
    return {str(key): value for key, value in parsed.items()}


def _normalized_contract_counts(
    table: pl.DataFrame,
    *,
    excluded_columns: Sequence[str],
) -> pl.DataFrame:
    """Return normalized business-row occurrence counts for scale comparison.

    Synthetic scale copies intentionally receive distinct lineage identities.
    Those opaque identities are excluded, while every reviewer-facing and
    financial value remains in the equivalence contract.
    """
    contract_columns = [
        column for column in table.columns if column not in excluded_columns
    ]
    if not contract_columns:
        raise RuntimeError("Audit scale contract table has no comparable columns.")
    normalized = table.select(contract_columns)
    string_columns = [
        column
        for column, data_type in normalized.schema.items()
        if data_type == pl.String
    ]
    if string_columns:
        normalized = normalized.with_columns(
            pl.col(column)
            .str.replace_all(_SCALE_PORTFOLIO_SUFFIX_PATTERN, "")
            # Missing-dividend X-ref narratives disclose that synthetic copies
            # share the issue. The business result remains the same 1x issue.
            .str.replace_all(_SYNTHETIC_AGGREGATION_PHRASE, "")
            .alias(column)
            for column in string_columns
        )
    return (
        normalized.group_by(contract_columns)
        .len(name="_occurrences")
        .sort(contract_columns, nulls_last=True)
    )


def _assert_scaled_table_equivalent(
    baseline_report_path: Path,
    scaled_report_path: Path,
    file_name: str,
    scale: int,
    *,
    excluded_columns: Sequence[str],
) -> None:
    """Require a scaled Audit table to be exact business-result replicas."""
    baseline = _normalized_contract_counts(
        read_supporting_csv(
            baseline_report_path,
            file_name,
            infer_types=False,
        ),
        excluded_columns=excluded_columns,
    ).with_columns((pl.col("_occurrences") * scale).alias("_occurrences"))
    scaled = _normalized_contract_counts(
        read_supporting_csv(
            scaled_report_path,
            file_name,
            infer_types=False,
        ),
        excluded_columns=excluded_columns,
    )
    try:
        assert_frame_equal(baseline, scaled)
    except AssertionError as error:
        raise RuntimeError(
            f"Scaled Audit {file_name} differs from {scale} exact business-result "
            "copies of the 1x baseline."
        ) from error


def assert_scaled_audit_equivalent(
    baseline_report_path: Path,
    scaled_report_path: Path,
    scale: int,
) -> None:
    """Require scaled findings and primary review tables to preserve semantics."""
    for file_name, excluded_columns in _EQUIVALENCE_TABLES.items():
        _assert_scaled_table_equivalent(
            baseline_report_path,
            scaled_report_path,
            file_name,
            scale,
            excluded_columns=excluded_columns,
        )


def _output_metrics(report_path: Path) -> tuple[int, dict[str, int]]:
    """Return visible bundle bytes and canonical review-sheet row counts."""
    total_bytes = sum(
        path.stat().st_size for path in report_path.iterdir() if path.is_file()
    )
    manifest = _read_supporting_json(report_path, "manifest.json")
    output_integrity = manifest.get("output_integrity")
    if not isinstance(output_integrity, dict):
        raise RuntimeError("Audit manifest output_integrity is missing or malformed.")
    review_sheets = output_integrity.get("review_sheets")
    if not isinstance(review_sheets, dict):
        raise RuntimeError("Audit manifest review_sheets is missing or malformed.")
    row_counts: dict[str, int] = {}
    for artifact_name, raw_metadata in review_sheets.items():
        if not isinstance(raw_metadata, dict):
            raise RuntimeError(
                f"Audit review-sheet metadata {artifact_name!r} is malformed."
            )
        rows = raw_metadata.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise RuntimeError(
                f"Audit review-sheet row count {artifact_name!r} is malformed."
            )
        row_counts[str(artifact_name)] = rows
    return total_bytes, row_counts


def print_output_metrics(report_name: str, report_path: Path) -> None:
    """Print compact scaled Audit output-size and review-row observations."""
    total_bytes, row_counts = _output_metrics(report_path)
    row_summary = ", ".join(
        f"{name}={rows:,}" for name, rows in sorted(row_counts.items())
    )
    print(
        f"  {report_name} output: visible_bytes={total_bytes:,}; {row_summary}"
    )
