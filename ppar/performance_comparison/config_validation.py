"""Validate performance comparison YAML configurations."""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import columns as _pc_cols
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.transactions import (
    TRANSACTION_SEMANTICS_SOURCE_MIXED,
    TRANSACTION_SEMANTICS_SOURCE_SOURCE,
    TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
    TransactionsLoader,
)

__all__ = [
    "main",
    "validate_config",
]


def main(argv: list[str] | None = None) -> int:
    """Validate a performance comparison YAML file from the command line.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means validation passed; ``1`` means the
        configuration has validation issues.
    """
    args = _argument_parser().parse_args(argv)
    try:
        summary = validate_config(args.comparison_path)
    except PpaError as error:
        print(f"Config validation failed: {args.comparison_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Config validation passed: {args.comparison_path}")
    print(f"Snapshot A: {summary['snapshot_a']}")
    print(f"Snapshot B: {summary['snapshot_b']}")
    print(f"Configured datasets: {summary['dataset_names']}")
    print(f"Missing optional files: {summary['missing_optional_files']}")
    print(f"Contribution impact methods: {summary['contribution_impact_methods']}")
    print(f"Position impact methods: {summary['position_impact_methods']}")
    print(f"Price impact methods: {summary['price_impact_methods']}")
    print(f"Cash impact methods: {summary['cash_impact_methods']}")
    print(f"Evidence-only impact methods: {summary['evidence_only_impact_methods']}")
    print(f"Transaction rules configured: {summary['transaction_rule_count']}")
    print(f"Transaction impact methods: {summary['transaction_impact_methods']}")
    print(f"Transaction files checked: {summary['transaction_files_checked']}")
    print(f"Transaction semantics sources: {summary['transaction_semantics_sources']}")
    return 0


def validate_config(comparison_path: Path) -> dict[str, object]:
    """Validate one comparison YAML file and return a compact summary.

    Args:
        comparison_path: Path to a performance comparison YAML file.

    Returns:
        Summary fields for the resolved snapshots, configured datasets, and
        transaction files checked.

    Raises:
        PpaError: If the comparison specification, configured files,
            transaction rules, or transaction impact methods are invalid.
    """
    specification = PerformanceComparisonSpecification(comparison_path)
    PerformanceComparison(specification)
    transaction_preview = _validate_transactions(specification)
    dataset_names = ", ".join(sorted(specification.files))
    return {
        "snapshot_a": specification.snapshot_a.path,
        "snapshot_b": specification.snapshot_b.path,
        "dataset_names": dataset_names,
        "missing_optional_files": _missing_optional_files(specification),
        "contribution_impact_methods": _contribution_impact_methods(specification),
        "position_impact_methods": _position_impact_methods(specification),
        "price_impact_methods": _price_impact_methods(specification),
        "cash_impact_methods": _cash_impact_methods(specification),
        "evidence_only_impact_methods": _evidence_only_impact_methods(specification),
        "transaction_rule_count": _transaction_rule_count(specification),
        "transaction_impact_methods": _transaction_impact_methods(specification),
        "transaction_files_checked": transaction_preview["files_checked"],
        "transaction_semantics_sources": transaction_preview["semantics_sources"],
    }


def _validate_transactions(
    specification: PerformanceComparisonSpecification,
) -> dict[str, object]:
    """Validate configured transaction files and return preview fields."""
    if _pc_cols.TRANSACTIONS not in specification.files:
        return {"files_checked": 0, "semantics_sources": "none"}
    loader = TransactionsLoader(specification)
    checked = 0
    semantics_source_counts: dict[str, int] = {}
    for snapshot_key in ("a", "b"):
        frame = loader.load(snapshot_key)
        if frame is None:
            continue
        checked += 1
        for value in frame.get_column(_pc_cols.TRANSACTION_SEMANTICS_SOURCE):
            if not isinstance(value, str) or not value:
                continue
            semantics_source_counts[value] = semantics_source_counts.get(value, 0) + 1
    return {
        "files_checked": checked,
        "semantics_sources": _format_semantics_source_counts(semantics_source_counts),
    }


def _missing_optional_files(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return a readable list of configured optional files that are absent."""
    missing_files: list[str] = []
    for comparison_file in specification.files.values():
        if comparison_file.required:
            continue
        for snapshot_key, file_path in (
            ("a", comparison_file.snapshot_a_path),
            ("b", comparison_file.snapshot_b_path),
        ):
            if not file_path.exists():
                missing_files.append(f"{comparison_file.name}:{snapshot_key}")
    return ", ".join(sorted(missing_files)) if missing_files else "none"


def _transaction_rule_count(
    specification: PerformanceComparisonSpecification,
) -> int:
    """Return the number of configured transaction code rules."""
    rules_value = specification.values.get("transaction_rules", {})
    return len(rules_value) if isinstance(rules_value, dict) else 0


def _transaction_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured transaction impact method keys."""
    methods_value = specification.values.get("transaction_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _contribution_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured contribution impact method keys."""
    methods_value = specification.values.get("contribution_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _position_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured position impact method keys."""
    methods_value = specification.values.get("position_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _price_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured price impact method keys."""
    methods_value = specification.values.get("price_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _cash_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured cash impact method keys."""
    methods_value = specification.values.get("cash_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _evidence_only_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured evidence-only impact method keys."""
    methods_value = specification.values.get("evidence_only_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _format_semantics_source_counts(counts: dict[str, int]) -> str:
    """Return stable transaction semantics-source counts."""
    ordered_sources = (
        TRANSACTION_SEMANTICS_SOURCE_SOURCE,
        TRANSACTION_SEMANTICS_SOURCE_MIXED,
        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
        TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    )
    parts = [
        f"{source}: {counts[source]}"
        for source in ordered_sources
        if counts.get(source, 0) > 0
    ]
    parts.extend(
        f"{source}: {counts[source]}"
        for source in sorted(counts)
        if source not in ordered_sources and counts.get(source, 0) > 0
    )
    return ", ".join(parts) if parts else "none"


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a performance comparison YAML configuration.",
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Path to a ppar_performance_comparison.yaml file.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
