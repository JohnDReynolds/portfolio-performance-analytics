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
from ppar.performance_comparison.transactions import TransactionsLoader

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
    print(f"Transaction files checked: {summary['transaction_files_checked']}")
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
    transaction_files_checked = _validate_transactions(specification)
    dataset_names = ", ".join(sorted(specification.files))
    return {
        "snapshot_a": specification.snapshot_a.path,
        "snapshot_b": specification.snapshot_b.path,
        "dataset_names": dataset_names,
        "transaction_files_checked": transaction_files_checked,
    }


def _validate_transactions(
    specification: PerformanceComparisonSpecification,
) -> int:
    """Validate configured transaction files and transaction rule semantics."""
    if _pc_cols.TRANSACTIONS not in specification.files:
        return 0
    loader = TransactionsLoader(specification)
    checked = 0
    for snapshot_key in ("a", "b"):
        if loader.load(snapshot_key) is not None:
            checked += 1
    return checked


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
