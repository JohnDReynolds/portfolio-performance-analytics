"""Validate a performance comparison YAML configuration."""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python imports
import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Project imports
from ppar.errors import PpaError  # noqa: E402
from ppar.performance_comparison import (  # noqa: E402
    PerformanceComparison,
    PerformanceComparisonSpecification,
    TransactionsLoader,
)
from ppar.performance_comparison import columns as _pc_cols  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Validate a performance comparison YAML file.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means validation passed; ``1`` means the
        configuration has validation issues.
    """
    args = _argument_parser().parse_args(argv)
    try:
        summary = _validate_config(args.comparison_path)
    except PpaError as error:
        print(f"Config validation failed: {args.comparison_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Config validation passed: {args.comparison_path}")
    print(f"Snapshot A: {summary['snapshot_a']}")
    print(f"Snapshot B: {summary['snapshot_b']}")
    print(f"Configured datasets: {summary['dataset_count']}")
    print(f"Transaction files checked: {summary['transaction_files_checked']}")
    return 0


def _validate_config(comparison_path: Path) -> dict[str, object]:
    """Validate one comparison YAML file and return a compact summary."""
    specification = PerformanceComparisonSpecification(comparison_path)
    PerformanceComparison(specification)
    transaction_files_checked = _validate_transactions(specification)
    return {
        "snapshot_a": specification.snapshot_a.path,
        "snapshot_b": specification.snapshot_b.path,
        "dataset_count": len(specification.files),
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
