"""Validate a performance comparison report bundle."""

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
from ppar.performance_comparison.report import (  # noqa: E402
    _report_bundle_validation_issues,
)


def main(argv: list[str] | None = None) -> int:
    """Validate a generated performance comparison report bundle.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means validation passed; ``1`` means the
        bundle has validation issues.
    """
    args = _argument_parser().parse_args(argv)
    issues = _report_bundle_validation_issues(args.bundle_directory)
    if not issues:
        print(f"Bundle validation passed: {args.bundle_directory}")
        return 0

    print(f"Bundle validation failed: {args.bundle_directory}", file=sys.stderr)
    for issue in issues:
        print(f"- {issue}", file=sys.stderr)
    return 1


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a performance comparison report bundle.",
    )
    parser.add_argument(
        "bundle_directory",
        type=Path,
        help="Path to a generated performance comparison report bundle directory.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
