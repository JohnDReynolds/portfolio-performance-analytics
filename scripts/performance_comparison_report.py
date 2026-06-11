"""Write a Markdown report for a performance comparison specification."""

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
from ppar.performance_comparison import (  # noqa: E402
    compare_snapshots,
    write_performance_comparison_markdown_report,
)


def main(argv: list[str] | None = None) -> int:
    """Write a Markdown report from a comparison YAML file.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that the report was written.
    """
    args = _argument_parser().parse_args(argv)
    findings = compare_snapshots(
        args.comparison_path,
        include_suppressed=not args.active_only,
    )
    report_path = write_performance_comparison_markdown_report(
        findings,
        args.output_path,
        title=args.title,
        include_suppressed_appendix=not args.no_suppressed_appendix,
        top_evidence_limit=args.top_evidence_limit,
    )
    print(f"Markdown report written to: {report_path}")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a Markdown performance comparison report.",
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Path to a ppar_performance_comparison.yaml file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination Markdown report path.",
    )
    parser.add_argument(
        "--title",
        default="Performance Comparison Report",
        help="Markdown H1 title for the report.",
    )
    parser.add_argument(
        "--top-evidence-limit",
        type=_non_negative_int,
        default=10,
        help="Maximum top-evidence rows to show per portfolio period.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Exclude suppressed findings from the comparison report input.",
    )
    parser.add_argument(
        "--no-suppressed-appendix",
        action="store_true",
        help="Omit the suppressed findings appendix section.",
    )
    return parser


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line argument."""
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed_value


if __name__ == "__main__":
    raise SystemExit(main())
