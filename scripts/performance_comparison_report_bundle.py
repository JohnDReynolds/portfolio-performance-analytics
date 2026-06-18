"""Write a review artifact bundle for a performance comparison."""

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
    compare_snapshots,
    write_performance_comparison_report_bundle,
)


def main(argv: list[str] | None = None) -> int:
    """Write a report bundle from a comparison YAML file.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that the report bundle was written.
    """
    args = _argument_parser().parse_args(argv)
    try:
        findings = compare_snapshots(
            args.comparison_path,
            include_suppressed=not args.active_only,
            require_causal_attribution=args.require_causal_attribution,
        )
        bundle_paths = write_performance_comparison_report_bundle(
            findings,
            args.output_directory,
            title=args.title,
            include_suppressed_appendix=not args.no_suppressed_appendix,
            top_evidence_limit=args.top_evidence_limit,
            include_workbook=args.include_workbook,
            require_causal_attribution=args.require_causal_attribution,
        )
    except PpaError as error:
        print(f"Report bundle failed: {error}", file=sys.stderr)
        return 1
    print(f"Report bundle written to: {bundle_paths['manifest'].parent}")
    print(f"README written to: {bundle_paths['readme']}")
    print(f"HTML report written to: {bundle_paths['html_report']}")
    if "review_workbook" in bundle_paths:
        print(f"Review workbook written to: {bundle_paths['review_workbook']}")
    print(f"Needs review summary written to: {bundle_paths['needs_review_summary']}")
    print(f"Manifest written to: {bundle_paths['manifest']}")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a performance comparison review artifact bundle.",
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Path to a ppar_performance_comparison.yaml file.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Destination report bundle directory.",
    )
    parser.add_argument(
        "--title",
        default="Performance Comparison Report",
        help="Report title for report.md and report.html.",
    )
    parser.add_argument(
        "--top-evidence-limit",
        type=_non_negative_int,
        default=10,
        help="Maximum top-evidence rows to include per portfolio period.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Exclude suppressed findings from the report bundle input.",
    )
    parser.add_argument(
        "--no-suppressed-appendix",
        action="store_true",
        help="Omit the suppressed findings appendix from report.md.",
    )
    parser.add_argument(
        "--include-workbook",
        action="store_true",
        help="Write optional review_workbook.xlsx. Requires ppar[excel].",
    )
    parser.add_argument(
        "--require-causal-attribution",
        action="store_true",
        help="Fail unless changed periods have all setup needed for causal attribution.",
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
