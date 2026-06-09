"""Write a Markdown-and-CSV report bundle for a performance comparison."""

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
    findings = compare_snapshots(
        args.comparison_path,
        include_suppressed=not args.active_only,
    )
    bundle_paths = write_performance_comparison_report_bundle(
        findings,
        args.output_directory,
        title=args.title,
        include_suppressed_appendix=not args.no_suppressed_appendix,
        top_evidence_limit=args.top_evidence_limit,
    )
    print(f"Report bundle written to: {bundle_paths['manifest'].parent}")
    print(f"Manifest written to: {bundle_paths['manifest']}")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a Markdown-and-CSV performance comparison report bundle.",
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
        help="Markdown H1 title for report.md.",
    )
    parser.add_argument(
        "--top-evidence-limit",
        type=int,
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
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
