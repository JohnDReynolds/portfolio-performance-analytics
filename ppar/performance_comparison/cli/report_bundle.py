"""Write a review artifact bundle for a performance comparison."""

# Python imports
import argparse
from pathlib import Path
import sys

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison import review_model as _pc_review_model
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
    PerformanceComparisonSpecification,
)

_COMPARISON_LEVEL_CHOICES = (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
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
        specification = PerformanceComparisonSpecification(
            args.comparison_path,
            comparison_level=args.comparison_level,
        )
        comparison_level = specification.comparison_level
        findings = compare_snapshots(
            args.comparison_path,
            comparison_level=comparison_level,
            include_suppressed=not args.active_only,
            require_causal_attribution=args.require_causal_attribution,
        )
        bundle_paths = write_performance_comparison_report_bundle(
            findings,
            args.output_directory,
            title=args.title,
            top_evidence_limit=args.top_evidence_limit,
            include_workbook=args.include_workbook,
            require_complete_yaml_setup=not args.allow_incomplete_yaml,
            require_causal_attribution=args.require_causal_attribution,
            comparison_path=args.comparison_path,
            comparison_level=comparison_level,
            include_reconstruction_diagnostics=(
                args.include_reconstruction_diagnostics
            ),
        )
    except PpaError as error:
        print(f"Report bundle failed: {error}", file=sys.stderr)
        return 1
    print(f"Report bundle written to: {bundle_paths['readme'].parent}")
    print(f"README written to: {bundle_paths['readme']}")
    if _pc_review_model.REVIEW_WORKBOOK_ARTIFACT in bundle_paths:
        print(
            "Review workbook written to: "
            f"{bundle_paths[_pc_review_model.REVIEW_WORKBOOK_ARTIFACT]}"
        )
    print(f"HTML report written to: {bundle_paths['html_report']}")
    print(f"Needs review summary written to: {bundle_paths['needs_review_summary']}")
    print(f"Manifest written to: {bundle_paths['manifest']}")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a Performance Auditing review artifact bundle.",
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Path to a Performance Auditing YAML file.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Destination report bundle directory.",
    )
    parser.add_argument(
        "--title",
        default="Performance Auditing Report",
        help="Report title for the level-specific HTML and optional XLSX audit.",
    )
    parser.add_argument(
        "--top-evidence-limit",
        type=_non_negative_int,
        default=10,
        help="Maximum top-evidence rows to include per portfolio period.",
    )
    parser.add_argument(
        "--comparison-level",
        choices=_COMPARISON_LEVEL_CHOICES,
        help=(
            "Primary performance result to compare. Overrides comparison.level "
            "in the YAML when supplied."
        ),
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Exclude suppressed findings from the report bundle input.",
    )
    parser.add_argument(
        "--include-workbook",
        action="store_true",
        help="Write the level-specific XLSX audit in addition to HTML and CSV files.",
    )
    parser.add_argument(
        "--include-reconstruction-diagnostics",
        action="store_true",
        help=(
            "Add optional Reconstruction Summary, Return Reconstruction Checks, "
            "and Security Return Checks sheets plus matching CSV artifacts."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-yaml",
        action="store_true",
        help=(
            "Write a diagnostic bundle even when changed source-data fields are "
            "not explicitly classified by additive, evidence-only, or suppression "
            "YAML."
        ),
    )
    parser.add_argument(
        "--require-causal-attribution",
        "--require-supported-attribution-setup",
        dest="require_causal_attribution",
        action="store_true",
        help=(
            "Fail unless changed periods have all YAML setup needed for supported "
            "causal attribution methods. This does not require every performance "
            "change to be fully explained."
        ),
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
