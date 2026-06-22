"""Shared helpers for performance comparison demo commands."""

# Python imports
from pathlib import Path

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification


def run_performance_comparison_demo(
    *,
    comparison_path: Path,
    bundle_path: Path,
    title: str,
) -> None:
    """Run one packaged performance comparison demo.

    Args:
        comparison_path: Packaged comparison YAML path.
        bundle_path: Output directory for the generated review bundle.
        title: Report title for generated artifacts and console output.
    """
    specification = PerformanceComparisonSpecification(comparison_path)
    findings = compare_snapshots(
        comparison_path,
        require_causal_attribution=specification.comparison_level == "portfolio",
    )
    written_bundle_paths = write_performance_comparison_report_bundle(
        findings,
        bundle_path,
        title=title,
        include_workbook=True,
        require_causal_attribution=specification.comparison_level == "portfolio",
        comparison_path=comparison_path,
        comparison_level=specification.comparison_level,
    )
    print(title)
    print()
    print(f"Report bundle written to: {written_bundle_paths['manifest'].parent}")
    _print_bundle_handoff(written_bundle_paths)


def _print_bundle_handoff(bundle_paths: dict[str, Path]) -> None:
    """Print generated bundle artifacts and the recommended review path."""
    print(f"Bundle artifacts: {len(bundle_paths)} files")
    print("Recommended review path")
    print(f"1. Open {bundle_paths['review_workbook']}")
    print(f"2. Open {bundle_paths['html_report']} if you prefer browser review")
    print(f"3. Use {bundle_paths['manifest']} to audit generated artifacts")
    print()
