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
    comparison_level: str,
) -> None:
    """Run one packaged performance comparison demo.

    Args:
        comparison_path: Packaged comparison YAML path.
        bundle_path: Output directory for the generated review bundle.
        title: Report title for generated artifacts and console output.
        comparison_level: Primary report level, such as ``"portfolio"`` or
            ``"security"``.
    """
    PerformanceComparisonSpecification(
        comparison_path,
        comparison_level=comparison_level,
    )
    findings = compare_snapshots(
        comparison_path,
        comparison_level=comparison_level,
    )
    written_bundle_paths = write_performance_comparison_report_bundle(
        findings,
        bundle_path,
        title=title,
        include_workbook=True,
        comparison_path=comparison_path,
        comparison_level=comparison_level,
    )
    _print_bundle_handoff(written_bundle_paths)


def _print_bundle_handoff(bundle_paths: dict[str, Path]) -> None:
    """Print generated workbook artifact paths."""
    print("Open these files to review performance_comparison output:")
    print(f"  {bundle_paths['review_workbook']}")
