"""Demonstrate performance comparison findings output."""

# Python imports
from importlib.resources import as_file, files
from pathlib import Path

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)


def main() -> None:
    """Run the performance comparison demonstration."""
    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        comparison_path = (
            axys_data_root / "ppar_performance_comparison_full_spec.yaml"
        )
        bundle_path = Path.cwd() / "_demo_output" / "performance_comparison"

        _run_comparison_demo(
            comparison_path=comparison_path,
            bundle_path=bundle_path,
        )


def _run_comparison_demo(
    *,
    comparison_path: Path,
    bundle_path: Path,
) -> None:
    """Run the performance comparison workflow for resolved demo paths."""
    findings = compare_snapshots(
        comparison_path,
        require_causal_attribution=True,
    )
    written_bundle_paths = write_performance_comparison_report_bundle(
        findings,
        bundle_path,
        title="Performance Comparison Demo",
        include_workbook=True,
        require_causal_attribution=True,
        comparison_path=comparison_path,
    )
    print("Performance comparison demo")
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


if __name__ == "__main__":
    main()
