"""Demonstrate performance comparison findings output."""

# Python imports
from importlib.resources import as_file, files
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    portfolio_period_impact_coverage_summary,
    summarize_findings,
    write_performance_comparison_report_bundle,
)

_LEGACY_STANDALONE_REPORT_NAMES = (
    "performance_comparison_restatement.md",
    "performance_comparison_restatement.html",
)


def _print_table(title: str, table: pl.DataFrame, *, wide: bool = False) -> None:
    """Print a titled table using demo-friendly Polars display settings."""
    print(title)
    if wide:
        with pl.Config(
            tbl_cols=-1,
            tbl_rows=-1,
            tbl_width_chars=240,
            fmt_str_lengths=80,
        ):
            print(table)
    else:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=160):
            print(table)
    print()


def main() -> None:
    """Run the performance comparison demonstration."""
    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        comparison_path = (
            axys_data_root / "ppar_performance_comparison_multi_restatement.yaml"
        )
        suppressed_comparison_path = (
            axys_data_root / "ppar_performance_comparison_suppressed.yaml"
        )

        output_root = Path.cwd() / "_demo_output"
        bundle_path = output_root / "performance_comparison_bundle"
        _remove_legacy_standalone_reports(output_root)

        _run_comparison_demo(
            comparison_path=comparison_path,
            suppressed_comparison_path=suppressed_comparison_path,
            bundle_path=bundle_path,
        )


def _remove_legacy_standalone_reports(output_root: Path) -> None:
    """Remove old top-level demo reports now superseded by bundle reports."""
    for report_name in _LEGACY_STANDALONE_REPORT_NAMES:
        (output_root / report_name).unlink(missing_ok=True)


def _run_comparison_demo(
    *,
    comparison_path: Path,
    suppressed_comparison_path: Path,
    bundle_path: Path,
) -> None:
    """Run the performance comparison workflow for resolved demo paths."""
    findings = compare_snapshots(comparison_path)
    impact_coverage = portfolio_period_impact_coverage_summary(findings)
    suppressed_findings = compare_snapshots(suppressed_comparison_path)
    suppressed_active_findings = compare_snapshots(
        suppressed_comparison_path,
        include_suppressed=False,
    )
    suppressed_summaries = summarize_findings(suppressed_findings)
    written_bundle_paths = write_performance_comparison_report_bundle(
        findings,
        bundle_path,
    )
    print("Multi-portfolio restatement comparison")
    print()
    print(f"Report bundle written to: {written_bundle_paths['manifest'].parent}")
    _print_bundle_handoff(written_bundle_paths)
    _print_table("Impact coverage summary", _impact_coverage_demo_summary(impact_coverage))
    _print_suppressed_summary(
        suppressed_findings=suppressed_findings,
        suppressed_active_findings=suppressed_active_findings,
        suppressed_summaries=suppressed_summaries,
    )


def _impact_coverage_demo_summary(impact_coverage: pl.DataFrame) -> pl.DataFrame:
    """Return a compact impact-coverage table for console demo output."""
    columns = [
        "portfolio_id",
        "from_date",
        "thru_date",
        "portfolio_return_delta",
        "estimated_cause_area_count",
        "evidence_only_cause_area_count",
        "missing_impact_inputs",
    ]
    return impact_coverage.select(columns)


def _print_suppressed_summary(
    *,
    suppressed_findings: pl.DataFrame,
    suppressed_active_findings: pl.DataFrame,
    suppressed_summaries: dict[str, pl.DataFrame],
) -> None:
    """Print suppression-focused demo output."""
    print("Suppressed restatement comparison")
    print()
    print(f"All findings: {suppressed_findings.height}")
    print(f"Active findings: {suppressed_active_findings.height}")
    print()
    _print_table("Finding count by suppression state", suppressed_summaries["by_suppressed"])


def _print_bundle_handoff(bundle_paths: dict[str, Path]) -> None:
    """Print generated bundle artifacts and the recommended review path."""
    print(f"Bundle artifacts: {len(bundle_paths)} files")
    needs_review_summary = pl.read_csv(bundle_paths["needs_review_summary"])
    _print_table("Needs review summary", needs_review_summary, wide=True)
    print("Recommended review path")
    print(f"1. Open {bundle_paths['html_report']}")
    print(f"2. Inspect {bundle_paths['needs_review_summary']}")
    print(f"3. Use {bundle_paths['manifest']} to audit generated artifacts")
    print()


if __name__ == "__main__":
    main()
