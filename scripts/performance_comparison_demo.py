"""Demonstrate performance comparison findings output."""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python imports
from pathlib import Path
import sys

# Third-party imports
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AXYS_DATA_ROOT = _REPO_ROOT / "tests" / "data" / "axys"
sys.path.insert(0, str(_REPO_ROOT))

# Project imports
from ppar.performance_comparison import (  # noqa: E402
    compact_findings_table,
    compare_snapshots,
    summarize_findings,
)


def main() -> None:
    """Run the performance comparison demonstration."""
    comparison_path = _AXYS_DATA_ROOT / "ppar_performance_comparison_restatement.yaml"
    suppressed_comparison_path = (
        _AXYS_DATA_ROOT / "ppar_performance_comparison_suppressed.yaml"
    )
    findings = compare_snapshots(comparison_path)
    active_findings = compare_snapshots(comparison_path, include_suppressed=False)
    compact_active_findings = compact_findings_table(findings)
    summaries = summarize_findings(findings)
    active_summaries = summarize_findings(active_findings)
    suppressed_findings = compare_snapshots(suppressed_comparison_path)
    suppressed_active_findings = compare_snapshots(
        suppressed_comparison_path,
        include_suppressed=False,
    )
    suppressed_summaries = summarize_findings(suppressed_findings)
    suppressed_active_summaries = summarize_findings(suppressed_active_findings)
    with pl.Config(tbl_cols=-1, tbl_rows=-1):
        print("Restatement comparison")
        print()
        print("Finding count by code")
        print(summaries["by_code"])
        print()
        print("Finding count by dataset")
        print(summaries["by_dataset"])
        print()
        print("Finding count by suppression state")
        print(summaries["by_suppressed"])
        print()
        print("Finding count by code and suppression state")
        print(summaries["by_code_suppressed"])
        print()
        print("Active finding count by code")
        print(active_summaries["by_code"])
        print()
        print("Compact active findings")
        print(compact_active_findings)
        print()
        print("Full audit findings")
        print(findings)
        print()
        print("Suppressed restatement comparison")
        print()
        print(f"All findings: {suppressed_findings.height}")
        print(f"Active findings: {suppressed_active_findings.height}")
        print()
        print("Finding count by suppression state")
        print(suppressed_summaries["by_suppressed"])
        print()
        print("Finding count by code and suppression state")
        print(suppressed_summaries["by_code_suppressed"])
        print()
        print("Active finding count by code")
        print(suppressed_active_summaries["by_code"])


if __name__ == "__main__":
    main()
