"""Run the local portfolio performance-comparison setup from Python."""

from __future__ import annotations

# Python imports
from pathlib import Path

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)


SITE_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = SITE_DIRECTORY / "ppar.yaml"
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output" / "portfolio"
COMPARISON_LEVEL = "portfolio"


def main() -> None:
    """Create the portfolio performance-comparison workbook and HTML report."""
    # ``compare_snapshots`` reads the two snapshot folders named in ``ppar.yaml``
    # and returns normalized finding rows for changed performance and source
    # data. Put mapping, transaction, and report decisions in the YAML.
    findings = compare_snapshots(CONFIG_PATH, comparison_level=COMPARISON_LEVEL)

    bundle_paths = write_performance_comparison_report_bundle(
        findings,
        OUTPUT_DIRECTORY,
        title="Portfolio Performance Comparison",
        include_workbook=True,
        comparison_path=CONFIG_PATH,
        comparison_level=COMPARISON_LEVEL,
    )

    print("Open these files to review performance_comparison output:")
    print(f"  {bundle_paths['review_workbook']}")


if __name__ == "__main__":
    main()
