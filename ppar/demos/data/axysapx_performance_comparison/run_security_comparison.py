"""Run the local security performance-comparison setup from Python."""

from __future__ import annotations

# Python imports
from pathlib import Path

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)


# Anchor paths to this setup folder so the script can be run from any current
# working directory.
SITE_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = SITE_DIRECTORY / "ppar.yaml"
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output" / "security"
COMPARISON_LEVEL = "security"


def main() -> None:
    """Create the security performance-comparison workbook and HTML report."""
    # Security comparison requires security-performance source files in both
    # snapshots. If your site starts with portfolio-only review, finish that
    # first, then add security source files and run this script.
    findings = compare_snapshots(CONFIG_PATH, comparison_level=COMPARISON_LEVEL)

    # The same YAML drives portfolio and security comparison; the
    # ``comparison_level`` selects the security-performance report path.
    bundle_paths = write_performance_comparison_report_bundle(
        findings,
        OUTPUT_DIRECTORY,
        title="Security Performance Comparison",
        include_workbook=True,
        comparison_path=CONFIG_PATH,
        comparison_level=COMPARISON_LEVEL,
    )

    # Print only the workbook path; the HTML file sits in the same output folder
    # when a browser-friendly view is useful.
    print("Open these files to review performance_comparison output:")
    print(f"  {bundle_paths['review_workbook']}")


if __name__ == "__main__":
    main()
