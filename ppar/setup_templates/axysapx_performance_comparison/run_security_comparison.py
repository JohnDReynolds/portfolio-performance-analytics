"""Run the local security performance-comparison setup from Python."""

from __future__ import annotations

# Python imports
from pathlib import Path
from typing import Any

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)


# Anchor paths to this setup folder so the script can be run from any current
# working directory.
SITE_DIRECTORY: Path = Path(__file__).resolve().parent
SPECIFICATIONS_PATH: Path = SITE_DIRECTORY / "ppar.yaml"
OUTPUT_DIRECTORY: Path = SITE_DIRECTORY / "output" / "security"
COMPARISON_LEVEL = "security"


def main() -> None:
    """Create the security performance-comparison workbook and HTML report.

    Returns:
        None. The workbook and companion HTML file are written to
        ``output/security/``.
    """
    # Security comparison requires security-performance source files in both
    # snapshots. If your site starts with portfolio-only review, finish that
    # first, then add security source files and run this script.
    findings: Any = compare_snapshots(
        SPECIFICATIONS_PATH,
        comparison_level=COMPARISON_LEVEL,
    )

    # The same YAML drives portfolio and security comparison; the
    # ``comparison_level`` selects the security-performance report path.
    bundle_paths: dict[str, Path] = write_performance_comparison_report_bundle(
        findings,
        OUTPUT_DIRECTORY,
        title="Security Performance Comparison",
        include_workbook=True,
        comparison_path=SPECIFICATIONS_PATH,
        comparison_level=COMPARISON_LEVEL,
    )

    # Print only the workbook path; the HTML version of the workbook sits in
    # the same output folder.
    print("Open these files to review performance-auditing output:")
    print(f"  {bundle_paths['review_workbook']}")


if __name__ == "__main__":
    main()
