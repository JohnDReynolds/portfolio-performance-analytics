"""Run the local portfolio performance-comparison setup from Python."""

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
OUTPUT_DIRECTORY: Path = SITE_DIRECTORY / "output" / "portfolio"
COMPARISON_LEVEL = "portfolio"


def main() -> None:
    """Create the portfolio performance-comparison workbook and HTML report.

    Returns:
        None. The workbook and companion HTML file are written to
        ``output/portfolio/``.
    """
    # ``compare_snapshots`` reads the two snapshot folders named in ``ppar.yaml``
    # and returns normalized finding rows for changed performance and source
    # data. Put mapping, transaction, and report decisions in the YAML.
    findings: Any = compare_snapshots(
        SPECIFICATIONS_PATH,
        comparison_level=COMPARISON_LEVEL,
    )

    # The report bundle turns those findings into the review workbook. The HTML
    # version of the workbook is written alongside it.
    bundle_paths: dict[str, Path] = write_performance_comparison_report_bundle(
        findings,
        OUTPUT_DIRECTORY,
        title="Portfolio Performance Comparison",
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
