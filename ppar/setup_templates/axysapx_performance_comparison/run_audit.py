"""Run the local Performance Auditing setup from Python."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReportSpec:
    """Describe one Performance Auditing report to write.

    Attributes:
        comparison_level: Report level passed to the PPAR comparison engine.
        title: Workbook and HTML report title.
        output_directory: Folder that receives the report package.
    """

    comparison_level: str
    title: str
    output_directory: Path


REPORTS: tuple[ReportSpec, ...] = (
    ReportSpec(
        comparison_level="portfolio",
        title="Portfolio Performance Auditing Report",
        output_directory=SITE_DIRECTORY / "output" / "portfolio",
    ),
    ReportSpec(
        comparison_level="security",
        title="Security Performance Auditing Report",
        output_directory=SITE_DIRECTORY / "output" / "security",
    ),
)


def main() -> None:
    """Create the portfolio and security Performance Auditing reports.

    Returns:
        None. Report packages are written under ``output/portfolio/`` and
        ``output/security/``.
    """
    workbook_paths = [_write_report(report) for report in REPORTS]

    print("Open these files to review Performance Auditing output:")
    for workbook_path in workbook_paths:
        print(f"  {workbook_path}")


def _write_report(report: ReportSpec) -> Path:
    """Write one Performance Auditing report and return its workbook path."""
    # ``compare_snapshots`` reads the two snapshot folders named in ``ppar.yaml``
    # and returns normalized finding rows for changed performance and source
    # data. Put mapping, transaction, and report decisions in the YAML.
    findings: Any = compare_snapshots(
        SPECIFICATIONS_PATH,
        comparison_level=report.comparison_level,
    )

    # The report bundle turns those findings into the review workbook. The HTML
    # version of the workbook is written alongside it.
    bundle_paths: dict[str, Path] = write_performance_comparison_report_bundle(
        findings,
        report.output_directory,
        title=report.title,
        include_workbook=True,
        comparison_path=SPECIFICATIONS_PATH,
        comparison_level=report.comparison_level,
    )
    return bundle_paths["review_workbook"]


if __name__ == "__main__":
    main()
