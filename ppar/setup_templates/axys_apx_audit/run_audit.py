"""Understand and customize the standard Audit workflow.

The normal command is ``ppar audit ./audit``. This script produces the same
review bundles while showing the comparison and report-writing steps. PPAR
handles command-line parsing and validation so the example can focus on the
auditing workflow.

This script accepts the same command-line options as ``ppar audit``. Run
``python run_audit.py -h`` to view them.
"""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    compare_snapshots,
    write_audit_report_bundle,
)
from ppar.audit import source_loader
from ppar.audit.cli.site_report import (
    is_missing_security_data,
    script_run_settings,
)
from ppar.audit.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)


# All relative paths are anchored to this file, so the script works from any
# current working directory.
SITE_DIRECTORY = Path(__file__).resolve().parent
SPECIFICATIONS_PATH = SITE_DIRECTORY / "ppar.yaml"


@dataclass(frozen=True)
class ReportSpec:
    """Describe one standard report; edit these values to customize output."""

    comparison_level: str
    title: str


REPORTS = (
    ReportSpec(PORTFOLIO_COMPARISON_LEVEL, "Portfolio Audit Report"),
    ReportSpec(SECURITY_COMPARISON_LEVEL, "Security Audit Report"),
)


@source_loader.source_frame_cache()
def main(argv: list[str] | None = None) -> int:
    """Run the visible Python equivalent of ``ppar audit``."""
    # ---------------------------------------------------------------------
    # 1. Resolve this run's presentation and validation choices
    # ---------------------------------------------------------------------
    # Source files, financial policy, Data Issues checks, and normal run choices
    # remain in ppar.yaml where they are reviewable and reproducible. CLI-style
    # options override matching run choices only for this invocation.
    settings = script_run_settings(SITE_DIRECTORY, argv)

    # ---------------------------------------------------------------------
    # 2. Compare the two configured source-data snapshots
    # ---------------------------------------------------------------------
    # compare_snapshots() is the main calculation boundary. It reads ppar.yaml,
    # loads Snapshot A and Snapshot B, applies comparison/audit policy, and
    # returns one normalized findings table for the selected review level.
    review_paths: list[Path] = []
    security_skipped = False
    for report in REPORTS:
        try:
            findings: Any = compare_snapshots(
                SPECIFICATIONS_PATH,
                include_suppressed=not settings.exclude_suppressed,
                require_causal_attribution=settings.require_causal_attribution,
                comparison_level=report.comparison_level,
            )

            # -------------------------------------------------------------
            # 3. Turn findings into the portable review bundle
            # -------------------------------------------------------------
            # This writes the level-specific audit workbook, optional HTML,
            # supporting CSVs, a manifest, and bundle guidance. Edit the call
            # below when building a custom reporting or downstream-review workflow.
            bundle_paths = write_audit_report_bundle(
                findings,
                settings.output_directory / report.comparison_level,
                title=settings.title or report.title,
                include_workbook=settings.include_workbook,
                include_html_output=settings.include_html_output,
                require_complete_yaml_setup=not settings.allow_incomplete_yaml,
                require_causal_attribution=settings.require_causal_attribution,
                comparison_path=SPECIFICATIONS_PATH,
                comparison_level=report.comparison_level,
                include_reconstruction_diagnostics=(
                    settings.include_reconstruction_diagnostics
                ),
                expand_all_supporting_files=settings.expand_all_supporting_files,
            )
            presentation_artifacts = tuple(
                artifact_name
                for artifact_name in ("review_workbook", "html_report")
                if artifact_name in bundle_paths
            )
            primary_artifacts = presentation_artifacts or (
                "performance_differences",
                "performance_difference_causes",
                "data_issues",
            )
            for artifact_name in primary_artifacts:
                review_path = bundle_paths.get(artifact_name)
                if review_path is not None:
                    review_paths.append(review_path)
        except PpaError as error:
            # The standard run remains useful for portfolio-only sites.
            if (
                report.comparison_level == SECURITY_COMPARISON_LEVEL
                and is_missing_security_data(error)
            ):
                security_skipped = True
                continue
            print(f"Report failed: {error}", file=sys.stderr)
            return 1

    # ---------------------------------------------------------------------
    # 4. Hand the primary review files back to the user
    # ---------------------------------------------------------------------
    print("Open these files to review Audit output:")
    for review_path in review_paths:
        print(f"  {review_path}")
    if security_skipped:
        print()
        print("Security output skipped because files.security_performance is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
