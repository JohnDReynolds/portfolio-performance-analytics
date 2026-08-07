"""Run and customize Audit from Python.

The normal command is ``ppar audit ./my_ppar_audit``. This script demonstrates
the equivalent Python integration point for users who want to schedule Audit,
wrap it in another process, or customize one run in code.

Source mappings, financial policy, and repeatable report settings remain in
``ppar.yaml``. Add only run-specific Python overrides to the ``run_report()``
call below.
"""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys

# Project imports
from ppar.errors import PpaError
from ppar.audit.cli.site_report import run_report


# All relative paths are anchored to this file, so the script works from any
# current working directory.
SITE_DIRECTORY = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    """Run the customizable Python Audit example."""
    _argument_parser().parse_args(argv)
    try:
        result = run_report(
            SITE_DIRECTORY,
            # Optional one-run customization examples:
            # output_directory=SITE_DIRECTORY / "custom_output",
            # title="My Audit Review",
            # top_evidence_limit=15,
            # include_workbook=True,
            # include_html_output=True,
        )
    except PpaError as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1

    print("Open these files to review Audit output:")
    for review_path in result["review_paths"]:
        print(f"  {review_path}")
    if "security_status" in result:
        print()
        print(
            "Security output skipped because "
            "files.security_performance is not available."
        )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Return help for the customizable Python example."""
    return argparse.ArgumentParser(
        prog="python run_audit.py",
        description=(
            "Run Audit through Python. Edit the run_report() call in this "
            "script to customize one run."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
