"""Command-line entry point for PPAR workflows."""

from __future__ import annotations

# Python imports
import argparse

# Project imports
from ppar.performance_comparison.cli import setup as _setup
from ppar.performance_comparison.cli import site_report as _site_report


def main(argv: list[str] | None = None) -> int:
    """Run the top-level ``ppar`` command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code from the selected subcommand.
    """
    args, remaining_args = _argument_parser().parse_known_args(argv)
    if args.command == "setup":
        return _setup.main(remaining_args)
    if args.command == "report":
        return _site_report.main(remaining_args)
    raise AssertionError(f"Unsupported command: {args.command}")


def _argument_parser() -> argparse.ArgumentParser:
    """Return the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="ppar",
        description="PPAR command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "setup",
        help="Create ppar.yaml and starter folders for a site.",
    )
    subparsers.add_parser(
        "report",
        help="Write report bundles from a configured site.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
