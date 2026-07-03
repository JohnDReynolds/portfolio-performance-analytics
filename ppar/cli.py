"""Command-line entry point for PPAR workflows."""

from __future__ import annotations

# Python imports
import argparse

# Project imports
from ppar.performance_comparison.cli import quickstart as _quickstart


def main(argv: list[str] | None = None) -> int:
    """Run the top-level ``ppar`` command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code from the selected subcommand.
    """
    args, remaining_args = _argument_parser().parse_known_args(argv)
    if args.command == "quickstart":
        return _quickstart.main(remaining_args)
    raise AssertionError(f"Unsupported command: {args.command}")


def _argument_parser() -> argparse.ArgumentParser:
    """Return the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="ppar",
        description="PPAR command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "quickstart",
        help="Create ppar.yaml and reports from a folder with snapshot_a/snapshot_b.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
