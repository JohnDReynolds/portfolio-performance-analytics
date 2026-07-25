"""Command-line entry point for PPAR workflows."""

from __future__ import annotations

import argparse
import sys


class _TopLevelHelpFormatter(argparse.RawTextHelpFormatter):
    """Format top-level help with enough room for long command names."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=42, width=120)


def main(argv: list[str] | None = None) -> int:
    """Run the top-level ``ppar`` command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code from the selected subcommand.
    """
    effective_argv = sys.argv[1:] if argv is None else argv
    if not effective_argv:
        print(_top_level_onboarding())
        return 0
    if _is_top_level_help_request(effective_argv):
        print(_top_level_help())
        return 0

    command = effective_argv[0]
    remaining_args = effective_argv[1:]
    if command == "analytics":
        from ppar.analytics import cli as _analytics

        return _analytics.main(remaining_args)
    if command == "setup":
        from ppar.audit.cli import setup as _setup

        return _setup.main(remaining_args)
    if command == "audit":
        from ppar.audit.cli import site_report as _site_report

        return _site_report.main(remaining_args, prog="ppar audit")

    _argument_parser().parse_args(effective_argv)
    raise AssertionError(f"Unsupported command: {command}")


def _is_top_level_help_request(argv: list[str]) -> bool:
    """Return whether the user asked for top-level help."""
    return len(argv) == 1 and argv[0] in {"-h", "--help"}


def _top_level_help() -> str:
    """Return Audit-focused help for the top-level command."""
    return (
        "usage: ppar <command> [options]\n"
        "\n"
        "commands:\n"
        "  setup                   Create a local PPAR Audit workspace.\n"
        "  audit                   "
        "Write Audit reports from a configured workspace.\n"
        "\n"
        "options:\n"
        "  -h, --help              Show this help message and exit.\n"
        "\n"
        "Examples:\n"
        "  ppar setup ./my_ppar_audit\n"
        "  ppar audit ./my_ppar_audit"
    )


def _top_level_onboarding() -> str:
    """Return the Audit first-run handoff for users who type ``ppar``."""
    return (
        "PPAR Audit explains why reported portfolio performance changed.\n"
        "\n"
        "First-time setup:\n"
        "  ppar setup ./my_ppar_audit\n"
        "\n"
        "Then run:\n"
        "  ppar audit ./my_ppar_audit\n"
        "\n"
        "For command help:\n"
        "  ppar -h"
    )


def _argument_parser() -> argparse.ArgumentParser:
    """Return the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="ppar",
        usage="ppar <command> [options]",
        description=None,
        epilog=(
            "Examples:\n"
            "  ppar setup ./my_ppar_audit\n"
            "  ppar audit ./my_ppar_audit"
        ),
        formatter_class=_TopLevelHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
        required=True,
    )
    subparsers.add_parser(
        "setup",
        help="Create a PPAR Audit workspace.",
    )
    subparsers.add_parser(
        "audit",
        help="Write Audit reports from a configured workspace.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
