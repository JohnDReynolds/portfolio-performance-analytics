"""Command-line entry point for PPAR workflows."""

from __future__ import annotations

# Python imports
import argparse
import os
from pathlib import Path
import sys

# Project imports
from ppar._chart_console import quiet_matplotlib_startup
from ppar.performance_comparison.cli import setup as _setup
from ppar.performance_comparison.cli import site_report as _site_report


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
    if _is_top_level_help_request(effective_argv):
        print(_top_level_help())
        return 0

    args, remaining_args = _argument_parser().parse_known_args(effective_argv)
    if args.command == "analytics":
        _prime_analytics_cache_environment(remaining_args)
        from ppar.analytics import cli as _analytics

        return _analytics.main(remaining_args)
    if args.command == "setup":
        return _setup.main(remaining_args)
    if args.command in {"performance_comparison", "perfcomp"}:
        return _site_report.main(remaining_args)
    raise AssertionError(f"Unsupported command: {args.command}")


def _is_top_level_help_request(argv: list[str]) -> bool:
    """Return whether the user asked for top-level help."""
    return len(argv) == 1 and argv[0] in {"-h", "--help"}


def _top_level_help() -> str:
    """Return user-facing help for the top-level command."""
    return (
        "usage: ppar <command> [options]\n"
        "\n"
        "commands:\n"
        "  setup                   Create a local PPAR setup folder.\n"
        "  analytics               Write analytics reports from a configured site.\n"
        "  performance_comparison  "
        "Write performance-comparison reports from a configured site.\n"
        "  perfcomp                Alias for performance_comparison.\n"
        "\n"
        "options:\n"
        "  -h, --help              Show this help message and exit.\n"
        "\n"
        "Examples:\n"
        "  ppar setup ./my_ppar_data\n"
        "  ppar analytics ./my_ppar_data/analytics\n"
        "  ppar performance_comparison ./my_ppar_data/performance_comparison"
    )


def _argument_parser() -> argparse.ArgumentParser:
    """Return the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="ppar",
        usage="ppar <command> [options]",
        description=None,
        epilog=(
            "Examples:\n"
            "  ppar setup ./my_ppar_data\n"
            "  ppar analytics ./my_ppar_data/analytics\n"
            "  ppar performance_comparison ./my_ppar_data/performance_comparison"
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
        "analytics",
        help="Write analytics reports from a configured site.",
    )
    subparsers.add_parser(
        "setup",
        help="Create ppar.yaml and starter folders for a site.",
    )
    subparsers.add_parser(
        "performance_comparison",
        help="Write performance-comparison reports from a configured site.",
    )
    subparsers.add_parser(
        "perfcomp",
        help="Alias for performance_comparison.",
    )
    return parser


def _prime_analytics_cache_environment(argv: list[str]) -> None:
    """Set chart-rendering cache paths before importing analytics modules."""
    output_directory = _analytics_output_directory(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_directory / ".cache"))
    quiet_matplotlib_startup()


def _analytics_output_directory(argv: list[str]) -> Path:
    """Infer the analytics output directory from top-level command arguments."""
    output_argument = _option_value(argv, "--output")
    if output_argument is not None:
        return Path(output_argument).expanduser()
    site_directory = _first_positional_argument(argv)
    if site_directory is None:
        return Path.cwd() / "output"
    return Path(site_directory).expanduser() / "output"


def _option_value(argv: list[str], option: str) -> str | None:
    """Return the value after an option if present."""
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _first_positional_argument(argv: list[str]) -> str | None:
    """Return the first argument that is not an option or option value."""
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value in {"--portfolio", "--benchmark", "--frequency", "-f", "--output"}:
            skip_next = True
            continue
        if value.startswith("-"):
            continue
        return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
